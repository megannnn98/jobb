"""Fast two-phase ruRoberta-large fine-tuning for sentiment classification.

V2 improvements over train_fast_bert.py:
  - 4 unfrozen encoder layers (vs 2)
  - warmup + cosine LR schedule
  - weight_decay
  - label_smoothing
  - early stopping

Works both locally (for quick tests with MAX_TRAIN_SAMPLES) and on Kaggle.
On Kaggle: auto-detects, reinstalls compatible PyTorch, uses Kaggle dataset paths.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Kaggle's "GPU T4 x2" accelerator exposes 2 physical GPUs; without this, HF
# Trainer auto-wraps the model in nn.DataParallel, whose per-step scatter/gather
# overhead (small per-device batch) makes training slower than a single GPU.
# Must be set before the first `import torch` — CUDA context init reads it once.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

IS_KAGGLE = os.path.exists("/kaggle/working")

if IS_KAGGLE:
    # Kaggle's default image ships torch 2.10.0+cu128, which has no compute
    # kernels for P100 (SM60/Pascal) — training ops fail with "no kernel image
    # is available for execution on the device" even though basic tensor
    # creation on GPU (`torch.tensor([1.0]).cuda()`) misleadingly succeeds.
    # This MUST run before the first `import torch` anywhere in this process:
    # once torch's compiled C extension is loaded, sys.modules caches it and a
    # later reinstall + re-`import torch` is a no-op (the .so stays loaded).
    print("Kaggle environment detected; reinstalling PyTorch for P100 (SM60) compatibility...")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
            "--index-url", "https://download.pytorch.org/whl/cu124",
            "--extra-index-url", "https://pypi.org/simple",
        ],
        check=True,
    )
    print("PyTorch 2.6.0+cu124 ready")

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# torch.utils.serialization was removed in PyTorch >= 2.6, but ruRoberta-large
# weights on HF were saved with an older PyTorch and reference the module in
# pickled tensors. Provide a compat shim as a proper package so unpickling works.
if IS_KAGGLE:
    import mmap as _mmap
    import types as _types

    def _shim_getattr(name: str):
        return getattr(torch.serialization, name)

    _serialization_shim = _types.ModuleType("torch.utils.serialization")
    _serialization_shim.__path__ = []
    _serialization_shim.__package__ = "torch.utils.serialization"
    _serialization_shim.__file__ = torch.serialization.__file__
    _serialization_shim.__getattr__ = _shim_getattr  # type: ignore[attr-defined]
    for _attr in dir(torch.serialization):
        if not _attr.startswith("_"):
            setattr(_serialization_shim, _attr, getattr(torch.serialization, _attr))
    sys.modules["torch.utils.serialization"] = _serialization_shim

    # torch/serialization.py (built into PyTorch 2.10, not editable) does
    # `from torch.utils.serialization import config` internally in load()/save()
    # to read tunables (config.load.mmap, config.load.calculate_storage_offsets,
    # config.save.compute_crc32, etc — an evolving, undocumented list). Rather
    # than enumerate every attribute PyTorch might touch, fall back to a safe
    # default (False) for anything not explicitly known, via __getattr__.
    class _ConfigAttrFallback:
        def __getattr__(self, name: str):
            return False

    class _LoadConfig(_ConfigAttrFallback):
        endianness = torch.serialization.LoadEndianness.NATIVE
        mmap_flags = _mmap.MAP_PRIVATE
        mmap = False
        calculate_storage_offsets = False

    class _SaveConfig(_ConfigAttrFallback):
        compute_crc32 = True
        storage_alignment = 64
        use_pinned_memory_for_d2h = False

    _config_shim = _types.ModuleType("torch.utils.serialization.config")
    _config_shim.load = _LoadConfig()
    _config_shim.save = _SaveConfig()
    _serialization_shim.config = _config_shim
    sys.modules["torch.utils.serialization.config"] = _config_shim

    print("torch.utils.serialization compat shim installed (with config submodule)")

    # Optional: use a Kaggle Secret named HF_TOKEN to avoid HF Hub rate limits.
    # Model is public, so training still works without it.
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secret")
    except Exception as _e:
        print(f"No HF_TOKEN secret available ({_e}); using unauthenticated HF Hub access")

KAGGLE_USER = os.environ.get("KAGGLE_USER", "megannnn98")
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "rubert-job-reviews-sentiment-5class")

MODEL_NAME = "ai-forever/ruRoberta-large"

if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/ruroberta-sentiment-5class-len192")

    def _find_tsv(filename: str) -> Path:
        """Search for a TSV file in Kaggle input — path differs by dataset type."""
        candidates = [
            Path(f"/kaggle/input/datasets/{KAGGLE_USER}/{KAGGLE_DATASET}/{filename}"),
        ]
        # Also search for the dataset in /kaggle/input/ (generic Kaggle mount)
        for root in Path("/kaggle/input").glob(f"**/{filename}"):
            if root not in candidates:
                candidates.append(root)
        candidates.append(Path("/kaggle/input") / KAGGLE_DATASET / filename)
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not find {filename} in /kaggle/input/. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    TRAIN_FILE = _find_tsv("sentiment_train.tsv")
    VAL_FILE = _find_tsv("sentiment_val.tsv")
    print(f"Dataset: train={TRAIN_FILE}, val={VAL_FILE}")
else:
    TRAIN_FILE = Path("data/sentiment5_train.tsv")
    VAL_FILE = Path("data/sentiment5_val.tsv")
    OUTPUT_DIR = Path("outputs/ruroberta-sentiment-5class-len192")

TOKENIZED_DIR = OUTPUT_DIR / "tokenized"
TOKENIZED_METADATA_FILE = TOKENIZED_DIR / "metadata.json"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
EXPECTED_COLUMNS = ["id", "name", TEXT_COLUMN, LABEL_COLUMN]

LABEL2ID = {"positive": 0, "negative": 1, "manual_review": 2, "spam": 3, "service_complaint": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NUM_LABELS = len(LABEL2ID)
MAX_LENGTH = 192
MIN_TEXT_LENGTH = 5
TOKENIZE_BATCH_SIZE = 1000
CACHE_SCHEMA_VERSION = 3

MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))
SAMPLE_SEED = 42

# —— ruRoberta-large (355M) uses smaller batches than RuBERT-base (178M) ——
PHASE1_BATCH_SIZE = 32
PHASE1_EPOCHS = 1
PHASE1_LR = 5e-4

PHASE2_BATCH_SIZE = 8
PHASE2_GRADIENT_ACCUMULATION = 4  # effective batch = 8 * 4 = 32
PHASE2_EPOCHS = 4
PHASE2_LR = 2e-5

# V2 improvements
UNFREEZE_LAYERS = 4          # was 2 in baseline
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 3


def file_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def tokenizer_fingerprints() -> list[dict[str, int | str]]:
    model_path = Path(MODEL_NAME)
    tokenizer_files = [
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
    ]
    return [file_fingerprint(path) for path in tokenizer_files if path.exists()]


def expected_cache_metadata() -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "min_text_length": MIN_TEXT_LENGTH,
        "label2id": LABEL2ID,
        "tokenizer_files": tokenizer_fingerprints(),
        "splits": {
            "train": file_fingerprint(TRAIN_FILE),
            "validation": file_fingerprint(VAL_FILE),
        },
    }


def read_cache_metadata() -> dict | None:
    if not TOKENIZED_METADATA_FILE.exists():
        return None
    with TOKENIZED_METADATA_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def write_cache_metadata(metadata: dict) -> None:
    with TOKENIZED_METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_tsv(path: Path, skip_short_text: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    repaired_tabs = 0
    skipped_short_text = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != EXPECTED_COLUMNS:
            raise ValueError(f"{path}: expected TSV header {EXPECTED_COLUMNS}, got {header}")

        for line_number, row in enumerate(reader, start=2):
            if len(row) < len(EXPECTED_COLUMNS):
                raise ValueError(f"{path}:{line_number}: expected 4 TSV fields, got {len(row)}")

            if len(row) > len(EXPECTED_COLUMNS):
                row = [row[0], row[1], "\t".join(row[2:-1]), row[-1]]
                repaired_tabs += 1

            review_id, name, text, label = row
            text = (text or "").strip()

            if label not in LABEL2ID:
                raise ValueError(f"{path}:{line_number}: unknown label {label!r}")

            if skip_short_text and len(text) <= MIN_TEXT_LENGTH:
                skipped_short_text += 1
                continue

            rows.append(
                {
                    "id": review_id,
                    "name": name,
                    TEXT_COLUMN: text,
                    LABEL_COLUMN: label,
                }
            )

    print(
        f"{path}: loaded={len(rows)} repaired_embedded_tabs={repaired_tabs} "
        f"skipped_short_text={skipped_short_text}"
    )
    return rows


def load_raw_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(parse_tsv(TRAIN_FILE, skip_short_text=True)),
            "validation": Dataset.from_list(parse_tsv(VAL_FILE, skip_short_text=False)),
        }
    )


def tokenize_batch(batch: dict[str, list[str]], tokenizer) -> dict:
    return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)


def build_tokenized_dataset(tokenizer) -> DatasetDict:
    dataset = load_raw_dataset()
    dataset = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=2,
        desc="Tokenizing reviews",
    )
    dataset = dataset.map(
        lambda batch: {"labels": [LABEL2ID[label] for label in batch[LABEL_COLUMN]]},
        batched=True,
        desc="Encoding labels",
    )
    return dataset.remove_columns(["id", "name", TEXT_COLUMN, LABEL_COLUMN])


def load_or_build_tokenized_dataset(tokenizer) -> DatasetDict:
    metadata = expected_cache_metadata()
    if TOKENIZED_DIR.exists() and read_cache_metadata() == metadata:
        print(f"Loading tokenized dataset from {TOKENIZED_DIR}")
        from datasets import load_from_disk
        return load_from_disk(str(TOKENIZED_DIR))

    print("Tokenized cache is missing or stale; rebuilding")
    dataset = build_tokenized_dataset(tokenizer)

    import shutil
    tmp_dir = TOKENIZED_DIR.with_name(f"{TOKENIZED_DIR.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    dataset.save_to_disk(str(tmp_dir))
    if TOKENIZED_DIR.exists():
        shutil.rmtree(TOKENIZED_DIR)
    tmp_dir.rename(TOKENIZED_DIR)
    write_cache_metadata(metadata)
    return dataset


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    counts = Counter(int(label) for label in dataset["labels"])
    total = sum(counts.values())
    weights = [
        total / (len(LABEL2ID) * counts[label_id])
        for label_id in range(len(LABEL2ID))
    ]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=weights, label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


def freeze_all_backbone(model) -> None:
    for param in model.base_model.parameters():
        param.requires_grad = False


def unfreeze_last_layers(model, n_layers: int) -> None:
    freeze_all_backbone(model)
    encoder = model.base_model.encoder
    for layer in encoder.layer[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def print_trainable_params(model) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


class GpuUtilizationCallback(TrainerCallback):
    """Logs GPU utilization/memory via nvidia-smi every `every_n_steps` steps."""

    def __init__(self, every_n_steps: int = 100):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            gpu_util, mem_util, mem_used, mem_total = (
                v.strip() for v in out.stdout.strip().split(",")
            )
            print(
                f"[GPU] step={state.global_step} util={gpu_util}% "
                f"mem_util={mem_util}% mem={mem_used}/{mem_total} MiB"
            )
        except Exception as e:
            print(f"[GPU] nvidia-smi failed: {e}")


def make_training_args(
    output_dir: Path,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int = 1,
    load_best_model: bool = False,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        fp16=True,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=100,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_f1_macro" if load_best_model else None,
        greater_is_better=True if load_best_model else None,
    )


def make_trainer(
    model,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    data_collator: DataCollatorWithPadding,
    class_weights: torch.Tensor,
    callbacks: list | None = None,
) -> Trainer:
    return WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        callbacks=callbacks or [],
    )


def maybe_limit_dataset(dataset: Dataset, max_samples: int, split_name: str) -> Dataset:
    if max_samples <= 0:
        return dataset
    selected = min(max_samples, len(dataset))
    print(f"{split_name}: using {selected} / {len(dataset)} samples")
    return dataset.shuffle(seed=SAMPLE_SEED).select(range(selected))


def log_gpu_info() -> None:
    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"Capability: {torch.cuda.get_device_capability(0)}")
        # `.cuda()` alone only allocates memory and can silently "succeed" even
        # without compute kernels for this GPU arch (SM60/P100 burned us this
        # way before). Run an actual matmul to catch "no kernel image" early.
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        (a @ b).sum().item()
        print("CUDA compute test (matmul): OK")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    log_gpu_info()

    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Phase1: batch={PHASE1_BATCH_SIZE}, epochs={PHASE1_EPOCHS}, lr={PHASE1_LR}")
    print(f"Phase2: batch={PHASE2_BATCH_SIZE}×{PHASE2_GRADIENT_ACCUMULATION}, "
          f"epochs={PHASE2_EPOCHS}, lr={PHASE2_LR}, unfreeze_layers={UNFREEZE_LAYERS}")
    print(f"V2: warmup={WARMUP_RATIO}, weight_decay={WEIGHT_DECAY}, "
          f"label_smoothing={LABEL_SMOOTHING}, early_stopping_patience={EARLY_STOPPING_PATIENCE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_or_build_tokenized_dataset(tokenizer)

    train_dataset = maybe_limit_dataset(dataset["train"], MAX_TRAIN_SAMPLES, "train")
    eval_dataset = maybe_limit_dataset(dataset["validation"], MAX_EVAL_SAMPLES, "validation")

    class_weights = compute_class_weights(train_dataset)
    print(
        "class weights: "
        + ", ".join(f"{ID2LABEL[i]}={class_weights[i]:.3f}" for i in range(len(ID2LABEL)))
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    # —— Phase 1: classifier head only ——
    freeze_all_backbone(model)
    print_trainable_params(model)

    phase1_trainer = make_trainer(
        model,
        make_training_args(
            output_dir=OUTPUT_DIR / "phase1",
            batch_size=PHASE1_BATCH_SIZE,
            eval_batch_size=PHASE1_BATCH_SIZE * 2,
            epochs=PHASE1_EPOCHS,
            learning_rate=PHASE1_LR,
        ),
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        class_weights,
        callbacks=[GpuUtilizationCallback(every_n_steps=100)],
    )

    print("=== Phase 1: classifier head ===")
    phase1_trainer.train()
    phase1_trainer.save_model(str(OUTPUT_DIR / "phase1_final"))
    print(f"Phase 1 done. eval: {phase1_trainer.evaluate()}")

    # —— Phase 2: unfreeze last N layers ——
    unfreeze_last_layers(model, n_layers=UNFREEZE_LAYERS)
    print_trainable_params(model)

    phase2_trainer = make_trainer(
        model,
        make_training_args(
            output_dir=OUTPUT_DIR / "phase2",
            batch_size=PHASE2_BATCH_SIZE,
            eval_batch_size=PHASE2_BATCH_SIZE * 4,
            epochs=PHASE2_EPOCHS,
            learning_rate=PHASE2_LR,
            gradient_accumulation_steps=PHASE2_GRADIENT_ACCUMULATION,
            load_best_model=True,
        ),
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            GpuUtilizationCallback(every_n_steps=100),
        ],
    )

    print("=== Phase 2: unfreeze last layers ===")
    phase2_trainer.train()
    phase2_trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Done. Model saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
