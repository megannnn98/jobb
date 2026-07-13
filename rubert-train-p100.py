"""Fast two-phase RuBERT fine-tuning on GPU for sentiment classification.

Uses pre-split TSV datasets generated from aj_reviews_export.jsonl.
Upload sentiment_train.tsv, sentiment_val.tsv, sentiment_test.tsv to Kaggle
alongside this script and the model weights directory.

Phase 1: freeze all BERT layers, train classifier head only.
Phase 2: unfreeze last 2 encoder layers, fine-tune with smaller LR.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Fix PyTorch for P100 GPU (SM60, Pascal) — need >= 2.6 for transformers
import subprocess, sys as _sys
subprocess.run(
    [_sys.executable, "-m", "pip", "install", "-q",
     "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
     "--index-url", "https://download.pytorch.org/whl/cu124"],
    check=True)
print("PyTorch 2.6.0+cu124 ready")

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

MODEL_NAME = "DeepPavlov/rubert-base-cased"
TRAIN_FILE = Path("/kaggle/input/datasets/megannnn98/rubert-job-reviews-sentiment/sentiment_train.tsv")
VAL_FILE = Path("/kaggle/input/datasets/megannnn98/rubert-job-reviews-sentiment/sentiment_val.tsv")
OUTPUT_DIR = Path("/kaggle/working/rubert-sentiment")
TOKENIZED_DIR = OUTPUT_DIR / "tokenized"
TOKENIZED_METADATA_FILE = TOKENIZED_DIR / "metadata.json"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
EXPECTED_COLUMNS = ["id", "name", TEXT_COLUMN, LABEL_COLUMN]

LABEL2ID = {"positive": 0, "negative": 1, "manual_review": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NUM_LABELS = len(LABEL2ID)
MAX_LENGTH = 128
MIN_TEXT_LENGTH = 5
TOKENIZE_BATCH_SIZE = 1000
CACHE_SCHEMA_VERSION = 2

MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))
SAMPLE_SEED = 42

PHASE1_BATCH_SIZE = 64
PHASE1_EPOCHS = 1
PHASE1_LR = 5e-4

PHASE2_BATCH_SIZE = 16
PHASE2_GRADIENT_ACCUMULATION = 2
PHASE2_EPOCHS = 2
PHASE2_LR = 2e-5


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
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def freeze_all_bert(model) -> None:
    for param in model.base_model.parameters():
        param.requires_grad = False


def unfreeze_last_bert_layers(model, n_layers: int) -> None:
    freeze_all_bert(model)
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


def make_training_args(
    output_dir: Path,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int = 1,
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
        logging_steps=100,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
    )


def make_trainer(
    model,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    data_collator: DataCollatorWithPadding,
    class_weights: torch.Tensor,
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
    )


def maybe_limit_dataset(dataset: Dataset, max_samples: int, split_name: str) -> Dataset:
    if max_samples <= 0:
        return dataset
    selected = min(max_samples, len(dataset))
    print(f"{split_name}: using {selected} / {len(dataset)} samples")
    return dataset.shuffle(seed=SAMPLE_SEED).select(range(selected))


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, Capability: {torch.cuda.get_device_capability(0)}")
        t = torch.tensor([1.0]).cuda()
        print(f"CUDA test: OK")

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

    # Phase 1: classifier head only
    freeze_all_bert(model)
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
    )

    phase1_trainer.train()
    phase1_trainer.save_model(str(OUTPUT_DIR / "phase1_final"))

    # Phase 2: unfreeze last 2 layers
    unfreeze_last_bert_layers(model, n_layers=2)
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
        ),
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        class_weights,
    )

    phase2_trainer.train()
    phase2_trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))


if __name__ == "__main__":
    main()
