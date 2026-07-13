"""Find every row where ruRoberta-large's raw argmax disagrees with the TSV label, on Kaggle GPU.

Same idea as scripts/find_disagreements.py, but the local script's job of turning raw JSONL rows
into (id, text, expected_label) is unnecessary here: the rubert-job-reviews-sentiment dataset's
sentiment_{train,val,test}.tsv already carry a `label` column (map_label() applied at split time),
so this script just loads all three and tags rows by which file they came from — no JSONL, no
map_label reimplementation.

Raw argmax (no threshold/policy fallback), full train+val+test (no --limit — GPU makes this cheap),
sorted by confidence descending (most-confident disagreements first — best material for a manual
audit). Train-split disagreements are label-noise candidates (model trained on that label and still
disputes it); test-split disagreements are honest generalization errors. See split column.
"""

from __future__ import annotations

import csv
import html
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

LABELS = ["positive", "negative", "manual_review"]
MAX_LENGTH = 128
BATCH_SIZE = 32


def _find_file(filename: str) -> Path:
    for root in Path("/kaggle/input").glob(f"**/{filename}"):
        return root
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input/")


def _find_model_dir() -> Path:
    for cfg in Path("/kaggle/input").glob("**/final/config.json"):
        return cfg.parent
    raise FileNotFoundError("Could not find a 'final/config.json' model dir under /kaggle/input/")


def looks_garbled(text: str) -> bool:
    """Heuristic for the old double-encoding mojibake (UTF-8 bytes read as latin-1)."""
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha < 10:
        return False
    cyrillic = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    return (text.count("Ð") + text.count("Ñ")) > 5 and cyrillic / alpha < 0.3


def decode_for_display(text: str, limit: int = 200) -> str:
    """Best-effort un-mojibake for human-readable output only (NOT fed to the model)."""
    shown = text
    try:
        shown = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    shown = html.unescape(shown)
    shown = " ".join(shown.split())
    return shown[:limit] + ("…" if len(shown) > limit else "")


def load_split(path: Path, split: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in rows if (r.get("text") or "").strip() and r.get("label") in LABELS]
    for r in rows:
        r["split"] = split
    return rows


def predict_batch(model, tokenizer, texts: list[str], id2label: dict[int, str], device: str) -> list[dict]:
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    results = []
    for row_probs in probs:
        best_id = int(torch.argmax(row_probs).item())
        results.append(
            {
                "predicted": id2label[best_id],
                "confidence": float(row_probs[best_id].item()),
                "probs": {id2label[i]: float(row_probs[i].item()) for i in range(len(row_probs))},
            }
        )
    return results


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_dir = _find_model_dir()
    print(f"Model dir: {model_dir}")

    rows: list[dict] = []
    split_totals = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        split_rows = load_split(_find_file(f"sentiment_{split}.tsv"), split)
        split_totals[split] = len(split_rows)
        rows.extend(split_rows)
        print(f"Loaded {len(split_rows)} {split} rows")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    disagreements = []
    split_disagree = {"train": 0, "val": 0, "test": 0}

    total = len(rows)
    start = time.monotonic()
    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        preds = predict_batch(model, tokenizer, [r["text"] for r in batch], id2label, device)
        for row, pred in zip(batch, preds):
            if pred["predicted"] == row["label"]:
                continue
            split_disagree[row["split"]] += 1
            disagreements.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "expected": row["label"],
                    "predicted": pred["predicted"],
                    "confidence": round(pred["confidence"], 4),
                    "positive_prob": round(pred["probs"].get("positive", 0.0), 4),
                    "negative_prob": round(pred["probs"].get("negative", 0.0), 4),
                    "manual_review_prob": round(pred["probs"].get("manual_review", 0.0), 4),
                    "garbled_text": looks_garbled(row["text"]),
                    "text": decode_for_display(row["text"]),
                }
            )
        done = batch_start + len(batch)
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
        print(
            f"\r{done}/{total} rows | {rate:.1f} rows/s | ETA {eta_min:.1f} min | "
            f"{len(disagreements)} disagreements so far",
            end="",
            flush=True,
        )
    print()

    disagreements.sort(key=lambda r: -r["confidence"])

    out_path = Path("/kaggle/working/disagreements.tsv")
    fieldnames = [
        "id",
        "split",
        "expected",
        "predicted",
        "confidence",
        "positive_prob",
        "negative_prob",
        "manual_review_prob",
        "garbled_text",
        "text",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(disagreements)

    print(f"\nwrote {len(disagreements)} disagreements to {out_path}")
    print("\nby split (disagreements / total, error rate):")
    for split in ("train", "val", "test"):
        t = split_totals[split]
        d = split_disagree[split]
        if t == 0:
            continue
        print(f"  {split:10s} {d:6d} / {t:6d} = {d / t:6.2%}")
    print(
        "\nNOTE: train-split disagreements are label-noise candidates (model was trained on "
        "these labels and still disputes them). test-split disagreements are honest "
        "generalization errors — that's the number that matters for 'how good is the model'."
    )


if __name__ == "__main__":
    main()
