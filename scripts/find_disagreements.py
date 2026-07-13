#!/usr/bin/env python3
"""Find every row where the sentiment model's raw argmax disagrees with map_label(row).

Runs batched inference over aj_reviews_export.jsonl (no threshold/policy fallback — raw
argmax, so the output reflects the model's actual belief, not a production routing
decision) and writes disagreements to a TSV, tagged with train/val/test split membership.

WARNING: ~80% of rows were used to train the model (data/sentiment_train.tsv). A
disagreement on a train-split row means the model still disputes a label it was
explicitly trained to reproduce — a strong label-noise signal. A disagreement on a
test-split row is an honest generalization error. Don't average the two together when
judging "how good is the model" (see split column / --split-only-test).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def map_label(row: dict) -> str:
    """Same mapping as split_jsonl_sentiment_dataset.py (status wins over isPositive)."""
    if row.get("status") == 1:
        return "negative"
    if row.get("isPositive") == 1:
        return "positive"
    return "manual_review"


def load_split_map(data_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"sentiment_{split}.tsv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                split_map[row["id"]] = split
    return split_map


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


def load_reviews(jsonl_path: Path) -> list[dict]:
    reviews = []
    skipped = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = (row.get("descr") or "").strip()
            if not text:
                skipped += 1
                continue
            reviews.append({"id": str(row.get("id", "")), "text": text, "expected": map_label(row)})
    print(f"loaded {len(reviews)} reviews, skipped {skipped} empty descr", file=sys.stderr)
    return reviews


def predict_batch(model, tokenizer, texts: list[str], id2label: dict[int, str], max_length: int) -> list[dict]:
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("aj_reviews_export.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("model/ruroberta-sentiment"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/disagreements.tsv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="only process first N reviews (smoke test)")
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    reviews = load_reviews(args.jsonl)
    if args.limit:
        reviews = reviews[: args.limit]
    split_map = load_split_map(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    disagreements = []
    split_totals = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    split_disagree = {"train": 0, "val": 0, "test": 0, "unknown": 0}

    total = len(reviews)
    start = time.monotonic()
    for batch_start in range(0, total, args.batch_size):
        batch = reviews[batch_start : batch_start + args.batch_size]
        preds = predict_batch(model, tokenizer, [r["text"] for r in batch], id2label, args.max_length)
        for row, pred in zip(batch, preds):
            split = split_map.get(row["id"], "unknown")
            split_totals[split] += 1
            if pred["predicted"] == row["expected"]:
                continue
            split_disagree[split] += 1
            disagreements.append(
                {
                    "id": row["id"],
                    "split": split,
                    "expected": row["expected"],
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
            file=sys.stderr,
            flush=True,
        )
    print(file=sys.stderr)

    disagreements.sort(key=lambda r: -r["confidence"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(disagreements)

    print(f"\nwrote {len(disagreements)} disagreements to {args.output}")
    print("\nby split (disagreements / total, error rate):")
    for split in ("train", "val", "test", "unknown"):
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
