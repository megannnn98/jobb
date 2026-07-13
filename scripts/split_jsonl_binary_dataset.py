#!/usr/bin/env python3
"""Build a clean binary positive/negative train/val/test from aj_reviews_export.jsonl.

Same source and stratified split as split_jsonl_sentiment_dataset.py, but the noisy
manual_review class (status==0 & isPositive==0) is dropped entirely: this trains the
model only on cleanly-labelled sentiment. Uncertain reviews are routed to manual_review
downstream by the threshold policy in scripts/predict_sentiment.py, not by the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SEED = 20260630
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"


def map_label(row: dict) -> str | None:
    """status wins over isPositive; manual_review rows (neither) are dropped (None)."""
    if row.get("status") == 1:
        return LABEL_NEGATIVE
    if row.get("isPositive") == 1:
        return LABEL_POSITIVE
    return None


def load_reviews(input_path: Path) -> list[dict[str, str]]:
    reviews: list[dict[str, str]] = []
    skipped_empty_text = 0
    skipped_dropped_label = 0

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = map_label(row)
            if label is None:
                skipped_dropped_label += 1
                continue
            text = (row.get("descr") or "").strip()
            if not text:
                skipped_empty_text += 1
                continue

            reviews.append(
                {
                    "id": str(row.get("id", "")),
                    "name": (row.get("name") or "").strip(),
                    "text": text,
                    "label": label,
                }
            )

    print(f"loaded {len(reviews)} reviews from {input_path}")
    print(f"skipped_empty_text: {skipped_empty_text}")
    print(f"skipped_dropped_label (manual_review): {skipped_dropped_label}")
    print(f"label distribution: {dict(Counter(row['label'] for row in reviews))}")
    return reviews


def stratified_split(
    reviews: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reviews:
        groups[row["label"]].append(row)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for label, rows in groups.items():
        rng.shuffle(rows)
        count = len(rows)
        train_count = int(count * TRAIN_RATIO)
        val_count = int(count * VAL_RATIO)
        train.extend(rows[:train_count])
        val.extend(rows[train_count : train_count + val_count])
        test.extend(rows[train_count + val_count :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "text", "label"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_split_counts(name: str, rows: list[dict[str, str]]) -> None:
    print(f"{name}: {len(rows)} rows {dict(Counter(row['label'] for row in rows))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("aj_reviews_export.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviews = load_reviews(args.input)
    train, val, test = stratified_split(reviews, args.seed)

    write_tsv(args.output_dir / "binary_train.tsv", train)
    write_tsv(args.output_dir / "binary_val.tsv", val)
    write_tsv(args.output_dir / "binary_test.tsv", test)

    print_split_counts("train", train)
    print_split_counts("val", val)
    print_split_counts("test", test)


if __name__ == "__main__":
    main()
