#!/usr/bin/env python3
"""Split JSONL reviews into stratified train/val/test TSV files."""

from __future__ import annotations

import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_SEED = 20260612
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_reviews(input_path: Path) -> list[dict[str, str]]:
    reviews: list[dict[str, str]] = []
    with input_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            row = json.loads(line)
            descr = (row.get("descr") or "").strip()
            if len(descr) < 20:
                continue
            reviews.append(
                {
                    "id": str(row.get("id", "")),
                    "name": (row.get("name") or "").strip(),
                    "text": " ".join(descr.split()),
                    "label": "publish" if row.get("status") == 1 else "reject",
                }
            )
    return reviews


def stratified_split(
    reviews: list[dict[str, str]], seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    import random

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in reviews:
        groups[r["label"]].append(r)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for label, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "name", "text", "label"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("aj_reviews_export.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    reviews = load_reviews(args.input)
    print(f"loaded {len(reviews)} reviews (filtered: descr >= 20 chars)")

    from collections import Counter
    labels = Counter(r["label"] for r in reviews)
    print(f"label distribution: {dict(labels)}")

    train, val, test = stratified_split(reviews, args.seed)

    write_tsv(args.output_dir / "train.tsv", train)
    write_tsv(args.output_dir / "val.tsv", val)
    write_tsv(args.output_dir / "test.tsv", test)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        c = Counter(r["label"] for r in split)
        print(f"{name}: {len(split)} rows — {dict(c)}")


if __name__ == "__main__":
    main()
