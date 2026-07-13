#!/usr/bin/env python3
"""Prepare a deterministic review-labeling sample from the raw TSV export."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED = 20260612
MIN_DESCR_CHARS = 40


@dataclass(frozen=True)
class ReviewRow:
    id: str
    name: str
    descr: str
    status: str
    isfunny: str
    is_self_delete: str
    is_about_work: str
    isPositive: str


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\r", "\n").split())


def is_valid_candidate(row: dict[str, str]) -> bool:
    if row.get("is_about_work") != "1":
        return False

    descr = normalize_text(row.get("descr") or "")
    if len(descr) < MIN_DESCR_CHARS:
        return False

    row_id = (row.get("id") or "").strip()
    if not row_id.isdigit():
        return False

    return True


def collect_candidates(input_path: Path) -> list[ReviewRow]:
    candidates: list[ReviewRow] = []

    with input_path.open(newline="", encoding="utf-8", errors="replace") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t", quotechar='"')
        for row in reader:
            if not is_valid_candidate(row):
                continue

            candidates.append(
                ReviewRow(
                    id=(row.get("id") or "").strip(),
                    name=normalize_text(row.get("name") or ""),
                    descr=normalize_text(row.get("descr") or ""),
                    status=(row.get("status") or "").strip(),
                    isfunny=(row.get("isfunny") or "").strip(),
                    is_self_delete=(row.get("is_self_delete") or "").strip(),
                    is_about_work=(row.get("is_about_work") or "").strip(),
                    isPositive=(row.get("isPositive") or "").strip(),
                )
            )

    return candidates


def write_sample(output_path: Path, sample: list[ReviewRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "name",
        "descr",
        "status",
        "isfunny",
        "is_self_delete",
        "is_about_work",
        "isPositive",
        "label",
        "confidence",
        "short_reason",
        "label_source",
        "needs_audit",
        "audit_reason",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "id": row.id,
                    "name": row.name,
                    "descr": row.descr,
                    "status": row.status,
                    "isfunny": row.isfunny,
                    "is_self_delete": row.is_self_delete,
                    "is_about_work": row.is_about_work,
                    "isPositive": row.isPositive,
                    "label": "",
                    "confidence": "",
                    "short_reason": "",
                    "label_source": "",
                    "needs_audit": "",
                    "audit_reason": "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("aj_reviews_export.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/labeling_sample_20000.tsv"))
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = collect_candidates(args.input)

    if len(candidates) < args.sample_size:
        raise SystemExit(
            f"Not enough valid candidates: need {args.sample_size}, got {len(candidates)}"
        )

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, args.sample_size)
    sample.sort(key=lambda row: int(row.id))

    write_sample(args.output, sample)
    print(
        f"wrote {len(sample)} rows to {args.output} "
        f"from {len(candidates)} valid is_about_work=1 candidates; seed={args.seed}"
    )


if __name__ == "__main__":
    main()
