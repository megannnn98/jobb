#!/usr/bin/env python3
"""Add conservative heuristic labels to the prepared review sample."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


DEFAULT_SEED = 20260612
RANDOM_AUDIT_RATE = 0.02

NEGATIVE_PATTERNS = [
    r"\bне\s+рекомендую\b",
    r"\bне\s+советую\b",
    r"\bне\s+стоит\s+(идти|устраиваться|работать|тратить)",
    r"\bобходите\b",
    r"\bбегите\b",
    r"\bкидалов",
    r"\bобман",
    r"\bмошен",
    r"\bшараг",
    r"\bне\s+тратьте\b",
    r"\bне\s+верьте\b",
    r"\bне\s+идите\b",
    r"\bне\s+плат",
    r"\bне\s+выплат",
    r"\bзадерж\w*",
    r"\bзадерж\w*\s+зарплат",
    r"\bчерн\w+\s+зарплат",
    r"\bсер\w+\s+зарплат",
    r"\bштраф",
    r"\bтекучк",
    r"\bхамств",
    r"\bужас",
    r"\bкошмар",
    r"\bрабств",
    r"\bад\b",
    r"\bсуд\b",
    r"\bпрокуратур",
    r"\bувол",
    r"\bжесток",
    r"\bугрож",
    r"\bлиш\w+\s+зарплат",
    r"\bотрицательн\w+\s+характеристик",
    r"\bнерадив\w+\s+работодател",
    r"\bобещан\w*\s+не\s+выполня",
    r"\bне\s+выполня\w+\s+обещан",
]

POSITIVE_PATTERNS = [
    r"\bрекомендую\s+(работодател|компани|работ)",
    r"\bсоветую\s+(работодател|компани|работ)",
    r"\bхорош\w+\s+(работодатель|компания|место\s+работы|коллектив|руководство)",
    r"\bотличн\w+\s+(работодатель|компания|место\s+работы|коллектив|руководство)",
    r"\bдостойн\w+\s+(зарплат|оплат)",
    r"\bзарплат\w*\s+(вовремя|своевременно|без\s+задерж)",
    r"\bбел\w+\s+зарплат",
    r"\bдружн\w+\s+коллектив",
    r"\bофициальн\w+\s+трудоустройств",
    r"\bсоц\w*\s+пакет",
]

VACANCY_PATTERNS = [
    r"\bтребуется\b",
    r"\bваканси",
    r"\bобязанности\b",
    r"\bтребования\b",
    r"\bусловия\s+работы\b",
    r"\bграфик\s+работы\b",
    r"\bзарплата\s+от\b",
    r"\bрезюме\b",
]

NOISE_PATTERNS = [
    r"^\s*\*{3,}",
    r"\bhttp[s]?://",
    r"\bwww\.",
    r"\be-?mail\b",
]


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


NEGATIVE_RX = compile_patterns(NEGATIVE_PATTERNS)
POSITIVE_RX = compile_patterns(POSITIVE_PATTERNS)
VACANCY_RX = compile_patterns(VACANCY_PATTERNS)
NOISE_RX = compile_patterns(NOISE_PATTERNS)


def count_matches(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def classify(row: dict[str, str]) -> tuple[str, str, str, str]:
    text = normalize_text(row["descr"]).lower()
    negative_score = count_matches(NEGATIVE_RX, text)
    positive_score = count_matches(POSITIVE_RX, text)
    vacancy_score = count_matches(VACANCY_RX, text)
    noise_score = count_matches(NOISE_RX, text)

    if len(text) < 80:
        return "manual_review", "0.55", "too_little_context", "too_little_context"

    if negative_score >= 2:
        if positive_score > 0:
            return "negative_about_work", "0.82", "strong_negative_mixed_with_positive", "mixed_sentiment"
        return "negative_about_work", "0.94", "multiple_negative_work_signals", ""

    if negative_score == 1 and vacancy_score <= 1:
        if positive_score > 0:
            return "manual_review", "0.62", "mixed_positive_negative_signals", "mixed_sentiment"
        return "negative_about_work", "0.86", "negative_work_signal", "low_confidence"

    if vacancy_score >= 2 and negative_score == 0 and positive_score == 0:
        return "manual_review", "0.76", "vacancy_or_job_posting", "not_review"

    if noise_score >= 2 and negative_score == 0 and positive_score == 0:
        return "manual_review", "0.68", "parser_or_repost_noise", "parser_noise"

    if positive_score >= 2 and negative_score == 0:
        return "manual_review", "0.72", "positive_candidate_needs_review", "potential_positive"

    if positive_score == 1 and negative_score == 0 and vacancy_score == 0:
        return "manual_review", "0.66", "single_positive_signal", "low_confidence"

    return "manual_review", "0.60", "insufficient_clear_sentiment", "low_confidence"


def append_audit_reason(existing_reason: str, extra_reason: str) -> str:
    if not existing_reason:
        return extra_reason
    if extra_reason in existing_reason.split(","):
        return existing_reason
    return f"{existing_reason},{extra_reason}"


def label_rows(
    rows: list[dict[str, str]], random_audit_rate: float, seed: int
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    labeled_rows: list[dict[str, str]] = []

    for row in rows:
        label, confidence, short_reason, audit_reason = classify(row)
        label_source = "heuristic"
        needs_audit = "1" if label == "manual_review" or audit_reason else "0"

        if row.get("isPositive") == "1":
            label = "positive_about_work"
            confidence = "0.80"
            short_reason = "source_positive_flag"
            label_source = "imported"
            needs_audit = "1"
            audit_reason = append_audit_reason(audit_reason, "source_column_positive")

        if needs_audit == "0" and rng.random() < random_audit_rate:
            needs_audit = "1"
            audit_reason = append_audit_reason(audit_reason, "random_audit")

        labeled = dict(row)
        labeled["label"] = label
        labeled["confidence"] = confidence
        labeled["short_reason"] = short_reason
        labeled["label_source"] = label_source
        labeled["needs_audit"] = needs_audit
        labeled["audit_reason"] = audit_reason
        labeled_rows.append(labeled)

    return labeled_rows


def read_rows(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in {input_path}")
        return reader.fieldnames, list(reader)


def write_rows(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/labeling_sample_20000.tsv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labeled_reviews_20000_heuristic.tsv"),
    )
    parser.add_argument("--audit-output", type=Path, default=Path("data/audit_reviews.tsv"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--random-audit-rate", type=float, default=RANDOM_AUDIT_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_rows(args.input)
    labeled_rows = label_rows(rows, args.random_audit_rate, args.seed)
    audit_rows = [row for row in labeled_rows if row["needs_audit"] == "1"]

    write_rows(args.output, fieldnames, labeled_rows)
    write_rows(args.audit_output, fieldnames, audit_rows)

    counts: dict[str, int] = {}
    for row in labeled_rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"wrote {len(labeled_rows)} rows to {args.output}")
    print(f"wrote {len(audit_rows)} audit rows to {args.audit_output}")
    print("label counts:", counts)


if __name__ == "__main__":
    main()
