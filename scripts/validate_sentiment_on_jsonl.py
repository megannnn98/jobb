#!/usr/bin/env python3
"""Validate the sentiment model against real reviews sampled from aj_reviews_export.jsonl.

Loads reviews exactly like split_jsonl_sentiment_dataset.py (raw ``descr`` text, same
``map_label`` logic), samples N per class, runs inference exactly like
predict_sentiment.py (softmax + argmax + threshold fallback to manual_review,
max_length=128) and reports accuracy, confusion matrix and error analysis.

The model was trained on 80% of this same JSONL, so sampling from the whole file
mixes train and test examples: reported accuracy is optimistic (data leakage).
"""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["positive", "negative", "manual_review"]


def map_label(row: dict) -> str:
    """Same mapping as split_jsonl_sentiment_dataset.py (status wins over isPositive)."""
    if row.get("status") == 1:
        return "negative"
    if row.get("isPositive") == 1:
        return "positive"
    return "manual_review"


def decode_for_display(text: str, limit: int = 160) -> str:
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
    reviews: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = (row.get("descr") or "").strip()
            if not text:
                continue
            reviews.append(
                {
                    "id": str(row.get("id", "")),
                    "text": text,  # raw mojibake, as the model was trained
                    "expected": map_label(row),
                }
            )
    return reviews


def sample_per_class(reviews: list[dict], n: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in reviews:
        groups[r["expected"]].append(r)

    rng = random.Random(seed)
    sampled: list[dict] = []
    available: dict[str, int] = {}
    for label in LABELS:
        rows = groups.get(label, [])
        available[label] = len(rows)
        rng.shuffle(rows)
        sampled.extend(rows[:n])
    rng.shuffle(sampled)
    return sampled, available


def predict(model, tokenizer, text: str, id2label: dict[int, str], max_length: int, threshold: float) -> dict:
    inputs = tokenizer(text, truncation=True, padding="longest", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    best_id = int(torch.argmax(probs).item())
    best_label = id2label[best_id]
    confidence = float(probs[best_id].item())
    decision = best_label if confidence >= threshold else "manual_review"
    sorted_probs = sorted(((id2label[i], float(p)) for i, p in enumerate(probs)), key=lambda x: -x[1])
    return {
        "argmax_label": best_label,
        "predicted": decision,
        "confidence": confidence,
        "margin": float(sorted_probs[0][1] - sorted_probs[1][1]),
        "probs": {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))},
    }


def fmt_probs(probs: dict[str, float]) -> str:
    return " ".join(f"{lbl[:3]}={probs[lbl]:.3f}" for lbl in LABELS)


def report(results: list[dict], available: dict[str, int], requested_n: int, args) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["expected"])

    # Confusion matrix [expected][predicted]
    matrix: dict[str, Counter] = {e: Counter() for e in LABELS}
    for r in results:
        matrix[r["expected"]][r["predicted"]] += 1

    print("=" * 78)
    print("ВАЛИДАЦИЯ SENTIMENT-МОДЕЛИ НА РЕАЛЬНЫХ ОТЗЫВАХ")
    print("=" * 78)
    print(f"JSONL:        {args.jsonl}")
    print(f"Модель:       {args.model_dir}")
    print(f"threshold={args.threshold}  max_length={args.max_length}  seed={args.seed}")
    print(f"Запрошено на класс: {requested_n}")
    print("ВНИМАНИЕ: модель обучалась на 80% этого же JSONL — accuracy оптимистична (data leakage).")
    print()
    print("Доступно примеров в JSONL по классам:")
    for lbl in LABELS:
        got = sum(1 for r in results if r["expected"] == lbl)
        flag = "" if available[lbl] >= requested_n else f"  <-- меньше, чем {requested_n}, взято всё"
        print(f"  {lbl:14s} доступно={available[lbl]:6d}  взято={got}{flag}")
    print()

    # Per-class + overall + macro
    print("-" * 78)
    print("ACCURACY")
    print("-" * 78)
    per_class = {}
    for lbl in LABELS:
        n = sum(1 for r in results if r["expected"] == lbl)
        c = matrix[lbl][lbl]
        acc = c / n if n else 0.0
        per_class[lbl] = acc
        print(f"  {lbl:14s} {c:4d}/{n:<4d} = {acc:6.2%}")
    macro = sum(per_class.values()) / len(per_class)
    print(f"  {'OVERALL':14s} {correct:4d}/{total:<4d} = {correct / total:6.2%}")
    print(f"  {'MACRO':14s}            = {macro:6.2%}")
    print()

    # Confusion matrix
    print("-" * 78)
    print("CONFUSION MATRIX (строки = expected, столбцы = predicted)")
    print("-" * 78)
    header = "expected \\ pred".ljust(18) + "".join(lbl[:12].rjust(14) for lbl in LABELS) + "    | total"
    print(header)
    for e in LABELS:
        row_total = sum(matrix[e].values())
        cells = "".join(str(matrix[e][p]).rjust(14) for p in LABELS)
        print(f"  {e:14s}" + cells + f"    | {row_total}")
    print()

    # Errors
    errors = [r for r in results if r["predicted"] != r["expected"]]
    print("-" * 78)
    print(f"ОШИБКИ: {len(errors)} из {total}")
    print("-" * 78)
    for r in errors:
        print(f"  id={r['id']:>7s} exp={r['expected']:13s} pred={r['predicted']:13s} "
              f"conf={r['confidence']:.3f} | {fmt_probs(r['probs'])}")
        print(f"            «{decode_for_display(r['text'])}»")
    print()

    # Top-10 most confident errors
    print("-" * 78)
    print("10 САМЫХ УВЕРЕННЫХ ОШИБОК")
    print("-" * 78)
    for r in sorted(errors, key=lambda x: -x["confidence"])[:10]:
        print(f"  id={r['id']:>7s} exp={r['expected']:13s} pred={r['predicted']:13s} conf={r['confidence']:.3f}")
        print(f"            «{decode_for_display(r['text'])}»")
    print()

    # Top-10 least confident predictions
    print("-" * 78)
    print("10 САМЫХ НЕУВЕРЕННЫХ ПРЕДСКАЗАНИЙ (по confidence топ-класса)")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["confidence"])[:10]:
        ok = "OK " if r["predicted"] == r["expected"] else "ERR"
        print(f"  [{ok}] id={r['id']:>7s} exp={r['expected']:13s} pred={r['predicted']:13s} "
              f"conf={r['confidence']:.3f} margin={r['margin']:.3f} | {fmt_probs(r['probs'])}")
    print()

    # Disputed / mixed cases
    print("-" * 78)
    print("СПОРНЫЕ / СМЕШАННЫЕ КЕЙСЫ")
    print("-" * 78)
    mixed = sorted([r for r in results if r["margin"] < 0.20], key=lambda x: x["margin"])
    print(f"a) Смешанный сигнал (маржа топ-2 классов < 0.20): {len(mixed)}")
    for r in mixed[:10]:
        print(f"   id={r['id']:>7s} exp={r['expected']:13s} pred={r['predicted']:13s} "
              f"margin={r['margin']:.3f} | {fmt_probs(r['probs'])}")
        print(f"            «{decode_for_display(r['text'])}»")

    conf_pos = [r for r in results if r["expected"] == "manual_review"
                and r["predicted"] == "positive" and r["confidence"] >= 0.80]
    print(f"\nb) Уверенно positive, а expected manual_review (conf>=0.80): {len(conf_pos)}")
    for r in sorted(conf_pos, key=lambda x: -x["confidence"])[:10]:
        print(f"   id={r['id']:>7s} conf={r['confidence']:.3f} | {fmt_probs(r['probs'])}")
        print(f"            «{decode_for_display(r['text'])}»")

    conf_neg = [r for r in results if r["expected"] == "manual_review"
                and r["predicted"] == "negative" and r["confidence"] >= 0.80]
    print(f"\nc) Уверенно negative, а expected manual_review (conf>=0.80): {len(conf_neg)}")
    for r in sorted(conf_neg, key=lambda x: -x["confidence"])[:10]:
        print(f"   id={r['id']:>7s} conf={r['confidence']:.3f} | {fmt_probs(r['probs'])}")
        print(f"            «{decode_for_display(r['text'])}»")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("aj_reviews_export.jsonl"))
    p.add_argument("--model-dir", type=Path, default=Path("outputs/rubert-sentiment-cpu/final"))
    p.add_argument("--samples-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260630)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    reviews = load_reviews(args.jsonl)
    sampled, available = sample_per_class(reviews, args.samples_per_class, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    results = []
    for row in sampled:
        pred = predict(model, tokenizer, row["text"], id2label, args.max_length, args.threshold)
        results.append({**row, **pred})

    report(results, available, args.samples_per_class, args)


if __name__ == "__main__":
    main()
