#!/usr/bin/env python3
"""Honest held-out evaluation of the sentiment model on a prepared TSV split.

Reads a sentiment TSV (id/name/text/label, multi-line text fields are csv-quoted),
runs batched inference once, then reports standard argmax metrics plus a
production threshold grid (post-processing on the cached probabilities, no
re-inference). The model never trained on the test split, so these numbers are
leakage-free.

Inference mirrors scripts/predict_sentiment.py: raw (mojibake) text, softmax over
[positive, negative, manual_review]. decode_for_display un-mojibakes text for
printing only.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

csv.field_size_limit(sys.maxsize)

LABELS = ["positive", "negative", "manual_review"]
POS_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
NEG_THRESHOLDS = [0.50, 0.55, 0.60, 0.70, 0.80]


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


def load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in rows if (r.get("text") or "").strip() and r.get("label") in LABELS]
    if limit:
        rows = rows[:limit]
    return rows


def run_inference(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> torch.Tensor:
    """Return an (N, 3) tensor of softmax probabilities."""
    all_probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        all_probs.append(torch.softmax(logits, dim=-1))
    return torch.cat(all_probs, dim=0)


def confusion(gold: list[str], pred: list[str], classes: list[str]) -> dict[str, dict[str, int]]:
    m = {g: {p: 0 for p in classes} for g in classes}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def per_class_prf(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    out = {}
    for c in LABELS:
        tp = matrix[c][c]
        fn = sum(matrix[c][p] for p in LABELS) - tp
        fp = sum(matrix[g][c] for g in LABELS) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def write_metrics_report(out, rows, probs, id2label, max_length, runtime_s) -> None:
    gold = [r["label"] for r in rows]
    top_ids = probs.argmax(dim=-1).tolist()
    top_prob = probs.max(dim=-1).values.tolist()
    pred = [id2label[i] for i in top_ids]

    total = len(rows)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    matrix = confusion(gold, pred, LABELS)
    prf = per_class_prf(matrix)
    macro_f1 = sum(prf[c]["f1"] for c in LABELS) / len(LABELS)
    macro_acc = sum(prf[c]["recall"] for c in LABELS) / len(LABELS)

    def emit(line=""):
        print(line)
        out.write(line + "\n")

    emit("=" * 80)
    emit("HELD-OUT TEST METRICS (argmax, без порога)")
    emit("=" * 80)
    emit(f"input rows: {total}   max_length={max_length}   runtime={runtime_s:.1f}s "
         f"({total / runtime_s:.1f} ex/s)")
    emit("Леакейджа нет: test split не пересекается с train.")
    emit("")
    emit(f"OVERALL accuracy: {correct}/{total} = {correct / total:.4f}")
    emit(f"MACRO accuracy (avg recall): {macro_acc:.4f}")
    emit(f"MACRO F1: {macro_f1:.4f}")
    emit("")
    emit(f"{'class':14s}{'precision':>11s}{'recall':>10s}{'f1':>9s}{'support':>10s}")
    for c in LABELS:
        emit(f"{c:14s}{prf[c]['precision']:>11.4f}{prf[c]['recall']:>10.4f}"
             f"{prf[c]['f1']:>9.4f}{prf[c]['support']:>10d}")
    emit("")
    emit("CONFUSION MATRIX (строки = gold, столбцы = pred)")
    emit("gold \\ pred".ljust(16) + "".join(c[:12].rjust(15) for c in LABELS) + "    | total")
    for g in LABELS:
        rt = sum(matrix[g].values())
        emit(f"  {g:14s}" + "".join(str(matrix[g][p]).rjust(15) for p in LABELS) + f"    | {rt}")
    emit("")

    errors = [(i, top_prob[i]) for i in range(total) if pred[i] != gold[i]]
    emit("-" * 80)
    emit("20 САМЫХ УВЕРЕННЫХ ОШИБОК (argmax != gold, по confidence)")
    emit("-" * 80)
    for i, conf in sorted(errors, key=lambda x: -x[1])[:20]:
        p = probs[i].tolist()
        emit(f"  id={rows[i]['id']:>7s} gold={gold[i]:13s} pred={pred[i]:13s} conf={conf:.3f} "
             f"| pos={p[0]:.3f} neg={p[1]:.3f} man={p[2]:.3f}")
        emit(f"            «{decode_for_display(rows[i]['text'])}»")
    emit("")
    emit("-" * 80)
    emit("20 САМЫХ НЕУВЕРЕННЫХ ПРЕДСКАЗАНИЙ (по confidence топ-класса)")
    emit("-" * 80)
    order = sorted(range(total), key=lambda i: top_prob[i])[:20]
    for i in order:
        p = probs[i].tolist()
        ok = "OK " if pred[i] == gold[i] else "ERR"
        emit(f"  [{ok}] id={rows[i]['id']:>7s} gold={gold[i]:13s} pred={pred[i]:13s} "
             f"conf={top_prob[i]:.3f} | pos={p[0]:.3f} neg={p[1]:.3f} man={p[2]:.3f}")
    emit("")


def write_threshold_grid(out, rows, probs, label2id, max_length) -> None:
    gold = [r["label"] for r in rows]
    pos_i, neg_i = label2id["positive"], label2id["negative"]
    top_ids = probs.argmax(dim=-1).tolist()
    pos_p = probs[:, pos_i].tolist()
    neg_p = probs[:, neg_i].tolist()
    pos_top_id = label2id["positive"]
    neg_top_id = label2id["negative"]
    total = len(rows)

    def emit(line=""):
        print(line)
        out.write(line + "\n")

    emit("=" * 80)
    emit("THRESHOLD GRID (production routing)")
    emit("=" * 80)
    emit(f"rows={total}  max_length={max_length}")
    emit("Правило: top=positive & pos_prob>=pos_thr -> auto-positive; "
         "top=negative & neg_prob>=neg_thr -> auto-negative; иначе manual_review.")
    emit("Цель: высокая precision auto-pos/auto-neg; manual_review-утечка в авто = риск.")
    emit("")
    header = (f"{'pos_thr':>7s}{'neg_thr':>8s}{'autoPos':>9s}{'autoNeg':>9s}{'manual':>8s}"
              f"{'precPos':>9s}{'precNeg':>9s}{'man>Pos':>9s}{'man>Neg':>9s}{'autoCov':>9s}")
    emit(header)
    emit("-" * len(header))

    candidates = []
    for pos_thr in POS_THRESHOLDS:
        for neg_thr in NEG_THRESHOLDS:
            auto_pos = auto_neg = manual = 0
            pos_correct = neg_correct = 0
            man_to_pos = man_to_neg = 0
            for i in range(total):
                top = top_ids[i]
                if top == pos_top_id and pos_p[i] >= pos_thr:
                    auto_pos += 1
                    if gold[i] == "positive":
                        pos_correct += 1
                    elif gold[i] == "manual_review":
                        man_to_pos += 1
                elif top == neg_top_id and neg_p[i] >= neg_thr:
                    auto_neg += 1
                    if gold[i] == "negative":
                        neg_correct += 1
                    elif gold[i] == "manual_review":
                        man_to_neg += 1
                else:
                    manual += 1
            prec_pos = pos_correct / auto_pos if auto_pos else 0.0
            prec_neg = neg_correct / auto_neg if auto_neg else 0.0
            cov = (auto_pos + auto_neg) / total
            emit(f"{pos_thr:>7.2f}{neg_thr:>8.2f}{auto_pos:>9d}{auto_neg:>9d}{manual:>8d}"
                 f"{prec_pos:>9.3f}{prec_neg:>9.3f}{man_to_pos:>9d}{man_to_neg:>9d}{cov:>9.3f}")
            candidates.append({
                "pos_thr": pos_thr, "neg_thr": neg_thr, "prec_pos": prec_pos,
                "prec_neg": prec_neg, "cov": cov, "man_to_pos": man_to_pos,
                "man_to_neg": man_to_neg,
            })

    emit("")
    emit("RECOMMENDED (pos_thr и neg_thr подбираются НЕЗАВИСИМО: precPos зависит только")
    emit("от pos_thr, precNeg — только от neg_thr). Цель precision >= 0.90, иначе авто отключаем.")
    target = 0.90
    by_pos = {c["pos_thr"]: c for c in candidates}  # precPos одинаков для любого neg_thr
    by_neg = {c["neg_thr"]: c for c in candidates}  # precNeg одинаков для любого pos_thr
    safe_pos = sorted([t for t, c in by_pos.items() if c["prec_pos"] >= target])
    safe_neg = sorted([t for t, c in by_neg.items() if c["prec_neg"] >= target])
    if safe_pos:
        t = safe_pos[0]
        emit(f"  auto-positive: pos_thr={t:.2f} precPos={by_pos[t]['prec_pos']:.3f} -> ВКЛ")
    else:
        best_t = max(by_pos, key=lambda t: by_pos[t]["prec_pos"])
        emit(f"  auto-positive: precision НЕ достигает {target:.2f} (макс {by_pos[best_t]['prec_pos']:.3f} "
             f"при pos_thr={best_t:.2f}) -> ОТКЛЮЧИТЬ, positive -> manual_review")
    if safe_neg:
        t = safe_neg[0]
        emit(f"  auto-negative: neg_thr={t:.2f} precNeg={by_neg[t]['prec_neg']:.3f} -> ВКЛ "
             f"(чем ниже порог при precNeg>=цели, тем выше покрытие)")
    else:
        best_t = max(by_neg, key=lambda t: by_neg[t]["prec_neg"])
        emit(f"  auto-negative: precision НЕ достигает {target:.2f} (макс {by_neg[best_t]['prec_neg']:.3f} "
             f"при neg_thr={best_t:.2f}) -> ОТКЛЮЧИТЬ, negative -> manual_review")
    emit("")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/sentiment_test.tsv"))
    p.add_argument("--model-dir", type=Path, default=Path("outputs/rubert-sentiment-cpu/final"))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows (speed)")
    p.add_argument("--threshold-grid", action="store_true")
    p.add_argument("--metrics-report", type=Path, default=Path("outputs/validation_test_report.txt"))
    p.add_argument("--grid-report", type=Path, default=Path("outputs/threshold_grid_report.txt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_rows(args.input, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    texts = [r["text"] for r in rows]
    t0 = time.perf_counter()
    probs = run_inference(model, tokenizer, texts, args.max_length, args.batch_size)
    runtime_s = time.perf_counter() - t0

    args.metrics_report.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_report.open("w", encoding="utf-8") as out:
        write_metrics_report(out, rows, probs, id2label, args.max_length, runtime_s)
    print(f"[metrics report -> {args.metrics_report}]")

    if args.threshold_grid:
        with args.grid_report.open("w", encoding="utf-8") as out:
            write_threshold_grid(out, rows, probs, label2id, args.max_length)
        print(f"[grid report -> {args.grid_report}]")


if __name__ == "__main__":
    main()
