"""Held-out test evaluation of the 5-class ruRoberta-large sentiment model, on Kaggle GPU.

Same metrics logic as kaggle/eval/eval.py (accuracy, macro F1, per-class precision/
recall/f1, confusion matrix, threshold grid) but for the 5-class label scheme
(positive/negative/manual_review/spam/service_complaint) — see CLAUDE.md for how
the manual_review subset was independently relabeled via DeepSeek.

Ground truth caveat: for former manual_review rows, "gold" here is the DeepSeek
verdict baked into data/sentiment5_test.tsv, not an independently-collected label —
spam/service_complaint never existed in the original status/isPositive flags at all.
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

LABELS = ["positive", "negative", "manual_review", "spam", "service_complaint"]
POS_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
NEG_THRESHOLDS = [0.50, 0.55, 0.60, 0.70, 0.80]

MAX_LENGTH = 128
BATCH_SIZE = 32
LIMIT = None  # full test set — GPU makes this cheap

# WeightedTrainer's extreme class weights in the 5-class scheme (manual_review weight ~10.68 vs
# negative's ~0.24, a ~44x ratio) collapse negative's softmax probability into a narrow ~0.31 band
# regardless of confidence (confirmed via local diagnostic — argmax is unaffected/still correct,
# only the probability magnitude is compressed), making the neg_prob>=thr auto-decide policy
# non-functional. Sharpening the softmax post-hoc (logits / TEMPERATURE, TEMPERATURE < 1) restores
# usable separation without retraining. TEMPERATURE=0.25 was picked from a local sweep (CPU, 4000-row
# sample): restores ~70% auto-negative coverage at ~0.99 precision, and even mildly improves
# auto-positive coverage at the same 0.90 precision target. This full-test-set run confirms it at
# scale before baking the value into scripts/predict_sentiment.py.
TEMPERATURE = 0.25


def _find_file(filename: str) -> Path:
    for root in Path("/kaggle/input").glob(f"**/{filename}"):
        return root
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input/")


def _find_model_dir() -> Path:
    for cfg in Path("/kaggle/input").glob("**/final/config.json"):
        return cfg.parent
    raise FileNotFoundError("Could not find a 'final/config.json' model dir under /kaggle/input/")


def decode_for_display(text: str, limit: int = 160) -> str:
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


def run_inference(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    all_logits = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_logits.append(logits.cpu())
        if start % (BATCH_SIZE * 50) == 0:
            print(f"  inference: {start}/{len(texts)}", flush=True)
    return torch.cat(all_logits, dim=0)


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


def write_metrics_report(out, rows, probs, id2label, runtime_s) -> None:
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
    emit("HELD-OUT TEST METRICS (argmax, ruRoberta-large 5-class, Kaggle GPU)")
    emit("=" * 80)
    emit(f"input rows: {total}   max_length={MAX_LENGTH}   runtime={runtime_s:.1f}s "
         f"({total / runtime_s:.1f} ex/s)")
    emit("")
    emit(f"OVERALL accuracy: {correct}/{total} = {correct / total:.4f}")
    emit(f"MACRO accuracy (avg recall): {macro_acc:.4f}")
    emit(f"MACRO F1: {macro_f1:.4f}")
    emit("")
    emit(f"{'class':18s}{'precision':>11s}{'recall':>10s}{'f1':>9s}{'support':>10s}")
    for c in LABELS:
        emit(f"{c:18s}{prf[c]['precision']:>11.4f}{prf[c]['recall']:>10.4f}"
             f"{prf[c]['f1']:>9.4f}{prf[c]['support']:>10d}")
    emit("")
    emit("CONFUSION MATRIX (строки = gold, столбцы = pred)")
    emit("gold \\ pred".ljust(18) + "".join(c[:14].rjust(16) for c in LABELS) + "    | total")
    for g in LABELS:
        rt = sum(matrix[g].values())
        emit(f"  {g:16s}" + "".join(str(matrix[g][p]).rjust(16) for p in LABELS) + f"    | {rt}")
    emit("")

    errors = [(i, top_prob[i]) for i in range(total) if pred[i] != gold[i]]
    emit("-" * 80)
    emit("20 САМЫХ УВЕРЕННЫХ ОШИБОК (argmax != gold, по confidence)")
    emit("-" * 80)
    for i, conf in sorted(errors, key=lambda x: -x[1])[:20]:
        probs_str = " ".join(f"{c[:3]}={probs[i][j]:.3f}" for j, c in enumerate(LABELS))
        emit(f"  id={rows[i]['id']:>7s} gold={gold[i]:16s} pred={pred[i]:16s} conf={conf:.3f} | {probs_str}")
        emit(f"            «{decode_for_display(rows[i]['text'])}»")
    emit("")


def write_threshold_grid(out, rows, probs, label2id) -> None:
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
    emit("THRESHOLD GRID (production routing) — ruRoberta-large 5-class")
    emit("=" * 80)
    emit(f"rows={total}  max_length={MAX_LENGTH}")
    emit("NOTE: only pos/neg auto-decide thresholds are graded here (same policy shape as "
         "the 3-class model); spam/service_complaint/manual_review always fall to manual review.")
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
                    elif gold[i] != "negative":
                        man_to_pos += 1
                elif top == neg_top_id and neg_p[i] >= neg_thr:
                    auto_neg += 1
                    if gold[i] == "negative":
                        neg_correct += 1
                    elif gold[i] != "positive":
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
                "prec_neg": prec_neg, "cov": cov,
            })

    emit("")
    target = 0.90
    by_pos = {c["pos_thr"]: c for c in candidates}
    by_neg = {c["neg_thr"]: c for c in candidates}
    safe_pos = sorted([t for t, c in by_pos.items() if c["prec_pos"] >= target])
    safe_neg = sorted([t for t, c in by_neg.items() if c["prec_neg"] >= target])
    if safe_pos:
        t = safe_pos[0]
        emit(f"  auto-positive: pos_thr={t:.2f} precPos={by_pos[t]['prec_pos']:.3f} -> ВКЛ")
    else:
        best_t = max(by_pos, key=lambda t: by_pos[t]["prec_pos"])
        emit(f"  auto-positive: precision НЕ достигает {target:.2f} (макс {by_pos[best_t]['prec_pos']:.3f} "
             f"при pos_thr={best_t:.2f}) -> ОТКЛЮЧИТЬ")
    if safe_neg:
        t = safe_neg[0]
        emit(f"  auto-negative: neg_thr={t:.2f} precNeg={by_neg[t]['prec_neg']:.3f} -> ВКЛ")
    else:
        best_t = max(by_neg, key=lambda t: by_neg[t]["prec_neg"])
        emit(f"  auto-negative: precision НЕ достигает {target:.2f} (макс {by_neg[best_t]['prec_neg']:.3f} "
             f"при neg_thr={best_t:.2f}) -> ОТКЛЮЧИТЬ")
    emit("")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_dir = _find_model_dir()
    test_file = _find_file("sentiment_test.tsv")
    print(f"Model dir: {model_dir}")
    print(f"Test file: {test_file}")

    rows = load_rows(test_file, LIMIT)
    print(f"Loaded {len(rows)} test rows")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    texts = [r["text"] for r in rows]
    t0 = time.perf_counter()
    logits = run_inference(model, tokenizer, texts, device)
    runtime_s = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=-1)
    probs_scaled = torch.softmax(logits / TEMPERATURE, dim=-1)

    out_dir = Path("/kaggle/working")
    with (out_dir / "ruroberta_test_report.txt").open("w", encoding="utf-8") as out:
        write_metrics_report(out, rows, probs, id2label, runtime_s)
    print("[metrics report -> /kaggle/working/ruroberta_test_report.txt]")

    with (out_dir / "ruroberta_threshold_grid.txt").open("w", encoding="utf-8") as out:
        emit_note = f"TEMPERATURE={TEMPERATURE} applied (logits / TEMPERATURE before softmax) — see comment at top of file.\n"
        print(emit_note)
        out.write(emit_note + "\n")
        write_threshold_grid(out, rows, probs_scaled, label2id)
    print("[grid report -> /kaggle/working/ruroberta_threshold_grid.txt]")


if __name__ == "__main__":
    main()
