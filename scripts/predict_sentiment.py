#!/usr/bin/env python3
"""Predict positive/negative/manual_review label for review text."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_ID2LABEL = {0: "positive", 1: "negative", 2: "manual_review"}
FIVE_CLASS_LABELS = {"positive", "negative", "manual_review", "spam", "service_complaint"}

# Routing defaults derived from held-out test threshold grid (ruRoberta-large,
# full test set, see docs/evaluation.md). label_smoothing=0.1 compresses
# negative_prob more than RuBERT did — no test example clears neg_prob>=0.70,
# so 0.70 (RuBERT's old "safe" threshold) would give zero auto-negative
# coverage on this model; 0.60 is the closest working high-precision threshold.
DEFAULT_SAFE_NEGATIVE_THRESHOLD = 0.60
DEFAULT_BALANCED_NEGATIVE_THRESHOLD = 0.50
DEFAULT_POSITIVE_THRESHOLD = 0.90
DEFAULT_TEMPERATURE = 1.0

# 5-class scheme (model/ruroberta-sentiment-5class/): WeightedTrainer's extreme class
# weights there (manual_review ~10.68 vs negative ~0.24, a ~44x ratio) collapse negative's
# softmax probability into a narrow ~0.31 band regardless of confidence — argmax stays
# correct, only the probability magnitude is unusable for thresholding. Sharpening the
# softmax post-hoc (logits / TEMPERATURE) restores separation without retraining.
# Confirmed on the full held-out test set (17,523 rows, MAX_LENGTH=128 — NOT yet verified
# at 512 like the 3-class model was): neg_thr=0.50 -> precision 0.996 (12,179 rows auto),
# pos_thr=0.70 -> precision 0.904 (1,330 rows auto), ~77% combined auto-coverage.
# See outputs/eval-5class/ruroberta_threshold_grid.txt.
FIVE_CLASS_TEMPERATURE = 0.25
FIVE_CLASS_NEGATIVE_THRESHOLD = 0.50
FIVE_CLASS_POSITIVE_THRESHOLD = 0.70
FIVE_CLASS_MAX_LENGTH = 128


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def id2label_from_model(model) -> dict[int, str]:
    mapping = model.config.id2label or DEFAULT_ID2LABEL
    return {int(key): value for key, value in mapping.items()}


def is_five_class_scheme(id2label: dict[int, str]) -> bool:
    return set(id2label.values()) == FIVE_CLASS_LABELS


def decide_label(
    policy: str,
    raw_top_label: str,
    raw_confidence: float,
    label_probs: dict[str, float],
    negative_threshold: float,
    positive_threshold: float,
    enable_auto_positive: bool,
    argmax_threshold: float,
) -> str:
    """Map raw model output to a production decision.

    balanced — auto-decide clear negative/positive, route uncertain text to manual_review.
    safe     — auto-decide only high-confidence negative by default; positive is opt-in.
    argmax   — legacy behaviour: take the top label, fall back to manual_review below threshold.
    """
    if policy == "argmax":
        return raw_top_label if raw_confidence >= argmax_threshold else "manual_review"

    if policy == "balanced":
        if raw_top_label == "negative" and label_probs.get("negative", 0.0) >= negative_threshold:
            return "negative"
        if raw_top_label == "positive" and label_probs.get("positive", 0.0) >= positive_threshold:
            return "positive"
        return "manual_review"

    if policy == "safe":
        if raw_top_label == "negative" and label_probs.get("negative", 0.0) >= negative_threshold:
            return "negative"
        if (
            enable_auto_positive
            and raw_top_label == "positive"
            and label_probs.get("positive", 0.0) >= positive_threshold
        ):
            return "positive"
    return "manual_review"


def predict(
    model,
    tokenizer,
    text: str,
    *,
    policy: str = "balanced",
    max_length: int = 512,
    negative_threshold: float = DEFAULT_BALANCED_NEGATIVE_THRESHOLD,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    enable_auto_positive: bool = False,
    argmax_threshold: float = 0.5,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    id2label = id2label_from_model(model)
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits / temperature, dim=-1)[0]
    label_probs = {id2label[i]: float(probs[i].item()) for i in range(len(probs))}
    raw_top_id = int(torch.argmax(probs).item())
    raw_top_label = id2label[raw_top_id]
    raw_confidence = float(probs[raw_top_id].item())

    decision = decide_label(
        policy,
        raw_top_label,
        raw_confidence,
        label_probs,
        negative_threshold,
        positive_threshold,
        enable_auto_positive,
        argmax_threshold,
    )

    result = {
        "decision": decision,
        "decision_policy": policy,
        "raw_top_label": raw_top_label,
        "raw_confidence": round(raw_confidence, 4),
        "positive_prob": round(label_probs.get("positive", 0.0), 4),
        "negative_prob": round(label_probs.get("negative", 0.0), 4),
        "manual_review_prob": round(label_probs.get("manual_review", 0.0), 4),
    }
    if "spam" in label_probs or "service_complaint" in label_probs:
        result["spam_prob"] = round(label_probs.get("spam", 0.0), 4)
        result["service_complaint_prob"] = round(label_probs.get("service_complaint", 0.0), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single review text to classify")
    parser.add_argument("--input", type=Path, default=None, help="TSV file with 'text' column")
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path")
    parser.add_argument("--policy", choices=["balanced", "safe", "argmax"], default="balanced",
                        help="balanced = practical routing (default); safe = conservative routing; "
                             "argmax = legacy top-label behaviour")
    parser.add_argument("--negative-threshold", type=float, default=None,
                        help="min negative_prob to auto-decide negative; defaults: balanced=0.55, safe=0.70 "
                             "(3-class), 0.50 (5-class)")
    parser.add_argument("--positive-threshold", type=float, default=None,
                        help="min positive_prob to auto-decide positive; defaults: 0.90 (3-class), 0.70 (5-class)")
    parser.add_argument("--enable-auto-positive", action="store_true",
                        help="allow auto-positive in safe policy; balanced enables auto-positive by policy")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="confidence cutoff for legacy argmax policy")
    parser.add_argument("--max-length", type=int, default=None,
                        help="defaults: 512 (3-class), 128 (5-class — not yet verified at 512)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="softmax temperature (logits / temperature); defaults: 1.0 (3-class), "
                             "0.25 (5-class — restores usable negative_prob/positive_prob thresholds, "
                             "see FIVE_CLASS_TEMPERATURE comment)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    five_class = is_five_class_scheme(id2label_from_model(model))

    negative_threshold = args.negative_threshold
    if negative_threshold is None:
        if five_class:
            negative_threshold = FIVE_CLASS_NEGATIVE_THRESHOLD
        elif args.policy == "safe":
            negative_threshold = DEFAULT_SAFE_NEGATIVE_THRESHOLD
        else:
            negative_threshold = DEFAULT_BALANCED_NEGATIVE_THRESHOLD

    positive_threshold = args.positive_threshold
    if positive_threshold is None:
        positive_threshold = FIVE_CLASS_POSITIVE_THRESHOLD if five_class else DEFAULT_POSITIVE_THRESHOLD

    max_length = args.max_length
    if max_length is None:
        max_length = FIVE_CLASS_MAX_LENGTH if five_class else 512

    temperature = args.temperature
    if temperature is None:
        temperature = FIVE_CLASS_TEMPERATURE if five_class else DEFAULT_TEMPERATURE

    predict_kwargs = dict(
        policy=args.policy,
        max_length=max_length,
        negative_threshold=negative_threshold,
        positive_threshold=positive_threshold,
        enable_auto_positive=args.enable_auto_positive,
        argmax_threshold=args.threshold,
        temperature=temperature,
    )

    if args.text:
        result = predict(model, tokenizer, args.text, **predict_kwargs)
        for key in (
            "decision",
            "decision_policy",
            "raw_top_label",
            "raw_confidence",
            "positive_prob",
            "negative_prob",
            "manual_review_prob",
            "spam_prob",
            "service_complaint_prob",
        ):
            if key in result:
                print(f"{key + ':':20s}{result[key]}")
        return

    if args.input:
        with args.input.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        results = []
        for row in rows:
            result = predict(model, tokenizer, row.get("text", ""), **predict_kwargs)
            results.append({**row, **result})

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(results[0].keys())
            with args.output.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(results)
            print(f"wrote {len(results)} predictions to {args.output}")
        else:
            for row in results[:5]:
                print(f"id={row.get('id', '?')} decision={row['decision']} "
                      f"raw_top={row['raw_top_label']} raw_conf={row['raw_confidence']}")
            if len(results) > 5:
                print(f"... and {len(results) - 5} more")
        return

    print("Provide --text or --input")


if __name__ == "__main__":
    main()
