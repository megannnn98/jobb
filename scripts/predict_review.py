#!/usr/bin/env python3
"""Predict whether a review should be published or rejected."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ID2LABEL = {0: "reject", 1: "publish"}


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def predict(
    model, tokenizer, text: str, max_length: int = 256, threshold: float = 0.5
) -> dict:
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    publish_prob = probs[1].item()
    reject_prob = probs[0].item()

    if publish_prob >= threshold:
        decision = "publish"
        confidence = publish_prob
    elif reject_prob >= threshold:
        decision = "reject"
        confidence = reject_prob
    else:
        decision = "human_review"
        confidence = max(publish_prob, reject_prob)

    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "publish_prob": round(publish_prob, 4),
        "reject_prob": round(reject_prob, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single review text to classify")
    parser.add_argument("--input", type=Path, default=None, help="TSV file with 'text' column")
    parser.add_argument("--output", type=Path, default=None, help="Output TSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for auto-decision")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)

    if args.text:
        result = predict(model, tokenizer, args.text, args.max_length, args.threshold)
        print(f"decision:    {result['decision']}")
        print(f"confidence:  {result['confidence']}")
        print(f"publish_prob: {result['publish_prob']}")
        print(f"reject_prob: {result['reject_prob']}")
        return

    if args.input:
        with args.input.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        results = []
        for row in rows:
            text = row.get("text", "")
            result = predict(model, tokenizer, text, args.max_length, args.threshold)
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
            for r in results[:5]:
                print(f"id={r.get('id', '?')} decision={r['decision']} confidence={r['confidence']}")
            if len(results) > 5:
                print(f"... and {len(results) - 5} more")
        return

    print("Provide --text or --input")


if __name__ == "__main__":
    main()
