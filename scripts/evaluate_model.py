#!/usr/bin/env python3
"""Evaluate trained publish/reject classifier on test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


ID2LABEL = {0: "reject", 1: "publish"}


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("data/test.tsv"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))

    dataset = load_dataset("csv", data_files=str(args.test_file), delimiter="\t")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    dataset = dataset.map(tokenize, batched=True, batch_size=1000)
    LABEL2ID = model.config.label2id
    dataset = dataset.map(lambda row: {"labels": [LABEL2ID[l] for l in row["label"]]}, batched=True)
    dataset = dataset.remove_columns(["label", "id", "name"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    result = trainer.evaluate(dataset["train"])
    print(f"Test metrics: {result}")

    predictions = trainer.predict(dataset["train"])
    preds = predictions.predictions.argmax(axis=-1)
    labels = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["reject", "publish"]))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"                 predicted")
    print(f"                 reject  publish")
    print(f"actual reject    {cm[0][0]:>6}  {cm[0][1]:>7}")
    print(f"actual publish   {cm[1][0]:>6}  {cm[1][1]:>7}")

    threshold_results = []
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        adjusted_preds = []
        for p in probs:
            if p >= threshold:
                adjusted_preds.append(1)
            else:
                adjusted_preds.append(0)
        f1 = f1_score(labels, adjusted_preds, average="macro")
        publish_count = sum(1 for p in adjusted_preds if p == 1)
        reject_count = sum(1 for p in adjusted_preds if p == 0)
        threshold_results.append({"threshold": threshold, "macro_f1": f1, "publish": publish_count, "reject": reject_count})

    print("\nThreshold sweep (on softmax publish probability):")
    print(f"{'threshold':>10} {'macro_f1':>10} {'publish':>10} {'reject':>10}")
    for r in threshold_results:
        print(f"{r['threshold']:>10.2f} {r['macro_f1']:>10.4f} {r['publish']:>10} {r['reject']:>10}")

    output_path = args.model_dir.parent / "test_metrics.json"
    with open(output_path, "w") as f:
        json.dump({
            "test_metrics": result,
            "threshold_sweep": threshold_results,
        }, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
