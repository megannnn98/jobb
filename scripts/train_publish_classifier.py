#!/usr/bin/env python3
"""Fine-tune RuBERT for binary publish/reject classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

LABEL2ID = {"reject": 0, "publish": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(len(LABEL2ID)):
        weights.append(total / (len(LABEL2ID) * counts[i]))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "accuracy": (preds == labels).mean().item(),
        "precision_publish": precision_score(labels, preds, pos_label=1),
        "recall_publish": recall_score(labels, preds, pos_label=1),
        "f1_publish": f1_score(labels, preds, pos_label=1),
        "f1_reject": f1_score(labels, preds, pos_label=0),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("model/rubert-base-cased"))
    parser.add_argument("--train-file", type=Path, default=Path("data/train.tsv"))
    parser.add_argument("--val-file", type=Path, default=Path("data/val.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/publish-classifier"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all data")
    args = parser.parse_args()

    dataset = load_dataset(
        "csv",
        data_files={"train": str(args.train_file), "validation": str(args.val_file)},
        delimiter="\t",
    )

    if args.max_samples > 0:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(min(args.max_samples, len(dataset["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    dataset = dataset.map(tokenize, batched=True, batch_size=1000)
    dataset = dataset.map(lambda row: {"labels": [LABEL2ID[l] for l in row["label"]]}, batched=True)
    dataset = dataset.remove_columns(["label", "id", "name"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_labels = dataset["train"]["labels"]
    class_weights = compute_class_weights([int(x) for x in train_labels])
    print(f"class weights: reject={class_weights[0]:.3f}, publish={class_weights[1]:.3f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_path),
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    args_training = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args_training,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    trainer.train()

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    with open(best_dir / "label_mapping.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)

    print(f"model saved to {best_dir}")


if __name__ == "__main__":
    main()
