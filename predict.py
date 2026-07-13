#!/usr/bin/env python3

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "model/ruroberta-sentiment-5class"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Текст отзыва")
    parser.add_argument("--json", action="store_true", help="Вывести JSON")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    id2label = model.config.id2label

    probabilities = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    label = max(probabilities, key=probabilities.get)

    if args.json:
        print(json.dumps({
            "label": label,
            "confidence": probabilities[label],
            "probabilities": probabilities,
        }, ensure_ascii=False, indent=2))
    else:
        print(label)


if __name__ == "__main__":
    main()
