#!/usr/bin/env python3
"""Binary (positive/negative) variant of the CPU fine-tuning pipeline.

Reuses the two-phase training machinery from train_fast_bert_cpu.py unchanged; only
the dataset paths, output dir and label space are re-pointed. manual_review is NOT a
class here — it is handled downstream as a threshold fallback in
scripts/predict_sentiment.py. Run scripts/split_jsonl_binary_dataset.py first.

The same env vars apply (MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES, TOKENIZE_NUM_PROC).
"""

from __future__ import annotations

from pathlib import Path

import train_fast_bert_cpu as base

base.TRAIN_FILE = Path("data/binary_train.tsv")
base.VAL_FILE = Path("data/binary_val.tsv")
base.OUTPUT_DIR = Path("outputs/rubert-binary-cpu")
base.TOKENIZED_DIR = base.OUTPUT_DIR / "tokenized"
base.TOKENIZED_METADATA_FILE = base.TOKENIZED_DIR / "metadata.json"
base.LABEL2ID = {"positive": 0, "negative": 1}
base.ID2LABEL = {0: "positive", 1: "negative"}
base.NUM_LABELS = len(base.LABEL2ID)


if __name__ == "__main__":
    base.main()
