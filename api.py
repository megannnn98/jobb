#!/usr/bin/env python3
"""HTTP API for the 5-class employer-review sentiment model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scripts.predict_sentiment import (
    FIVE_CLASS_MAX_LENGTH,
    FIVE_CLASS_NEGATIVE_THRESHOLD,
    FIVE_CLASS_POSITIVE_THRESHOLD,
    FIVE_CLASS_TEMPERATURE,
    load_model,
    predict,
)


DEFAULT_MODEL_DIR = Path("model/ruroberta-sentiment-5class")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    decision: str
    decision_policy: str
    raw_top_label: str
    raw_confidence: float
    positive_prob: float
    negative_prob: float
    manual_review_prob: float
    spam_prob: float | None = None
    service_complaint_prob: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_dir: str


model: Any | None = None
tokenizer: Any | None = None
model_dir = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    if not model_dir.exists():
        raise RuntimeError(f"model directory does not exist: {model_dir}")
    model, tokenizer = load_model(model_dir)
    yield


app = FastAPI(
    title="Employer Review Sentiment API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_dir=str(model_dir))


@app.post("/predict", response_model=PredictResponse)
def predict_review(request: PredictRequest) -> dict[str, Any]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    return predict(
        model,
        tokenizer,
        request.text,
        policy="balanced",
        max_length=FIVE_CLASS_MAX_LENGTH,
        negative_threshold=FIVE_CLASS_NEGATIVE_THRESHOLD,
        positive_threshold=FIVE_CLASS_POSITIVE_THRESHOLD,
        temperature=FIVE_CLASS_TEMPERATURE,
    )
