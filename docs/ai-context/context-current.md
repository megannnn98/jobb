# Context — 2026-07-30

## What's done
- Added Docker/FastAPI inference layer for the current 5-class ruRoberta model.
- `api.py` exposes `GET /health` and `POST /predict`; startup loads the model from `MODEL_DIR`.
- `Dockerfile` builds a CPU inference image and copies `model/ruroberta-sentiment-5class/` into the image by default.
- README and inference/architecture docs now describe container build/run, endpoint contract, and mounted-model override.

## Next
- Integrate external callers through `POST /predict` and consume `decision` as the routing result.
- For server deployment, either transfer the built `jobb-sentiment-api:latest` image or rebuild on the server after placing the model under `model/ruroberta-sentiment-5class/`.

## Blockers
- Docker image is large because it embeds ruRoberta-large weights.
- `docs/evaluation.md` and some older docs still contain historical 3-class wording outside this Docker API update scope.

## Key files
- `api.py` — FastAPI service for health and single-text prediction.
- `Dockerfile` — CPU runtime image for the inference API.
- `.dockerignore` — keeps datasets, outputs, venv, and unrelated model variants out of Docker context.
- `requirements-api.txt` — runtime Python dependencies for the API image.
- `scripts/predict_sentiment.py` — source of production routing policy reused by the API.
- `model/ruroberta-sentiment-5class/` — expected current model directory.
