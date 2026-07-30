FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model/ruroberta-sentiment-5class

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.7,<3"

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api.py predict.py main.py ./
COPY scripts ./scripts
COPY model/ruroberta-sentiment-5class ./model/ruroberta-sentiment-5class

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
