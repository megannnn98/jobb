# Использование модели (inference)

Три способа:

- `predict.py` — короткий CLI: одно решение по тексту отзыва (см. ниже);
- `scripts/predict_sentiment.py` — policy-инференс с диагностикой и batch-режимом;
- `api.py` в Docker-контейнере — HTTP API для интеграции с внешним сервисом.

## Основная модель: ruRoberta-large

Текущая главная модель — `model/ruroberta-sentiment-5class/` (fine-tuned
`ai-forever/ruRoberta-large`, см. `CLAUDE.md`). Она классифицирует отзывы на
`positive`, `negative`, `manual_review`, `spam`, `service_complaint`.
Предыдущая 3-классовая ruRoberta (`model/ruroberta-sentiment/`) оставлена как исторический
baseline и локально больше не является main.

Проверка загрузки:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("model/ruroberta-sentiment-5class")
model = AutoModelForSequenceClassification.from_pretrained("model/ruroberta-sentiment-5class")
```

## CLI: `predict.py`

Одно решение одним словом:

```bash
python predict.py "Зарплату задерживают уже три месяца."
# negative
```

Режим JSON (label, confidence, probabilities):

```bash
python predict.py --json "Зарплату задерживают уже три месяца."
```

```json
{
  "label": "negative",
  "confidence": 0.5378,
  "probabilities": {
    "positive": 0.0014,
    "negative": 0.5378,
    "manual_review": 0.3604,
    "spam": 0.0139,
    "service_complaint": 0.0865
  }
}
```

`predict.py` использует чистый `argmax`. Для пороговой routing-policy
(auto-negative / auto-positive / `manual_review`) используйте `scripts/predict_sentiment.py`
или Docker API (ниже).

## Какую модель указывать (`scripts/predict_sentiment.py`)

В `--model-dir` указывается **каталог** обученной модели (не отдельный файл):

| Каталог для `--model-dir` | Классы |
|---|---|
| `model/ruroberta-sentiment-5class` | positive / negative / manual_review / spam / service_complaint (**основная модель**) |
| `model/ruroberta-sentiment` | positive / negative / manual_review (предыдущая 3-классовая модель, исторический baseline) |
| `outputs/rubert-binary-cpu/final` | positive / negative (см. [docs/legacy.md](legacy.md)) |

Не путать с `model/rubert-base-cased/` — это была базовая модель RuBERT (вход обучения, не
результат); удалена локально 2026-07-12 вместе с устаревшими RuBERT-чекпоинтами (см. `CLAUDE.md`).

`scripts/predict_sentiment.py` **автоматически определяет схему модели** по `id2label`: для
5-классовой модели дефолты порогов/температуры/`max-length` переключаются автоматически (см.
раздел «5-class variant» в `CLAUDE.md`), для 3-классовой поведение не меняется.

## Один отзыв

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment-5class \
  --text "Зарплату задерживают месяцами, руководство хамит, сотрудников штрафуют без причины."
```

Главное поле в выводе — `decision`. Полный набор полей (остальные — диагностика):

```text
decision
decision_policy
raw_top_label
raw_confidence
positive_prob
negative_prob
manual_review_prob
spam_prob                # только для 5-классовой модели
service_complaint_prob   # только для 5-классовой модели
```

## Inference policies

Способ, которым `decision` выводится из вероятностей, задаётся флагом `--policy`.

### `balanced` — default

Практичный режим для обычной сортировки:

```text
raw_top == negative && negative_prob >= 0.50 -> negative
raw_top == positive && positive_prob >= 0.70 -> positive
иначе                                        -> manual_review
```

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment-5class \
  --policy balanced \
  --text "Отличная компания, зарплата всегда вовремя."
```

Пороги по умолчанию для 5-классовой модели: `negative_threshold = 0.50`,
`positive_threshold = 0.70`, `temperature = 0.25`, `max_length = 128`. Без temperature scaling
`negative_prob` был бы бесполезен для порогов (не превышает ~0.33 независимо от уверенности
модели — см. `CLAUDE.md`).

### `safe`

Более строгий режим: автоматически пропускает только уверенный `negative`, остальное отправляет в `manual_review`. Auto-positive выключен, если явно не указан `--enable-auto-positive`.

```bash
python scripts/predict_sentiment.py \
  --policy safe \
  --model-dir model/ruroberta-sentiment-5class \
  --text "Зарплату задерживают месяцами, руководство хамит."
```

Пороги по умолчанию для 5-классовой модели остаются `negative_threshold = 0.50`,
`positive_threshold = 0.70`; в `safe` auto-positive выключен, если явно не указан
`--enable-auto-positive`.

### 3-классовая модель

При `--model-dir model/ruroberta-sentiment` скрипт переключается на старые 3-классовые дефолты:
`negative_threshold=0.50` для `balanced`, `negative_threshold=0.60` для `safe`,
`positive_threshold=0.90`, `temperature=1.0`, `max_length=512`.

### `argmax`

Legacy/debug-режим: берёт top-label модели и отправляет в `manual_review`, если confidence ниже `--threshold`.

```bash
python scripts/predict_sentiment.py \
  --policy argmax \
  --threshold 0.5 \
  --model-dir model/ruroberta-sentiment-5class \
  --text "Работал недолго, были плюсы и минусы."
```

## Batch inference

Входной TSV должен содержать колонку `text` (лишние колонки сохраняются):

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment-5class \
  --input data/sentiment_test.tsv \
  --output outputs/predictions.tsv
```

В выходной TSV к исходным колонкам добавляются `decision`, `decision_policy`, `raw_top_label`, `raw_confidence`, `positive_prob`, `negative_prob`, `manual_review_prob`. Без `--output` первые строки печатаются в консоль.

## Docker API

`api.py` публикуется через Dockerfile как FastAPI-сервис. Контейнер использует ту же `balanced`
policy, что и `scripts/predict_sentiment.py`, и возвращает те же диагностические поля.

Сборка и запуск:

```bash
docker build -t jobb-sentiment-api .
docker run --rm -p 8000:8000 jobb-sentiment-api
```

По умолчанию Dockerfile копирует модель из `model/ruroberta-sentiment-5class/` внутрь image.
Модель загружается на startup; если каталог отсутствует, сервис не стартует.

Проверка:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Зарплату задерживают уже три месяца."}'
```

Для хранения весов отдельно от image переопределите `MODEL_DIR` и примонтируйте каталог модели:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models/ruroberta-sentiment-5class \
  -v /srv/jobb/model/ruroberta-sentiment-5class:/models/ruroberta-sentiment-5class:ro \
  jobb-sentiment-api
```

## Проверочные примеры

Positive:

```bash
python scripts/predict_sentiment.py --model-dir model/ruroberta-sentiment-5class --text "Отличная компания, зарплата всегда вовремя, оформление официальное, руководство адекватное, коллектив дружный."
```

Negative:

```bash
python scripts/predict_sentiment.py --model-dir model/ruroberta-sentiment-5class --text "Не устраивайтесь сюда: договор не дают, начальник орёт, штрафуют без объяснений, переработки не оплачивают."
```

Manual review:

```bash
python scripts/predict_sentiment.py --model-dir model/ruroberta-sentiment-5class --text "Работал недолго, были плюсы и минусы, зарплату платили, но руководство часто меняло правила."
```
