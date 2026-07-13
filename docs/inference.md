# Использование модели (inference)

Два способа:

- `predict.py` — короткий CLI: одно решение по тексту отзыва (см. ниже);
- `scripts/predict_sentiment.py` — policy-инференс с диагностикой и batch-режимом.

## Основная модель: ruRoberta-large

Текущая главная модель — `model/ruroberta-sentiment/` (fine-tuned `ai-forever/ruRoberta-large`,
см. `CLAUDE.md`). Три более слабых RuBERT-чекпоинта (`gpu1`/`gpu2`/`v2`) и базовая
`model/rubert-base-cased/` были удалены локально 2026-07-12 как устаревшие — воспроизводимы через
их training-скрипты при необходимости (см. `CLAUDE.md`, таблица lineage).

Проверка загрузки:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("model/ruroberta-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("model/ruroberta-sentiment")
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
  "confidence": 0.5005,
  "probabilities": {
    "positive": 0.2293,
    "negative": 0.5005,
    "manual_review": 0.2702
  }
}
```

`predict.py` использует чистый `argmax` (модель `model/ruroberta-sentiment/`). Для пороговой
routing-policy (auto-negative / auto-positive / `manual_review`) используйте
`scripts/predict_sentiment.py` (ниже).

## Какую модель указывать (`scripts/predict_sentiment.py`)

В `--model-dir` указывается **каталог** обученной модели (не отдельный файл):

| Каталог для `--model-dir` | Классы |
|---|---|
| `model/ruroberta-sentiment` | positive / negative / manual_review (**основная модель**) |
| `model/ruroberta-sentiment-5class` | positive / negative / manual_review / spam / service_complaint (экспериментальная, см. `CLAUDE.md`) |
| `outputs/rubert-binary-cpu/final` | positive / negative (см. [docs/legacy.md](legacy.md)) |

Не путать с `model/rubert-base-cased/` — это была базовая модель RuBERT (вход обучения, не
результат); удалена локально 2026-07-12 вместе с устаревшими RuBERT-чекпоинтами (см. `CLAUDE.md`).

`scripts/predict_sentiment.py` **автоматически определяет схему модели** по `id2label`: для
5-классовой модели дефолты порогов/температуры/`max-length` переключаются автоматически (см.
раздел «5-class variant» в `CLAUDE.md`), для 3-классовой поведение не меняется.

## Один отзыв

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment \
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
raw_top == positive && positive_prob >= 0.90 -> positive
иначе                                        -> manual_review
```

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment \
  --policy balanced \
  --text "Отличная компания, зарплата всегда вовремя."
```

Пороги по умолчанию: `negative_threshold = 0.50`, `positive_threshold = 0.90` (ruRoberta-large;
RuBERT использовал другие значения — см. `CLAUDE.md`, там же — почему `neg_thr=0.70` не работает
на ruRoberta: `label_smoothing=0.1` делает вероятности заметно менее «острыми»).

### `safe`

Более строгий режим: автоматически пропускает только уверенный `negative`, остальное отправляет в `manual_review`. Auto-positive выключен, если явно не указан `--enable-auto-positive`.

```bash
python scripts/predict_sentiment.py \
  --policy safe \
  --model-dir model/ruroberta-sentiment \
  --text "Зарплату задерживают месяцами, руководство хамит."
```

Пороги по умолчанию: `negative_threshold = 0.60`, `positive_threshold = 0.90`.

### Экспериментальная 5-классовая модель

При `--model-dir model/ruroberta-sentiment-5class` пороги, temperature и `--max-length`
переключаются на 5-классовые дефолты автоматически (см. `CLAUDE.md`, «5-class variant»):
`negative_threshold=0.50`, `positive_threshold=0.70`, `--temperature=0.25`, `--max-length=128`.
Без temperature scaling `negative_prob` там был бы бесполезен для порогов (не превышает ~0.33
независимо от уверенности модели — см. `CLAUDE.md`). Вывод дополнительно включает `spam_prob` и
`service_complaint_prob`.

### `argmax`

Legacy/debug-режим: берёт top-label модели и отправляет в `manual_review`, если confidence ниже `--threshold`.

```bash
python scripts/predict_sentiment.py \
  --policy argmax \
  --threshold 0.5 \
  --model-dir model/ruroberta-sentiment \
  --text "Работал недолго, были плюсы и минусы."
```

## Batch inference

Входной TSV должен содержать колонку `text` (лишние колонки сохраняются):

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment \
  --input data/sentiment_test.tsv \
  --output outputs/predictions.tsv
```

В выходной TSV к исходным колонкам добавляются `decision`, `decision_policy`, `raw_top_label`, `raw_confidence`, `positive_prob`, `negative_prob`, `manual_review_prob`. Без `--output` первые строки печатаются в консоль.

## Проверочные примеры

Positive:

```bash
python scripts/predict_sentiment.py --model-dir model/ruroberta-sentiment --text "Отличная компания, зарплата всегда вовремя, оформление официальное, руководство адекватное, коллектив дружный."
```

Negative:

```bash
python scripts/predict_sentiment.py --model-dir model/ruroberta-sentiment --text "Не устраивайтесь сюда: договор не дают, начальник орёт, штрафуют без объяснений, переработки не оплачивают."
```

Manual review:

```bash
python scripts/predict_sentiment.py --model-dir model/ruroberta-sentiment --text "Работал недолго, были плюсы и минусы, зарплату платили, но руководство часто меняло правила."
```
