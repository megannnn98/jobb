# Классификация отзывов о работодателях

Модель классифицирует отзывы о работодателях на пять категорий:

- `positive` — явно позитивный отзыв о работодателе
- `negative` — явно негативный отзыв о работодателе
- `manual_review` — спорный/неоднозначный случай для ручной проверки
- `spam` — не отзыв о работодателе вообще (флуд, реклама, мусор)
- `service_complaint` — жалоба клиента на товар/услугу компании, не на неё как работодателя

**Текущая основная модель — 5-классовая ruRoberta-large** (fine-tuned `ai-forever/ruRoberta-large`,
`spam`/`service_complaint` независимо размечены через DeepSeek поверх исходной 3-классовой
разметки — см. [CLAUDE.md](CLAUDE.md), раздел «5-class variant»). Предыдущая 3-классовая версия
(`positive`/`negative`/`manual_review`) обходила все версии RuBERT по качеству, но её локальные
веса удалены после промоушена 5-класса — воспроизводима через `train_ruroberta.py` или публичный
Kaggle Dataset [`megannnn98/ruroberta-sentiment`](https://www.kaggle.com/datasets/megannnn98/ruroberta-sentiment)
(см. [«Качество модели»](#качество-модели)).

## Быстрый старт

### 1. Скачай модель

**Ссылка:** [megannnn98/ruroberta-sentiment-5class](https://www.kaggle.com/datasets/megannnn98/ruroberta-sentiment-5class)

Модель нужно положить в папку `model/ruroberta-sentiment-5class/` внутри репозитория.

*Руками:*
1. Скачай архив с Kaggle
2. Распакуй в `model/ruroberta-sentiment-5class/` (рядом с `predict.py`):

```text
jobb/
├── predict.py
├── model/
│   └── ruroberta-sentiment-5class/   ← сюда распаковать
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── ...
```

*Через Kaggle CLI:*
```bash
pip install kaggle
kaggle datasets download -d megannnn98/ruroberta-sentiment-5class -p model/ruroberta-sentiment-5class/ --unzip
```

### 2. Установи зависимости

```bash
pip install torch transformers
```

### 3. Запусти

```bash
python predict.py "Зарплату задерживают уже три месяца."
# → negative
```

Или с подробностями:
```bash
python predict.py --json "Отличная компания, зарплата всегда вовремя."
```

```json
{
  "label": "positive",
  "confidence": 0.829,
  "probabilities": {
    "positive": 0.829,
    "negative": 0.004,
    "manual_review": 0.080,
    "spam": 0.035,
    "service_complaint": 0.053
  }
}
```

> `predict.py`/`main.py` — чистый argmax без порогов. У этой модели (сильный дисбаланс class
> weights из-за маленькой доли `manual_review`) argmax-вероятности `positive`/`negative` часто
> выглядят невысокими (0.3–0.6) даже когда предсказание верное — это ожидаемо, не баг (подробнее —
> [CLAUDE.md](CLAUDE.md), «5-class variant»). Для продакшен-решений с адекватными порогами
> используй `scripts/predict_sentiment.py` (ниже) — он сам исправляет это через temperature scaling.

## Примеры

```bash
# Негативный отзыв
python predict.py "Начальник хамит, штрафуют без причины, переработки не оплачивают."
# → negative

# Позитивный отзыв
python predict.py "Официальное оформление, зарплата вовремя, дружный коллектив."
# → positive
```

Спорные отзывы `scripts/predict_sentiment.py` уводит в `manual_review`, а не пытается угадать
позитив/негатив:

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment-5class \
  --text "Работал недолго, были плюсы и минусы."
# decision: manual_review (raw_top_label тоже manual_review, confidence 0.55)
```

## Policy-инференс (для продакшена)

Для автоматической модерации с порогами используй `scripts/predict_sentiment.py` — он сам
определяет 5-классовую схему по модели и включает нужные пороги/temperature scaling:

```bash
python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment-5class \
  --text "Зарплату задерживают месяцами."
```

### Пороги (held-out test, полный test-сплит, 5-классовая ruRoberta-large, temperature=0.25)

| Политика | auto-negative | auto-positive | precNeg | precPos | суммарное покрытие |
|---|---|---|---|---|---|
| `balanced` (дефолт) | `neg_prob >= 0.50` | `pos_prob >= 0.70` | 0.996 | 0.904 | ~77% |

- Без temperature scaling `negative_prob` был бы бесполезен для порога (экстремальные class
  weights схлопывают его в узкую полосу ~0.31 независимо от уверенности) — см. `CLAUDE.md`.
- `spam`/`service_complaint`/`manual_review` как top-label всегда уходят в `manual_review`.
- Всё, что не прошло порог — тоже `manual_review`.

Подробнее — [docs/inference.md](docs/inference.md).

## Docker API

Контейнер поднимает FastAPI-сервис поверх production policy из `scripts/predict_sentiment.py`.
По умолчанию модель встраивается в image из локальной папки `model/ruroberta-sentiment-5class/`,
поэтому перед сборкой она должна уже лежать в репозитории.

```bash
docker build -t jobb-sentiment-api .
docker run --rm -p 8000:8000 jobb-sentiment-api
```

Сервис грузит модель на startup. Если папка модели не найдена, контейнер завершится с ошибкой.
Локально собранный image получается большим, потому что содержит веса ruRoberta-large.

### Endpoints

- `GET /health` — проверка, что API поднялся и какая папка модели используется.
- `POST /predict` — классификация одного текста через `balanced` policy.

Проверка после запуска:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Зарплату задерживают уже три месяца."}'
```

Пример ответа `/predict`:

```json
{
  "decision": "negative",
  "decision_policy": "balanced",
  "raw_top_label": "negative",
  "raw_confidence": 0.5378,
  "positive_prob": 0.0014,
  "negative_prob": 0.5378,
  "manual_review_prob": 0.3604,
  "spam_prob": 0.0139,
  "service_complaint_prob": 0.0865
}
```

Для интеграции вместо GPT отправляй текст отзыва в `/predict`; использовать нужно поле
`decision`. `raw_top_label` и вероятности — диагностические поля для логов/разбора спорных
случаев. `manual_review` означает, что автоматическое решение не принято.

### Модель снаружи контейнера

Dockerfile по умолчанию копирует модель внутрь image. Если на сервере удобнее хранить веса
отдельно, можно переопределить `MODEL_DIR` и примонтировать каталог:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models/ruroberta-sentiment-5class \
  -v /srv/jobb/model/ruroberta-sentiment-5class:/models/ruroberta-sentiment-5class:ro \
  jobb-sentiment-api
```

Каталог должен содержать минимум `config.json`, `model.safetensors`, `tokenizer.json` и
`tokenizer_config.json`.

## Качество модели

**5-классовая модель (текущая основная), held-out test, весь test-сплит (17523), MAX_LENGTH=128:**

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| positive | 0.898 | 0.862 | 0.880 | 1407 |
| negative | 0.993 | 0.927 | 0.959 | 14491 |
| manual_review | 0.318 | 0.793 | 0.454 | 347 |
| spam | 0.799 | 0.899 | 0.846 | 741 |
| service_complaint | 0.496 | 0.868 | 0.631 | 537 |

Accuracy 0.916, macro F1 0.754 — не сравнимо напрямую с 3-классовыми числами ниже (другой набор
классов, `manual_review` сжался с ~3300 до 347 строк после независимой LLM-разметки через
DeepSeek — см. [CLAUDE.md](CLAUDE.md)). MAX_LENGTH=512 для этой модели не перепроверялся
(решение — см. `docs/TODO.md`).

**3-классовая модель (предыдущая основная, оставлена для сравнения):**

| Версия | Test-сэмпл | Accuracy | Macro F1 | Pos F1 | Neg F1 | Man F1 |
|---|---|---|---|---|---|---|
| RuBERT GPU1 (T4x2) | 4000 | 0.808 | 0.705 | 0.705 | 0.888 | 0.523 |
| RuBERT GPU2 (P100) | 4000 | 0.815 | 0.710 | 0.714 | 0.894 | 0.522 |
| RuBERT V2 (P100) | 4000 | 0.825 | 0.733 | 0.748 | 0.898 | 0.554 |
| **ruRoberta-large** (T4, MAX_LENGTH=512) | **весь test (17523)** | **0.865** | **0.781** | **0.804** | **0.927** | **0.612** |

ruRoberta-large обходила лучший RuBERT (V2) по всем метрикам, особенно на самом слабом
и проблемном классе `manual_review` (+5.8 п.п. F1). Оценивалась на **полном** test-сплите,
не на сэмпле 4000 (это стало возможно только на GPU — на CPU/малой выборке полный прогон
был бы слишком медленным). Инференс использовал `MAX_LENGTH=512` (не 128, как при обучении —
см. [docs/evaluation.md](docs/evaluation.md)); при 128 accuracy/macro F1 чуть ниже (0.854/0.771).

Подробнее — [docs/evaluation.md](docs/evaluation.md).

## Структура проекта

```text
model/ruroberta-sentiment-5class/  # основная модель (скачать с Kaggle)
predict.py                         # CLI: классификация одного отзыва
scripts/predict_sentiment.py       # policy-инференс с порогами
api.py                             # FastAPI: /health и /predict
Dockerfile                         # контейнер для API
main.py                            # demo
docs/                              # документация
```

## Документация

- [docs/inference.md](docs/inference.md) — использование модели, inference policies, batch inference
- [docs/evaluation.md](docs/evaluation.md) — метрики модели (5-класс и 3-класс ruRoberta-large)
- [docs/architecture.md](docs/architecture.md) — устройство проекта
- [docs/gpu_training.md](docs/gpu_training.md) — история обучения RuBERT (GPU1/GPU2/V2)
- [docs/model-tuning.md](docs/model-tuning.md) — тюнинг гиперпараметров RuBERT GPU1→GPU2→V2
