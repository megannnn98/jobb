# Устройство проекта

## Pipeline

```text
aj_reviews_export.jsonl
        |
        v
scripts/split_jsonl_sentiment_dataset.py      (см. docs/data.md)
        |
        +-- data/sentiment_train.tsv
        +-- data/sentiment_val.tsv
        +-- data/sentiment_test.tsv
        |
        v
train_ruroberta.py / kaggle/ruroberta/train.py (Kaggle P100, см. docs/training.md, CLAUDE.md)
        |
        v
model/ruroberta-sentiment-5class/    ← основная модель (ruRoberta-large, 5 классов)
        |
        +-- predict.py                          (CLI, см. docs/inference.md)
        +-- scripts/predict_sentiment.py        (policy, см. docs/inference.md)
        +-- api.py + Dockerfile                 (HTTP API, см. docs/inference.md)
        +-- scripts/validate_sentiment_on_tsv.py (см. docs/evaluation.md)
        +-- scripts/validate_sentiment_on_jsonl.py
```

Актуальный основной артефакт — `model/ruroberta-sentiment-5class/` (ruRoberta-large,
`positive`/`negative`/`manual_review`/`spam`/`service_complaint`). 5-классовая модель обучена на
Kaggle и использует независимую LLM-разметку (DeepSeek) подмножества прежнего `manual_review` для
выделения `spam` и `service_complaint`; детали lineage и threshold policy см. в `CLAUDE.md` и
[docs/inference.md](inference.md). Предыдущая 3-классовая ruRoberta осталась историческим
baseline и локально больше не является main.

Для интеграции с внешним сервисом добавлен контейнерный слой: `api.py` поднимает FastAPI,
загружает модель на startup и публикует `GET /health` и `POST /predict`. `Dockerfile` собирает
CPU image с PyTorch/Transformers и по умолчанию копирует `model/ruroberta-sentiment-5class/`
внутрь image; при запуске можно переопределить `MODEL_DIR` и примонтировать веса отдельно.

## Ключевые проектные решения

### `manual_review` — статус модерации, не текстовый класс

В текущей разметке `manual_review` отражает скорее статус модерации (`status == 0 && isPositive != 1`), чем чистую текстовую тональность. Часть таких отзывов семантически выглядит как явный `positive`/`negative`. Поэтому:

- inference не полагается на чистый `argmax`, а использует пороговую routing policy (см. [docs/inference.md](inference.md));
- спорные и неуверенные случаи уводятся в `manual_review`.

### Пороговая политика вместо argmax

В основной 5-классовой модели routing policy использует temperature scaling поверх softmax
(`temperature=0.25`) и дефолтные пороги `negative_threshold=0.50`,
`positive_threshold=0.70`. Без scaling `negative_prob` не годился для порога из-за экстремальных
class weights. `spam`, `service_complaint`, `manual_review` и всё, что не прошло пороги,
уходит в `manual_review`.

### Кодировка текста

Текущий экспорт фактически чистый UTF-8 (mojibake — единичные легаси-строки). На inference подавайте обычный UTF-8, перекодировать не нужно. Подробнее — [docs/data.md](data.md).

### Двухфазное обучение

Frozen backbone → размораживание последних N encoder-слоёв, weighted cross-entropy для дисбаланса
(`WeightedTrainer`). N=2 в исходных RuBERT GPU1/GPU2, N=4 начиная с RuBERT V2 и в
`train_ruroberta.py`/5-классовом варианте. Схема одинакова для всех вариантов; отличаются
окружение, батчи, бюджет эпох и (в 5-классовом случае) экстремальность class weights из-за
маленькой доли `manual_review`. Подробнее — [docs/training.md](training.md), `CLAUDE.md`
(«Training architecture», «5-class variant»), история GPU-прогона —
[docs/gpu_training.md](gpu_training.md).
