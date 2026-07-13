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
model/ruroberta-sentiment/    ← основная модель (ruRoberta-large, 355M)
        |
        +-- predict.py                          (CLI, см. docs/inference.md)
        +-- scripts/predict_sentiment.py        (policy, см. docs/inference.md)
        +-- scripts/validate_sentiment_on_tsv.py (см. docs/evaluation.md)
        +-- scripts/validate_sentiment_on_jsonl.py
```

Актуальный основной артефакт — `model/ruroberta-sentiment/` (ruRoberta-large, обучена на Kaggle
P100, macro F1 0.7811 на полном held-out test при MAX_LENGTH=512 — см.
[docs/evaluation.md](evaluation.md)). Три более ранних RuBERT-чекпоинта (`gpu1`/`gpu2`/`v2`)
превосходились по метрикам на каждом шаге и были удалены локально 2026-07-12 (воспроизводимы из
committed training-скриптов при необходимости — см. `CLAUDE.md`, таблица lineage). Детали —
[docs/training.md](training.md) и [docs/gpu_training.md](gpu_training.md).

Параллельно существует **экспериментальная 5-классовая ветка** (`positive`/`negative`/
`manual_review`/`spam`/`service_complaint`, `model/ruroberta-sentiment-5class/`,
`kaggle/ruroberta-5class/`) с независимой LLM-разметкой (DeepSeek) подмножества `manual_review` —
не промотирована в main, см. `CLAUDE.md`, раздел «5-class variant».

## Ключевые проектные решения

### `manual_review` — статус модерации, не текстовый класс

В текущей разметке `manual_review` отражает скорее статус модерации (`status == 0 && isPositive != 1`), чем чистую текстовую тональность. Часть таких отзывов семантически выглядит как явный `positive`/`negative`. Поэтому:

- inference не полагается на чистый `argmax`, а использует пороговую routing policy (см. [docs/inference.md](inference.md));
- спорные и неуверенные случаи уводятся в `manual_review`.

### Пороговая политика вместо argmax

По held-out (см. [docs/evaluation.md](evaluation.md), ruRoberta-large) `negative` уверенно
отделяется (precision ~0.928 при `negative_threshold=0.50`), а `positive` — нет (precision
максимум ~0.847, ниже целевых 0.90). Отсюда асимметрия в policy: auto-negative включается раньше,
auto-positive требует высокого порога или выключен. В экспериментальной 5-классовой модели та же
идея реализована через temperature scaling поверх softmax (см. `CLAUDE.md`) — там без этой правки
`negative_prob` вообще не годился для порога из-за экстремальных class weights.

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
