# Валидация и метрики

## Held-out test

Честная оценка на `data/sentiment_test.tsv` (для GPU-модели):

```bash
python scripts/validate_sentiment_on_tsv.py \
  --model-dir model/rubert-sentiment-gpu1 \
  --input data/sentiment_test.tsv \
  --max-length 128 \
  --limit 4000
```

(добавьте `--threshold-grid` для подбора порогов policy.)

Выходные отчёты:

```text
outputs/validation_test_report.txt
outputs/threshold_grid_report.txt
```

## Текущая основная модель: ruRoberta-large

Обучена на Kaggle P100 (`train_ruroberta.py` / `kaggle/ruroberta/train.py`, ~3ч08м), оценена на
Kaggle **T4** GPU через отдельный кернел `kaggle/eval/` — на **полном** test-сплите (17 523 строки,
не выборка). Инференс использует `MAX_LENGTH=512` (не 128, как при обучении — см.
`CLAUDE.md`, раздел «Inference-time MAX_LENGTH=512»), 1455с (12.0 ex/s):

```text
accuracy:                    0.8650  (15158/17523)
macro accuracy / avg recall: 0.7871
macro F1:                    0.7811
positive:        P 0.7482 / R 0.8684 / F1 0.8038  (support 1170)
negative:        P 0.9095 / R 0.9451 / F1 0.9269  (support 13049)
manual_review:   P 0.6946 / R 0.5478 / F1 0.6125  (support 3304)
```

(При `MAX_LENGTH=128`, то есть той же длине, что и при обучении: accuracy 0.8543, macro F1
0.7705 — см. `outputs/eval-len512/` для отчёта при 512 и исходный `kaggle/eval/` прогон для 128.)

Обходит лучшую версию RuBERT (V2: accuracy 0.825 / macro F1 0.733) по всем метрикам, особенно
заметно на `manual_review` (+5.8 п.п. F1 при 512) — историческом самом слабом месте проекта.

### Threshold grid (полный test, ruRoberta-large, MAX_LENGTH=512)

```text
auto-negative: neg_thr=0.50 -> precNeg=0.928 (ВКЛ)
auto-positive: max precPos=0.847 при pos_thr=0.90 (< 0.90 цели) -> ОТКЛЮЧИТЬ
```

**Важное отличие от RuBERT:** ruRoberta обучена с `label_smoothing=0.1`, и её вероятности заметно
менее «острые». В полном test-сплите `neg_prob` практически никогда не превышает ~0.65–0.69 — то
есть порог `neg_thr=0.70`, который был `safe`-порогом для RuBERT V2, на ruRoberta даёт **0%
покрытия** (ни одного примера). Ближайший рабочий аналог `safe`-порога для ruRoberta — `neg_thr=0.60`
(precNeg=0.960, но покрытие только ~7% test, а не как у RuBERT). Пороги при переходе с 128 на 512
почти не сдвинулись (не потребовали пересчёта). Актуальные дефолты — `scripts/predict_sentiment.py`.

## Экспериментальная 5-классовая модель (не текущая основная)

`model/ruroberta-sentiment-5class/` — параллельный вариант с классами `positive` / `negative` /
`manual_review` / `spam` / `service_complaint` (см. `CLAUDE.md`, раздел «5-class variant», для
методологии релейблинга через DeepSeek). Оценка на полном test (17 523 строки, `kaggle/eval-5class/`,
**MAX_LENGTH=128** — 512 пока не перепроверялся):

```text
accuracy:      0.9161  (16053/17523)
macro F1:      0.7538   (не сравнимо напрямую с 3-классовой 0.7811 — другой набор классов)

positive:            P 0.898 / R 0.862 / F1 0.880  (support 1407)
negative:            P 0.993 / R 0.927 / F1 0.959  (support 14491)
manual_review:       P 0.318 / R 0.793 / F1 0.454  (support 347)
spam:                P 0.799 / R 0.899 / F1 0.846  (support 741)
service_complaint:   P 0.496 / R 0.868 / F1 0.631  (support 537)
```

**Важная находка:** в 5-классовой схеме экстремальный дисбаланс class weights (`manual_review`
~10.68 против `negative` ~0.24, отношение ~44x) полностью схлопывал `negative_prob` в узкую полосу
~0.31 — auto-negative порог давал 0% покрытия при любом значении. Исправлено temperature scaling
(`softmax(logits / 0.25)`, без переобучения) — восстанавливает `neg_thr=0.50 → precision 0.996`
(12 179/17 523 строк) и `pos_thr=0.70 → precision 0.904` (впервые проходит цель 0.90 для positive).
Суммарное авто-покрытие ≈77%. См. `outputs/eval-5class/ruroberta_threshold_grid.txt` и подробности
в `CLAUDE.md`.

### RuBERT GPU1 — история (не текущая модель)

Модель обучена на **80/10/10** стратифицированном сплите (train/val/test), seed=20260630.
Held-out test (sample 4000):

```text
accuracy:                    0.8077
macro accuracy / avg recall: 0.7454
macro F1:                    0.7049
positive:        P 0.5990 / R 0.8551 / F1 0.7045  (support 276)
negative:        P 0.8969 / R 0.8783 / F1 0.8875  (support 2990)
manual_review:   P 0.5442 / R 0.5027 / F1 0.5227  (support 734)
```

Подробнее — [docs/gpu_training.md](gpu_training.md).

### CPU baseline — для сравнения (legacy)

Тот же held-out sample 4000, но на CPU-модели (`outputs/rubert-sentiment-cpu/final`):

```text
overall accuracy: 0.7548
macro F1:         0.6234
negative:         P 0.867 / R 0.834 / F1 0.850
positive:         P 0.452 / R 0.768 / F1 0.569
manual_review:    P 0.479 / R 0.426 / F1 0.451
```

GPU-модель заметно сильнее baseline (macro F1 0.70 против 0.62).

### Threshold grid

```text
auto-negative: neg_thr=0.55 -> precNeg=0.902 (ВКЛ)
auto-positive: max precPos=0.829 при pos_thr=0.90 (< 0.90 цели) -> ОТКЛЮЧИТЬ
```

Поэтому `safe` режим отключает auto-positive, а `balanced` включает auto-positive только при `positive_prob >= 0.90` (см. [docs/inference.md](inference.md)).

## Проверка на JSONL

Валидация на реальных примерах из исходного JSONL:

```bash
python scripts/validate_sentiment_on_jsonl.py \
  --jsonl aj_reviews_export.jsonl \
  --model-dir outputs/rubert-sentiment-cpu/final \
  --samples-per-class 100
```

Отчёт:

```text
outputs/validation_report_100.txt
```

Эта проверка полезна для просмотра реальных ошибок, но может включать строки из train, поэтому held-out `data/sentiment_test.tsv` важнее для честной метрики.

## Ограничения валидации (важно)

- `validate_sentiment_on_tsv.py` на **CPU** медленный (исторически ~7.6 отзывов/сек на RuBERT;
  ruRoberta-large на CPU ещё медленнее). Полный test **без** `--limit` на CPU может выглядеть как
  зависание — используйте `--limit` (test-файл предварительно перемешан, `--limit N` даёт
  репрезентативную пропорциональную выборку).
- **На GPU полный test — не проблема.** `kaggle/eval/eval.py` прогоняет весь test (17 523 строки)
  за ~450с на T4 (batched inference, `device="cuda"`). Это решает старый TODO «прогнать полный test
  без `--limit»` — просто нужно делать это на GPU (Kaggle), а не локально на CPU.
- `manual_review` остаётся самым слабым классом и, вероятно, содержит шумную/смешанную разметку —
  расхождение по этому классу ожидаемо, а не баг модели (хотя у ruRoberta он заметно лучше, чем
  у любой версии RuBERT).

## Эксперименты

Результаты бинарного эксперимента (positive/negative) вынесены в [docs/legacy.md](legacy.md).
