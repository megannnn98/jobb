# GPU-обучение RuBERT (Kaggle, T4x2)

Короткая история эксперимента по переходу с CPU-baseline на GPU-обучение.

## Проблема

CPU-обучение (`train_fast_bert_cpu.py`) было слишком медленным для прогона на полном train-наборе:
итерации занимали часы, поэтому baseline обучался на урезанных данных и давал слабые метрики
(macro F1 ~0.62 на held-out 4000, см. [docs/evaluation.md](evaluation.md)).

## Решение

Перенос обучения в **Kaggle Notebook** с GPU-акселератором:

- **Accelerator:** GPU **T4x2** (фактически использовалась одна GPU, окружение GPU-enabled);
- **базовая модель:** `DeepPavlov/rubert-base-cased` (тянется с HuggingFace Hub);
- **данные:** 80/10/10 стратифицированный сплит (train/val/test), seed=20260630;
- `max_length = 128`;
- двухфазная схема:
  - **phase 1:** frozen backbone, обучается только classifier head;
  - **phase 2:** размораживаются последние 2 слоя BERT, fine-tune.

## Настройки (отличия от CPU baseline)

| Параметр | CPU | GPU |
|---|---|---|
| `MODEL_NAME` | `model/rubert-base-cased` | `DeepPavlov/rubert-base-cased` |
| `use_cpu` | `True` | `False` |
| `fp16` | — | `True` |
| `dataloader_pin_memory` | `False` | `True` |
| `DATALOADER_NUM_WORKERS` | `0` | `2` |
| `PHASE1_BATCH_SIZE` | `32` | `64` |
| `PHASE2_BATCH_SIZE` | `8` | `16` |
| `PHASE2_GRADIENT_ACCUMULATION` | `4` | `2` |
| `PHASE1_EPOCHS` | `2` | `1` |
| `PHASE2_EPOCHS` | `1` | `2` |
| Output dir | `outputs/rubert-sentiment-cpu` | `outputs/rubert-sentiment` |

Полная таблица с обоснованием — [docs/training.md](training.md).

## Результаты

### Held-out test (sample 4000)

Модель обучена на **80/10/10** сплите (ранее — на 100% данных, метрики сравнимы).

```text
accuracy:                    0.8077
macro accuracy / avg recall: 0.7454
macro F1:                    0.7049
```

По классам:

| Класс | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.5990 | 0.8551 | 0.7045 | 276 |
| negative | 0.8969 | 0.8783 | 0.8875 | 2990 |
| manual_review | 0.5442 | 0.5027 | 0.5227 | 734 |

GPU-модель заметно сильнее CPU baseline (held-out macro F1 0.705 против ~0.62). `negative` —
устойчиво сильный класс; `manual_review` — самый слабый (вероятно, шумная/смешанная разметка,
см. [docs/architecture.md](architecture.md)).

## Что дальше

- ускорить `validate_sentiment_on_tsv.py` (сейчас ~7.6 отзывов/сек на sample 4000; полный test без
  `--limit` выглядит как зависание) — переписать на batch inference (`Trainer.predict` / `DataLoader`);
- прогнать полный test без `--limit` после ускорения валидации;
- попробовать разморозить последние **4** слоя BERT в phase 2;
- ревизовать `manual_review` (шумная/смешанная разметка);
- сохранить experiment bundle: model + `train_fast_bert.py` + validation report.

Полный список — [docs/TODO.md](TODO.md).
