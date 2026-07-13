# Legacy: старые эксперименты и скрипты

Здесь собраны эксперименты и скрипты, которые не входят в основной 3-классовый пайплайн (`positive`/`negative`/`manual_review`). Оставлены для истории и воспроизводимости.

## Бинарный вариант (positive / negative)

Эксперимент: убрать шумный `manual_review` из обучения и учить только чистые `positive`/`negative`; спорные случаи на инференсе уводит в `manual_review` пороговая policy в `scripts/predict_sentiment.py`.

Подготовка split (строки `manual_review` отбрасываются):

```bash
python scripts/split_jsonl_binary_dataset.py \
  --input aj_reviews_export.jsonl \
  --output-dir data
# -> data/binary_train.tsv, data/binary_val.tsv, data/binary_test.tsv
```

Обучение — тонкая обёртка над `train_fast_bert_cpu.py` (та же two-phase схема, переопределены только пути, выходной каталог и метки):

```bash
MAX_TRAIN_SAMPLES=6000 MAX_EVAL_SAMPLES=1500 python train_fast_bert_binary_cpu.py
# -> outputs/rubert-binary-cpu/final
```

Оценка на held-out (скрипт 3-классовый; строку `manual_review` с support=0 игнорировать, macro считать по двум классам):

```bash
python scripts/validate_sentiment_on_tsv.py \
  --input data/binary_test.tsv \
  --model-dir outputs/rubert-binary-cpu/final \
  --limit 4000 --max-length 128
```

Последний результат на sample 4000 (train cap 6000):

```text
negative:         P 0.978 / R 0.931 / F1 0.954
positive:         P 0.511 / R 0.771 / F1 0.615
2-class macro F1: 0.784
```

Вывод: удаление `manual_review` подняло `positive` F1 0.569 -> 0.615 и `negative` 0.850 -> 0.954. Но сравнение не строго apples-to-apples (бинарный test не содержит `manual_review`), а `positive` precision всё ещё ~0.51 — узкое место сместилось с шума разметки на дисбаланс/нехватку positive-данных (в train их всего ~9.3k).

## Старый pipeline publish/reject

Ранняя 2-классовая постановка задачи (publish/reject). В текущем пайплайне не используется.

| Файл | Назначение |
|---|---|
| `scripts/split_dataset.py` | Старый split по `status` (publish/reject) |
| `scripts/train_publish_classifier.py` | Старый binary publish/reject fine-tuning |
| `scripts/evaluate_model.py` | Старая оценка publish/reject модели |
| `scripts/predict_review.py` | Старый binary inference |

## Старые скрипты разметки

| Файл | Назначение |
|---|---|
| `scripts/prepare_labeling_sample.py` | Детерминированный сэмпл для ручной разметки из сырого TSV |
| `scripts/heuristic_label_reviews.py` | Эвристические метки на сэмпле |

## Старые документы

- `docs/labeling_rubric.md` — рубрика разметки (использует старые имена меток `positive_about_work`/`negative_about_work`).
- `docs/training_guide_ru.md` — раннее руководство по обучению (предполагает другие скрипты и параметры, чем реализовано).

> Эти документы устарели и частично противоречат текущей реализации — см. [docs/TODO.md](TODO.md).
