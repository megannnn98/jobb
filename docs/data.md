# Данные и подготовка датасета

## Источник: `aj_reviews_export.jsonl`

Локальный JSONL-экспорт отзывов. Не хранится в Git (большой). Перед подготовкой данных положите его в корень проекта.

| Поле | Назначение |
|---|---|
| `id` | ID отзыва |
| `name` | Название компании |
| `descr` | Текст отзыва (вход модели) |
| `status` | Статус модерации |
| `isPositive` | Флаг позитивности |
| `isfunny` | Служебный флаг |
| `is_about_work` | Служебный флаг |
| `is_self_delete` | Служебный флаг |

### Кодировка текста

Текущий экспорт (`aj_reviews_export.jsonl`, 175 804 строк) — фактически **чистый UTF-8**: скан
показывает ~99% корректной кириллицы и лишь единичные строки (≈2) в старом mojibake (двойная
UTF-8, прочитанная как latin-1, например `Ð¤Ð¸Ñ€Ð¼Ð°`). Скрипт сплита подаёт `descr` как есть, поэтому
модель обучена практически на чистом тексте.

**На inference подавайте обычный UTF-8** — перекодировать/«чинить» ничего не нужно. Проверено
эмпирически на `model/rubert-sentiment-gpu1`: чистый текст даёт корректные уверенные предсказания,
а искусственная mojibake-перекодировка (`text.encode("latin-1").decode("utf-8")` наоборот) ломает
их и уводит почти всё в `manual_review`.

> Историческая заметка: ранний экспорт был массово mojibake, отсюда хелпер `decode_for_display`
> (`latin-1`→`utf-8` + `html.unescape`) в validation-скриптах. Для текущих чистых данных он
> почти всегда no-op (на чистой кириллице `encode("latin-1")` упал бы, поэтому он best-effort и
> только для читаемого отчёта). Прежнее требование «подавать mojibake и на inference» — **неверно**
> для текущего экспорта.

## Разметка (label mapping)

Текущий mapping в `scripts/split_jsonl_sentiment_dataset.py`:

```text
status == 1                  -> negative
status == 0 && isPositive==1 -> positive
остальное                    -> manual_review
```

Это бизнес-эвристика по текущему экспорту. Перед следующим крупным обучением стоит отдельно ревизовать `manual_review`: часть таких строк семантически выглядит как явный `positive` или `negative` (см. [docs/architecture.md](architecture.md)).

## Подготовка split-файлов

```bash
python scripts/split_jsonl_sentiment_dataset.py \
  --input aj_reviews_export.jsonl \
  --output-dir data
```

Выход (генерируется, игнорируется Git-ом):

```text
data/sentiment_train.tsv
data/sentiment_val.tsv
data/sentiment_test.tsv
```

## Базовая модель: `model/rubert-base-cased/`

Код ожидает локальную RuBERT-модель в `model/rubert-base-cased/`. Малые tokenizer/config-файлы можно хранить в репозитории, но веса не коммитятся из-за размера. Перед обучением/инференсом файл весов должен существовать локально (любой из форматов):

```text
model/rubert-base-cased/model.safetensors      # или
model/rubert-base-cased/pytorch_model.bin
```

Если весов нет, их можно докачать: `huggingface_hub.hf_hub_download("DeepPavlov/rubert-base-cased", "pytorch_model.bin")` и скопировать в `model/rubert-base-cased/`.

> Базовая модель — это **вход** для обучения. Обученные модели лежат в `outputs/.../final` (см. [docs/training.md](training.md)).

## Воспроизведение состояния с нуля

1. Положить локально `aj_reviews_export.jsonl`.
2. Положить локально веса базовой модели в `model/rubert-base-cased/`.
3. Сгенерировать sentiment TSV через `scripts/split_jsonl_sentiment_dataset.py`.
4. Запустить обучение (см. [docs/training.md](training.md)).
5. Использовать `outputs/rubert-sentiment-cpu/final` для inference/validation.

Артефакты, не входящие в Git: `aj_reviews_export.jsonl`, `aj_reviews_export.jsonl.zip`, `outputs/`, `data/*.tsv`, веса в `model/rubert-base-cased/`.
