# Как обучать модель на размеченных отзывах

Документ описывает практический путь от текущих TSV-файлов до обученной модели на базе RuBERT.

Целевая задача в этом пайплайне:

```text
positive_about_work
negative_about_work
manual_review
```

То есть модель должна отвечать на вопрос:

```text
Отзыв положительно описывает работу у работодателя, отрицательно или требует ручной проверки?
```

Важно: текущий файл `data/labeled_reviews_20000_heuristic.tsv` еще нельзя считать финальным датасетом для обучения 3-классовой модели. В нем почти нет подтвержденного positive-класса.

## 1. Что уже есть

### Исходная выгрузка

```text
aj_reviews_export.tsv
```

Содержит отзывы и служебные поля:

```text
id
name
descr
status
isfunny
is_self_delete
is_about_work
isPositive
```

Для текущей задачи используется только `is_about_work=1`.

### Выборка для разметки

```text
data/labeling_sample_20000.tsv
```

Создается командой:

```bash
python scripts/prepare_labeling_sample.py
```

Это 20 000 отзывов из валидных строк `is_about_work=1`.

### Предварительная разметка

```text
data/labeled_reviews_20000_heuristic.tsv
```

Создается командой:

```bash
python scripts/heuristic_label_reviews.py
```

Текущий результат:

```text
negative_about_work: 15006
manual_review:        4992
positive_about_work:     2
всего:              20000
```

### Файл аудита

```text
data/audit_reviews.tsv
```

Содержит строки, которые нужно проверить вручную или через LLM API.

Текущий размер:

```text
11691 строка
```

## 2. Главный блокер перед обучением

Сейчас датасет сильно несбалансирован:

```text
positive_about_work: 2
negative_about_work: 15006
manual_review: 4992
```

На таком датасете нельзя обучить нормальную 3-классовую модель.

Что произойдет, если обучать сразу:

1. Модель почти не увидит positive-класс.
2. Она будет предсказывать в основном `negative_about_work` или `manual_review`.
3. Accuracy может выглядеть приемлемо, но macro F1 по positive-классу будет почти нулевой.
4. Модель будет непригодна для реальной классификации положительных отзывов.

Минимальное условие для старта обучения:

```text
positive_about_work: минимум 300-500 подтвержденных примеров
negative_about_work: минимум 300-500 подтвержденных примеров
manual_review: минимум 300-500 подтвержденных примеров
```

Рекомендуемый стартовый уровень:

```text
positive_about_work: 1000-3000
negative_about_work: 1000-3000
manual_review: 1000-3000
```

Для более надежного качества:

```text
positive_about_work: 5000+
negative_about_work: 5000+
manual_review: 5000+
```

## 3. Как подготовить финальный датасет

Нужно получить файл с подтвержденными метками.

Рекомендуемое имя:

```text
data/final_labeled_reviews.tsv
```

Минимальные колонки:

```text
id
name
descr
label
label_source
confidence
needs_audit
```

Для обучения использовать только строки:

```text
needs_audit=0
```

и только метки:

```text
positive_about_work
negative_about_work
manual_review
```

### 3.1. Что делать с текущим файлом

Файл:

```text
data/labeled_reviews_20000_heuristic.tsv
```

можно использовать как стартовую точку.

Что взять сразу:

- строки `negative_about_work` с `needs_audit=0` - как кандидаты в negative;
- строки из `data/audit_reviews.tsv` - для ручной или LLM-проверки;
- строки с `potential_positive` и `source_column_positive` - проверить в первую очередь.

Что нельзя делать:

- нельзя считать все heuristic-метки окончательной истиной;
- нельзя обучать 3-классовую модель без добора positive;
- нельзя использовать `status` как целевую метку без подтверждения бизнес-семантики.

### 3.2. Как добрать positive-класс

Нужно найти реальные положительные отзывы о работе.

Варианты:

1. Сделать новую выгрузку из БД, где уже есть надежный признак положительного отзыва.
2. Отобрать кандидатов по `isPositive=1`, если в полной базе таких строк больше, чем в текущей выборке.
3. Использовать поиск по тексту как кандидатный пул, но не как финальную метку.
4. Разметить кандидатов вручную или через LLM API.

Примеры поисковых фраз для candidate mining:

```text
рекомендую работодателя
хороший работодатель
отличный коллектив
зарплата вовремя
белая зарплата
адекватное руководство
приятно работать
работала дальше
хорошие условия
официальное трудоустройство
```

Важно: такие фразы дают много ложных срабатываний. Например, негативный отзыв может цитировать обещания работодателя: "обещали белую зарплату и дружный коллектив, но обманули". Поэтому все positive-кандидаты нужно проверять.

### 3.3. Как обработать manual_review

`manual_review` нужен для случаев, где модель не должна уверенно принимать решение.

Туда можно относить:

- вакансии и объявления;
- тексты не про работу;
- слишком короткие или битые тексты;
- смешанные отзывы, где нет итогового вывода;
- отзывы, где человек говорит о продукте/услуге, а не о работе;
- случаи, где два разметчика могли бы спорить.

Не нужно превращать все сложные случаи в positive или negative. Для production-пайплайна `manual_review` полезен как безопасный класс.

## 4. Проверка финального датасета

Перед обучением нужно проверить распределение классов.

Команда:

```bash
python - <<'PY'
import csv
from collections import Counter

path = "data/final_labeled_reviews.tsv"

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

usable = [
    row for row in rows
    if row["needs_audit"] == "0"
    and row["label"] in {
        "positive_about_work",
        "negative_about_work",
        "manual_review",
    }
]

print("all rows:", len(rows))
print("usable rows:", len(usable))
print("labels:", Counter(row["label"] for row in usable))
PY
```

Минимально приемлемый результат:

```text
positive_about_work: 300+
negative_about_work: 300+
manual_review: 300+
```

Лучше:

```text
positive_about_work: 1000+
negative_about_work: 1000+
manual_review: 1000+
```

## 5. Разделение на train / validation / test

Данные нужно разделить стратифицированно, чтобы каждый класс был представлен во всех частях.

Рекомендуемая пропорция:

```text
train: 80%
validation: 10%
test: 10%
```

Или при малом датасете:

```text
train: 70%
validation: 15%
test: 15%
```

Правила:

1. Делить нужно по `label` стратифицированно.
2. Один и тот же `id` не должен попасть в разные split.
3. Если у одной компании много похожих отзывов, желательно проверить утечку по `name`.
4. Test-сет нельзя использовать для подбора параметров и порога confidence.

Рекомендуемые выходные файлы:

```text
data/train.tsv
data/val.tsv
data/test.tsv
```

Минимальные колонки:

```text
text
label
```

Можно добавить служебные:

```text
id
name
label_source
```

## 6. Что подавать модели на вход

Основной текстовый вход:

```text
descr
```

Можно добавить название компании:

```text
Компания: <name>
Отзыв: <descr>
```

Рекомендация для первого baseline:

```text
<descr>
```

Причина: так проще понять качество модели именно по тексту отзыва. Поле `name` может дать утечки, если одни и те же компании часто встречаются с одной меткой.

После baseline можно сравнить два варианта:

1. Только `descr`.
2. `name + descr`.

Оставить тот вариант, где выше macro F1 на validation и test.

## 7. Установка окружения

В проекте уже есть `.venv/`.

Активировать:

```bash
source .venv/bin/activate
```

Если окружение нужно создать заново:

```bash
python -m venv .venv
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install torch transformers datasets scikit-learn pandas accelerate
```

Если обучение будет на GPU, нужно установить версию PyTorch под вашу CUDA. Проверяйте команду установки на официальном сайте PyTorch для вашей видеокарты и версии CUDA.

Проверить GPU:

```bash
python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## 8. Модель

Базовая модель:

```text
model/rubert-base-cased/
```

Это локальная копия:

```text
DeepPavlov/rubert-base-cased
```

Для классификации нужно использовать:

```python
AutoModelForSequenceClassification
```

Количество классов:

```text
3
```

Маппинг меток:

```python
label2id = {
    "positive_about_work": 0,
    "negative_about_work": 1,
    "manual_review": 2,
}

id2label = {
    0: "positive_about_work",
    1: "negative_about_work",
    2: "manual_review",
}
```

Важно сохранить этот маппинг вместе с моделью. Иначе при инференсе можно перепутать классы.

## 9. Рекомендуемая структура будущих скриптов

Сейчас в проекте есть скрипты подготовки и эвристической разметки. Для обучения лучше добавить отдельные файлы:

```text
scripts/split_dataset.py
scripts/train_rubert.py
scripts/evaluate_rubert.py
scripts/predict_review.py
```

Назначение:

- `split_dataset.py` - готовит `train.tsv`, `val.tsv`, `test.tsv`;
- `train_rubert.py` - запускает fine-tuning;
- `evaluate_rubert.py` - считает метрики на test-сете;
- `predict_review.py` - проверяет один отзыв или TSV с отзывами.

## 10. Настройки обучения

Стартовые гиперпараметры:

```text
learning_rate: 2e-5
num_train_epochs: 4
per_device_train_batch_size: 16
per_device_eval_batch_size: 32
warmup_ratio: 0.1
weight_decay: 0.01
max_length: 256 или 512
metric_for_best_model: macro_f1
```

Если не хватает GPU-памяти:

1. Уменьшить `per_device_train_batch_size` до `8` или `4`.
2. Включить gradient accumulation.
3. Уменьшить `max_length` до `256`.

Пример:

```text
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
```

Это примерно имитирует batch size 16.

## 11. Дисбаланс классов

Для этой задачи дисбаланс критичен.

Что нужно сделать:

1. Считать `macro F1`, а не только accuracy.
2. Использовать class weights или balanced sampling.
3. Следить за F1 по каждому классу отдельно.

Почему accuracy плохая:

Если 90% датасета это `negative_about_work`, модель может всегда предсказывать negative и получить высокую accuracy, но она не будет решать задачу.

Основная метрика:

```text
macro F1
```

Дополнительно смотреть:

```text
precision по каждому классу
recall по каждому классу
confusion matrix
```

## 12. Пример логики обучения

Ниже не готовый файл проекта, а схема того, что должен делать `train_rubert.py`.

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_PATH = "model/rubert-base-cased"

label2id = {
    "positive_about_work": 0,
    "negative_about_work": 1,
    "manual_review": 2,
}
id2label = {value: key for key, value in label2id.items()}

dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/train.tsv",
        "validation": "data/val.tsv",
        "test": "data/test.tsv",
    },
    delimiter="\t",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def encode(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

dataset = dataset.map(encode, batched=True)
dataset = dataset.map(lambda row: {"labels": label2id[row["label"]]})

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3,
    label2id=label2id,
    id2label=id2label,
)

training_args = TrainingArguments(
    output_dir="outputs/rubert-work-reviews",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("outputs/rubert-work-reviews/best")
tokenizer.save_pretrained("outputs/rubert-work-reviews/best")
```

В реальном скрипте нужно обязательно добавить:

- расчет `macro_f1`;
- class weights или balanced sampling;
- сохранение метрик;
- оценку на test-сете после выбора лучшей модели.

## 13. Метрики

На validation считать после каждой эпохи:

```text
macro_f1
weighted_f1
accuracy
precision per class
recall per class
f1 per class
```

На test считать один раз после выбора лучшей модели:

```text
classification_report
confusion_matrix
```

Главный критерий:

```text
macro_f1
```

Особенно смотреть:

```text
F1 positive_about_work
F1 manual_review
```

Если `positive_about_work` плохой, почти всегда причина в нехватке или плохом качестве positive-разметки.

## 14. Порог уверенности

После обучения модель будет выдавать вероятности по 3 классам.

Production-правило:

```text
если max_softmax_probability < threshold:
    отправить в manual_review
иначе:
    использовать предсказанный класс
```

Стартовый threshold:

```text
0.85
```

Но его нужно подобрать по validation-сету.

Как подбирать:

1. Прогнать validation-сет.
2. Для каждого threshold от `0.50` до `0.95` посчитать:
   - сколько отзывов ушло в `manual_review`;
   - precision/recall по positive и negative;
   - macro F1 после применения порога.
3. Выбрать threshold, который дает приемлемый баланс между автоматизацией и ошибками.

Для модерации лучше иметь меньше автоматических решений, но меньше опасных ошибок.

## 15. Что сохранять после обучения

Рекомендуемая директория:

```text
outputs/rubert-work-reviews/best/
```

В ней должны быть:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
training_args.bin
```

Дополнительно сохранить:

```text
outputs/rubert-work-reviews/metrics.json
outputs/rubert-work-reviews/label_mapping.json
outputs/rubert-work-reviews/threshold.json
```

`threshold.json` пример:

```json
{
  "threshold": 0.85,
  "selected_on": "validation",
  "metric": "macro_f1_with_manual_review_fallback"
}
```

## 16. Проверка инференса

После обучения нужно проверить модель на ручных примерах.

Примеры:

```text
Не рекомендую эту компанию, зарплату задерживают, руководство хамит.
```

Ожидаемо:

```text
negative_about_work
```

```text
Работала несколько лет, зарплата вовремя, коллектив хороший, руководство адекватное.
```

Ожидаемо:

```text
positive_about_work
```

```text
Требуется менеджер, график 5/2, зарплата от 50000.
```

Ожидаемо:

```text
manual_review
```

Если модель уверенно классифицирует вакансии как positive или negative, значит в датасете плохой manual_review-класс.

## 17. Типичные ошибки

### Ошибка 1: учить на текущем heuristic TSV как есть

Проблема:

```text
positive_about_work почти отсутствует
```

Результат:

```text
модель не научится positive-классу
```

### Ошибка 2: смотреть только accuracy

Проблема:

```text
accuracy скрывает провал редких классов
```

Решение:

```text
использовать macro F1 и отчет по каждому классу
```

### Ошибка 3: считать ключевые слова финальной разметкой

Проблема:

```text
в негативных отзывах часто цитируют позитивные обещания работодателя
```

Пример:

```text
Обещали белую зарплату и дружный коллектив, но обманули.
```

Такой текст не positive, а negative.

### Ошибка 4: смешивать audit и train

Проблема:

```text
needs_audit=1 содержит спорные и шумные строки
```

Решение:

```text
до обучения использовать только подтвержденные строки needs_audit=0
```

### Ошибка 5: подбирать threshold на test-сете

Проблема:

```text
test перестает быть независимой проверкой
```

Решение:

```text
threshold подбирать на validation, test использовать один раз в конце
```

## 18. Рекомендуемый порядок работ

### Этап 1: завершить разметку

1. Проверить `data/audit_reviews.tsv`.
2. Добрать positive-класс.
3. Сформировать `data/final_labeled_reviews.tsv`.
4. Проверить распределение классов.

Критерий готовности:

```text
минимум 300-500 подтвержденных строк на каждый класс
```

### Этап 2: подготовить split

1. Создать `scripts/split_dataset.py`.
2. Сформировать:

```text
data/train.tsv
data/val.tsv
data/test.tsv
```

3. Проверить распределение классов в каждом split.

### Этап 3: обучить baseline

1. Создать `scripts/train_rubert.py`.
2. Обучить `model/rubert-base-cased`.
3. Сохранить лучшую модель по `macro_f1`.
4. Посмотреть ошибки на validation.

### Этап 4: оценить качество

1. Запустить test evaluation.
2. Сохранить classification report.
3. Построить confusion matrix.
4. Отдельно проверить positive и manual_review.

### Этап 5: подобрать threshold

1. Прогнать validation.
2. Подобрать confidence threshold.
3. Зафиксировать threshold в `threshold.json`.

### Этап 6: подготовить инференс

1. Создать `scripts/predict_review.py`.
2. Проверить ручные примеры.
3. Проверить батч предсказаний на свежих отзывах.
4. Только после этого думать о деплое.

## 19. Что делать прямо сейчас

Самый правильный следующий шаг:

```text
не запускать обучение сразу
```

Сначала нужно:

1. Проверить `data/audit_reviews.tsv`.
2. Найти и подтвердить positive-класс.
3. Собрать `data/final_labeled_reviews.tsv`.
4. Убедиться, что в каждом классе есть хотя бы 300-500 строк.

Только после этого имеет смысл писать `split_dataset.py` и `train_rubert.py`.

Если очень хочется проверить технический пайплайн обучения раньше, можно сделать маленький smoke test на текущих данных, но результат нельзя интерпретировать как качество модели.

Smoke test нужен только для проверки:

- запуска tokenizer/model;
- чтения TSV;
- работы Trainer;
- сохранения модели.

Он не отвечает на вопрос, хорошая ли модель.
