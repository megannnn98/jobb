# Обучение модели

Подготовка данных — см. [docs/data.md](data.md). История GPU-эксперимента и его результаты —
см. [docs/gpu_training.md](gpu_training.md) и [docs/evaluation.md](evaluation.md).

> **Текущая основная модель — ruRoberta-large** (`train_ruroberta.py` / `kaggle/ruroberta/train.py`,
> Kaggle P100, → `model/ruroberta-sentiment/`), не RuBERT GPU1 из таблицы ниже. Таблица и раздел
> ниже описывают историю RuBERT-экспериментов (GPU1/GPU2/CPU baseline), которые были превзойдены
> по метрикам последовательно RuBERT V2, затем ruRoberta-large, и в итоге удалены локально
> 2026-07-12 (см. `CLAUDE.md`, таблица lineage) — воспроизводимы из этих committed скриптов при
> необходимости. Актуальная схема ruRoberta-large и параллельная 5-классовая ветка — см.
> «ruRoberta-large и 5-класс» ниже.

Исторически было два варианта обучения одной и той же 3-классовой задачи
(`positive` / `negative` / `manual_review`) на RuBERT:

| Вариант | Скрипт | Где запускался | Output | Статус |
|---|---|---|---|---|
| GPU1 | `train_fast_bert.py` | Kaggle Notebook, GPU T4x2 (фактически 1 GPU) | `outputs/rubert-sentiment/final` | превзойдён (см. lineage в `CLAUDE.md`) |
| CPU baseline | `train_fast_bert_cpu.py` | локально, CPU | `outputs/rubert-sentiment-cpu/final` | legacy / первый baseline |

> Имена `train_fast_bert.py` и `outputs/rubert-sentiment/final` — это соглашение Kaggle-ноутбука.
> В репозитории закоммичены оба скрипта: GPU-версия `train_fast_bert.py` и CPU-базовый `train_fast_bert_cpu.py`.
> GPU-веса были распакованы локально в `model/rubert-sentiment-gpu1/`, удалены 2026-07-12.

## Общая схема (одинакова для GPU и CPU)

- базовая модель RuBERT (`DeepPavlov/rubert-base-cased`);
- HuggingFace `Trainer`, метрика — **macro F1** (не accuracy);
- двухфазная схема:
  - **phase 1:** frozen backbone, обучается только classifier head;
  - **phase 2:** размораживаются последние 2 encoder-слоя BERT, fine-tune;
- weighted cross-entropy для дисбаланса классов (без весов модель схлопывается в `negative`);
- динамический padding через `DataCollatorWithPadding`;
- `MAX_LENGTH=128` (256/512 не дают заметного выигрыша — сигнал в первых ~128 токенах);
- кэш токенизированного датасета с fingerprint-инвалидацией;
- env caps (`MAX_TRAIN_SAMPLES`, `MAX_EVAL_SAMPLES`) для быстрых экспериментов.

## Отличия GPU-версии от CPU baseline

GPU-вариант — это тот же `train_fast_bert_cpu.py`, в котором изменены настройки под GPU-окружение
и переразложены бюджеты эпох/батчей:

| Параметр | CPU baseline | GPU (Kaggle T4x2) | Зачем |
|---|---|---|---|
| `MODEL_NAME` | `model/rubert-base-cased` (локально) | `DeepPavlov/rubert-base-cased` (hub) | в Kaggle базовая модель тянется с HuggingFace Hub |
| `use_cpu` | `True` | `False` | задействовать GPU |
| `fp16` | — | `True` | mixed precision, быстрее и меньше памяти на T4 |
| `dataloader_pin_memory` | `False` | `True` | быстрее host→GPU копирование |
| `DATALOADER_NUM_WORKERS` | `0` | `2` | параллельная подготовка батчей |
| `PHASE1_BATCH_SIZE` | `32` | `64` | больше памяти GPU |
| `PHASE2_BATCH_SIZE` | `8` | `16` | больше памяти GPU |
| `PHASE2_GRADIENT_ACCUMULATION` | `4` | `2` | эффективный батч сохранён при большем реальном батче |
| `PHASE1_EPOCHS` | `2` | `1` | head сходится быстро, бюджет перенесён в phase 2 |
| `PHASE2_EPOCHS` | `1` | `2` | больше fine-tune размороженных слоёв |
| Output dir | `outputs/rubert-sentiment-cpu` | `outputs/rubert-sentiment` | разные артефакты |

## Команды запуска

### GPU (основной вариант, Kaggle)

```bash
python train_fast_bert.py            # → outputs/rubert-sentiment/final
```

Обучение проводилось на **80/10/10** стратифицированном сплите. Результаты — [docs/evaluation.md](evaluation.md).

### CPU baseline (legacy, локально)

Быстрый smoke train:

```bash
MAX_TRAIN_SAMPLES=512 MAX_EVAL_SAMPLES=256 python train_fast_bert_cpu.py
```

Рабочий CPU-прогон на части данных:

```bash
MAX_TRAIN_SAMPLES=10000 MAX_EVAL_SAMPLES=2000 python train_fast_bert_cpu.py
```

Полный CPU-прогон (медленный):

```bash
python train_fast_bert_cpu.py
```

## Результат (checkpoints)

Финальная модель сохраняется в каталог (формат HuggingFace, не один файл):

```text
outputs/rubert-sentiment/final/        # GPU (Kaggle); скачан как rubert-sentiment-final.zip
outputs/rubert-sentiment-cpu/final/    # CPU baseline (локально)
```

Структура каталога модели:

```text
<final>/
├── model.safetensors      # ← веса модели
├── config.json            # ← архитектура + id2label (какой индекс какой класс)
├── tokenizer.json
└── tokenizer_config.json
```

`outputs/` игнорируется Git-ом: внутри лежат большие модельные артефакты. RuBERT GPU1/GPU2/V2 и
базовая `model/rubert-base-cased/` были удалены локально 2026-07-12 (см. `CLAUDE.md`).

## Замечание про CPU baseline

CPU training медленный, поэтому он остаётся только как первый baseline (исторический — актуальная
модель обучалась на GPU). Для быстрых итераций на CPU используйте `MAX_TRAIN_SAMPLES` /
`MAX_EVAL_SAMPLES`. Скрипт принудительно работает на CPU (`use_cpu=True`).

## ruRoberta-large и 5-класс (актуальные варианты)

Начиная с RuBERT V2 обучение переехало с "GPU vs CPU baseline" на итеративное улучшение одной и
той же двухфазной схемы (frozen backbone → unfreeze последних N слоёв, weighted cross-entropy,
warmup+cosine LR, label smoothing, early stopping — см. `CLAUDE.md`, «Training architecture»).
Полная сравнительная таблица RuBERT GPU1/GPU2/V2 vs ruRoberta-large (параметры, hardware, macro F1)
— в `CLAUDE.md`, не дублируется здесь.

**ruRoberta-large** (`train_ruroberta.py` / `kaggle/ruroberta/train.py`, текущая основная модель):
`ai-forever/ruRoberta-large` (355M), Kaggle P100, phase1 batch 32/1 эпоха, phase2 batch 8 ×
grad-accum 4 / 4 эпохи, 4 размороженных слоя, ~3ч08м. `CUDA_VISIBLE_DEVICES=0` пинуется перед
первым `import torch`, чтобы HF `Trainer` не обернул модель в `nn.DataParallel` на Kaggle
"GPU T4 x2" (2 физических GPU) — без этого маленький per-device batch делает обучение медленнее
из-за overhead scatter/gather. Подробности и P100 SM60 compat-фикс — `CLAUDE.md`.

**5-классовый вариант** (`kaggle/ruroberta-5class/train.py`, экспериментальный, не promoted):
идентичен `train_ruroberta.py`, кроме `LABEL2ID` (5 классов вместо 3) и путей датасета/output.
Датасет — `data/sentiment5_{train,val,test}.tsv`, построен независимой LLM-разметкой (DeepSeek)
подмножества `manual_review` (см. `CLAUDE.md`, «5-class variant»). Из-за того что `manual_review`
в этой схеме сжался до ~1.87% train-данных, `WeightedTrainer` даёт ему экстремальный вес (~10.68
против ~0.24 у `negative`), что видно как высокий (но не «сломанный») абсолютный loss в Phase 2 —
ориентируйтесь на `eval_f1_macro`/`eval_accuracy` по чекпоинтам, не на сырой loss.
