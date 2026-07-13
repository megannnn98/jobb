# Обучение RuBERT на Kaggle GPU: полная инструкция

## Обзор

Kaggle предоставляет бесплатные GPU (P100, T4, иногда L4/A100 — зависит от загрузки)
для интерактивных ноутбуков и скриптов. Наш пайплайн загружает предварительно
нарезанные TSV-файлы как Dataset, а скрипт обучения — как Kernel (тип `script`),
линкует датасет и запускает двухфазный fine-tuning на GPU.

**Ожидаемое время:** ~1–2 часа (P100, 175K примеров, Phase1 + Phase2).

---

## 1. Подготовка: Kaggle-аккаунт и API-токен

1. Зарегистрироваться на [kaggle.com](https://kaggle.com).
2. Settings → API → **Create New Token** — скачается `kaggle.json`.
3. Положить токен в `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Формат токена:
```json
{"username": "your_username", "key": "KGAT_..."}
```

4. Установить Kaggle CLI в venv проекта:

```bash
.venv/bin/pip install kaggle
.venv/bin/kaggle --version          # должно быть >= 2.2
.venv/bin/kaggle config view        # проверяем, что username подхватился
```

---

## 2. Подготовка данных: сплит и загрузка Dataset

### 2.1. Нарезать JSONL на train/val/test

```bash
# Один раз (TSV-файлы уже есть в data/ — перегенерировать при изменении JSONL)
.venv/bin/python scripts/split_jsonl_sentiment_dataset.py
```

Результат: `data/sentiment_train.tsv` (~210 MB), `data/sentiment_val.tsv` (~26 MB),
`data/sentiment_test.tsv` (~26 MB).

### 2.2. Загрузить TSV как Kaggle Dataset

Создать временную директорию с файлами и метаданными:

```bash
mkdir -p /tmp/kaggle-dataset
cp data/sentiment_{train,val,test}.tsv /tmp/kaggle-dataset/
```

Файл `/tmp/kaggle-dataset/dataset-metadata.json`:
```json
{
  "title": "rubert-job-reviews-sentiment",
  "id": "your_username/rubert-job-reviews-sentiment",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Загрузить:
```bash
.venv/bin/kaggle datasets create -p /tmp/kaggle-dataset -u -t
```

Флаги:
- `-p` — путь к папке с файлами и `dataset-metadata.json`
- `-u` — сделать публичным (нужно, чтобы Kernel мог использовать)
- `-t` — **не конвертировать TSV в CSV** (важно! без этого Kaggle переименует файлы)

После загрузки датасет доступен по URL:
`https://www.kaggle.com/datasets/your_username/rubert-job-reviews-sentiment`

**Внимание:** имя датасета в `id` должно быть уникальным в пределах аккаунта. Слаг
формируется из `id` — используй `your_username/короткое-имя`.

---

## 3. Подготовка скрипта обучения

### 3.1. Каркас скрипта

Берём локальный `train_fast_bert.py` и адаптируем под Kaggle. Ниже — полный шаблон
с исправлениями, которые понадобились в реальном прогоне.

Ключевые изменения относительно локального скрипта:

| Строка | Локально | На Kaggle |
|--------|----------|-----------|
| `TRAIN_FILE` | `Path("data/sentiment_train.tsv")` | `Path("/kaggle/input/datasets/<user>/<dataset>/sentiment_train.tsv")` |
| `VAL_FILE` | `Path("data/sentiment_val.tsv")` | `Path("/kaggle/input/datasets/<user>/<dataset>/sentiment_val.tsv")` |
| `OUTPUT_DIR` | `Path("outputs/rubert-sentiment")` | `Path("/kaggle/working/rubert-sentiment")` |
| `MODEL_NAME` | Без изменений | `"DeepPavlov/rubert-base-cased"` (нужен интернет) |

### 3.2. Фикс PyTorch (критически важно!)

Kaggle-образ поставляется с `torch 2.10.0+cu128`, у которого **нет ядер под P100 (SM60/Pascal)**.
Без переустановки получишь:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Кроме того, библиотека `transformers` (последняя на Kaggle) **требует PyTorch >= 2.6**
из-за CVE-2025-32434.

Поэтому **первым делом в скрипте** (до `import torch`) нужно переустановить PyTorch:

```python
import subprocess, sys as _sys
subprocess.run(
    [_sys.executable, "-m", "pip", "install", "-q",
     "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
     "--index-url", "https://download.pytorch.org/whl/cu124"],
    check=True)
print("PyTorch 2.6.0+cu124 ready")

import torch  # теперь импортируется правильная версия
```

**Пояснение:**
- `cu124` — CUDA 12.4, стабильно работает с P100 (SM60)
- Версии torch/torchvision/torchaudio должны совпадать (2.6.0 / 0.21.0 / 2.6.0)
- `--index-url` указывает на официальные wheels PyTorch (не из PyPI)

### 3.3. Проверка GPU в `main()`

Полезно добавить в начало `main()` для диагностики:

```python
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, Capability: {torch.cuda.get_device_capability(0)}")
    t = torch.tensor([1.0]).cuda()
    print(f"CUDA test: OK")
```

### 3.4. Полный шаблон `kernel-metadata.json`

```json
{
  "id": "your_username/rubert-train-p100",
  "title": "rubert-train-p100",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "",
  "dataset_sources": ["your_username/rubert-job-reviews-sentiment"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**Важно:** все boolean-поля — **строки** `"true"`/`"false"`, не JSON-boolean.
На `kernel_type: "notebook"` полезут свои грабли (papermill, логи через раз).
`"script"` — самый надёжный вариант.

### 3.5. `docker_image_pinning_type`

**Не заполняй** поле `machine_shape` строкой `"Gpu"` и не указывай
`docker_image_pinning_type`. Kaggle сам подберёт образ. Ручные эксперименты
с образом `gcr.io/kaggle-private-byod/...` привели к нестабильности.

---

## 4. Запуск обучения

### 4.1. Запушить Kernel

```bash
# Папка содержит kernel-metadata.json и train.py
.venv/bin/kaggle kernels push -p /path/to/kernel-folder
```

После пуша Kernel автоматически запускается (статус `RUNNING`).

### 4.2. Мониторинг

**Через Kaggle UI:**
Открыть `https://www.kaggle.com/code/your_username/rubert-train-p100` → вкладка **Logs**.

**Через CLI/API (для скриптов, а не ноутбуков):**

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)

api = KaggleApi()
api.authenticate()

with api.build_kaggle_client() as kaggle:
    client = kaggle.kernels.kernels_api_client

    # Статус
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = "your_username"
    req.kernel_slug = "rubert-train-p100"
    status = client.get_kernel_session_status(req)
    print(status.to_dict())

    # Логи
    req_out = ApiListKernelSessionOutputRequest()
    req_out.user_name = "your_username"
    req_out.kernel_slug = "rubert-train-p100"
    output = client.list_kernel_session_output(req_out)
    # output.to_dict()["log"] — JSON-строка с массивом записей
```

### 4.3. Что должно быть в логах

Нормальный прогон:

```
PyTorch 2.6.0+cu124 ready
Torch: 2.6.0+cu124, CUDA: True
GPU: Tesla P100-PCIE-16GB, Capability: (6, 0)
CUDA test: OK
Loading tokenizer...
.../sentiment_train.tsv: loaded=140643 repaired_embedded_tabs=...
class weights: positive=..., negative=..., manual_review=...
Trainable params: ...
=== Phase 1: classifier head ===
Trainable params: 4,611 / 178,053,891 (0.00%)
...
{'loss': 1.343, 'grad_norm': 15.39, 'learning_rate': 1.567e-05, 'epoch': 0.4338}
...
=== Phase 2: unfreeze last 2 layers ===
Trainable params: 22,756,611 / 178,053,891 (12.78%)
...
Done. Model saved to /kaggle/working/rubert-sentiment/final
```

---

## 5. Скачивание результата

После завершения (статус `COMPLETE`) модель лежит в `/kaggle/working/rubert-sentiment/final/`.

### 5.1. Скачать через Kaggle UI

На странице Kernel → вкладка **Output** → кнопка **Download** (или файлы по отдельности).

### 5.2. Скачать через CLI/API

Метод `kernels output` требует публичного Kernel. Для приватных — через SDK:

```python
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelFilesRequest

req = ApiListKernelFilesRequest()
req.user_name = "your_username"
req.kernel_slug = "rubert-train-p100"
files = client.list_kernel_files(req)
# Список файлов в /kaggle/working/
```

Затем скачать нужные файлы через метод `download_kernel_output`.

### 5.3. Практический вариант

Самый простой способ — сделать Kernel **публичным** перед скачиванием:

1. Открыть Kernel на Kaggle
2. Settings → Sharing → Public
3. `kaggle kernels output your_username/rubert-train-p100 -p ./downloaded-model/`

Или просто скачать через веб-интерфейс (Output → Download).

---

## 6. Типичные ошибки

### 6.1. `FileNotFoundError: data/sentiment_train.tsv`

**Причина:** путь к датасету на Kaggle отличается от ожидаемого.

**Диагностика:** добавить `subprocess.run(["ls", "-laR", "/kaggle/input/"])` в начало скрипта.

**Фактический путь (июль 2025):**
```
/kaggle/input/datasets/<username>/<dataset-slug>/sentiment_train.tsv
```

**Решение:** использовать полный абсолютный путь в `TRAIN_FILE` / `VAL_FILE`.

### 6.2. `CUDA error: no kernel image is available for execution on the device`

**Причина:** PyTorch в Kaggle-образе (`2.10.0+cu128`) не включает ядра под архитектуру
текущего GPU. Чаще всего GPU — **P100 (SM60/Pascal)**, на котором `cu128`-сборка не работает.

**Диагностика:**
```python
import torch
print(torch.cuda.get_device_capability(0))  # (6, 0) = P100
```

**Решение:** переустановить PyTorch (см. раздел 3.2).

### 6.3. `ValueError: require users to upgrade torch to at least v2.6`

**Причина:** `transformers` последней версии требует PyTorch >= 2.6 (CVE-2025-32434).

**Решение:** использовать `torch==2.6.0` (см. раздел 3.2).

### 6.4. `Maximum batch GPU session count of 2 reached`

**Причина:** бесплатный аккаунт Kaggle — максимум 2 одновременных GPU-сессии.

**Решение:** отменить висящие сессии:

```python
from kagglesdk.kernels.types.kernels_api_service import ApiDeleteKernelRequest

for slug in ["old-kernel-slug"]:
    req = ApiDeleteKernelRequest()
    req.user_name = "your_username"
    req.kernel_slug = slug
    client.delete_kernel(req)
```

### 6.5. `Expecting value: line 1 column 1` при push

**Причина:** `kernel_type: "notebook"` в метаданных, а `code_file` указывает на `.py`,
не на `.ipynb`. Для `notebook` ожидается JSON-файл `.ipynb`.

**Решение:** использовать `kernel_type: "script"` для `.py`-файлов.

### 6.6. Kernel висит в статусе RUNNING без логов

**Причина:** `pip install` подвис (особенно `uninstall` старого torch).

**Решение:** не делать `pip uninstall`, только `pip install` с `--force-reinstall`
не нужен — Kaggle-образ позволяет перезаписать пакет обычным `install`.

---

## 7. Быстрый старт (cheatsheet)

```bash
# 1. Токен
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
.venv/bin/pip install kaggle

# 2. Сплит (если ещё нет)
.venv/bin/python scripts/split_jsonl_sentiment_dataset.py

# 3. Загрузить датасет
cp data/sentiment_{train,val,test}.tsv /tmp/kd/
# + /tmp/kd/dataset-metadata.json
.venv/bin/kaggle datasets create -p /tmp/kd -u -t

# 4. Подготовить скрипт (адаптировать TRAIN_FILE, VAL_FILE, OUTPUT_DIR,
#    добавить фикс PyTorch 2.6.0+cu124, kernel-metadata.json)

# 5. Запушить и запустить
.venv/bin/kaggle kernels push -p /path/to/kernel-folder

# 6. Мониторить на https://www.kaggle.com/code/<user>/<kernel-slug>

# 7. Скачать модель (после COMPLETE)
.venv/bin/kaggle kernels output <user>/<kernel-slug> -p ./downloaded-model/
```

---

## 8. Версии пакетов (рабочая конфигурация, июль 2025)

| Пакет | Версия | Примечание |
|-------|--------|-----------|
| Python | 3.12.13 | Встроен в Kaggle-образ |
| torch | 2.6.0+cu124 | **Переустанавливается** (образ: 2.10.0+cu128 — несовместим) |
| torchvision | 0.21.0+cu124 | В паре с torch 2.6.0 |
| torchaudio | 2.6.0+cu124 | В паре с torch 2.6.0 |
| transformers | ~4.50+ | Встроен в образ, требует torch >= 2.6 |
| datasets | ~3.0+ | Встроен в образ |
| CUDA Driver | 580.159.04 | Поддерживает CUDA до 13.0 |
| GPU | Tesla P100 (SM60) | 16 GB VRAM |
| DeepPavlov/rubert-base-cased | — | Качается с HuggingFace Hub при старте |
