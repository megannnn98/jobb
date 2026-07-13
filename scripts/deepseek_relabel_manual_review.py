"""Independent-judge relabeling of manual_review reviews via DeepSeek API.

Reads outputs/manual_review_reviews.jsonl (33,631 rows where status==0 & isPositive==0),
asks DeepSeek to independently judge each review's true sentiment (positive/negative/ambiguous),
and writes verdicts incrementally (resumable — already-processed ids are skipped on restart).

Unlike the earlier heuristic relabeling (based on our own sentiment model's confidence — see
CLAUDE.md "Label-noise relabeling experiment"), this uses an independent judge that never saw
our model's predictions, avoiding that circularity.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["DEEPSEEK_API_KEY"]
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

LABELS = {"positive", "negative", "manual_review", "spam", "service_complaint"}

SYSTEM_PROMPT = """\
Ты — независимый эксперт-разметчик отзывов о работодателях на русском языке.
Тебе дают текст с сайта отзывов о работе. Определи одну из пяти категорий.

Ответь СТРОГО в формате JSON: {"label": "positive"|"negative"|"manual_review"|"spam"|"service_complaint", "reason": "краткое обоснование на русском, не длиннее 15 слов"}

Категории:
- "positive": текст — это отзыв именно о работодателе (компании как месте работы), и автор явно доволен.
- "negative": текст — это отзыв именно о работодателе (компании как месте работы), и автор явно недоволен, жалуется на компанию/руководство в целом.
- "service_complaint": автор жалуется на компанию не как работодателя, а как на поставщика товара/услуги — то есть жалоба клиента/потребителя (плохой сервис, качество товара, обман клиентов), даже если упоминается что-то про работу вскользь. Ключевой признак: жалоба идёт от лица клиента, а не сотрудника.
- "manual_review": текст — это отзыв о работе/работодателе, но: (a) отношение неоднозначное, смешанное или нейтральное без явного перевеса; ИЛИ (b) жалоба/похвала адресована не работодателю как таковому, а конкретному коллеге, ассистенту, другому сотруднику или третьему лицу; ИЛИ (c) текст написан не на русском языке; ИЛИ (d) текст состоит в основном из вопросов или обращений к другим комментаторам НА ТЕМУ РАБОТЫ/РАБОТОДАТЕЛЯ ("кто ещё сталкивался с этой компанией?", "напишите контакты работодателя"), а не выражает собственное отношение автора — но если вопрос вообще не связан с работой/работодателем (например, вопрос про возврат денег/вклада, кредиты и т.п. без связи с трудоустройством), это НЕ manual_review, а "spam"; ИЛИ (e) текст написан представителем/сотрудником самой компании в ответ на отзывы (защита репутации, опровержение); ИЛИ (f) текст — это просьба удалить/снять ранее размещённый комментарий, отписаться от рассылки, либо оспаривание/угроза по поводу чужого отзыва (например с отсылкой к статье о клевете) — во всех случаях (e) и (f) нужна ручная проверка, а не авто-решение.
- "spam": текст НЕ является отзывом о работодателе вообще. Сюда относится: пустой текст; нечитаемый набор символов (например "ываываываыв ыва ыва"); текст, испорченный неправильной кодировкой/мохибейк (например "Ð&#157;ÐµÐ´Ð°Ð²Ð½Ð¾") — даже если можно догадаться о смысле, всё равно относи сюда, не пытайся декодировать; спам-рассылка со ссылками (например на сайты знакомств); рекламное объявление любого рода (медицинские услуги, товары, окна и т.д.); текст, где одна и та же фраза/слово повторяется много раз подряд (флуд/накрутка) — даже если фраза несёт явную тональность, повторение — признак спама, а не подлинного отзыва; любой другой текст, не содержащий отзыва о работодателе.

Оценивай только по смыслу текста, игнорируй длину и грамматику."""

_session = requests.Session()
_stats_lock = threading.Lock()
_stats = {"prompt_tokens": 0, "completion_tokens": 0, "errors": 0}


def classify(text: str) -> tuple[dict, dict]:
    resp = _session.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:4000]},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return json.loads(content), usage


def process_row(row: dict) -> dict:
    text = row.get("descr") or ""
    if not text.strip():
        return {"id": row["id"], "verdict_label": "spam", "verdict_reason": "empty text", "text": text}
    try:
        verdict, usage = classify(text)
        with _stats_lock:
            _stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            _stats["completion_tokens"] += usage.get("completion_tokens", 0)
        label = verdict.get("label")
        if label not in LABELS:
            with _stats_lock:
                _stats["errors"] += 1
            return {"id": row["id"], "verdict_label": "error",
                     "verdict_reason": f"invalid label from model: {label!r}", "text": text}
        return {
            "id": row["id"],
            "verdict_label": label,
            "verdict_reason": verdict.get("reason"),
            "text": text,
        }
    except Exception as e:
        with _stats_lock:
            _stats["errors"] += 1
        return {"id": row["id"], "verdict_label": "error", "verdict_reason": str(e), "text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--in-path", default="outputs/manual_review_reviews.jsonl")
    parser.add_argument("--out-path", default="outputs/deepseek_verdicts.jsonl")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    done_ids = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} rows already done in {out_path}")

    rows = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] not in done_ids:
                rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    print(f"Processing {len(rows)} rows with {args.workers} workers -> {out_path}")
    t0 = time.perf_counter()
    write_lock = threading.Lock()
    n_done = 0
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_row, row) for row in rows]
        for fut in as_completed(futures):
            record = fut.result()
            with write_lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
                if n_done % 200 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = n_done / elapsed
                    eta = (len(rows) - n_done) / rate if rate else 0
                    print(f"  {n_done}/{len(rows)} done, {elapsed:.0f}s elapsed, "
                          f"{rate:.1f} rows/s, ETA {eta:.0f}s, errors={_stats['errors']}")

    elapsed = time.perf_counter() - t0
    print(f"Done: {len(rows)} rows in {elapsed:.1f}s ({len(rows) / elapsed:.2f} rows/s)")
    print(f"Tokens: prompt={_stats['prompt_tokens']} completion={_stats['completion_tokens']} "
          f"errors={_stats['errors']}")


if __name__ == "__main__":
    main()
