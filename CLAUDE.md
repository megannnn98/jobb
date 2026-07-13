# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Classifying Russian job reviews into `positive` / `negative` / `manual_review` / `spam` /
`service_complaint`. The **current main model is the 5-class ruRoberta-large**
(fine-tuned `ai-forever/ruRoberta-large`, 355M params, `model/ruroberta-sentiment-5class/`) —
`main.py` and `predict.py` default to it (promoted 2026-07-13, see "5-class variant" section
below for training/eval/threshold details). The earlier **3-class ruRoberta-large** (`model/ruroberta-sentiment/`) was main until the 5-class
promotion — it beat every RuBERT checkpoint on held-out test (macro F1 0.771 vs RuBERT V2's 0.733
— see lineage table below) — but its local weights were **deleted on 2026-07-13** after the
5-class promotion (superseded, same pattern as the RuBERT cleanup below); reproducible via
`train_ruroberta.py` / `kaggle/ruroberta/train.py` or re-downloadable from the public Kaggle
Dataset `megannnn98/ruroberta-sentiment` if needed again. Three RuBERT
(`DeepPavlov/rubert-base-cased`, 178M) checkpoints existed for comparison before that
(`model/rubert-sentiment-gpu1/`, `-gpu2/`, `-v2/`) but were **deleted locally on 2026-07-12** as
outdated/weaker than ruRoberta-large — see note after the lineage table below. Always use the
project venv: `.venv/bin/python`.
There is no test suite and no linter config — `python -m py_compile` is the de-facto syntax check
used here.

Documentation lives in `README.md` (quick-start) + `docs/*.md` (architecture, data, training,
inference, evaluation, legacy). `docs/TODO.md` tracks which docs are known-stale — check it before
trusting `docs/training_guide_ru.md`, `docs/labeling_rubric.md`, or root `rubert-reviews-context.md`
(all use an old label naming scheme and/or a wrong assumption about the data; superseded by the
code and by `docs/*.md`). `docs/evaluation.md`, `docs/gpu_training.md`, `docs/model-tuning.md`
predate ruRoberta and describe RuBERT-only history — read them as that, not as current-model docs.

## Model lineage — RuBERT (3 checkpoints) superseded by ruRoberta-large

| Version | Params | Script (committed) | Hardware | Macro F1 (held-out) | Test sample |
|---|---|---|---|---|---|
| RuBERT GPU1 | 178M | `train_fast_bert.py` | Kaggle T4x2 | 0.705 | 4000 |
| RuBERT GPU2 | 178M | `rubert-train-p100.py` | Kaggle P100 | 0.710 | 4000 |
| RuBERT V2 | 178M | *not committed* — see below | Kaggle P100 | 0.733 | 4000 |
| **ruRoberta-large (main)** | 355M | `train_ruroberta.py` / `kaggle/ruroberta/train.py` | Kaggle P100 (training) | **0.771** | **full test (17,523)** |

- `rubert-train-p100.py` is a **standalone Kaggle script**, not a variant of `train_fast_bert.py`
  despite the similar two-phase structure — it hardcodes Kaggle input paths.
- The **RuBERT V2 training script itself is not in this repo** — `docs/model-tuning.md` points to
  `/tmp/kaggle-train-v2/train.py` on the machine that ran it, not reproducible from committed code.
- ruRoberta-large's macro F1 (0.771) isn't directly apples-to-apples with RuBERT's 0.733 — it was
  measured on the **full** test split (17,523 rows), not a 4000-row sample, because GPU eval made
  that cheap (~450s on a T4). Both splits come from the same stratified test file, so the
  comparison is still meaningful, just not identical methodology.
- **Local weights for the 3 RuBERT checkpoints and the base `model/rubert-base-cased/` were
  deleted on 2026-07-12** (2.7GB freed, all weaker than the ruRoberta-large main model). This
  table/lineage history stays as documentation; the weights themselves are gone locally. If needed
  again, RuBERT GPU1/GPU2 are reproducible via their committed training scripts
  (`train_fast_bert.py`, `rubert-train-p100.py`); V2's script was never committed (see above) so
  it is **not** reproducible without the original `/tmp/kaggle-train-v2/train.py`. Base weights
  (`DeepPavlov/rubert-base-cased`) are re-downloadable from HF Hub if any RuBERT script needs
  local CPU training again (`train_fast_bert_cpu.py` expects `model/rubert-base-cased/` to exist).

## Label-noise relabeling experiment (2026-07-12) — tried, rejected

Attempted to fix `manual_review` label noise by heuristically relabeling ~10,138 train rows where
the model's own confidence exceeded a threshold (`manual_review→negative` at confidence≥0.50,
`manual_review→positive` at confidence≥0.45, excluding exact-duplicate spam text) — see
`outputs/relabel_candidates.tsv`. Retrained ruRoberta-large from scratch on this corrected dataset
(Kaggle dataset `rubert-job-reviews-sentiment-relabeled`, kernel `ruroberta-train-relabeled`), then
evaluated on the **original, unmodified** `sentiment_test.tsv` at MAX_LENGTH=512 for a fair
comparison with baseline.

**Result: measurably worse, not better.** Accuracy 0.8534 vs baseline 0.865 (−1.16pp), macro F1
0.7715 vs baseline 0.7811 (−0.96pp) — see `outputs/eval-relabeled/ruroberta_test_report.txt`. Root
cause: the heuristic relabeled train using the model's own confidence, then eval scored against the
same original test labels containing the same kind of noise — 43% of test's `manual_review` rows
(1426/3304) got reclassified by the new model into positive/negative, many semantically correct
(see confusion matrix / top-confidence "errors" in the report above), but counted as wrong because
the test set itself wasn't independently cleaned. **Lesson: relabeling train data with the model's
own predictions as pseudo-labels, then evaluating on an unchanged noisy test set, structurally
penalizes exactly the cases the relabeling was meant to fix — it cannot demonstrate a fair
improvement this way.** A real fix would require independently (non-heuristic) verifying at least
the test set's `manual_review` rows; not attempted here.

**Decision: kept baseline (`model/ruroberta-sentiment/`) in production.** Local experiment
artifacts (`kaggle/eval-relabeled/`, `kaggle/ruroberta-relabeled/`, `model/ruroberta-sentiment-relabeled/`)
were deleted after this conclusion; `outputs/relabel_candidates.tsv` and `outputs/eval-relabeled/`
reports were kept as evidence. The Kaggle-side dataset/kernels
(`rubert-job-reviews-sentiment-relabeled`, `ruroberta-train-relabeled`, `ruroberta-eval-relabeled`)
were left as-is on Kaggle (not deleted).

## 5-class variant (2026-07-13) — independent-judge relabeling + threshold fix, experimental

Follow-up to the rejected relabeling experiment above, using a methodologically sound approach
this time: an **independent LLM judge (DeepSeek API)** that never saw the target model's own
predictions, avoiding the circularity that sank the heuristic relabel.

**Relabeling.** Extracted all 33,631 rows where `status==0 & isPositive==0` (the `manual_review`
moderation-status subset — see "`manual_review` is a moderation status" below) into
`outputs/manual_review_reviews.jsonl`, then had DeepSeek (`deepseek-chat`, `temperature=0`,
`response_format=json_object`) independently judge each into one of 5 categories via
`scripts/deepseek_relabel_manual_review.py` (resumable, `ThreadPoolExecutor`, 16 workers):
- `positive` / `negative` — genuine employer review, clear sentiment.
- `manual_review` — genuinely ambiguous/mixed, third-party-directed, non-Russian, or a dispute
  needing human judgement (company rep response, request to remove/dispute a review).
- `spam` — not a review at all (empty, unreadable, mojibake, ad, flood/repetition).
- `service_complaint` — customer complaint about the company's product/service, not about it as
  an employer (new category, distinct from `negative`).

Verdicts: `outputs/deepseek_verdicts.jsonl` (33,631 rows, 0 errors after retry). Distribution:
`negative=14,446 (43.0%), spam=8,290 (24.6%), service_complaint=5,199 (15.5%),
manual_review=3,293 (9.8%), positive=2,403 (7.1%)` — i.e. most of the old `manual_review` bucket
was genuinely positive/negative/spam, confirming it was a moderation-status artifact, not a real
text class (as already suspected — see below).

**Dataset.** `data/sentiment5_{train,val,test}.tsv` — same train/val/test row membership as the
3-class splits; only rows where the original label was `manual_review` got overwritten with the
DeepSeek verdict, everything else (original `positive`/`negative` rows) is untouched. Built by
overlaying `outputs/deepseek_verdicts.jsonl` onto `data/sentiment_{train,val,test}.tsv` (see
`kaggle/ruroberta-5class/` for the training-side dataset upload).

**Training.** `kaggle/ruroberta-5class/train.py` — identical to `train_ruroberta.py` (same
two-phase schedule, same `CUDA_VISIBLE_DEVICES=0` fix, MAX_LENGTH=128) except
`LABEL2ID = {"positive": 0, "negative": 1, "manual_review": 2, "spam": 3, "service_complaint": 4}`
and dataset/output paths repointed to the 5-class variant. Kaggle kernel
`megannnn98/ruroberta-train-5class` (T4 accelerator), dataset
`megannnn98/rubert-job-reviews-sentiment-5class`. Output downloaded to
`model/ruroberta-sentiment-5class/` (gitignored, like the 3-class model).

**Eval (full test set, 17,523 rows, MAX_LENGTH=128 — not yet re-verified at 512 like the 3-class
model was):** accuracy 0.9161, macro F1 0.7538 (not directly comparable to the 3-class model's
0.7811 — different class definitions; `manual_review` shrank from ~3,300 to 347 gold rows).

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| positive | 0.898 | 0.862 | 0.880 | 1407 |
| negative | 0.993 | 0.927 | 0.959 | 14491 |
| manual_review | 0.318 | 0.793 | 0.454 | 347 |
| spam | 0.799 | 0.899 | 0.846 | 741 |
| service_complaint | 0.496 | 0.868 | 0.631 | 537 |

See `outputs/eval-5class/ruroberta_test_report.txt`. Confirmed remaining noise **outside** the
relabeled subset: the top confident errors are almost all original `positive`/`negative` rows
whose text is actually a "please remove this defamatory review" / business-dispute letter
(never sent to DeepSeek, since their `status`/`isPositive` weren't `0`/`0`) — the model correctly
flags these as `manual_review` but they're scored wrong against the untouched, noisy original
label. Quantified via a keyword scan of the untouched ~142,173 non-`manual_review` rows: real
"defamation/removal-request" pattern hits are concentrated in the small `positive` class (~1%,
~100-150 rows) and much rarer in `negative` (~0.05-0.1%, ~100-130 rows) once generic
"Уважаемая редакция"-style salutations (a normal genre convention for genuine negative reviews
too) are excluded as false positives — **under 0.3% of the full corpus**. **Decision: not worth a
full 175,804-row DeepSeek relabel** for this; a full pass would cost ~5x the already-completed
33,631-row run to fix a few hundred rows with negligible effect on macro metrics. Left as known,
documented residual noise; not fixed.

**Threshold-policy bug found and fixed: extreme class weights break probability-threshold
routing for `negative`.** `WeightedTrainer`'s weight formula gives `manual_review` weight ~10.68
vs `negative`'s ~0.24 (a ~44x ratio, far more extreme than the 3-class scheme's ~9.6x, since
`manual_review` shrank to ~1.87% of train data). This collapses `negative`'s softmax probability
into a narrow ~0.27–0.33 band regardless of input confidence — confirmed via local diagnostic
(loaded the downloaded model, sampled 3,000–4,000 test rows, inspected raw logits): argmax stays
correct (92.7% recall on `negative`), but the probability *magnitude* carries almost no signal
(p10=0.308, p90=0.319 across all `negative`-argmax rows; mean top1–top2 margin only 0.027). Net
effect: `neg_prob >= thr` auto-decide gives **0% coverage at any threshold**, even 0.50 — the
policy was completely non-functional, not merely miscalibrated.

**Fix: post-hoc temperature scaling, no retraining needed.** Sharpening the softmax
(`softmax(logits / TEMPERATURE)`, `TEMPERATURE < 1`) restores separation because argmax (and thus
accuracy) is temperature-invariant — only the confidence *readout* changes. `TEMPERATURE=0.25` was
picked via a local CPU sweep (3,000–4,000-row sample) and confirmed on the **full** test set via
`kaggle/eval-5class/` (T4): `neg_thr=0.50` → precision **0.996** (12,179/17,523 rows auto-decided),
`pos_thr=0.70` → precision **0.904** (1,330 rows) — the first time this project's positive
threshold has cleared the 0.90 "safe" bar (the 3-class model's positive precision capped at
~0.847, never reaching it). Combined auto-coverage ≈ **77%** of the full test set, vs ~7% before
the fix (only `positive` partially worked) and vs the 3-class model's more limited
negative-only routing. See `outputs/eval-5class/ruroberta_threshold_grid.txt`.

**Implementation** (`scripts/predict_sentiment.py`): `predict()` takes a `temperature` parameter
(`softmax(logits / temperature)`); defaults to `1.0` (no-op) unless the loaded model's `id2label`
matches the 5-class scheme, auto-detected via `is_five_class_scheme()`. For a detected 5-class
model, defaults switch to `TEMPERATURE=0.25`, `neg_thr=0.50`, `pos_thr=0.70`, `MAX_LENGTH=128` —
all overridable via `--temperature`/`--negative-threshold`/`--positive-threshold`/`--max-length`.
3-class model behaviour is unchanged (verified: same output before/after this change). `spam_prob`/
`service_complaint_prob` are added to the output dict when the model exposes those classes;
`decide_label()` needed no changes since it already only branches on `raw_top_label in
{"negative", "positive"}`, routing everything else (`manual_review`/`spam`/`service_complaint` as
top label) to `manual_review`, matching the intended policy.

**Status: promoted to main on 2026-07-13.** `main.py`/`predict.py` now default to
`model/ruroberta-sentiment-5class/`; `predict.py`'s hardcoded inference `max_length` was changed
from 512 to 128 to match what was actually validated for this model (MAX_LENGTH=512 was
deliberately **not** re-verified for the 5-class model before promoting — a product-side decision,
not a technical blocker: downstream consumption of `decision` will accept whatever labels come out,
so the new `spam`/`service_complaint` values don't need any compatibility shim). The 3-class
model's local weights (`model/ruroberta-sentiment/`) were deleted on 2026-07-13, same day as
promotion — see the note in "What this repo is" above for how to get it back if ever needed.

## ruRoberta-large pipeline (current main model)

`train_ruroberta.py` (repo root) and `kaggle/ruroberta/train.py` are **identical, untracked**
files (`git status` still shows them as `??` — not yet committed; ask before assuming they should
be). Same two-phase scheme as RuBERT V2 (4 unfrozen layers, warmup+cosine, weight decay, label
smoothing 0.1, early stopping), adapted for `ai-forever/ruRoberta-large`:

- **P100 SM60 compat is order-sensitive.** Kaggle's default image ships torch 2.10.0+cu128, which
  has no compute kernels for P100 (Pascal/SM60) — `torch.tensor([1.0]).cuda()` succeeds (just
  allocates memory) but any real op raises `CUDA error: no kernel image is available`. The fix
  (pip-reinstall `torch==2.6.0+cu124`) **must run before the first `import torch` anywhere in the
  process** — `sys.modules` caches the already-`dlopen`'d C extension, so reinstalling after import
  is a silent no-op. `train_ruroberta.py` does the reinstall as literally the first executable code.
- **`model.safetensors`, not pickle** — the trained checkpoint is saved via `safetensors` (HF
  `Trainer` default), so loading it for inference never hits PyTorch pickle/`torch.utils.serialization`
  compat issues. Those issues *do* apply when loading the **base** `ai-forever/ruRoberta-large`
  weights from HF Hub at training time (its `pytorch_model.bin` was pickled with pre-2.6 PyTorch) —
  `train_ruroberta.py` installs a `torch.utils.serialization` + `.config` compat shim for that.
- **Kaggle GPU accelerator can be requested explicitly**: `kaggle kernels push --accelerator
  NvidiaTeslaT4` (undocumented in `kagglesdk`, but works) requests T4 over the default P100 lottery.
  T4 (Turing, SM75) has Tensor Cores and is natively supported by Kaggle's stock torch (no reinstall
  needed) — used for `kaggle/eval/` (see below).
- **`kaggle/eval/`** (`eval.py` + `kernel-metadata.json`) is a separate Kaggle kernel for evaluation
  only: references the training kernel's output via `kernel_sources` (auto-mounted under
  `/kaggle/input/`) instead of re-uploading the model as a dataset, runs on GPU, defaults to the
  **full** test set (no `--limit`) since GPU throughput makes that fast. Same metrics logic as
  `scripts/validate_sentiment_on_tsv.py`, duplicated rather than imported (Kaggle kernels are
  single-file).
- The model is also published as a public Kaggle Dataset (`megannnn98/ruroberta-sentiment`, config
  + safetensors + tokenizer only — no optimizer/checkpoint bloat) so `README.md`'s download
  instructions have somewhere real to point.
- **Probabilities are much less peaked than RuBERT's** (`label_smoothing=0.1` effect is stronger
  here) — e.g. `neg_prob` on the held-out test essentially never exceeds ~0.65–0.69, so RuBERT's
  old `safe`-policy threshold (`neg_thr=0.70`) gives **zero** auto-negative coverage on this model.
  See the updated thresholds in `scripts/predict_sentiment.py` / below.

## Two pipelines live side by side — know which one you're touching

1. **Current / active: sentiment classification** (5-class as of 2026-07-13).
   - `scripts/split_jsonl_sentiment_dataset.py` → `data/sentiment_{train,val,test}.tsv`
   - **Main model:** 5-class ruRoberta-large → `model/ruroberta-sentiment-5class/` (`positive` /
     `negative` / `manual_review` / `spam` / `service_complaint`), trained via
     `kaggle/ruroberta-5class/train.py` on `data/sentiment5_{train,val,test}.tsv` (independently
     relabeled `manual_review` subset via DeepSeek), evaluated via `kaggle/eval-5class/eval.py`.
   - **Previous main (3-class ruRoberta-large, see lineage table above):** local weights deleted
     2026-07-13 after promotion — reproducible via `train_ruroberta.py` or the public Kaggle
     Dataset `megannnn98/ruroberta-sentiment`. RuBERT V2 (`model/rubert-sentiment-v2/`) was
     deleted earlier still, 2026-07-12.
   - `scripts/predict_sentiment.py` (inference + production policy — auto-detects 3-class vs
     5-class via `id2label` and switches temperature/threshold defaults accordingly),
     `predict.py` (quick argmax CLI)
   - `scripts/validate_sentiment_on_tsv.py`, `scripts/validate_sentiment_on_jsonl.py` (evaluation)

   - **Binary variant** (clean positive/negative, `manual_review` dropped from training and
     handled as a threshold fallback): `scripts/split_jsonl_binary_dataset.py` →
     `data/binary_{train,val,test}.tsv`; `train_fast_bert_binary_cpu.py` (thin wrapper that
     re-points paths/labels on top of `train_fast_bert_cpu.py`, no changes to that file) →
     `outputs/rubert-binary-cpu/final/`. Details in `docs/legacy.md`.

2. **Legacy: 2-class publish/reject** (documented in `docs/legacy.md`, still present, not the
   current focus).
   - `scripts/split_dataset.py` → `data/{train,val,test}.tsv`
   - `scripts/train_publish_classifier.py` → `outputs/publish-classifier/best/`
   - `scripts/evaluate_model.py`, `scripts/predict_review.py`
   - `scripts/prepare_labeling_sample.py`, `scripts/heuristic_label_reviews.py` are from an even
     earlier task and unused.

3. **Quick inference scripts on the main model** (`main.py`, `predict.py`). Both load
   `model/ruroberta-sentiment-5class/` and do **plain argmax** at `MAX_LENGTH=128`: `main.py` runs
   a hardcoded demo text; `predict.py "<text>" [--json]` is the one-review CLI (prints the label,
   or label/confidence/probabilities with `--json`). Note argmax on the raw (non-temperature-scaled)
   softmax will show a low-looking `negative`/`positive` confidence even when correct — see
   "5-class variant" section above for why (extreme class weights compress those probabilities;
   `scripts/predict_sentiment.py`'s temperature scaling fixes this for threshold decisions, but
   these two quick-CLI scripts intentionally bypass that policy entirely — use them for a fast
   sanity check, not for production decisions).

## Data: `aj_reviews_export.jsonl` is the source of truth

Note: the JSONL is large and kept **out of the git repo** (GitHub-safe); it may be absent on a
fresh clone. The committed working artifacts are the split TSVs. To rebuild any split you need the
JSONL present locally; the binary splits can alternatively be derived by filtering `data/sentiment_*.tsv`
(drop `manual_review`), which preserves the same train/val/test boundaries.

175,804 JSONL rows. Fields: `id, name, descr, status, isfunny, isPositive, is_about_work, is_self_delete`.
`descr` is the review text. Real distribution in the current export: `status` 0→45,322 / 1→130,482;
`isPositive` 0→164,010 / 1→11,794.

**Encoding — feed plain UTF-8 at inference.** The current export (`aj_reviews_export.jsonl`,
175,804 rows) is essentially **clean UTF-8**: a scan finds ~99% normal Cyrillic and only ~2 rows
in the old double-encoded mojibake (UTF-8 read as latin-1, e.g. `Ð¤Ð¸Ñ€Ð¼Ð°`). The split script feeds
`descr` as-is, so the model is trained on (effectively) clean text. **At inference pass ordinary
UTF-8 — do not re-encode or "fix" anything.** Verified empirically: clean text gives correct
confident predictions, while artificially re-mojibaking the input
(`text.encode("utf-8").decode("latin-1")`) collapses everything to `manual_review`.

> Historical note: an earlier export was mostly mojibake — hence `decode_for_display`
> (`latin-1`→`utf-8` + `html.unescape`) in the validation scripts. On the current clean data it is
> effectively a no-op / best-effort (clean Cyrillic can't `.encode("latin-1")`), used only for
> readable reports. The old "feed mojibake at inference too" rule no longer applies.

**Sentiment label mapping** (`map_label` in `split_jsonl_sentiment_dataset.py`) — order matters,
`status` wins over `isPositive`:
- `status == 1` → `negative`
- else `isPositive == 1` → `positive`
- else → `manual_review`

**`manual_review` is a moderation status, not a learnable text class.** It is `status==0 & isPositive==0`
(rejected/unmoderated). Its text overlaps fully with positive/negative, so the model cannot separate
it from text alone (it stays the weakest class across every version — held-out F1 0.52 GPU1/GPU2,
0.554 V2; much of gold `manual_review` gets confidently read as pos/neg). Treat divergence from this
label as expected, not a model bug. Confirmed directly by the 5-class independent-judge relabeling
(see "5-class variant" section above): only 9.8% of this bucket was genuinely ambiguous — the rest
was really `negative` (43.0%), `spam` (24.6%), `service_complaint` (15.5%), or `positive` (7.1%).

**TSV gotcha:** `text` fields are multi-line and csv-quoted. Always read splits with the `csv` module
(as the scripts do); `cut`/`wc -l`/`head` will mis-split rows and give garbage counts. `data/sentiment_*.tsv`
and `outputs/` are git-ignored generated artifacts.

**`.gitignore` blind spot — do not `git add` new fine-tuned model dirs.** `.gitignore` excludes
base model weights (`model/rubert-base-cased/`) and the four known fine-tuned dirs
(`model/rubert-sentiment-gpu1/`, `-gpu2/`, `-v2/`, `model/ruroberta-sentiment/`). If you produce a
**new** local model directory (another checkpoint, a different architecture, etc.), add it to
`.gitignore` before running any blanket `git add` — weight files easily exceed GitHub's 100 MB
limit and will break the push.

## Common commands

```bash
# Regenerate sentiment splits from the JSONL (stratified 80/10/10, seed 20260630)
.venv/bin/python scripts/split_jsonl_sentiment_dataset.py

# Train locally (CPU baseline, slow — cap data via env vars for a quick pass)
.venv/bin/python train_fast_bert_cpu.py
MAX_TRAIN_SAMPLES=10000 MAX_EVAL_SAMPLES=2000 .venv/bin/python train_fast_bert_cpu.py

# Retrain RuBERT on Kaggle: push rubert-train-p100.py (GPU1/GPU2 config) via
# `kaggle kernels push`, or adapt it for the V2 hyperparameters in docs/model-tuning.md
# (the V2 script that actually produced model/rubert-sentiment-v2 is not committed here).

# Retrain ruRoberta-large on Kaggle (~3h on P100, see lineage section above)
cp train_ruroberta.py kaggle/ruroberta/train.py
.venv/bin/kaggle kernels push -p kaggle/ruroberta
# download the trained model from the kernel's Output tab, or:
.venv/bin/kaggle kernels output megannnn98/ruroberta-train -p model/ruroberta-sentiment/

# Evaluate ruRoberta-large on Kaggle GPU (full test set, no --limit needed — see kaggle/eval/)
.venv/bin/kaggle kernels push -p kaggle/eval --accelerator NvidiaTeslaT4

# Honest held-out evaluation locally (CPU — slow for ruRoberta-large; use --limit or prefer
# kaggle/eval/ above). The test file is pre-shuffled, so --limit N is a representative sample.
.venv/bin/python scripts/validate_sentiment_on_tsv.py \
  --input data/sentiment_test.tsv --model-dir model/ruroberta-sentiment \
  --limit 4000 --max-length 128 --threshold-grid

# Quick one-review CLI (plain argmax on the main model)
.venv/bin/python predict.py "Зарплату задерживают уже три месяца."          # → negative
.venv/bin/python predict.py --json "Зарплату задерживают уже три месяца."   # label/confidence/probabilities

# One review with the production policy (see policy section below)
.venv/bin/python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment --text "..."

# Batch inference: input TSV must have a 'text' column; prediction columns are appended
.venv/bin/python scripts/predict_sentiment.py \
  --model-dir model/ruroberta-sentiment --input data/sentiment_test.tsv --output preds.tsv

# Syntax check (closest thing to a test)
.venv/bin/python -m py_compile scripts/predict_sentiment.py
```

## Training architecture (two-phase frozen-backbone schedule)

Shared by every `train_fast_bert*`/`rubert-train-p100.py` variant:
- **Phase 1:** freeze all BERT layers, train only the classifier head.
- **Phase 2:** unfreeze the last N encoder layers, fine-tune (N=2 for GPU1/GPU2, N=4 for V2 — see
  lineage table above).

| | CPU baseline | GPU1 (`train_fast_bert.py`) | GPU2 (`rubert-train-p100.py`) |
|---|---|---|---|
| `MODEL_NAME` | `model/rubert-base-cased` | `DeepPavlov/rubert-base-cased` (hub) | `DeepPavlov/rubert-base-cased` (hub) |
| `use_cpu` / `fp16` | `True` / off | `False` / `True` | `False` / `True` |
| Hardware | local CPU | Kaggle T4x2 | Kaggle P100 |
| phase1 batch / epochs | 32 / 2 | 64 / 1 | 64 / 1 |
| phase2 batch × grad-accum / epochs | 8 × 4 / 1 | 16 × 2 / 2 | 16 × 2 / 2 |
| output dir | `outputs/rubert-sentiment-cpu` | `outputs/rubert-sentiment` (Kaggle-side) | `/kaggle/working/rubert-sentiment` |

CPU held-out macro F1 ~0.62; superseded by every GPU version. RuBERT V2 (not in this table — see
lineage table above) adds 4 unfrozen layers, warmup+cosine LR, early stopping, weight decay, label
smoothing, and grad clipping on top of the GPU2 base config.

**ruRoberta-large** (`train_ruroberta.py`) follows the same two-phase shape but scaled for a 355M
model: `MODEL_NAME = "ai-forever/ruRoberta-large"`, phase1 batch 32/1 epoch, phase2 batch 8 ×
grad-accum 4 (effective 32) / 4 epochs, 4 unfrozen layers, same warmup+cosine/weight
decay/label smoothing/early stopping as RuBERT V2. Runtime on Kaggle P100: ~3h08m total. Output:
`/kaggle/working/ruroberta-sentiment/final` → downloaded as `model/ruroberta-sentiment/`.

Other essentials: **training** uses `MAX_LENGTH=128` (unchanged — retraining with a longer window is
untested and not required, see below). `WeightedTrainer` applies balanced class weights (the data is imbalanced;
without weights the model collapses to `negative`). Metric is **macro F1**, not accuracy. Tokenized data
is cached under the run's `tokenized/` dir with a fingerprint (`metadata.json`) that auto-invalidates
when the model, splits, `MAX_LENGTH`, or label map change.

## Inference-time MAX_LENGTH=512 (not 128 — training/inference differ here)

**Inference scripts (`scripts/predict_sentiment.py`, `predict.py`, `scripts/validate_sentiment_on_tsv.py`)
default `--max-length` to 512, not the 128 used for training.** This looks inconsistent but is
empirically correct: a disagreement audit (`scripts/find_disagreements.py` / `kaggle/disagreements/`)
found high-confidence errors concentrated on long, multi-paragraph "плюсы/минусы"-style reviews where
the true sentiment payload sits past token 128 — truncation at 128 cut the model off mid-preamble
before the actual complaint. Re-running the **same already-trained checkpoint** at `max_length=512`
(no retraining) on the full test split (17,523 rows, via `kaggle/eval/` with `MAX_LENGTH=512`) confirmed
the fix: accuracy 0.854→0.865, macro F1 0.771→0.7811, with every per-class F1 improving (`manual_review`
+1.35pp, the largest gain) and none regressing. Runtime cost is real (~450s→~1455s on a T4 for the full
test set, due to per-batch padding to the longest sequence) but irrelevant for single-review inference.
Training itself still uses `MAX_LENGTH=128` (see above) — retraining with a longer window is a separate,
untested question; this fix is inference-only and needed no retrain.

## Inference decision policy (`scripts/predict_sentiment.py`)

Thresholds come from the held-out **full-test** threshold grid on ruRoberta-large (run via
`kaggle/eval/`, see lineage section): negative precision reaches ~0.927 at `neg_prob >= 0.50`;
positive precision tops out ~0.844 at `pos_prob >= 0.90` (never reaching a safe 0.90 target —
positive stays under threshold gating by design, same conclusion as RuBERT). These thresholds were
re-verified at `MAX_LENGTH=512` and barely move (neg precision 0.927→0.928, pos precision cap
0.844→0.847) — no threshold retuning needed after the MAX_LENGTH bump. `decide_label()`
exposes three policies via `--policy`:
- `balanced` (default): auto-decide `negative` if `neg_prob >= 0.50`, `positive` if `pos_prob >= 0.90`, else `manual_review`.
- `safe`: auto-decide only high-confidence `negative` (`>= 0.60`); auto-positive is off unless `--enable-auto-positive`.
- `argmax`: legacy top-label behaviour with the old `--threshold` (default 0.5) fallback.

**These thresholds are ruRoberta-specific, not portable to RuBERT models.** ruRoberta was trained
with `label_smoothing=0.1` and its probabilities are noticeably less peaked than RuBERT's — on the
held-out test, `neg_prob` essentially never exceeds ~0.65–0.69, so RuBERT's old `safe` threshold
(`neg_thr=0.70`) gives **zero** auto-negative coverage on ruRoberta. If evaluating a RuBERT
checkpoint, use RuBERT's thresholds (`neg_thr=0.55` balanced / `0.70` safe) instead.

Output fields: `decision, decision_policy, raw_top_label, raw_confidence, positive_prob, negative_prob, manual_review_prob`.
The asymmetric design is deliberate: a wrong auto-decision is costlier than an extra `manual_review`,
so uncertain and positive-leaning text is routed to manual review.
