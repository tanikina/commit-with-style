# commit-with-style

Prototype for commit-message **completion**: given a code diff, a repo's recent commit
history, and a user-typed prefix of a commit message, generate continuations with different
LLMs/strategies, then score them with an LLM-as-a-judge on:

- **coherence** — does the message match the style/tone of the repo's commit history?
- **fluency** — is it grammatically correct and natural?
- **correctness** — are its claims actually supported by the diff (no invented facts, no
  omitted main change)?

It also includes a LoRA fine-tuning phase: fine-tune a small Qwen model on the same task,
hyperparameter-search over a few LoRA configs, and evaluate the winning adapter with the
same judge/metrics harness so it's directly comparable to the 3 prompting strategies.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .

# only if you'll call the Grazie backend (--backend grazie / --judge-backend grazie)
.venv/bin/pip install -e ".[grazie]"
```

## Generation strategies

1. **`vanilla`** — single-shot continuation from diff + history + prefix.
2. **`structured`** — three steps: (a) extract a compact structured change-plan
   (`change_type`, `component`, `main_change`, `supported_motivation` — null unless actually
   backed by the diff/history, `important_identifiers`, style info), (b) generate a
   continuation from the plan + history + prefix, (c) verify the candidate against the diff
   (claims supported? anything omitted? prefix preserved? style match? concise?) and apply at
   most **one** revision — no repeated self-refinement loops.
3. **`reference_guided`** — sees the diff, history, *and* the original developer's actual
   commit message; may improve clarity/style/grammar but must not add unsupported facts.

## Data

Reads examples directly from the local file
`data/evaluation_data_8000-samples_2025-09-02.jsonl` (8000 rows, 4 languages, already split
into `prefix`/`completion` such that `prefix + completion == msg`). No separate
download/build step is needed — `commit_style/data.py` streams and stratifies samples from
this file.

For anything that trains or evaluates a model (as opposed to a quick `load_examples`-based
smoke test), `load_project_splits` draws three sets from one shared partition:

- **`train`** (default 1000 rows) — fine-tuning training data.
- **`strategy_val`** (default 200 rows) — fixed set for comparing the 3 prompting strategies
  (Phase 1).
- **`finetune_test`** (default 300 rows) — fixed set for LoRA hp-search config selection and
  the final fine-tuned-model evaluation (Phase 2).

All three are **mutually disjoint by `repo_name`**: every repo's commits are assigned
wholesale to exactly one split, so a model is never evaluated on commits from a repo it saw
during training (or during strategy selection) — a real leakage risk with a naive row-level
random split, since a model could pick up repo-specific style rather than generalizing.
Within each split, rows are stratified by both language and prefix length (short/medium/long
bins), so each set has varied representation instead of whatever distribution random sampling
happens to produce. Since it's a pure function of `(path, seed, size args)`, calling it with
the same arguments from different scripts (`run_prototype.py --strategy-val` and
`finetune_qwen.py`) deterministically yields the same non-overlapping partition without any
shared state.

A 4th, independent holdout for the multi-condition fine-tuning campaign (see
[Multi-condition fine-tuning campaign](#multi-condition-fine-tuning-campaign) below) is drawn
by `load_final_eval_split`. It calls `load_project_splits` first (unperturbed — the 3 sets
above are byte-identical whether or not this function is ever called), collects the repos
already used by `train`/`strategy_val`/`finetune_test`, and builds the new split only from the
remaining, genuinely-unused repos — so it's disjoint from all three by construction, not just
by the language/prefix stratification. Leftover repo availability is skewed by language, so
per-language quotas are water-filled proportionally to each language's headroom rather than
split evenly.

## LLM backends

`commit_style/llm/` exposes one `LLMClient` interface with three implementations, selected
via `--backend`:

- `mock` (default) — deterministic, no network/model, used to validate pipeline wiring.
- `grazie` — JetBrains' internal Grazie API gateway, e.g. `google-gemini-3-5-flash`. Requires
  a JWT token (see below).
- `local_hf` — a small local Hugging Face model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) via
  `transformers`, optionally with a LoRA adapter (`adapter_path`, see Fine-tuning below). A
  full Qwen3.5-4B run is deferred to a future, larger GPU VM.

Structured output (`ChangePlan`, `VerificationResult`, `JudgeScores`) is enforced the same
way across all backends: instruct the model to emit JSON matching the pydantic schema, parse
it (tolerating surrounding commentary), validate, and retry once on failure.

### Grazie token

The Grazie client needs a JWT token. Either:

- export `GRAZIE_JWT_TOKEN` (or `GRAZIE_USER_JWT_TOKEN`/`GRAZIE_SERVICE_JWT_TOKEN`) in the
  shell you run the script from, or
- pass `--grazie-token-file path/to/token.txt` — the file may contain either the raw token or
  a shell-export line like `GRAZIE_JWT_TOKEN="..."`.

Never commit a token file (`jwt_token.txt` is gitignored).

## Running it

```bash
# mock dry run (no network) — 12 stratified examples x 3 strategies
.venv/bin/python scripts/run_prototype.py

# live run against Grazie/Gemini
.venv/bin/python scripts/run_prototype.py \
  --n 2 \
  --backend grazie --model google-gemini-3-5-flash \
  --grazie-token-file jwt_token.txt \
  --out outputs/smoke_grazie.jsonl
```

Useful flags: `--n` (sample count), `--seed`, `--strategies` (subset of
`vanilla structured reference_guided`), `--judge` (repeatable, see below), `--with-bertscore`
(adds BERTScore F1 to metrics; lazy-loads a ~1.4GB model on first use).

For strategy comparison (as opposed to a quick smoke test), use `--strategy-val` to run
against the fixed, repo-disjoint 200-sample `strategy_val` set from `load_project_splits`
instead of `--n` ad-hoc sampling:

```bash
.venv/bin/python scripts/run_prototype.py \
  --strategy-val \
  --backend grazie --model google-gemini-3-5-flash \
  --grazie-token-file jwt_token.txt \
  --out outputs/strategy_selection.jsonl
```

Output is one JSON row per (example, strategy) written to `--out` (default
`outputs/dry_run.jsonl`), plus a printed summary table of mean judge scores per strategy.
Each row includes the diff-derived context, the candidate continuation/full message, the
reference completion/message, judge scores + rationale, per-example metrics (`chrf`,
`meteor`, optionally `bertscore_f1`), and strategy-specific artifacts (e.g. the `structured`
strategy's change plan and verification result).

### Comparing multiple judges

`--judge BACKEND:MODEL` is repeatable, letting you score every candidate with several judge
models at once — e.g. to compare whether `openai-gpt-5` and `anthropic-claude-sonnet-5` agree
on the same completions. If `--judge` is omitted entirely, it falls back to a single judge
using the generation backend/model (today's default behavior).

```bash
.venv/bin/python scripts/run_prototype.py \
  --strategy-val \
  --backend grazie --model google-gemini-3-5-flash \
  --judge grazie:openai-gpt-5 --judge grazie:anthropic-claude-sonnet-5 \
  --grazie-token-file jwt_token.txt \
  --out outputs/strategy_multi_judge.jsonl
```

With multiple judges, each output row's `judge_scores` is a dict keyed by judge id (the
model name, or the backend name if no model name is given — e.g. `mock`):
`{"openai-gpt-5": {coherence, fluency, correctness, rationale}, "anthropic-claude-sonnet-5":
{...}}`. `summarize()` prints one table block per judge id.

## Fine-tuning (LoRA)

`scripts/finetune_qwen.py` LoRA-fine-tunes a small causal LM on the vanilla-strategy prompt
(same `HISTORY:`/`DIFF:`/`PREFIX:` shape used above), runs a small manual hyperparameter grid,
and evaluates the winning adapter with the exact same judge/metrics harness as the 3
prompting strategies, so results are directly comparable.

```bash
.venv/bin/python scripts/finetune_qwen.py \
  --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 3 --batch-size 1 --grad-accum 8 --max-seq-length 768 \
  --output-dir outputs/finetune \
  --out outputs/finetune_eval.jsonl
```

What it does:

1. `load_project_splits` draws the shared, repo-disjoint, language-and-prefix-length-
   stratified partition described in [Data](#data) — `--n-train` (default 1000) for
   training, `--n-test` (default 300) for both hp-search config selection and the final
   fine-tuned-model evaluation. `--n-strategy-val` (default 200) isn't used directly by this
   script; it's passed so the partition computed here has the same `strategy_val` set carved
   out as `run_prototype.py --strategy-val` (same `--data`/`--seed`), keeping the test set
   disjoint from both the training set *and* the Phase-1 strategy-selection set.
2. Trains on the target-model's `reference_completion` (see [Training targets](#training-targets)
   below for the opt-in synthetic-completion mode), masking the prompt out of the loss
   (`commit_style/finetune/dataset.py`), for each config in a small manual LoRA grid
   (`commit_style/finetune/hp_search.py`: 3 hand-picked `r`/`alpha`/`dropout`/`lr`
   combinations — not a full search), via a plain `transformers.Trainer` + `peft` LoRA
   (`commit_style/finetune/train.py`). Each config's adapter is saved under
   `--output-dir/config_N`; validation loss is printed as a table and written to
   `hp_search_results.json`.
3. Picks the config with the lowest `eval_loss`, loads it as a `LocalHFClient` adapter
   (`adapter_path=...`) — the fine-tuned model becomes just another pluggable `LLMClient` —
   and evaluates it on the full `--n-test` set with `judge_candidate` and the
   `chrf`/`meteor`(/`bertscore`) metrics, unchanged from the prompting-strategy pipeline
   (`commit_style/finetune/evaluate.py`).
4. Writes `--out` (default `outputs/finetune_eval.jsonl`, rows with `strategy ==
   "finetuned"`) and prints the same summary table as `run_prototype.py` — concatenate it
   with a `dry_run.jsonl` from `run_prototype.py` for a side-by-side comparison.

Each row also carries a `usage` field —
`{"input_tokens": int, "output_tokens": int, "latency_s": float}` for the fine-tuned model's
single generation call (measured directly against its own tokenizer/wall-clock, so it's exact
rather than estimated). `summarize()` prints a `--- cost ---` block with the mean
`in_tok`/`out_tok`/`latency_s` per `strategy` value (i.e. per `--run-label`), which is what
you want when comparing inference cost across the 5 conditions in the
[multi-condition campaign](#multi-condition-fine-tuning-campaign) below. `run_prototype.py`'s
rows don't get a `usage` field (a strategy there can make several LLM calls — e.g.
`structured` does 3 — so a single last-call snapshot wouldn't represent the strategy's total
cost); `summarize()` simply omits the cost block when no row has `usage`.

Notes:

- Defaults target the **local 4GB-VRAM GPU**: `Qwen/Qwen2.5-0.5B-Instruct` (not the
  `Qwen3.5-4B` originally asked for — too large to LoRA-tune locally), `--batch-size 1
  --grad-accum 8 --max-seq-length 768`, gradient checkpointing on. Moving to a bigger model
  on a GPU VM later is meant to be just `--model-name Qwen/Qwen3.5-4B-Instruct` (plus larger
  `--batch-size`/`--max-seq-length`).
- `--load-in-4bit` (QLoRA via `bitsandbytes`) is wired for that future larger-model/GPU-VM
  case but **untested** this session — install it via `.venv/bin/pip install -e ".[qlora]"`
  first.
- `--judge`/`--grazie-token-file` work exactly like `run_prototype.py`'s equivalent flags
  (repeatable `--judge BACKEND:MODEL`; defaults to a single `mock` judge if omitted). See
  [Comparing multiple judges](#comparing-multiple-judges) above.

### Training targets

By default, fine-tuning trains on the dataset's human-written `reference_completion` for
every `--n-train` example, unchanged from Phase 2. `--training-target generated` instead
generates each training example's target by running one prompting strategy (default
`structured`, any of `vanilla`/`structured`/`reference_guided` via
`--training-target-strategy`) through a generator LLM (default `grazie:google-gemini-3-5-flash`,
override with `--training-target-backend`/`--training-target-model`) and substituting the
strategy's `continuation` for `reference_completion` — a form of distillation from the
prompted model into the fine-tuned Qwen model, rather than training on the raw human label.
Everything else about an `Example` (diff, history, prefix, `reference_full_message`) is left
untouched, so evaluation still compares against the original human reference.

```bash
.venv/bin/python scripts/finetune_qwen.py \
  --training-target generated \
  --training-target-strategy structured \
  --training-target-backend grazie --training-target-model google-gemini-3-5-flash \
  --grazie-token-file jwt_token.txt \
  --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --output-dir outputs/finetune_generated \
  --out outputs/finetune_eval_generated.jsonl
```

**Cost note:** generating targets for 1000 training examples means ~1000 real API calls for
`vanilla`/`reference_guided`, or ~3000 for `structured` (plan + generate + verify per
example) — do a small `--n-train` real-backend check before running the full 1000. Results
are cached to `--training-targets-cache` (default
`outputs/training_targets_cache.jsonl`, keyed by example id); a rerun with the same
`--n-train`/`--seed`/`--data` reuses the cache instead of re-paying for generation, and only
regenerates (overwriting the cache) if any example id in the current run isn't already
covered by it.

#### Generating targets standalone

`scripts/generate_training_targets.py` runs just the generation step above — no hp-search,
no training, no eval — so you can pre-populate the cache (e.g. once per strategy you want to
compare) without committing to a full fine-tuning run each time:

```bash
.venv/bin/python scripts/generate_training_targets.py \
  --strategy structured \
  --backend grazie --model google-gemini-3-5-flash \
  --grazie-token-file jwt_token.txt \
  --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --out outputs/training_targets_structured.jsonl
```

It takes the same `--data`/`--n-train`/`--n-strategy-val`/`--n-test`/`--seed` args as
`finetune_qwen.py` (so it derives the identical partition) plus `--strategy`/`--backend`/
`--model` (equivalent to `finetune_qwen.py`'s `--training-target-strategy`/`-backend`/
`-model`). If `--out` already covers every id in the current `--n-train` set, it exits
immediately without spending any API calls. Point `finetune_qwen.py --training-target
generated --training-targets-cache outputs/training_targets_structured.jsonl` at the result
to train on it.

### Notes on the hyperparameter search

`HP_GRID` in `commit_style/finetune/hp_search.py` is a **small hand-picked grid**, not an
automated search (e.g. Optuna) — 3 configs varying LoRA rank, alpha (kept at `2*r`, the
common rule of thumb), dropout, and learning rate together:

| config | r  | alpha | dropout | lr   |
|--------|----|-------|---------|------|
| 0      | 8  | 16    | 0.05    | 2e-4 |
| 1      | 16 | 32    | 0.05    | 1e-4 |
| 2      | 8  | 16    | 0.10    | 5e-5 |

Selection is by validation loss alone (`Trainer.evaluate()`'s `eval_loss`) — the LLM-judge/
metrics pass only runs once, on the winning config, to avoid paying for judge calls on
configs that lose anyway. Results for a given run are in the printed table and
`--output-dir/hp_search_results.json`.

**Reusing one search across strategies.** The winning hp-search config depends on the model
and the shape of the training data, not on which strategy generated the training-target
completions — so if you're comparing `--training-target-strategy` options (e.g. `vanilla`
vs. `structured`), you don't need to re-run the 3-config grid for each one. Every
`finetune_qwen.py` run (grid-searched or reused) writes its winning params to
`<output-dir>/best_lora_params.json`; pass that file to `--lora-params-file` on a later run
to skip straight to training a single config instead of searching:

```bash
# first run: full grid search, e.g. on the default reference-completion target
.venv/bin/python scripts/finetune_qwen.py --output-dir outputs/finetune_vanilla_target ...

# later runs: reuse that config instead of re-searching
.venv/bin/python scripts/finetune_qwen.py \
  --training-target generated --training-target-strategy structured \
  --training-targets-cache outputs/training_targets_structured.jsonl \
  --lora-params-file outputs/finetune_vanilla_target/best_lora_params.json \
  --output-dir outputs/finetune_structured_target \
  --out outputs/finetune_eval_structured.jsonl
```

**Should this be repeated with the real `Qwen3.5-4B` target, with more options tried?**
Yes, worth doing, but not by simply scaling this exact 3-config grid up unchanged. The
0.5B stand-in mainly validates the *pipeline* (data splits, label masking, training loop,
adapter loading into the eval harness) — a 4B model has different capacity and will likely
land on a different optimal rank/lr than what's best here, and each config costs much more
to train on a bigger model/GPU. A better-calibrated next step: use this session's winning
config as a rough starting point, then spend the larger compute budget on a slightly wider
but still-bounded search — e.g. 5-8 configs, or a short random/Bayesian search (Optuna) over
`{r, lr}` centered near that starting point — rather than a full grid, since marginal gains
from hp tuning tend to flatten out quickly relative to just using a bigger/better base model
or more training data.

## Multi-condition fine-tuning campaign

Beyond a single fine-tuning run, `scripts/finetune_qwen.py` supports a campaign that fairly
compares 5 training conditions — **original** (human) completions vs. distilled completions
from the **vanilla**/**structured**/**reference_guided** strategies, plus a 50/50
**mixture** of original+structured — each trained with 3 different seeds (15 fine-tune runs)
and judged on the same independent holdout. This is meant to run on a bigger-GPU machine
against the real target model (e.g. `Qwen/Qwen3-4B`); this repo only needs to be cloned there
with the dataset file alongside it.

### 1. Fix the training-id set across conditions

Each of the 3 distillation caches (`outputs/training_targets_gemini_{vanilla,structured,
reference_guided}.jsonl`, produced by `scripts/generate_training_targets.py` — see
[Generating targets standalone](#generating-targets-standalone)) may cover slightly different
ids depending on which generations succeeded. For the comparison to be fair, all 5 conditions
must train on the exact same fixed set of examples. `scripts/prepare_finetune_conditions.py`
computes the intersection of ids across the `train` pool and all 3 caches, and builds the
fixed-halves mixture cache:

```bash
.venv/bin/python scripts/prepare_finetune_conditions.py \
  --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --vanilla-cache outputs/training_targets_gemini_vanilla.jsonl \
  --structured-cache outputs/training_targets_gemini_structured.jsonl \
  --reference-guided-cache outputs/training_targets_gemini_reference_guided.jsonl \
  --mixture-seed 1234 \
  --ids-out outputs/training_ids_intersection.txt \
  --mixture-out outputs/training_targets_gemini_mixture.jsonl
```

It writes `--ids-out` (one id per line — pass this to every condition's
`--training-ids-file`, below) and `--mixture-out` (a `write_training_targets_cache`-shaped
JSONL: a `random.Random(--mixture-seed)` shuffle of the intersection ids split into two fixed,
non-overlapping halves — the first half keeps its original `reference_completion`, the second
half takes the `structured` cache's completion for that id). Run it once, after all 3 caches
exist; it prints a report (train-pool size, per-cache id counts, intersection size, mixture
half-sizes) to sanity-check before committing to the full campaign.

### 2. `--training-ids-file`

Restricts `train_examples` to a fixed external id list — applied right after
`load_project_splits`/before any `--training-target generated` cache lookup, so every
condition trains on exactly the ids in `outputs/training_ids_intersection.txt` and each
generated-target cache's strict "covers every id in the current run" check trivially passes:

```bash
--training-ids-file outputs/training_ids_intersection.txt
```

The **original** condition needs no cache file at all — just `--training-target reference`
(the default) plus this flag; the filtered `train_examples` already carry their own
unmodified human `reference_completion`.

### 3. `--train-seed`

Decoupled from `--seed` (which stays fixed across the whole campaign — it only controls the
data partition). `--train-seed` (default 42) seeds training randomness only (weight init,
dataloader shuffling) via `transformers.set_seed` + `TrainingArguments(seed=, data_seed=)`.
Vary it (e.g. 1/2/3) across the 3 seed-replicates per condition.

### 4. `--final-eval-split final_eval` and `--run-label`

`--final-eval-split {finetune_test,final_eval}` (default `finetune_test`, unchanged from a
single run) controls only the *judged* comparison pass at the end of a run — training-time
eval loss (`Trainer`'s `eval_dataset`, and hp-search's config selection) always uses
`--n-test`/`finetune_test` regardless of this flag. Pass `final_eval` (plus `--n-final-eval`,
default 200) to instead judge the winning adapter against the brand-new, independent
`load_final_eval_split` holdout described in [Data](#data) — used for the 15 real condition
runs, not for the one-time hp-search.

`--run-label` (default `finetuned`) is written to each output row's `strategy` field, so the
15 conditions' `eval.jsonl` files can be concatenated and stay distinguishable to
`commit_style.pipeline.summarize()` (which groups by `strategy`) — e.g. `vanilla_seed1`,
`mixture_seed3`.

### 5. Full command sequence

```bash
# 1. hp-search once, on vanilla, intersection-restricted -> outputs/ft/hp_search/best_lora_params.json
.venv/bin/python scripts/finetune_qwen.py \
  --model-name Qwen/Qwen3-4B --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --training-ids-file outputs/training_ids_intersection.txt \
  --training-target generated --training-target-strategy vanilla \
  --training-targets-cache outputs/training_targets_gemini_vanilla.jsonl \
  --train-seed 42 --final-eval-split finetune_test --run-label hp_search \
  --output-dir outputs/ft/hp_search --out outputs/ft/hp_search/eval.jsonl \
  --judge grazie:google-gemini-3-5-flash --grazie-token-file jwt_token.txt

# 2. 15 runs = 5 conditions x 3 train-seeds, all reusing the hp-search's best_lora_params.json
BEST=outputs/ft/hp_search/best_lora_params.json
IDS=outputs/training_ids_intersection.txt
for seed in 1 2 3; do
  for cond in original vanilla structured reference_guided mixture; do
    case $cond in
      original)          TARGET="--training-target reference" ;;
      vanilla)            TARGET="--training-target generated --training-target-strategy vanilla --training-targets-cache outputs/training_targets_gemini_vanilla.jsonl" ;;
      structured)         TARGET="--training-target generated --training-target-strategy structured --training-targets-cache outputs/training_targets_gemini_structured.jsonl" ;;
      reference_guided)   TARGET="--training-target generated --training-target-strategy reference_guided --training-targets-cache outputs/training_targets_gemini_reference_guided.jsonl" ;;
      mixture)            TARGET="--training-target generated --training-target-strategy structured --training-targets-cache outputs/training_targets_gemini_mixture.jsonl" ;;
    esac
    .venv/bin/python scripts/finetune_qwen.py \
      --model-name Qwen/Qwen3-4B --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
      --training-ids-file $IDS $TARGET \
      --lora-params-file $BEST --train-seed $seed \
      --final-eval-split final_eval --n-final-eval 200 \
      --run-label ${cond}_seed${seed} \
      --output-dir outputs/ft/${cond}/seed${seed} \
      --out outputs/ft/${cond}/seed${seed}/eval.jsonl \
      --judge grazie:google-gemini-3-5-flash --grazie-token-file jwt_token.txt
  done
done
```

`mixture`'s `--training-target-strategy structured` is only a print-log label — the actual
completions come entirely from `--training-targets-cache
outputs/training_targets_gemini_mixture.jsonl`, which `load_training_targets_cache` applies
regardless of the strategy flag's value. `--seed 0` is identical everywhere (the data split
never changes); only `--train-seed` and the training-target wiring vary. Each of the 16
invocations writes to a distinct `outputs/ft/.../` directory and `--out` path, so nothing
clobbers anything else.

Substitute `Qwen/Qwen3-4B` with the real HF hub id available on the target machine, and add
`--load-in-4bit` if the GPU needs it (see [Notes](#fine-tuning-lora) above — untested this
session, install via `.venv/bin/pip install -e ".[qlora]"` first).

### 6. Out-of-the-box baseline (no fine-tuning)

To see what fine-tuning actually buys you, `scripts/eval_base_model.py` runs the same
single-shot vanilla-prompt generation (`commit_style/finetune/evaluate.py`'s
`run_finetuned_eval`, unchanged) against the **unmodified base model** — no adapter, no
training — and judges it on the exact same `load_final_eval_split` holdout as the 15
fine-tuned condition runs (same `--data`/`--seed`/`--n-train`/`--n-strategy-val`/`--n-test`/
`--n-final-eval` reproduce the identical partition). Run this **once** — there's no training
randomness to average over, so there's no `--train-seed`/seed-replicate loop here:

```bash
.venv/bin/python scripts/eval_base_model.py \
  --model-name Qwen/Qwen3-4B --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --n-final-eval 200 --run-label base_qwen3_4b \
  --out outputs/ft/base_qwen3_4b/eval.jsonl \
  --judge grazie:google-gemini-3-5-flash --grazie-token-file jwt_token.txt
```

Output rows are shaped identically to the fine-tuned runs' (including the `usage`
cost/latency field), so `outputs/ft/base_qwen3_4b/eval.jsonl` can be concatenated with the 15
conditions' `eval.jsonl` files and fed straight into `commit_style.pipeline.summarize()` for a
single side-by-side comparison table, keyed off `--run-label`/`strategy`.
