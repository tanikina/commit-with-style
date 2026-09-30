# commit-with-style

Commit-message **completion**: given a code diff, a repository's recent commit history, and a
user-typed prefix of a commit message, generate continuations with different LLMs/strategies
and score them with an LLM-as-a-judge on:

- **coherence**: does the message match the style/tone of the repo's commit history?
- **fluency**: is it grammatically correct and natural?
- **correctness**: are its claims supported by the diff (no invented facts, main change not omitted)?

The repository also covers LoRA fine-tuning of a small Qwen model on the same task, evaluated with
the same judge/metrics harness so results are directly comparable to the prompting strategies.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -e ".[vllm]"    # local GPU generation (default backend)
.venv/bin/pip install -e ".[qlora]"   # 4-bit training / bitsandbytes inference
.venv/bin/pip install -e ".[grazie]"  # API-gateway backend (optional)
```

## Generation strategies

1. **`vanilla`**: single-shot continuation from diff + history + prefix.
2. **`structured`**: (a) extract a structured change plan (`change_type`, `component`,
   `main_change`, `supported_motivation`, `important_identifiers`, style info), (b) generate a
   continuation from the plan + history + prefix, (c) verify the candidate against the diff and
   apply at most one revision.
3. **`reference_guided`**: additionally sees the original developer's commit message; may improve
   clarity/style/grammar but must not add unsupported facts.

## Data

Examples are read from `data/evaluation_data_8000-samples_2025-09-02.jsonl` (8000 rows, pre-split
into `prefix`/`completion` with `prefix + completion == msg`); `commit_style/data.py` streams and
stratifies samples from it. Statistics below are produced by `util_scripts/describe_data.py`.

| language | rows |
|---|---|
| Java | 2000 |
| JavaScript | 2000 |
| Kotlin | 2000 |
| Python | 2000 |

187 repositories; commits span 2022-01-01 to 2025-05-28.

| field | mean | min | max |
|---|---|---|---|
| prefix length (chars) | 19.3 | 2 | 146 |
| full message length (chars) | 49.3 | 14 | 247 |
| history length (# prior messages) | 84.5 | 0 | 917 |
| diff length (chars) | 6852.3 | 132 | 204182 |
| files changed per commit | 5.5 | 1 | 834 |

Prefixes are character-level cuts at ~20% / 40% / 60% of the message length (roughly even thirds);
about two-thirds end mid-word. At inference time, history is truncated to the last 5 messages
(`max_history_messages`) and the diff to 4000 characters (`max_diff_chars`).

### Splits

`load_project_splits` draws **repo-disjoint** splits (every repo's commits go to exactly one split),
stratified by language and prefix length. It is a pure function of `(path, seed, size args)`, so all
scripts reproduce the same partition. `load_final_eval_split` draws an additional holdout only from
repos unused by the other three splits. Realized sizes at `seed=0`:

| split | purpose | n | repos |
|---|---|---|---|
| `train` | fine-tuning | 996 | 61 |
| `strategy_val` | prompting-strategy comparison | 192 | 37 |
| `finetune_test` | LoRA config selection | 300 | 43 |
| `final_eval` | final holdout for fine-tuned models | 200 | 45 |

## LLM backends

`commit_style/llm/` exposes one `LLMClient` interface, selected via `--backend`:

- `vllm` (default): local Hugging Face model via vLLM, optionally with a LoRA adapter. If the
  engine fails with a `flashinfer`/`nvcc`/`CUDA_HOME` error, set `VLLM_USE_FLASHINFER_SAMPLER=0`.
- `local_hf`: plain `transformers`, optionally with a LoRA adapter.
- `grazie`: hosted models (e.g. Gemini, GPT, Claude) through an LLM API gateway. Requires a JWT
  token via `GRAZIE_JWT_TOKEN` or `--grazie-token-file`.
- `mock`: deterministic, no model; for pipeline dry runs.

Notes:

- Qwen3-family models are run with `enable_thinking=False`; `strip_thinking()` additionally removes
  any stray `<think>...</think>` block. `--debug` prints raw continuations as they are generated.
- Structured outputs (`ChangePlan`, `VerificationResult`, `JudgeScores`) are requested as JSON,
  validated against pydantic schemas, and retried once on failure.
- `--quantization bitsandbytes` loads vLLM models in 4-bit NF4 when bf16 weights do not fit.
- LoRA training always uses `transformers` + `peft`; the backend flags only affect generation.

## Prompting strategies

```bash
# dry run (no model)
.venv/bin/python scripts/run_prototype.py --backend mock

# strategy comparison on strategy_val with multiple judges
.venv/bin/python scripts/run_prototype.py \
  --strategy-val \
  --backend grazie --model google-gemini-3-5-flash \
  --judge grazie:openai-gpt-5 --judge grazie:anthropic-claude-sonnet-5 \
  --grazie-token-file token.txt \
  --out outputs/strategy_multi_judge.jsonl
```

Other flags: `--n`, `--seed`, `--strategies`, `--with-bertscore`. `--judge BACKEND:MODEL` is
repeatable; without it, the generation model is used as the judge. Each output row contains the
candidate, reference, per-judge scores and rationale, metrics (`chrf`, `meteor`, optionally
`bertscore_f1`), and strategy artifacts (e.g. change plan and verification result).

## Fine-tuning (LoRA)

`scripts/finetune_qwen.py` fine-tunes on the vanilla-strategy prompt with the prompt masked out of
the loss, runs a small LoRA grid, and evaluates the best adapter (lowest `eval_loss`) with the same
judge/metrics harness. Output rows include a `usage` field (input/output tokens, latency).

| config | r  | alpha | dropout | lr   |
|--------|----|-------|---------|------|
| 0      | 8  | 16    | 0.05    | 2e-4 |
| 1      | 16 | 32    | 0.05    | 1e-4 |
| 2      | 8  | 16    | 0.10    | 5e-5 |

Key options:

- `--training-target {reference,generated}`: train on human completions or on completions
  distilled from a prompting strategy (`--training-target-strategy`, `--training-target-backend`,
  `--training-target-model`). Generated targets are cached in `--training-targets-cache`.
- `--lora-params-file`: reuse a previous run's `best_lora_params.json` instead of re-searching.
- `--training-ids-file`: restrict training to a fixed list of example ids.
- `--train-seed`: seed for training randomness only (`--seed` controls the data partition).
- `--final-eval-split {finetune_test,final_eval}` and `--run-label`: choose the judged evaluation
  split and the label written to each row's `strategy` field.
- `--load-in-4bit`: QLoRA training.

### Training targets

- `scripts/generate_training_targets.py` pre-generates distilled targets for the `train` split.
- Generated completions are passed through `clean_reasoning_leakage()`
  (`commit_style/finetune/synth_targets.py`), which removes self-reflection fragments
  (e.g. `"Wait, the diff only shows..."`) and falls back to the human completion when a clean
  continuation cannot be extracted. `scripts/clean_training_targets_cache.py` applies the same
  cleanup to existing caches.
- `scripts/prepare_finetune_conditions.py` computes the id intersection across all caches (so all
  conditions train on identical examples) and builds a 50/50 mixture cache of original and
  `structured` completions.

### Multi-condition experiment

Five training conditions (`original`, `vanilla`, `structured`, `reference_guided`, `mixture`) x
3 train seeds, all evaluated on `final_eval`:

```bash
# 1. hp search once
.venv/bin/python scripts/finetune_qwen.py \
  --model-name Qwen/Qwen3-4B --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --training-ids-file outputs/training_ids_intersection.txt \
  --training-target generated --training-target-strategy vanilla \
  --training-targets-cache outputs/training_targets_vanilla.jsonl \
  --run-label hp_search --output-dir outputs/ft/hp_search --out outputs/ft/hp_search/eval.jsonl \
  --judge grazie:google-gemini-3-5-flash --grazie-token-file token.txt

# 2. 5 conditions x 3 seeds, reusing the best config
for seed in 1 2 3; do
  for cond in original vanilla structured reference_guided mixture; do
    case $cond in
      original) TARGET="--training-target reference" ;;
      mixture)  TARGET="--training-target generated --training-targets-cache outputs/training_targets_mixture.jsonl" ;;
      *)        TARGET="--training-target generated --training-target-strategy $cond --training-targets-cache outputs/training_targets_${cond}.jsonl" ;;
    esac
    .venv/bin/python scripts/finetune_qwen.py \
      --model-name Qwen/Qwen3-4B --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
      --training-ids-file outputs/training_ids_intersection.txt $TARGET \
      --lora-params-file outputs/ft/hp_search/best_lora_params.json --train-seed $seed \
      --final-eval-split final_eval --n-final-eval 200 --run-label ${cond}_seed${seed} \
      --output-dir outputs/ft/${cond}/seed${seed} --out outputs/ft/${cond}/seed${seed}/eval.jsonl \
      --judge grazie:google-gemini-3-5-flash --grazie-token-file token.txt
  done
done

# 3. base-model baseline (no fine-tuning), same holdout
.venv/bin/python scripts/eval_base_model.py \
  --model-name Qwen/Qwen3-4B --seed 0 --n-train 1000 --n-strategy-val 200 --n-test 300 \
  --n-final-eval 200 --run-label base --out outputs/ft/base/eval.jsonl \
  --judge grazie:google-gemini-3-5-flash --grazie-token-file token.txt
```

All `eval.jsonl` files share one schema and can be concatenated and passed to
`commit_style.pipeline.summarize()`.

## Statistical significance testing

- `util_scripts/significance_test_ft_eval.py` compares fine-tuning conditions per seed and metric:
  Friedman omnibus test, then pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction,
  reporting mean differences with bootstrap 95% CIs.
- `util_scripts/compare_to_baseline.py` compares each condition to the base model with paired
  Wilcoxon tests and Holm-Bonferroni correction.

```bash
python util_scripts/significance_test_ft_eval.py outputs/ft/*/seed*/eval.jsonl \
  --metrics coherence correctness fluency chrf --judge openai-gpt-5 -o results/significance_tests.json

python util_scripts/compare_to_baseline.py outputs/ft/base/eval.jsonl outputs/ft/*/seed*/eval.jsonl \
  --metrics coherence correctness fluency chrf --output-csv results/baseline_comparison.csv
```

### Results (Qwen3-4B)

All fine-tuning conditions significantly outperform the base model (p < 0.001) on every metric:
chrF +11–14, fluency +1.3–1.6, coherence +1.0–1.6, correctness +0.3–1.0.

Between conditions, `structured` improves correctness over `original` by ~0.5 in all 3 seeds but
lowers coherence by ~0.4–0.6 in 2 of 3 seeds. Fluency and chrF do not differ significantly
(p > 0.19).
