import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_final_eval_split, load_project_splits
from commit_style.finetune.evaluate import run_finetuned_eval
from commit_style.finetune.hp_search import run_hp_search, pick_best
from commit_style.finetune.synth_targets import (
    generate_training_targets,
    load_training_targets_cache,
    write_training_targets_cache,
)
from commit_style.finetune.train import load_lora_params, train_one_config, write_lora_params
from commit_style.llm.factory import build_judges, build_llm
from commit_style.pipeline import summarize, write_jsonl
from commit_style.strategies import STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a small Qwen model + hp-search + eval")
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument(
        "--n-strategy-val",
        type=int,
        default=200,
        help="Kept in sync with run_prototype.py --strategy-val so both scripts derive the "
        "same repo-disjoint partition; not used directly by this script.",
    )
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training-ids-file",
        default=None,
        help="Optional path to a plain-text file, one example id per line. Restricts "
        "train_examples to this fixed set (e.g. an intersection of ids covered by several "
        "--training-target generated caches) before any cache lookup, so multiple conditions "
        "can be trained on the exact same fixed set of examples.",
    )
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true", help="Untested this session; for a future larger GPU.")
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Seed for training randomness only (weight init, dataloader shuffling) via "
        "transformers.set_seed + TrainingArguments(seed=, data_seed=). Fully decoupled from "
        "--seed, which only controls the data partition.",
    )
    parser.add_argument(
        "--training-target",
        choices=["reference", "generated"],
        default="reference",
        help="'reference' (default) trains on the dataset's human reference_completion, "
        "unchanged. 'generated' replaces it with a strategy-generated completion from "
        "--training-target-backend/-model for every training example (a form of distillation "
        "into the fine-tuned model) — see --training-target-strategy/-cache.",
    )
    parser.add_argument(
        "--training-target-strategy",
        choices=list(STRATEGIES.keys()),
        default="structured",
        help="Strategy used to generate training targets when --training-target generated.",
    )
    parser.add_argument(
        "--training-target-backend", choices=["mock", "grazie", "local_hf", "vllm"], default="vllm"
    )
    parser.add_argument(
        "--training-target-model", default="",
        help="Defaults to the tiny local stand-in model for vllm/local_hf backends. Pass "
        "e.g. --training-target-model google-gemini-3-5-flash with --training-target-backend "
        "grazie, or a real hub id like Qwen/Qwen3.5-9B with --training-target-backend vllm.",
    )
    parser.add_argument(
        "--training-targets-cache",
        default="outputs/training_targets_cache.jsonl",
        help="JSONL cache of generated training targets keyed by example id, to avoid "
        "re-paying for live API calls on reruns. Reused only if it covers every id in the "
        "current --n-train set; regenerated (and overwritten) otherwise.",
    )
    parser.add_argument(
        "--training-target-quantization",
        default=None,
        help="Passed through to --training-target-backend vllm, e.g. 'bitsandbytes' for "
        "on-the-fly 4-bit loading — use this when the model's full-precision weights don't "
        "fit the GPU (see README). Ignored by other backends.",
    )
    parser.add_argument(
        "--lora-params-file",
        default=None,
        help="Skip the 3-config hp-search grid and train once with LoRA params loaded from "
        "this JSON file instead. Every run (searched or reused) writes its winning params to "
        "<output-dir>/best_lora_params.json — pass that path here on a later run (e.g. with a "
        "different --training-target-strategy) to reuse the same LoRA config instead of "
        "re-searching, since the hyperparameters mainly depend on the model/data shape, not "
        "on which strategy generated the training targets.",
    )
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument(
        "--gen-backend", choices=["mock", "grazie", "local_hf", "vllm"], default="vllm",
        help="Backend for the final judged-eval pass only (training always uses "
        "transformers+peft, unaffected by this flag). Defaults to vllm for fast generation.",
    )
    parser.add_argument(
        "--gen-quantization",
        default=None,
        help="Passed through to --gen-backend vllm, e.g. 'bitsandbytes' for on-the-fly 4-bit "
        "loading — use this when the model's full-precision weights don't fit the GPU (see "
        "README). Ignored by other backends.",
    )
    parser.add_argument(
        "--judge", action="append", metavar="BACKEND:MODEL",
        help="Repeatable, e.g. --judge grazie:openai-gpt-5 --judge grazie:anthropic-claude-sonnet-5. "
        "If omitted, judges with mock.",
    )
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument("--with-bertscore", action="store_true")
    parser.add_argument("--out", default="outputs/finetune_eval.jsonl")
    parser.add_argument(
        "--final-eval-split",
        choices=["finetune_test", "final_eval"],
        default="finetune_test",
        help="'finetune_test' (default) judges the final model on the same --n-test split "
        "that also feeds Trainer's eval_dataset/hp-search loss. 'final_eval' instead uses a "
        "brand-new, independent load_final_eval_split holdout (see --n-final-eval) for the "
        "judged comparison, while --n-test still feeds training-time eval loss unchanged.",
    )
    parser.add_argument("--n-final-eval", type=int, default=200)
    parser.add_argument(
        "--run-label",
        default="finetuned",
        help="Value written to each output row's 'strategy' field, so multiple runs' "
        "eval.jsonl files can be concatenated and distinguished by commit_style.pipeline."
        "summarize() (e.g. 'vanilla_seed1').",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print each candidate's continuation as it's generated (training-target "
        "generation and the final judged-eval pass).",
    )
    args = parser.parse_args()

    train_examples, _strategy_val_examples, test_examples = load_project_splits(
        args.data,
        n_train=args.n_train,
        n_strategy_val=args.n_strategy_val,
        n_finetune_test=args.n_test,
        seed=args.seed,
        max_diff_chars=args.max_diff_chars,
        max_history_messages=args.max_history_messages,
    )
    print(
        f"Loaded {len(train_examples)} train / {len(test_examples)} test examples "
        f"(repo-disjoint from each other and from the {args.n_strategy_val}-sample "
        f"strategy_val set; {', '.join(sorted({e.language for e in train_examples}))})"
    )

    if args.training_ids_file:
        keep_ids = {line.strip() for line in open(args.training_ids_file) if line.strip()}
        train_examples = [e for e in train_examples if e.id in keep_ids]
        print(f"Filtered train_examples to {len(train_examples)} via --training-ids-file")

    if args.training_target == "generated":
        cached = load_training_targets_cache(train_examples, args.training_targets_cache)
        if cached is not None:
            print(f"Loaded {len(cached)} generated training targets from {args.training_targets_cache}")
            train_examples = cached
        else:
            print(
                f"Generating {len(train_examples)} training targets via strategy="
                f"{args.training_target_strategy} backend={args.training_target_backend} "
                f"model={args.training_target_model} (no usable cache found)..."
            )
            target_llm = build_llm(
                args.training_target_backend,
                args.training_target_model,
                args.grazie_token_file,
                quantization=args.training_target_quantization,
            )
            train_examples = generate_training_targets(
                train_examples, args.training_target_strategy, target_llm, debug=args.debug
            )
            write_training_targets_cache(train_examples, args.training_targets_cache)
            print(f"Wrote generated training targets to {args.training_targets_cache}")
            del target_llm
            gc.collect()

    if args.lora_params_file:
        lora_params = load_lora_params(args.lora_params_file)
        print(f"\nSkipping hp-search; reusing LoRA params from {args.lora_params_file}: {lora_params}")
        reused_dir = str(Path(args.output_dir) / "reused_config")
        best = train_one_config(
            args.model_name,
            lora_params,
            train_examples,
            test_examples,
            reused_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            seed=args.train_seed,
        )
        results = [best]
        print(f"eval_loss={best.eval_loss:.4f}")
    else:
        results = run_hp_search(
            args.model_name,
            train_examples,
            test_examples,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            seed=args.train_seed,
        )
        best = pick_best(results)

        print("\nHP search results:")
        print(f"{'config':<10}{'r':>4}{'alpha':>7}{'dropout':>9}{'lr':>10}{'eval_loss':>12}")
        for i, result in enumerate(results):
            marker = " *" if result is best else ""
            p = result.lora_params
            print(
                f"config_{i:<3}{p.r:>4}{p.alpha:>7}{p.dropout:>9}{p.lr:>10.1e}{result.eval_loss:>12.4f}{marker}"
            )

    write_lora_params(best.lora_params, str(Path(args.output_dir) / "best_lora_params.json"))

    with open(Path(args.output_dir) / "hp_search_results.json", "w") as f:
        json.dump(
            [
                {
                    "adapter_path": r.adapter_path,
                    "eval_loss": r.eval_loss,
                    "lora_params": vars(r.lora_params),
                }
                for r in results
            ],
            f,
            indent=2,
        )

    gc.collect()
    torch.cuda.empty_cache()

    if args.final_eval_split == "final_eval":
        final_eval_examples = load_final_eval_split(
            args.data,
            n_final_eval=args.n_final_eval,
            seed=args.seed,
            n_train=args.n_train,
            n_strategy_val=args.n_strategy_val,
            n_finetune_test=args.n_test,
            max_diff_chars=args.max_diff_chars,
            max_history_messages=args.max_history_messages,
        )
        print(f"Loaded {len(final_eval_examples)} final_eval examples for the judged comparison pass")
    else:
        final_eval_examples = test_examples

    judge_llms = build_judges(
        args.judge, args.grazie_token_file, fallback_backend="mock", fallback_model=""
    )
    gen_llm = build_llm(
        args.gen_backend,
        args.model_name,
        args.grazie_token_file,
        adapter_path=best.adapter_path,
        quantization=args.gen_quantization,
    )

    rows = run_finetuned_eval(
        final_eval_examples,
        gen_llm,
        judge_llms,
        gen_model_name=args.model_name,
        gen_backend_name=args.gen_backend,
        with_bertscore=args.with_bertscore,
        run_label=args.run_label,
        debug=args.debug,
    )
    write_jsonl(rows, args.out)
    print(f"\nWrote {len(rows)} rows to {args.out}\n")
    print(summarize(rows))


if __name__ == "__main__":
    main()
