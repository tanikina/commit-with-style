import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_project_splits
from commit_style.finetune.evaluate import run_finetuned_eval
from commit_style.finetune.hp_search import run_hp_search, pick_best
from commit_style.finetune.synth_targets import (
    generate_training_targets,
    load_training_targets_cache,
    write_training_targets_cache,
)
from commit_style.finetune.train import load_lora_params, train_one_config, write_lora_params
from commit_style.llm.factory import build_judges, build_llm
from commit_style.llm.local_hf import LocalHFClient
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
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true", help="Untested this session; for a future larger GPU.")
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
    parser.add_argument("--training-target-backend", choices=["mock", "grazie", "local_hf"], default="grazie")
    parser.add_argument("--training-target-model", default="google-gemini-3-5-flash")
    parser.add_argument(
        "--training-targets-cache",
        default="outputs/training_targets_cache.jsonl",
        help="JSONL cache of generated training targets keyed by example id, to avoid "
        "re-paying for live API calls on reruns. Reused only if it covers every id in the "
        "current --n-train set; regenerated (and overwritten) otherwise.",
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
        "--judge", action="append", metavar="BACKEND:MODEL",
        help="Repeatable, e.g. --judge grazie:openai-gpt-5 --judge grazie:anthropic-claude-sonnet-5. "
        "If omitted, judges with mock.",
    )
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument("--with-bertscore", action="store_true")
    parser.add_argument("--out", default="outputs/finetune_eval.jsonl")
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
                args.training_target_backend, args.training_target_model, args.grazie_token_file
            )
            train_examples = generate_training_targets(
                train_examples, args.training_target_strategy, target_llm
            )
            write_training_targets_cache(train_examples, args.training_targets_cache)
            print(f"Wrote generated training targets to {args.training_targets_cache}")

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

    judge_llms = build_judges(
        args.judge, args.grazie_token_file, fallback_backend="mock", fallback_model=""
    )
    gen_llm = LocalHFClient(model_name=args.model_name, adapter_path=best.adapter_path)

    rows = run_finetuned_eval(
        test_examples,
        gen_llm,
        judge_llms,
        gen_model_name=args.model_name,
        with_bertscore=args.with_bertscore,
    )
    write_jsonl(rows, args.out)
    print(f"\nWrote {len(rows)} rows to {args.out}\n")
    print(summarize(rows))


if __name__ == "__main__":
    main()
