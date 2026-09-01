import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_project_splits
from commit_style.finetune.synth_targets import (
    generate_training_targets,
    load_training_targets_cache,
    write_training_targets_cache,
)
from commit_style.llm.factory import build_llm
from commit_style.strategies import STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and cache strategy-based training-target completions for the "
        "--n-train examples, without running the rest of scripts/finetune_qwen.py's "
        "hp-search/train/eval pipeline. Run this once per strategy, then point "
        "finetune_qwen.py --training-target generated --training-targets-cache at the "
        "resulting file (or reuse --out directly, its default matches finetune_qwen.py's "
        "default cache path)."
    )
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument(
        "--n-strategy-val",
        type=int,
        default=200,
        help="Kept in sync with run_prototype.py --strategy-val / finetune_qwen.py so all "
        "three scripts derive the same repo-disjoint partition; not used directly here.",
    )
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="structured")
    parser.add_argument("--backend", choices=["mock", "grazie", "local_hf"], default="grazie")
    parser.add_argument("--model", default="google-gemini-3-5-flash")
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument("--out", default="outputs/training_targets_cache.jsonl")
    args = parser.parse_args()

    train_examples, _strategy_val, _test = load_project_splits(
        args.data,
        n_train=args.n_train,
        n_strategy_val=args.n_strategy_val,
        n_finetune_test=args.n_test,
        seed=args.seed,
        max_diff_chars=args.max_diff_chars,
        max_history_messages=args.max_history_messages,
    )
    print(f"Loaded {len(train_examples)} train examples")

    cached = load_training_targets_cache(train_examples, args.out)
    if cached is not None:
        print(f"{args.out} already covers all {len(cached)} example ids — nothing to generate.")
        return

    print(
        f"Generating {len(train_examples)} training targets via strategy={args.strategy} "
        f"backend={args.backend} model={args.model}..."
    )
    llm = build_llm(args.backend, args.model, args.grazie_token_file)
    generated = generate_training_targets(train_examples, args.strategy, llm)
    write_training_targets_cache(generated, args.out)
    print(f"Wrote {len(generated)} generated training targets to {args.out}")


if __name__ == "__main__":
    main()
