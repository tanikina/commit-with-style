import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_examples, load_project_splits
from commit_style.llm.factory import build_judges, build_llm
from commit_style.pipeline import run_pipeline, summarize, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit-message completion prototype pipeline")
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strategy-val",
        action="store_true",
        help="Use the fixed, repo-disjoint 200-sample strategy-selection set from "
        "load_project_splits instead of --n ad-hoc sampling (also disjoint by repo from the "
        "fine-tuning train/test sets in scripts/finetune_qwen.py, given the same --seed).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["vanilla", "structured", "reference_guided"],
    )
    parser.add_argument(
        "--backend", choices=["mock", "grazie", "local_hf", "vllm"], default="vllm",
        help="Defaults to vllm for fast local generation; pass --backend mock for a "
        "zero-dependency, no-GPU dry run.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--judge", action="append", metavar="BACKEND:MODEL",
        help="Repeatable, e.g. --judge grazie:openai-gpt-5 --judge grazie:anthropic-claude-sonnet-5. "
        "If omitted, judges with the generation backend/model.",
    )
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument("--out", default="outputs/dry_run.jsonl")
    parser.add_argument("--with-bertscore", action="store_true")
    args = parser.parse_args()

    if args.strategy_val:
        _train, examples, _test = load_project_splits(args.data, seed=args.seed)
    else:
        examples = load_examples(args.data, n=args.n, seed=args.seed)
    print(f"Loaded {len(examples)} examples "
          f"({', '.join(sorted({e.language for e in examples}))})")

    gen_llm = build_llm(args.backend, args.model, args.grazie_token_file)
    judge_llms = build_judges(
        args.judge, args.grazie_token_file, fallback_backend=args.backend, fallback_model=args.model
    )

    rows = run_pipeline(
        examples,
        args.strategies,
        gen_llm,
        judge_llms,
        gen_backend_name=args.backend,
        gen_model_name=args.model,
        with_bertscore=args.with_bertscore,
    )
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}\n")
    print(summarize(rows))


if __name__ == "__main__":
    main()
