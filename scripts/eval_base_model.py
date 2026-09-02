import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_final_eval_split
from commit_style.finetune.evaluate import run_finetuned_eval
from commit_style.llm.factory import build_judges, build_llm
from commit_style.pipeline import summarize, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an out-of-the-box (non-fine-tuned) model on the same "
        "final_eval holdout used by the multi-condition fine-tuning campaign."
    )
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--backend", choices=["mock", "grazie", "local_hf", "vllm"], default="vllm",
        help="Defaults to vllm for fast local generation; pass --backend mock for a "
        "zero-dependency, no-GPU dry run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-strategy-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-final-eval", type=int, default=200)
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)
    parser.add_argument(
        "--judge",
        action="append",
        metavar="BACKEND:MODEL",
        help="Repeatable, e.g. --judge grazie:openai-gpt-5. If omitted, judges with mock.",
    )
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument(
        "--quantization",
        default=None,
        help="Passed through to the vllm backend, e.g. 'bitsandbytes' for on-the-fly 4-bit "
        "loading — use this when the model's full-precision weights don't fit the GPU (see "
        "README). Ignored by other backends.",
    )
    parser.add_argument("--with-bertscore", action="store_true")
    parser.add_argument("--out", default="outputs/base_model_eval.jsonl")
    parser.add_argument(
        "--run-label",
        default="base_qwen3_4b",
        help="Value written to each output row's 'strategy' field, so this run stays "
        "distinguishable from the 15 fine-tuned conditions when eval.jsonl files are "
        "concatenated and summarized together.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print each candidate's continuation as it's generated.",
    )
    args = parser.parse_args()

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
    print(f"Loaded {len(final_eval_examples)} final_eval examples")

    judge_llms = build_judges(args.judge, args.grazie_token_file, fallback_backend="mock", fallback_model="")
    gen_llm = build_llm(args.backend, args.model_name, args.grazie_token_file, quantization=args.quantization)

    rows = run_finetuned_eval(
        final_eval_examples,
        gen_llm,
        judge_llms,
        gen_model_name=args.model_name,
        gen_backend_name=args.backend,
        with_bertscore=args.with_bertscore,
        run_label=args.run_label,
        debug=args.debug,
    )
    write_jsonl(rows, args.out)
    print(f"\nWrote {len(rows)} rows to {args.out}\n")
    print(summarize(rows))


if __name__ == "__main__":
    main()
