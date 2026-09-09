#!/usr/bin/env python3
"""Refresh coherence/correctness judge scores on already-generated fine-tuning eval sets.

`outputs/ft/{strategy}/seed{seed}/eval.jsonl` files (plus, via `--files`, standalone eval sets on
the same final_eval split, e.g. `outputs/qwen3_4b_eval.jsonl`) carry `judge_scores` produced with
an older judge prompt/schema (single blended `rationale`, pre coherence/correctness rewrites). This
script re-judges only coherence and correctness for each row's already-generated candidate (no
regeneration), leaving each row's existing fluency score untouched, and writes the result to a
sibling `<stem>_reannotated.jsonl` (never overwrites the original file).

Since these eval files don't store diff/history, the same final_eval split used to produce them is
reloaded via `load_final_eval_split` (every file targeted by this script — the 15 ft grid files and
`outputs/qwen3_4b_eval.jsonl`/`outputs/ft/base_qwen3_4b/eval.jsonl` alike — shares the same
200-example split: `--seed 0 --n-final-eval 200` are constant across all of them, confirmed against
`scripts/finetune_qwen.py` and `scripts/eval_base_model.py`) and indexed by example id.

Resumable: re-running after an interruption skips rows that already have a `coherence_rationale`
for the target judge id (loaded from a prior `eval_reannotated.jsonl` if present), so no credits
are re-spent on already-reannotated rows.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_final_eval_split
from commit_style.judge import judge_candidate
from commit_style.llm.factory import build_llm, judge_id
from commit_style.pipeline import summarize
from commit_style.strategies import Candidate

STRATEGIES = ["mixture", "original", "reference_guided", "structured", "vanilla"]


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_row(row: dict, f) -> None:
    f.write(json.dumps(row) + "\n")
    f.flush()


def output_path(eval_path: Path) -> Path:
    return eval_path.with_name(f"{eval_path.stem}_reannotated{eval_path.suffix}")


def reannotate_file(
    eval_path: Path,
    llm,
    jid: str,
    id_to_example: dict,
    max_rows: int | None,
    debug: bool,
) -> list[dict]:
    reannotated_path = output_path(eval_path)
    rows = load_rows(reannotated_path) if reannotated_path.exists() else load_rows(eval_path)

    tmp_path = reannotated_path.with_suffix(reannotated_path.suffix + ".tmp")
    n_judged = 0
    with tmp_path.open("w") as f:
        for i, row in enumerate(rows):
            existing = row["judge_scores"].get(jid, {})
            if existing.get("coherence_rationale") or (max_rows is not None and n_judged >= max_rows):
                write_row(row, f)
                continue

            example = id_to_example.get(row["example_id"])
            if example is None:
                print(f"[reannotate_ft_judge_scores] {eval_path}: no example for id={row['example_id']!r}, skipping row")
                write_row(row, f)
                continue

            candidate = Candidate(
                strategy=row["strategy"],
                continuation=row["candidate_continuation"],
                full_message=row["candidate_full_message"],
            )
            try:
                scores = judge_candidate(llm, example, candidate)
            except Exception as e:
                print(f"[reannotate_ft_judge_scores] {eval_path}: ({i + 1}/{len(rows)}) judge error, skipping: {e}")
                write_row(row, f)
                continue

            n_judged += 1
            print(f"[reannotate_ft_judge_scores] {eval_path}: ({i + 1}/{len(rows)}) reannotated example={row['example_id']}")
            if debug:
                print(f"[debug] example={row['example_id']} scores={scores.model_dump()}")

            merged = dict(existing)
            merged["coherence"] = scores.coherence
            merged["coherence_rationale"] = scores.coherence_rationale
            merged["correctness"] = scores.correctness
            merged["correctness_rationale"] = scores.correctness_rationale
            if "fluency" not in merged:
                merged["fluency"] = scores.fluency
                merged["fluency_rationale"] = scores.fluency_rationale
            row["judge_scores"][jid] = merged
            write_row(row, f)

    tmp_path.replace(reannotated_path)
    print(f"[reannotate_ft_judge_scores] wrote {len(rows)} rows to {reannotated_path}\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-strategy-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-final-eval", type=int, default=200)
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)

    parser.add_argument("--ft-dir", default="outputs/ft")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Defaults to the full 5-strategy grid, unless --files is passed and --strategies is "
        "not, in which case the grid is skipped and only --files is reannotated. Pass explicitly "
        "(with or without values) to override either way.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--files",
        nargs="+",
        default=[],
        help="Extra standalone eval files on the same final_eval split to reannotate, e.g. "
        "outputs/qwen3_4b_eval.jsonl or outputs/ft/base_qwen3_4b/eval.jsonl. If --strategies is "
        "not also passed, only these files are reannotated (the strategy/seed grid is skipped).",
    )

    parser.add_argument("--backend", choices=["mock", "grazie", "local_hf", "vllm"], default="grazie")
    parser.add_argument("--model-name", default="openai-gpt-5")
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap how many rows of the FIRST file are reannotated, for a calibration run.",
    )
    args = parser.parse_args()
    if args.strategies is None:
        args.strategies = [] if args.files else STRATEGIES

    print(f"[reannotate_ft_judge_scores] loading final_eval split from {args.data}...")
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
    id_to_example = {e.id: e for e in final_eval_examples}
    print(f"[reannotate_ft_judge_scores] loaded {len(id_to_example)} examples")

    llm = build_llm(args.backend, args.model_name, args.grazie_token_file)
    jid = judge_id(args.backend, args.model_name)

    eval_paths = [
        Path(args.ft_dir) / strategy / f"seed{seed}" / "eval.jsonl"
        for strategy in args.strategies
        for seed in args.seeds
    ]
    eval_paths += [Path(f) for f in args.files]

    all_rows = []
    for i, eval_path in enumerate(eval_paths):
        if not eval_path.exists():
            print(f"[reannotate_ft_judge_scores] missing {eval_path}, skipping")
            continue
        max_rows = args.max_rows if i == 0 else None
        rows = reannotate_file(eval_path, llm, jid, id_to_example, max_rows, args.debug)
        all_rows.extend(rows)

    if all_rows:
        print(summarize(all_rows))


if __name__ == "__main__":
    main()
