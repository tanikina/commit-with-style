#!/usr/bin/env python3
"""Refresh coherence/correctness judge scores on the multi-judge strategy-comparison eval sets.

`outputs/multijudge_strategies/*.jsonl` (produced by `scripts/run_prototype.py --strategy-val
--judge grazie:openai-gpt-5 --judge grazie:anthropic-claude-sonnet-5`) carry `judge_scores` for
two judges, both with the old schema (single blended `rationale`, pre coherence/correctness
rewrites). This re-judges only coherence and correctness for each row's already-generated
candidate (no regeneration) with each configured judge, leaving each row's existing fluency score
untouched, and writes the result to a sibling `<stem>_reannotated.jsonl` (never overwrites the
original file).

Rows don't store diff/history, so the same repo-disjoint strategy_val split used to produce them
is reloaded via `load_project_splits(...)[1]` (the `--strategy-val` flag in run_prototype.py) and
indexed by example id.

Resumable per (row, judge id): re-running after an interruption skips any row/judge pair that
already has a `coherence_rationale`, so no credits are re-spent on already-reannotated pairs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_project_splits
from commit_style.judge import judge_candidate
from commit_style.llm.factory import build_judges
from commit_style.pipeline import summarize
from commit_style.strategies import Candidate

DEFAULT_FILES = [
    "outputs/multijudge_strategies/strategy_gemini_multi_judge.jsonl",
    #"outputs/multijudge_strategies/strategy_qwen3_14_multi_judge.jsonl",
    "outputs/multijudge_strategies/strategy_qwen35_9b_multi_judge.jsonl",
]
DEFAULT_JUDGES = ["grazie:openai-gpt-5", "grazie:anthropic-claude-sonnet-5"]


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(rows: list[dict], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(path)


def write_row(row: dict, f) -> None:
    """Write a single row to an open file handle."""
    f.write(json.dumps(row) + "\n")
    f.flush()  # Ensure data is written immediately


def output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_reannotated{input_path.suffix}")


def reannotate_file(
    input_path: Path,
    judges: dict,
    id_to_example: dict,
    max_calls: int | None,
    debug: bool,
) -> tuple[list[dict], int]:
    out_path = output_path(input_path)
    rows = load_rows(out_path) if out_path.exists() else load_rows(input_path)

    # Use tmp file for incremental writes
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    n_calls = 0
    with tmp_path.open("w") as f:
        for i, row in enumerate(rows):
            row_modified = False
            example = id_to_example.get(row["example_id"])

            for jid, llm in judges.items():
                existing = row["judge_scores"].get(jid, {})
                if existing.get("coherence_rationale"):
                    continue
                if max_calls is not None and n_calls >= max_calls:
                    # Write current row if modified, then finalize
                    if row_modified:
                        write_row(row, f)
                    # Write remaining unmodified rows
                    for remaining_row in rows[i + (0 if row_modified else 1):]:
                        write_row(remaining_row, f)
                    tmp_path.replace(out_path)
                    print(f"[reannotate_multijudge_strategy_scores] wrote {len(rows)} rows to {out_path} (max-calls reached)\n")
                    return rows, n_calls
                if example is None:
                    print(f"[reannotate_multijudge_strategy_scores] {input_path}: no example for id={row['example_id']!r}, skipping row")
                    break

                candidate = Candidate(
                    strategy=row["strategy"],
                    continuation=row["candidate_continuation"],
                    full_message=row["candidate_full_message"],
                )
                try:
                    scores = judge_candidate(llm, example, candidate)
                except Exception as e:
                    print(f"[reannotate_multijudge_strategy_scores] {input_path}: ({i + 1}/{len(rows)}) judge={jid} error, skipping: {e}")
                    continue

                n_calls += 1
                print(f"[reannotate_multijudge_strategy_scores] {input_path}: ({i + 1}/{len(rows)}) judge={jid} reannotated example={row['example_id']}")
                if debug:
                    print(f"[debug] example={row['example_id']} judge={jid} scores={scores.model_dump()}")

                merged = dict(existing)
                merged["coherence"] = scores.coherence
                merged["coherence_rationale"] = scores.coherence_rationale
                merged["correctness"] = scores.correctness
                merged["correctness_rationale"] = scores.correctness_rationale
                if "fluency" not in merged:
                    merged["fluency"] = scores.fluency
                    merged["fluency_rationale"] = scores.fluency_rationale
                row["judge_scores"][jid] = merged
                row_modified = True

            # Write row after processing all judges for this row
            write_row(row, f)

    # Atomically replace output file with tmp file
    tmp_path.replace(out_path)
    print(f"[reannotate_multijudge_strategy_scores] wrote {len(rows)} rows to {out_path}\n")
    return rows, n_calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-strategy-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)

    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument(
        "--judges", nargs="+", default=DEFAULT_JUDGES, metavar="BACKEND:MODEL",
        help="Repeatable, e.g. grazie:openai-gpt-5 grazie:anthropic-claude-sonnet-5. Each entry's "
        "backend:model determines both which LLM client is built and the judge_scores key that "
        "gets reannotated. Pass e.g. mock:openai-gpt-5 mock:anthropic-claude-sonnet-5 for a dry "
        "run without real credentials.",
    )
    parser.add_argument("--grazie-token-file", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Cap how many individual judge calls are made in the FIRST file, for a calibration run.",
    )
    args = parser.parse_args()

    print(f"[reannotate_multijudge_strategy_scores] loading strategy_val split from {args.data}...")
    _train, strategy_val_examples, _test = load_project_splits(
        args.data,
        n_train=args.n_train,
        n_strategy_val=args.n_strategy_val,
        n_finetune_test=args.n_test,
        seed=args.seed,
        max_diff_chars=args.max_diff_chars,
        max_history_messages=args.max_history_messages,
    )
    id_to_example = {e.id: e for e in strategy_val_examples}
    print(f"[reannotate_multijudge_strategy_scores] loaded {len(id_to_example)} examples")

    judges = build_judges(args.judges, args.grazie_token_file, "grazie", "openai-gpt-5")

    all_rows = []
    remaining_calls = args.max_calls
    for file_idx, file_str in enumerate(args.files):
        input_path = Path(file_str)
        if not input_path.exists():
            print(f"[reannotate_multijudge_strategy_scores] missing {input_path}, skipping")
            continue
        max_calls = remaining_calls if file_idx == 0 else None
        rows, _n_calls = reannotate_file(input_path, judges, id_to_example, max_calls, args.debug)
        all_rows.extend(rows)

    if all_rows:
        print(summarize(all_rows))


if __name__ == "__main__":
    main()
