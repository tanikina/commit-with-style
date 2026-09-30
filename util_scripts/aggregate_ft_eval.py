#!/usr/bin/env python3
"""Aggregate fine-tuning eval.jsonl files directly into a comparison CSV.

Unlike extract_judge_scores.py (which scrapes a printed judge-score table out
of a text log), this reads the raw per-example JSONL eval outputs, pulls in
chrF/METEOR and usage (latency, token counts) alongside judge scores, and
restricts every average to the intersection of example ids scored across all
included files so runs with different drop-out counts stay comparable.

Pass --base-model-eval to fold in one or more out-of-the-box (non-fine-tuned) baseline eval
files for the same comparison, e.g. --base-model-eval outputs/qwen3_4b_eval_reannotated.jsonl.
These are labelled base_<model> (derived from the filename) rather than <strategy>_seed<seed>,
and join the same intersection-restricted aggregation as every other input.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SEED_RE = re.compile(r"^(?P<rest>.*)_seed(?P<seed>\d+)$")
CSV_FIELDS = [
    "strategy",
    "seed",
    "judge",
    "n",
    "n_raw",
    "coherence",
    "fluency",
    "correctness",
    "chrf",
    "meteor",
    "input_tokens",
    "output_tokens",
    "latency_s",
]


def label_for(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "eval":
        parts = parts[:-1]
    return "_".join(parts)


def base_model_label(path: Path) -> str:
    name = path.stem
    for suffix in ("_reannotated", "_eval"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return f"base_{name}"


def split_seed(label: str) -> tuple[str, str]:
    match = SEED_RE.fullmatch(label)
    if match:
        return match.group("rest"), match.group("seed")
    return label, ""


def load_rows(path: Path) -> list[dict] | None:
    with path.open() as f:
        print("Loading:", path)
        rows = [json.loads(line) for line in f if line.strip()]
    judge_ids = {jid for row in rows for jid in row.get("judge_scores", {})}
    if judge_ids == {"mock"}:
        print(f"[aggregate_ft_eval] skipping {path} (mock-judge only)", file=sys.stderr)
        return None
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(labelled_rows: dict[str, list[dict]]) -> list[dict]:
    id_sets = {label: {row["example_id"] for row in rows} for label, rows in labelled_rows.items()}
    overlap = set.intersection(*id_sets.values()) if len(id_sets) > 1 else None

    out_rows = []
    for label, rows in labelled_rows.items():
        n_raw = len(rows)
        if overlap is not None:
            rows = [row for row in rows if row["example_id"] in overlap]
        n = len(rows)
        print(f"[aggregate_ft_eval] {label}: n_raw={n_raw} -> n={n}", file=sys.stderr)

        strategy, seed = split_seed(label)
        chrf = mean([row["metrics"]["chrf"] for row in rows if "chrf" in row.get("metrics", {})])
        meteor = mean([row["metrics"]["meteor"] for row in rows if "meteor" in row.get("metrics", {})])
        in_tok = mean([row["usage"]["input_tokens"] for row in rows if row.get("usage", {}).get("input_tokens") is not None])
        out_tok = mean([row["usage"]["output_tokens"] for row in rows if row.get("usage", {}).get("output_tokens") is not None])
        latency = mean([row["usage"]["latency_s"] for row in rows if row.get("usage", {}).get("latency_s") is not None])

        judge_ids: list[str] = []
        for row in rows:
            for jid in row.get("judge_scores", {}):
                if jid not in judge_ids:
                    judge_ids.append(jid)

        for jid in judge_ids:
            scores = [row["judge_scores"][jid] for row in rows if jid in row.get("judge_scores", {})]
            out_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "judge": jid,
                    "n": n,
                    "n_raw": n_raw,
                    "coherence": mean([s["coherence"] for s in scores]),
                    "fluency": mean([s["fluency"] for s in scores]),
                    "correctness": mean([s["correctness"] for s in scores]),
                    "chrf": chrf,
                    "meteor": meteor,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "latency_s": latency,
                }
            )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate fine-tuning eval.jsonl files into a comparison CSV."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Paths to eval.jsonl files.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output CSV.")
    parser.add_argument(
        "--root",
        default="outputs/ft",
        type=Path,
        help="Prefix stripped from each input path when deriving its label (default: outputs/ft).",
    )
    parser.add_argument(
        "--base-model-eval",
        nargs="+",
        type=Path,
        default=[],
        metavar="PATH",
        help="One or more out-of-the-box baseline eval.jsonl files to include alongside the "
        "fine-tuned inputs, e.g. outputs/qwen3_4b_eval_reannotated.jsonl. Labelled base_<model> "
        "instead of <strategy>_seed<seed>, and joins the same intersection-restricted aggregate.",
    )
    args = parser.parse_args()

    labelled_rows: dict[str, list[dict]] = defaultdict(list)
    for path in args.inputs:
        rows = load_rows(path)
        if rows is None:
            continue
        labelled_rows[label_for(path, args.root)].extend(rows)

    for path in args.base_model_eval:
        rows = load_rows(path)
        if rows is None:
            continue
        labelled_rows[base_model_label(path)].extend(rows)

    if not labelled_rows:
        raise SystemExit("No usable rows found in the given input files.")

    out_rows = aggregate(labelled_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
