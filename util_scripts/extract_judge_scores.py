#!/usr/bin/env python3
"""Extract judge scores from an evaluation log into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


JUDGE_HEADING_RE = re.compile(r"^---\s*judge:\s*.+?\s*---$")
SEED_RE = re.compile(r"^(?P<strategy>.+)_seed(?P<seed>\d+)$")
EXPECTED_HEADER = ["strategy", "n", "coherence", "fluency", "correctness"]


def parse_scores(text: str) -> list[dict[str, str | int | float]]:
    """Parse rows belonging to judge score tables and ignore other tables."""
    lines = text.splitlines()
    rows: list[dict[str, str | int | float]] = []

    for index, line in enumerate(lines):
        if line.split() != EXPECTED_HEADER:
            continue

        # A matching header must belong to a judge section, not another table.
        previous_nonempty = next(
            (candidate.strip() for candidate in reversed(lines[:index]) if candidate.strip()),
            "",
        )
        if not JUDGE_HEADING_RE.match(previous_nonempty):
            continue

        for data_line in lines[index + 1 :]:
            if not data_line.strip() or data_line.lstrip().startswith("---"):
                break

            fields = data_line.split()
            if len(fields) != 5:
                raise ValueError(f"Malformed judge row: {data_line!r}")

            strategy_and_seed, n, coherence, fluency, correctness = fields
            match = SEED_RE.fullmatch(strategy_and_seed)
            if match:
                strategy = match.group("strategy")
                seed = int(match.group("seed"))
            else:
                strategy = strategy_and_seed
                seed = ""

            rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "n": int(n),
                    "coherence": float(coherence),
                    "fluency": float(fluency),
                    "correctness": float(correctness),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract coherence, fluency, and correctness scores from a text log."
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt log")
    parser.add_argument("output", type=Path, help="Path to the output .csv file")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(
            f"Input file not found: {input_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    rows = parse_scores(args.input.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("No judge score rows found in the input file.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "strategy",
                "seed",
                "n",
                "coherence",
                "fluency",
                "correctness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
