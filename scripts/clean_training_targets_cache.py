import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.finetune.synth_targets import clean_reasoning_leakage


def load_original_references(data_path: str) -> dict[str, str]:
    refs = {}
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            refs[row["commit_sha"]] = row["completion"]
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and heuristically clean reasoning-leakage rows in an "
        "already-generated training-targets cache (see generate_training_targets.py), "
        "without any further LLM calls. Unsalvageable rows fall back to the example's "
        "original human reference_completion, recovered from --data by id, so cache "
        "id-coverage (and row count) never changes."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument(
        "--out", default=None,
        help="Defaults to <cache> with a .cleaned.jsonl suffix (non-destructive). Pass "
        "--in-place to overwrite --cache directly instead.",
    )
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    out_path = args.cache if args.in_place else (args.out or str(Path(args.cache).with_suffix(".cleaned.jsonl")))
    refs = load_original_references(args.data)

    n_total = n_leaked = n_extracted = n_fallback = 0
    rows = []
    with open(args.cache) as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            text = row["reference_completion"]
            cleaned = clean_reasoning_leakage(text)
            if cleaned is None:
                n_leaked += 1
                n_fallback += 1
                cleaned = refs.get(row["id"], text)
            elif cleaned != text:
                n_leaked += 1
                n_extracted += 1
            rows.append({"id": row["id"], "reference_completion": cleaned})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(
        f"{args.cache}: {n_total} rows, {n_leaked} contaminated "
        f"({n_extracted} cleanly extracted, {n_fallback} fell back to human reference)"
    )
    print(f"Wrote cleaned cache to {out_path}")


if __name__ == "__main__":
    main()
