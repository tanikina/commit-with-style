import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.data import load_project_splits
from commit_style.finetune.synth_targets import write_training_targets_cache


def _load_cache_ids(path: str) -> dict[str, str]:
    cached = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            cached[row["id"]] = row["reference_completion"]
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the fixed training-id intersection across the vanilla/structured/"
        "reference_guided distillation caches and the --n-train pool, and build the 50/50 "
        "fixed-halves original+structured mixture cache, for the multi-condition fine-tuning "
        "campaign. Run this once, after all 3 --strategy caches exist, then point every "
        "condition's --training-ids-file at --ids-out."
    )
    parser.add_argument("--data", default="data/evaluation_data_8000-samples_2025-09-02.jsonl")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-strategy-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-diff-chars", type=int, default=4000)
    parser.add_argument("--max-history-messages", type=int, default=5)
    parser.add_argument("--vanilla-cache", default="outputs/training_targets_gemini_vanilla.jsonl")
    parser.add_argument("--structured-cache", default="outputs/training_targets_gemini_structured.jsonl")
    parser.add_argument(
        "--reference-guided-cache", default="outputs/training_targets_gemini_reference_guided.jsonl"
    )
    parser.add_argument(
        "--mixture-seed",
        type=int,
        default=1234,
        help="Seeds the fixed, non-overlapping-halves split of the intersection ids into an "
        "original-completion half and a structured-completion half.",
    )
    parser.add_argument("--ids-out", default="outputs/training_ids_intersection.txt")
    parser.add_argument("--mixture-out", default="outputs/training_targets_gemini_mixture.jsonl")
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
    train_by_id = {e.id: e for e in train_examples}

    vanilla_ids = set(_load_cache_ids(args.vanilla_cache))
    structured_cache = _load_cache_ids(args.structured_cache)
    reference_guided_ids = set(_load_cache_ids(args.reference_guided_cache))

    intersection_ids = sorted(set(train_by_id) & vanilla_ids & set(structured_cache) & reference_guided_ids)

    Path(args.ids_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.ids_out, "w") as f:
        for id_ in intersection_ids:
            f.write(id_ + "\n")

    rng = random.Random(args.mixture_seed)
    shuffled = list(intersection_ids)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    original_half, structured_half = shuffled[:half], shuffled[half:]

    mixture_examples = [train_by_id[id_] for id_ in original_half] + [
        replace(train_by_id[id_], reference_completion=structured_cache[id_]) for id_ in structured_half
    ]
    write_training_targets_cache(mixture_examples, args.mixture_out)

    print(f"train pool: {len(train_by_id)}")
    print(f"vanilla cache ids: {len(vanilla_ids)}")
    print(f"structured cache ids: {len(structured_cache)}")
    print(f"reference_guided cache ids: {len(reference_guided_ids)}")
    print(f"intersection ids: {len(intersection_ids)}")
    print(f"wrote {len(intersection_ids)} ids to {args.ids_out}")
    print(
        f"mixture: {len(original_half)} original + {len(structured_half)} structured = "
        f"{len(mixture_examples)} -> {args.mixture_out}"
    )


if __name__ == "__main__":
    main()
