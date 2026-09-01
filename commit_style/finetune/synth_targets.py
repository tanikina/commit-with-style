import json
from dataclasses import replace
from pathlib import Path

from commit_style.data import Example
from commit_style.llm.base import LLMClient
from commit_style.strategies import STRATEGIES


def generate_training_targets(
    examples: list[Example], strategy_name: str, gen_llm: LLMClient
) -> list[Example]:
    """Replace each example's `reference_completion` with a strategy-generated one.

    Runs `STRATEGIES[strategy_name]` (e.g. `structured`) once per example and swaps in
    `candidate.continuation` as the new training target, so the fine-tuned model distills
    from the prompted generator instead of the dataset's human-written completion. Only
    `reference_completion` changes — `reference_full_message` and everything else on the
    `Example` is left untouched (metrics/judging elsewhere still compare against the human
    reference).
    """
    strategy_fn = STRATEGIES[strategy_name]
    generated = []
    for example in examples:
        try:
            candidate = strategy_fn(gen_llm, example)
        except Exception as e:
            print(f"[generate_training_targets] skipping example={example.id} after a persistent error: {e}")
            continue
        generated.append(replace(example, reference_completion=candidate.continuation))
    return generated


def write_training_targets_cache(examples: list[Example], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps({"id": example.id, "reference_completion": example.reference_completion}) + "\n")


def load_training_targets_cache(examples: list[Example], path: str) -> list[Example] | None:
    """Returns `examples` with cached generated completions substituted in, or None on any
    cache miss (missing file, or an id present in `examples` that isn't in the cache) so the
    caller falls back to regenerating from scratch rather than silently mixing cached and
    human reference completions.
    """
    if not Path(path).exists():
        return None
    cached: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            cached[row["id"]] = row["reference_completion"]
    if not all(example.id in cached for example in examples):
        return None
    return [replace(example, reference_completion=cached[example.id]) for example in examples]
