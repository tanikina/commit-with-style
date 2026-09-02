import json
import re
from dataclasses import replace
from pathlib import Path

from commit_style.data import Example
from commit_style.llm.base import LLMClient
from commit_style.strategies import STRATEGIES

_LEAK_CUE_RE = re.compile(
    r"\bwait,|\blet me\b|\blet's\b|\bhmm\b|\bactually,|\bor maybe\b|\bthe prefix is\b|"
    r"\bthe diff (?:only )?shows?\b|\bis it \"|\bso the continuation\b|\bso the commit message\b|"
    r"\bmaybe this is\b|\bmaybe the continuation\b|\bno, \"|\bi think the\b|"
    r"\bmatches the original message\b|\bthe original message (?:has|had)\b|"
    r"\bit could be completed as\b|\ban elegant and clear way\b|\banswering the user's request\b|"
    r"\bthe continuation (?:of the commit message )?should be\b",
    re.IGNORECASE,
)
_ARROW_ANSWER_RE = re.compile(r'"[^"\n]*"\s*(?:->|→)\s*"([^"\n]+)"\s*$')


def clean_reasoning_leakage(text: str) -> str | None:
    """Detect self-questioning/reasoning leakage in a generated completion (e.g. `"Wait, the
    diff only shows..."`) via a cue-phrase match. A no-op (returns `text` unchanged) when no
    cue matches. When a cue matches, attempts one narrow, high-confidence extraction — a
    terminal `"X" -> "Y"` arrow construct where `Y` is a complete, closed quote at the very
    end of the text — and returns just `Y`. Returns `None` when a cue matches but nothing can
    be safely extracted (most cases: the text is truncated mid-thought or lists multiple
    unresolved candidate phrasings) — the caller is expected to fall back to a known-good
    value (e.g. the example's human `reference_completion`) rather than use `None` as data.
    This cue list is precision-favoring, not exhaustive — some contamination without any of
    these exact phrases will slip through undetected.
    """
    if not _LEAK_CUE_RE.search(text):
        return text
    m = _ARROW_ANSWER_RE.search(text.strip())
    return m.group(1).strip() if m else None


def generate_training_targets(
    examples: list[Example], strategy_name: str, gen_llm: LLMClient, debug: bool = False
) -> list[Example]:
    """Replace each example's `reference_completion` with a strategy-generated one.

    Runs `STRATEGIES[strategy_name]` (e.g. `structured`) once per example and swaps in
    `candidate.continuation` as the new training target, so the fine-tuned model distills
    from the prompted generator instead of the dataset's human-written completion. Only
    `reference_completion` changes — `reference_full_message` and everything else on the
    `Example` is left untouched (metrics/judging elsewhere still compare against the human
    reference).

    Generated completions are passed through `clean_reasoning_leakage()` first: cleanly
    extractable leakage is fixed in place; unsalvageable leakage falls back to the example's
    original human `reference_completion` rather than keeping the contaminated text or
    dropping the row (dropping would break cache id-coverage guarantees relied on elsewhere).
    """
    strategy_fn = STRATEGIES[strategy_name]
    generated = []
    total = len(examples)
    for count, example in enumerate(examples, start=1):
        try:
            candidate = strategy_fn(gen_llm, example)
        except Exception as e:
            print(
                f"[generate_training_targets] ({count}/{total}) skipping example={example.id} "
                f"after a persistent error: {e}"
            )
            continue
        print(f"[generate_training_targets] ({count}/{total}) generated example={example.id}")
        if debug:
            print(f"[debug] example={example.id} continuation={candidate.continuation!r}")
        cleaned = clean_reasoning_leakage(candidate.continuation)
        if cleaned is None:
            print(
                f"[generate_training_targets] example={example.id}: reasoning leakage detected, "
                f"no clean answer extractable — falling back to the human reference_completion"
            )
            cleaned = example.reference_completion
        elif cleaned != candidate.continuation:
            print(f"[generate_training_targets] example={example.id}: extracted clean answer from reasoning leakage")
        generated.append(replace(example, reference_completion=cleaned))
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
