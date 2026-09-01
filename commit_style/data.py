import json
import random
from dataclasses import dataclass


@dataclass
class Example:
    id: str
    repo_name: str
    language: str
    diff: str
    history: list[str]
    prefix: str
    reference_completion: str
    reference_full_message: str


def render_diff(changes: list[dict], max_chars: int = 4000) -> str:
    parts = []
    for mod in changes:
        header = f"--- {mod.get('old_path')} -> {mod.get('new_path')} ({mod.get('change_type')})"
        parts.append(f"{header}\n{mod.get('diff', '')}")
    rendered = "\n".join(parts)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "\n... [truncated]"
    return rendered


def truncate_history(history: list[str], k: int = 5) -> list[str]:
    if not history:
        return []
    return history[-k:]


def format_history(history: list[str]) -> str:
    if not history:
        return "(no history available)"
    return "\n".join(f"- {h}" for h in history)


def _row_to_example(row: dict, max_diff_chars: int, max_history_messages: int) -> Example:
    return Example(
        id=row["commit_sha"],
        repo_name=row["repo_name"],
        language=row["repo_main_language"],
        diff=render_diff(row.get("changes") or [], max_chars=max_diff_chars),
        history=truncate_history(row.get("history") or [], k=max_history_messages),
        prefix=row["prefix"],
        reference_completion=row["completion"],
        reference_full_message=row["msg"],
    )


def load_examples(
    path: str,
    n: int | None = 12,
    seed: int = 0,
    languages: list[str] | None = None,
    stratify: bool = True,
    max_diff_chars: int = 4000,
    max_history_messages: int = 5,
) -> list[Example]:
    """Stream the JSONL file and sample `n` examples.

    When `stratify` is True and `languages` is not given, examples are drawn evenly across
    whatever languages are present in the file (discovered while streaming).
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            lang = row["repo_main_language"]
            if languages is not None and lang not in languages:
                continue
            buckets.setdefault(lang, []).append(row)

    if n is None:
        selected_rows = [row for rows in buckets.values() for row in rows]
    elif stratify:
        langs = sorted(buckets.keys())
        per_lang = max(1, n // len(langs)) if langs else 0
        selected_rows = []
        for lang in langs:
            rows = buckets[lang]
            rng.shuffle(rows)
            selected_rows.extend(rows[:per_lang])
        rng.shuffle(selected_rows)
        selected_rows = selected_rows[:n]
    else:
        all_rows = [row for rows in buckets.values() for row in rows]
        rng.shuffle(all_rows)
        selected_rows = all_rows[:n]

    return [
        _row_to_example(row, max_diff_chars=max_diff_chars, max_history_messages=max_history_messages)
        for row in selected_rows
    ]


def _assign_repos_to_splits(
    repos: list[str],
    repo_rows: dict[str, list[dict]],
    quotas: dict[str, int],
) -> dict[str, list[dict]]:
    """Greedily assign each repo's rows wholesale to the split with the largest deficit.

    Guarantees repo-level disjointness: every row from a given repo lands in exactly one
    split (never split across train/val/test), which is what avoids leakage.
    """
    assigned: dict[str, int] = {split: 0 for split in quotas}
    pools: dict[str, list[dict]] = {split: [] for split in quotas}
    for repo in repos:
        split = max(quotas, key=lambda s: quotas[s] - assigned[s])
        rows = repo_rows[repo]
        pools[split].extend(rows)
        assigned[split] += len(rows)
    return pools


def _stratify_by_prefix_length(rows: list[dict], quota: int, n_bins: int, rng: random.Random) -> list[dict]:
    """Sample `quota` rows drawn evenly across `n_bins` prefix-length buckets (short/medium/long)."""
    if len(rows) <= quota:
        return rows

    ordered = sorted(rows, key=lambda r: len(r["prefix"]))
    n = len(ordered)
    edges = [round(i * n / n_bins) for i in range(n_bins + 1)]
    bins = [ordered[edges[i] : edges[i + 1]] for i in range(n_bins)]
    for b in bins:
        rng.shuffle(b)

    per_bin = max(1, quota // len(bins))
    selected: list[dict] = []
    for b in bins:
        selected.extend(b[:per_bin])
    rng.shuffle(selected)
    return selected[:quota]


def load_project_splits(
    path: str,
    n_train: int = 1000,
    n_strategy_val: int = 200,
    n_finetune_test: int = 300,
    seed: int = 0,
    languages: list[str] | None = None,
    max_diff_chars: int = 4000,
    max_history_messages: int = 5,
    n_prefix_bins: int = 3,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Returns (train, strategy_val, finetune_test) drawn from one shared partition.

    The three sets are mutually disjoint by `repo_name` (whole repos are assigned to exactly
    one split, so no repo's commits leak across train/val/test) and stratified by both
    language and prefix length within each split. `strategy_val` is meant for comparing the
    3 prompting strategies (Phase 1); `finetune_test` is meant for LoRA hp-search selection
    and the final fine-tuned-model evaluation (Phase 2).
    """
    rng = random.Random(seed)
    repo_rows: dict[str, list[dict]] = {}
    repo_lang: dict[str, str] = {}

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            lang = row["repo_main_language"]
            if languages is not None and lang not in languages:
                continue
            repo = row["repo_name"]
            repo_rows.setdefault(repo, []).append(row)
            repo_lang.setdefault(repo, lang)

    repos_by_lang: dict[str, list[str]] = {}
    for repo, lang in repo_lang.items():
        repos_by_lang.setdefault(lang, []).append(repo)

    langs = sorted(repos_by_lang.keys())
    n_langs = len(langs) or 1
    split_targets = {"train": n_train, "strategy_val": n_strategy_val, "finetune_test": n_finetune_test}

    split_rows: dict[str, list[dict]] = {"train": [], "strategy_val": [], "finetune_test": []}
    for lang in langs:
        repos = repos_by_lang[lang]
        rng.shuffle(repos)
        quotas = {split: max(1, target // n_langs) for split, target in split_targets.items()}
        pools = _assign_repos_to_splits(repos, repo_rows, quotas)
        for split, quota in quotas.items():
            pool = pools[split]
            if len(pool) < quota:
                print(
                    f"[load_project_splits] warning: only {len(pool)}/{quota} rows available "
                    f"for split={split} lang={lang}"
                )
            split_rows[split].extend(_stratify_by_prefix_length(pool, quota, n_prefix_bins, rng))

    examples: dict[str, list[Example]] = {}
    for split, rows in split_rows.items():
        rng.shuffle(rows)
        rows = rows[: split_targets[split]]
        examples[split] = [
            _row_to_example(row, max_diff_chars=max_diff_chars, max_history_messages=max_history_messages)
            for row in rows
        ]
    return examples["train"], examples["strategy_val"], examples["finetune_test"]
