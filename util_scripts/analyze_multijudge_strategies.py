#!/usr/bin/env python3
"""Score summary + low-score rationale clustering for the reannotated multi-judge strategy eval sets.

Reads outputs/multijudge_strategies/*_reannotated.jsonl (produced by
scripts/reannotate_multijudge_strategy_scores.py) — no LLM calls, this only reads existing
judge_scores. Prints:

1. A per-file and pooled coherence/fluency/correctness score table per strategy/judge (reusing
   commit_style.pipeline.summarize), plus a chrF/METEOR table per strategy from each row's
   `metrics` field (judge-independent, so it's printed once per file/pool rather than per judge).
2. For coherence and correctness (fluency is skipped — its rubric wasn't touched this session),
   the top --top-k clusters of "bad" (score <= --threshold) rationale texts, pooled across
   strategies/files/judges into one embedding space so cluster membership is directly comparable
   across strategies. Each cluster reports its size, a representative rationale (closest to the
   cluster centroid), and a per-strategy count breakdown — answering whether the same failure
   mode recurs across strategies or is strategy-specific.

Only rows/judges carrying the new-schema {dimension}_rationale fields are used for clustering
(old-schema entries with just a blended `rationale` are skipped there, though they still count
towards the score summary via commit_style.pipeline.summarize).

Each file's rows are restricted (by default) to the intersection of example ids scored under
every strategy present in that file, so the per-strategy n is identical within each file — a few
generations fail per strategy for different ids, and comparing means over different id sets isn't
a fair vanilla-vs-structured-vs-reference_guided comparison. Pass --no-strategy-intersect to fall
back to each strategy's full (unequal) row count instead.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commit_style.pipeline import summarize

DEFAULT_GLOB = "outputs/multijudge_strategies/*_reannotated.jsonl"
DIMENSIONS = ["coherence", "correctness"]


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def file_label(path: Path) -> str:
    name = path.stem
    for suffix in ("_reannotated", "_multi_judge"):
        name = name.replace(suffix, "")
    return name


def restrict_to_strategy_intersection(rows: list[dict]) -> list[dict]:
    id_sets: dict[str, set] = {}
    for row in rows:
        id_sets.setdefault(row["strategy"], set()).add(row["example_id"])
    if len(id_sets) < 2:
        return rows
    overlap = set.intersection(*id_sets.values())
    return [row for row in rows if row["example_id"] in overlap]


def embed_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectors = TfidfVectorizer(stop_words="english", max_features=4000).fit_transform(texts).toarray()
    return normalize(vectors)


def embed_transformers(texts: list[str], model_name: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    batch_size = 32
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.numpy())
    return normalize(np.concatenate(embeddings, axis=0))


def collect_bad_rationale_items(labelled_rows: dict[str, list[dict]], dimension: str, threshold: int) -> list[dict]:
    rationale_key = f"{dimension}_rationale"
    items = []
    for label, rows in labelled_rows.items():
        for row in rows:
            for jid, scores in row.get("judge_scores", {}).items():
                if not scores.get(rationale_key):
                    continue
                if scores.get(dimension, 5) > threshold:
                    continue
                items.append(
                    {
                        "text": scores[rationale_key],
                        "strategy": row["strategy"],
                        "file": label,
                        "judge": jid,
                        "example_id": row.get("example_id"),
                    }
                )
    return items


def summarize_metrics(rows: list[dict]) -> str:
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("metrics"):
            by_strategy[row["strategy"]].append(row["metrics"])
    if not by_strategy:
        return ""

    def _mean(metrics_list: list[dict], key: str) -> float | None:
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    lines = ["--- chrF / METEOR ---", f"{'strategy':<18}{'n':>4}{'chrf':>10}{'meteor':>10}"]
    for strategy, metrics_list in by_strategy.items():
        n = len(metrics_list)
        lines.append(
            f"{strategy:<18}{n:>4}{_fmt(_mean(metrics_list, 'chrf')):>10}{_fmt(_mean(metrics_list, 'meteor')):>10}"
        )
    return "\n".join(lines)


def cluster_bad_rationales(items: list[dict], top_k: int, embedding_backend: str, embedding_model: str) -> list[dict]:
    if not items:
        return []

    texts = [it["text"] for it in items]
    vectors = embed_tfidf(texts) if embedding_backend == "tfidf" else embed_transformers(texts, embedding_model)

    n_clusters = min(top_k, len(items))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(vectors)

    clusters = []
    for cluster_id in range(n_clusters):
        member_idxs = [i for i, label in enumerate(labels) if label == cluster_id]
        member_vecs = vectors[member_idxs]
        centroid = member_vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = member_vecs @ centroid
        rep_idx = member_idxs[int(np.argmax(sims))]
        clusters.append(
            {
                "size": len(member_idxs),
                "representative": items[rep_idx]["text"],
                "strategy_counts": Counter(items[i]["strategy"] for i in member_idxs),
            }
        )
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters[:top_k]


def print_cluster_report(dimension: str, clusters: list[dict], n_total: int, threshold: int) -> None:
    print(f"\n=== bad {dimension} rationale clusters (n={n_total} cases with {dimension} <= {threshold}) ===")
    if not clusters:
        print("(no cases at or below threshold)")
        return
    for i, cluster in enumerate(clusters, 1):
        breakdown = ", ".join(f"{strategy}={count}" for strategy, count in cluster["strategy_counts"].most_common())
        print(f"\n#{i} size={cluster['size']} ({breakdown})")
        print(f"  {cluster['representative']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=f"Defaults to every file matching {DEFAULT_GLOB!r}.",
    )
    parser.add_argument("--threshold", type=int, default=2, help="A dimension score <= this counts as 'bad'.")
    parser.add_argument(
        "--no-strategy-intersect",
        action="store_true",
        help="Skip restricting each file's rows to the id intersection across its strategies; use "
        "each strategy's full (unequal) row count instead.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-backend", choices=["tfidf", "transformers"], default="tfidf")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Only used with --embedding-backend transformers.",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(Path().glob(DEFAULT_GLOB))
        if not paths:
            raise SystemExit(f"No files found matching {DEFAULT_GLOB!r}")

    labelled_rows: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    for path in paths:
        rows = load_rows(path)
        n_raw = len(rows)
        if not args.no_strategy_intersect:
            rows = restrict_to_strategy_intersection(rows)
        label = file_label(path)
        labelled_rows[label] = rows
        all_rows.extend(rows)
        print(f"[analyze_multijudge_strategies] {path}: n_raw={n_raw} -> n={len(rows)}")

    print("\n=== per-file score summary ===")
    for label, rows in labelled_rows.items():
        print(f"\n--- {label} ---")
        print(summarize(rows))
        print(summarize_metrics(rows))

    if len(labelled_rows) > 1:
        print("\n=== pooled score summary (all files) ===")
        print(summarize(all_rows))
        print(summarize_metrics(all_rows))

    for dimension in DIMENSIONS:
        items = collect_bad_rationale_items(labelled_rows, dimension, args.threshold)
        clusters = cluster_bad_rationales(items, args.top_k, args.embedding_backend, args.embedding_model)
        print_cluster_report(dimension, clusters, len(items), args.threshold)


if __name__ == "__main__":
    main()
