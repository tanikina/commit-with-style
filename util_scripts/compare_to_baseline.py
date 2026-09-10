#!/usr/bin/env python3
"""Compare fine-tuned strategies to baseline model using paired statistical tests.

For each seed/metric combination:
1. Compare each FT strategy against the base model on matched examples
2. Use Wilcoxon signed-rank test (paired)
3. Apply Holm-Bonferroni correction across all comparisons
4. Report mean difference, bootstrap 95% CI, and adjusted p-value
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    random_state: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for paired difference (x - y)."""
    rng = np.random.RandomState(random_state)
    n = len(x)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(np.mean(x[idx] - y[idx]))
    alpha = 1 - ci
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (original_idx, p_val) in enumerate(sorted_pairs):
        adjusted[original_idx] = min(1.0, p_val * (n - rank))
    # Enforce monotonicity
    sorted_adjusted = sorted(adjusted)
    for i in range(1, n):
        if sorted_adjusted[i] < sorted_adjusted[i - 1]:
            sorted_adjusted[i] = sorted_adjusted[i - 1]
    adjusted_final = [0.0] * n
    for i, val in enumerate(adjusted):
        adjusted_final[i] = sorted_adjusted[sorted(adjusted).index(val)]
    return adjusted_final


def load_baseline(path: Path) -> dict[str, dict]:
    """Load baseline model scores.

    Returns: {example_id: {metric: score}}
    """
    scores = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = row["example_id"]
            scores[example_id] = {
                "coherence": row.get("judge_scores", {}).get("openai-gpt-5", {}).get("coherence"),
                "fluency": row.get("judge_scores", {}).get("openai-gpt-5", {}).get("fluency"),
                "correctness": row.get("judge_scores", {}).get("openai-gpt-5", {}).get("correctness"),
                "chrf": row.get("metrics", {}).get("chrf"),
                "meteor": row.get("metrics", {}).get("meteor"),
            }
    return scores


def load_ft_data(paths: list[Path]) -> dict[str, dict[str, list[dict]]]:
    """Load fine-tuned model eval files and group by seed and strategy.

    Returns: {seed: {strategy: [rows]}}
    """
    by_seed: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for path in paths:
        parts = path.parts

        # Parse seed
        seed = None
        for part in parts:
            if part.startswith("seed"):
                seed = part
                break

        if seed is None:
            print(f"[compare_to_baseline] warning: cannot parse seed from {path}", file=sys.stderr)
            continue

        # Parse strategy
        strategy = None
        for part in parts:
            if part in ["original", "vanilla", "structured", "mixture", "reference_guided"]:
                strategy = part
                break

        if strategy is None:
            print(f"[compare_to_baseline] warning: cannot parse strategy from {path}", file=sys.stderr)
            continue

        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        by_seed[seed][strategy] = rows

    return by_seed


def extract_ft_scores(
    rows: list[dict],
    metric: str,
    judge: str = "openai-gpt-5",
) -> dict[str, float]:
    """Extract scores for a metric, indexed by example_id.

    Returns: {example_id: score}
    """
    scores = {}
    for row in rows:
        example_id = row["example_id"]
        judge_scores = row.get("judge_scores", {}).get(judge, {})

        if metric in ["coherence", "fluency", "correctness"]:
            score = judge_scores.get(metric)
        elif metric in ["chrf", "meteor"]:
            score = row.get("metrics", {}).get(metric)
        else:
            continue

        if score is not None:
            scores[example_id] = score

    return scores


def compare_strategy_to_baseline(
    ft_scores: dict[str, float],
    baseline_scores: dict[str, dict],
    metric: str,
    strategy: str,
    seed: str,
) -> dict[str, Any]:
    """Compare one FT strategy to baseline for a specific metric."""

    # Find common example IDs
    common_ids = set(ft_scores.keys()) & set(baseline_scores.keys())

    # Filter to examples where both have non-None scores
    valid_ids = [
        ex_id for ex_id in common_ids
        if baseline_scores[ex_id][metric] is not None
    ]

    if len(valid_ids) < 3:
        return {
            "strategy": strategy,
            "seed": seed,
            "metric": metric,
            "error": f"only {len(valid_ids)} common examples",
        }

    # Build aligned arrays
    ft_array = np.array([ft_scores[ex_id] for ex_id in valid_ids])
    baseline_array = np.array([baseline_scores[ex_id][metric] for ex_id in valid_ids])

    # Wilcoxon signed-rank test
    stat, p_val = stats.wilcoxon(ft_array, baseline_array, alternative="two-sided")

    # Mean difference (FT - baseline, so positive means FT is better)
    mean_diff = float(np.mean(ft_array - baseline_array))

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(ft_array, baseline_array, random_state=int(seed[-1]) if seed[-1].isdigit() else 0)

    return {
        "strategy": strategy,
        "seed": seed,
        "metric": metric,
        "n_examples": len(valid_ids),
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_val),
    }


def format_results(results: list[dict[str, Any]]) -> str:
    """Format results as a readable table."""
    lines = []

    # Group by seed and metric
    by_seed_metric = defaultdict(list)
    for res in results:
        if "error" in res:
            continue
        key = (res["seed"], res["metric"])
        by_seed_metric[key].append(res)

    for (seed, metric), comparisons in sorted(by_seed_metric.items()):
        lines.append(f"\n{'='*90}")
        lines.append(f"Seed: {seed} | Metric: {metric}")
        lines.append(f"Comparison to base_qwen3_4b (positive Δ means FT is better)")
        lines.append("")
        lines.append(f"{'Strategy':<20} {'N':>5} {'Δ mean':>8} {'95% CI':>22} {'p (adj)':>10} {'Sig':>5}")
        lines.append("-" * 90)

        for comp in sorted(comparisons, key=lambda x: x.get("p_value_adjusted", 1)):
            strategy = comp["strategy"]
            n = comp["n_examples"]
            mean_diff = comp["mean_diff"]
            ci = f"[{comp['ci_lower']:+.3f}, {comp['ci_upper']:+.3f}]"
            p_adj = comp.get("p_value_adjusted", comp["p_value"])
            sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else ""))

            lines.append(f"{strategy:<20} {n:>5} {mean_diff:>+8.3f} {ci:>22} {p_adj:>10.4f} {sig:>5}")

    return "\n".join(lines)


def results_to_csv(results: list[dict]) -> list[dict]:
    """Convert results to CSV rows."""
    csv_rows = []
    for res in results:
        if "error" in res:
            row = {
                "seed": res["seed"],
                "metric": res["metric"],
                "strategy": res["strategy"],
                "error": res["error"],
            }
        else:
            p_adj = res.get("p_value_adjusted", res["p_value"])
            row = {
                "seed": res["seed"],
                "metric": res["metric"],
                "strategy": res["strategy"],
                "n_examples": res["n_examples"],
                "mean_diff": f"{res['mean_diff']:+.4f}",
                "ci_lower": f"{res['ci_lower']:+.4f}",
                "ci_upper": f"{res['ci_upper']:+.4f}",
                "p_value": f"{res['p_value']:.6f}",
                "p_value_adjusted": f"{p_adj:.6f}",
                "significant": "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns")),
            }
        csv_rows.append(row)
    return csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline",
        type=Path,
        help="Path to baseline eval file (e.g., outputs/qwen3_4b_eval_reannotated.jsonl)",
    )
    parser.add_argument(
        "ft_inputs",
        nargs="+",
        type=Path,
        help="Paths to fine-tuned eval files",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["coherence", "correctness", "fluency", "chrf"],
        help="Metrics to test",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output JSON file for results",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional output CSV file for results",
    )
    args = parser.parse_args()

    # Load baseline
    print(f"[compare_to_baseline] Loading baseline from {args.baseline}...", file=sys.stderr)
    baseline_scores = load_baseline(args.baseline)
    print(f"[compare_to_baseline] Loaded {len(baseline_scores)} baseline examples", file=sys.stderr)

    # Load fine-tuned data
    print("[compare_to_baseline] Loading fine-tuned data...", file=sys.stderr)
    by_seed = load_ft_data(args.ft_inputs)

    if not by_seed:
        raise SystemExit("No fine-tuned data loaded")

    print(f"[compare_to_baseline] Found {len(by_seed)} seeds", file=sys.stderr)
    for seed, strategies in by_seed.items():
        print(f"  {seed}: {len(strategies)} strategies", file=sys.stderr)

    # Run comparisons
    all_results = []

    for seed in sorted(by_seed.keys()):
        for metric in args.metrics:
            print(f"[compare_to_baseline] Testing {seed} / {metric}...", file=sys.stderr)

            comparisons = []
            for strategy, rows in by_seed[seed].items():
                ft_scores = extract_ft_scores(rows, metric)
                result = compare_strategy_to_baseline(
                    ft_scores, baseline_scores, metric, strategy, seed
                )
                comparisons.append(result)

            # Apply Holm correction across all strategies for this seed/metric
            valid_comparisons = [c for c in comparisons if "error" not in c]
            if valid_comparisons:
                p_values = [c["p_value"] for c in valid_comparisons]
                adjusted_p = holm_bonferroni(p_values)

                for comp, adj_p in zip(valid_comparisons, adjusted_p):
                    comp["p_value_adjusted"] = adj_p

            all_results.extend(comparisons)

    # Format and print
    print(format_results(all_results))

    # Save JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[compare_to_baseline] Saved detailed results to {args.output}", file=sys.stderr)

    # Save CSV if requested
    if args.output_csv:
        csv_rows = results_to_csv(all_results)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)

        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with args.output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

            print(f"[compare_to_baseline] Saved {len(csv_rows)} rows to {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
