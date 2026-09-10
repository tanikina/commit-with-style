#!/usr/bin/env python3
"""Statistical significance testing for fine-tuning evaluation results.

For each seed and metric:
1. Run Friedman test across all strategies on matched examples
2. If significant, run pairwise Wilcoxon signed-rank tests
3. Apply Holm-Bonferroni correction to control FWER
4. Report mean difference and bootstrap 95% CI for each pair
"""

from __future__ import annotations

import argparse
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
    # Sort p-values and track original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (original_idx, p_val) in enumerate(sorted_pairs):
        # Holm: multiply by (n - rank)
        adjusted[original_idx] = min(1.0, p_val * (n - rank))
    # Enforce monotonicity
    sorted_adjusted = sorted(adjusted)
    for i in range(1, n):
        if sorted_adjusted[i] < sorted_adjusted[i - 1]:
            sorted_adjusted[i] = sorted_adjusted[i - 1]
    # Map back to original order
    adjusted_final = [0.0] * n
    for i, val in enumerate(adjusted):
        adjusted_final[i] = sorted_adjusted[sorted(adjusted).index(val)]
    return adjusted_final


def load_data(paths: list[Path]) -> dict[str, dict[str, list[dict]]]:
    """Load eval files and group by seed and strategy.

    Returns: {seed: {strategy: [rows]}}
    """
    by_seed: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for path in paths:
        # Parse path: outputs/ft/{strategy}/seed{N}/eval_reannotated.jsonl
        parts = path.parts
        if "seed1" in str(path):
            seed = "seed1"
        elif "seed2" in str(path):
            seed = "seed2"
        elif "seed3" in str(path):
            seed = "seed3"
        else:
            print(f"[significance_test] warning: cannot parse seed from {path}", file=sys.stderr)
            continue

        # Find strategy
        strategy = None
        for part in parts:
            if part in ["original", "vanilla", "structured", "mixture", "reference_guided"]:
                strategy = part
                break

        if strategy is None:
            print(f"[significance_test] warning: cannot parse strategy from {path}", file=sys.stderr)
            continue

        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        by_seed[seed][strategy] = rows

    return by_seed


def extract_scores(
    rows_by_strategy: dict[str, list[dict]],
    metric: str,
    judge: str,
) -> dict[str, dict[str, float]]:
    """Extract scores for a metric, indexed by example_id.

    Returns: {strategy: {example_id: score}}
    """
    scores_by_strategy = {}
    for strategy, rows in rows_by_strategy.items():
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

        scores_by_strategy[strategy] = scores

    return scores_by_strategy


def test_seed_metric(
    rows_by_strategy: dict[str, list[dict]],
    seed: str,
    metric: str,
    judge: str,
) -> dict[str, Any]:
    """Run Friedman + pairwise Wilcoxon for one seed and metric."""
    # Extract scores
    scores_by_strategy = extract_scores(rows_by_strategy, metric, judge)

    if len(scores_by_strategy) < 2:
        return {"error": "fewer than 2 strategies"}

    # Find intersection of example IDs
    id_sets = [set(scores.keys()) for scores in scores_by_strategy.values()]
    common_ids = set.intersection(*id_sets)

    if len(common_ids) < 3:
        return {"error": f"only {len(common_ids)} common examples"}

    # Build aligned data matrix: rows = examples, cols = strategies
    strategies = sorted(scores_by_strategy.keys())
    data = []
    for strategy in strategies:
        strategy_scores = scores_by_strategy[strategy]
        data.append([strategy_scores[ex_id] for ex_id in sorted(common_ids)])

    data = np.array(data).T  # shape: (n_examples, n_strategies)

    # Friedman test
    friedman_stat, friedman_p = stats.friedmanchisquare(*data.T)

    result = {
        "seed": seed,
        "metric": metric,
        "judge": judge,
        "n_examples": len(common_ids),
        "n_strategies": len(strategies),
        "strategies": strategies,
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "pairwise": [],
    }

    # If Friedman is significant, do pairwise tests
    if friedman_p < 0.05:
        pairs = []
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                strat_a, strat_b = strategies[i], strategies[j]
                scores_a = data[:, i]
                scores_b = data[:, j]

                # Wilcoxon signed-rank test
                stat, p_val = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")

                # Mean difference
                mean_diff = float(np.mean(scores_a - scores_b))

                # Bootstrap CI
                ci_lower, ci_upper = bootstrap_ci(scores_a, scores_b, random_state=int(seed[-1]))

                pairs.append({
                    "strategy_a": strat_a,
                    "strategy_b": strat_b,
                    "mean_diff": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_val),
                })

        # Holm correction
        p_values = [pair["p_value"] for pair in pairs]
        adjusted_p = holm_bonferroni(p_values)

        for pair, adj_p in zip(pairs, adjusted_p):
            pair["p_value_adjusted"] = adj_p

        result["pairwise"] = pairs

    return result


def format_results(results: list[dict[str, Any]]) -> str:
    """Format results as a readable table."""
    lines = []

    for res in results:
        if "error" in res:
            lines.append(f"\n{res.get('seed', '?')} / {res.get('metric', '?')}: {res['error']}")
            continue

        lines.append(f"\n{'='*80}")
        lines.append(f"Seed: {res['seed']} | Metric: {res['metric']} | Judge: {res['judge']}")
        lines.append(f"N={res['n_examples']} | Strategies: {', '.join(res['strategies'])}")
        lines.append(f"Friedman χ²={res['friedman_stat']:.2f}, p={res['friedman_p']:.4f}")

        if res['friedman_p'] >= 0.05:
            lines.append("  → Not significant, skipping pairwise tests")
        elif not res['pairwise']:
            lines.append("  → Significant, but no pairwise tests run")
        else:
            lines.append("\nPairwise comparisons (Wilcoxon signed-rank + Holm correction):")
            lines.append(f"{'Comparison':<40} {'Δ mean':>8} {'95% CI':>20} {'p (adj)':>10} {'Sig':>5}")
            lines.append("-" * 85)

            for pair in sorted(res['pairwise'], key=lambda x: x['p_value_adjusted']):
                comparison = f"{pair['strategy_a']} vs {pair['strategy_b']}"
                mean_diff = pair['mean_diff']
                ci = f"[{pair['ci_lower']:+.3f}, {pair['ci_upper']:+.3f}]"
                p_adj = pair['p_value_adjusted']
                sig = "*" if p_adj < 0.05 else ""

                lines.append(f"{comparison:<40} {mean_diff:>+8.3f} {ci:>20} {p_adj:>10.4f} {sig:>5}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Paths to eval_reannotated.jsonl files",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["coherence", "correctness", "fluency", "chrf"],
        help="Metrics to test",
    )
    parser.add_argument(
        "--judge",
        default="openai-gpt-5",
        help="Judge ID for judge scores",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output JSON file for results",
    )
    args = parser.parse_args()

    # Load data
    print("[significance_test] Loading data...", file=sys.stderr)
    by_seed = load_data(args.inputs)

    if not by_seed:
        raise SystemExit("No data loaded")

    print(f"[significance_test] Found {len(by_seed)} seeds", file=sys.stderr)
    for seed, strategies in by_seed.items():
        print(f"  {seed}: {len(strategies)} strategies, {[len(rows) for rows in strategies.values()]} rows", file=sys.stderr)

    # Run tests
    results = []
    for seed in sorted(by_seed.keys()):
        for metric in args.metrics:
            print(f"[significance_test] Testing {seed} / {metric}...", file=sys.stderr)
            result = test_seed_metric(by_seed[seed], seed, metric, args.judge)
            results.append(result)

    # Format and print
    print(format_results(results))

    # Save JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[significance_test] Saved detailed results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
