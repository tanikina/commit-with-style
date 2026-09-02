import json
import os
from collections import defaultdict

from commit_style.data import Example
from commit_style.judge import judge_candidates
from commit_style.llm.base import LLMClient
from commit_style.metrics import bertscore_batch, chrf_score, meteor_score
from commit_style.strategies import STRATEGIES


def run_pipeline(
    examples: list[Example],
    strategies: list[str],
    gen_llm: LLMClient,
    judge_llms: dict[str, LLMClient],
    gen_backend_name: str = "",
    gen_model_name: str = "",
    with_bertscore: bool = False,
    debug: bool = False,
) -> list[dict]:
    rows = []
    total = len(examples) * len(strategies)
    count = 0
    for example in examples:
        for strategy_name in strategies:
            count += 1
            try:
                strategy_fn = STRATEGIES[strategy_name]
                candidate = strategy_fn(gen_llm, example)
                judge_scores = judge_candidates(judge_llms, example, candidate)
                metrics = {
                    "chrf": chrf_score(candidate.continuation, example.reference_completion),
                    "meteor": meteor_score(candidate.continuation, example.reference_completion),
                }
            except Exception as e:
                print(
                    f"[run_pipeline] ({count}/{total}) skipping example={example.id} "
                    f"strategy={strategy_name} after a persistent error: {e}"
                )
                continue
            print(f"[run_pipeline] ({count}/{total}) generated example={example.id} strategy={strategy_name}")
            if debug:
                print(f"[debug] example={example.id} strategy={strategy_name} continuation={candidate.continuation!r}")
            rows.append(
                {
                    "example_id": example.id,
                    "repo_name": example.repo_name,
                    "language": example.language,
                    "strategy": candidate.strategy,
                    "prefix": example.prefix,
                    "reference_completion": example.reference_completion,
                    "reference_full_message": example.reference_full_message,
                    "candidate_continuation": candidate.continuation,
                    "candidate_full_message": candidate.full_message,
                    "artifacts": candidate.artifacts,
                    "judge_scores": {jid: s.model_dump() for jid, s in judge_scores.items()},
                    "metrics": metrics,
                    "backend": gen_backend_name,
                    "model_name": gen_model_name,
                }
            )

    if with_bertscore and rows:
        preds = [row["candidate_continuation"] for row in rows]
        refs = [row["reference_completion"] for row in rows]
        bert = bertscore_batch(preds, refs)
        for i, row in enumerate(rows):
            row["metrics"]["bertscore_f1"] = bert["f1"][i]

    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def summarize(rows: list[dict]) -> str:
    judge_ids: list[str] = []
    for row in rows:
        for jid in row["judge_scores"]:
            if jid not in judge_ids:
                judge_ids.append(jid)

    blocks = []
    for jid in judge_ids:
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if jid in row["judge_scores"]:
                by_strategy[row["strategy"]].append(row["judge_scores"][jid])

        lines = [f"--- judge: {jid} ---"]
        lines.append(f"{'strategy':<18}{'n':>4}{'coherence':>12}{'fluency':>10}{'correctness':>14}")
        for strategy_name, scores in by_strategy.items():
            n = len(scores)
            mean_coherence = sum(s["coherence"] for s in scores) / n
            mean_fluency = sum(s["fluency"] for s in scores) / n
            mean_correctness = sum(s["correctness"] for s in scores) / n
            lines.append(
                f"{strategy_name:<18}{n:>4}{mean_coherence:>12.2f}{mean_fluency:>10.2f}{mean_correctness:>14.2f}"
            )
        blocks.append("\n".join(lines))

    usage_rows = [row for row in rows if row.get("usage")]
    if usage_rows:
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for row in usage_rows:
            by_strategy[row["strategy"]].append(row["usage"])

        def _mean(usages: list[dict], key: str) -> float | None:
            vals = [u[key] for u in usages if u.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        def _fmt(value: float | None, spec: str) -> str:
            return format(value, spec) if value is not None else "n/a"

        lines = ["--- cost ---"]
        lines.append(f"{'strategy':<18}{'n':>4}{'in_tok':>10}{'out_tok':>10}{'latency_s':>12}")
        for strategy_name, usages in by_strategy.items():
            n = len(usages)
            in_tok = _fmt(_mean(usages, "input_tokens"), ".1f")
            out_tok = _fmt(_mean(usages, "output_tokens"), ".1f")
            latency = _fmt(_mean(usages, "latency_s"), ".3f")
            lines.append(f"{strategy_name:<18}{n:>4}{in_tok:>10}{out_tok:>10}{latency:>12}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
