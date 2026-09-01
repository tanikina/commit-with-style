from commit_style.data import Example, format_history
from commit_style.judge import judge_candidates
from commit_style.llm.base import LLMClient
from commit_style.metrics import bertscore_batch, chrf_score, meteor_score
from commit_style.strategies import Candidate, _VANILLA_SYSTEM


def _generate(gen_llm: LLMClient, example: Example) -> Candidate:
    user = (
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"DIFF:\n{example.diff}\n\n"
        f"PREFIX:\n{example.prefix}"
    )
    continuation = gen_llm.complete(_VANILLA_SYSTEM, user)
    return Candidate(
        strategy="finetuned",
        continuation=continuation,
        full_message=example.prefix + continuation,
    )


def run_finetuned_eval(
    val_examples: list[Example],
    gen_llm: LLMClient,
    judge_llms: dict[str, LLMClient],
    gen_model_name: str = "",
    with_bertscore: bool = False,
) -> list[dict]:
    rows = []
    for example in val_examples:
        try:
            candidate = _generate(gen_llm, example)
            judge_scores = judge_candidates(judge_llms, example, candidate)
            metrics = {
                "chrf": chrf_score(candidate.continuation, example.reference_completion),
                "meteor": meteor_score(candidate.continuation, example.reference_completion),
            }
        except Exception as e:
            print(f"[run_finetuned_eval] skipping example={example.id} after a persistent error: {e}")
            continue
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
                "backend": "local_hf",
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
