from commit_style.data import Example, format_history
from commit_style.llm.base import LLMClient
from commit_style.schemas import JudgeScores
from commit_style.strategies import Candidate

_JUDGE_SYSTEM = (
    "You are an expert judge of git commit messages. Score the CANDIDATE_MESSAGE on three "
    "1-5 integer scales: "
    "coherence (does it match the style/tone/conventions of HISTORY, the repo's recent commit "
    "messages?), "
    "fluency (is it grammatically correct and natural to read?), and "
    "correctness (are its claims actually supported by DIFF — no invented facts, no omission of "
    "the main change?). "
    "REFERENCE_MESSAGE is the original developer's message, given only to help you judge "
    "correctness against the diff — do not penalize the candidate merely for phrasing "
    "differently than the reference. Give a short rationale."
)


def judge_candidate(llm: LLMClient, example: Example, candidate: Candidate) -> JudgeScores:
    user = (
        f"DIFF:\n{example.diff}\n\n"
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"PREFIX:\n{example.prefix}\n\n"
        f"CANDIDATE_MESSAGE:\n{candidate.full_message}\n\n"
        f"REFERENCE_MESSAGE:\n{example.reference_full_message}"
    )
    return llm.complete_json(_JUDGE_SYSTEM, user, JudgeScores)


def judge_candidates(
    judges: dict[str, LLMClient], example: Example, candidate: Candidate
) -> dict[str, JudgeScores]:
    """Score `candidate` with every judge, keyed by judge id (see `commit_style.llm.factory.judge_id`)."""
    return {judge_id: judge_candidate(llm, example, candidate) for judge_id, llm in judges.items()}
