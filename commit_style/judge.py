from commit_style.data import Example, format_history
from commit_style.llm.base import LLMClient
from commit_style.schemas import JudgeScores
from commit_style.strategies import Candidate

_JUDGE_SYSTEM = (
    "You are an expert judge of git commit messages. Score the CANDIDATE_MESSAGE on three "
    "1-5 integer scales, each with its own rationale in 1-2 sentences (be concise — do not "
    "write more than 2 sentences per rationale field): "
    "coherence, fluency, and correctness.\n\n"
    "For coherence_rationale, in your first sentence state the typical length (roughly how many "
    "words) and structural form (e.g. short imperative subject line, prefix-tag style, bullet "
    "list) of the entries in HISTORY vs. CANDIDATE_MESSAGE's own length and form — do this "
    "explicitly before giving the coherence score, do not just assert a match. Score coherence 1-2 "
    "(not higher) whenever the candidate is meaningfully longer or more detailed than typical "
    "HISTORY entries, or takes a different structural form (e.g. a parenthetical aside or a "
    "TODO-style note when HISTORY entries are short imperative subject lines) — even if individual "
    "words look plausible and even if the content is otherwise accurate. Judge coherence purely on "
    "length/detail/structure vs. HISTORY, independently of correctness — do not let factual "
    "accuracy issues affect coherence_rationale or the coherence score.\n\n"
    "For fluency_rationale: is CANDIDATE_MESSAGE grammatically correct and natural to read?\n\n"
    "For correctness_rationale: does CANDIDATE_MESSAGE accurately and adequately describe DIFF? "
    "Score correctness down for: invented facts that contradict DIFF; omission of the main change; "
    "vague or uninformative wording that doesn't convey what actually changed even if nothing in "
    "it is technically false; and specific claims that cannot be verified from DIFF alone — e.g. "
    "ticket/issue numbers, external motivations, or context not visible in the code — since these "
    "look unverifiable/potentially hallucinated regardless of whether they happen to be accurate. "
    "REFERENCE_MESSAGE is the original developer's message, given only to help you judge "
    "correctness against the diff — do not penalize the candidate merely for phrasing "
    "differently than the reference."
)

_JUDGE_MAX_TOKENS = 3000


def judge_candidate(llm: LLMClient, example: Example, candidate: Candidate) -> JudgeScores:
    user = (
        f"DIFF:\n{example.diff}\n\n"
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"PREFIX:\n{example.prefix}\n\n"
        f"CANDIDATE_MESSAGE:\n{candidate.full_message}\n\n"
        f"REFERENCE_MESSAGE:\n{example.reference_full_message}"
    )
    return llm.complete_json(_JUDGE_SYSTEM, user, JudgeScores, max_tokens=_JUDGE_MAX_TOKENS)


def judge_candidates(
    judges: dict[str, LLMClient], example: Example, candidate: Candidate
) -> dict[str, JudgeScores]:
    """Score `candidate` with every judge, keyed by judge id (see `commit_style.llm.factory.judge_id`)."""
    return {judge_id: judge_candidate(llm, example, candidate) for judge_id, llm in judges.items()}
