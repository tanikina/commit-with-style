from commit_style.data import Example, format_history
from commit_style.llm.base import LLMClient
from commit_style.schemas import JudgeScores
from commit_style.strategies import Candidate

_JUDGE_SYSTEM = (
    "You are an expert judge of git commit messages. Score the CANDIDATE_MESSAGE on three "
    "1-5 integer scales, each with its own rationale in 1-2 sentences (be concise — do not "
    "write more than 2 sentences per rationale field): "
    "coherence, fluency, and correctness.\n\n"
    "For coherence_rationale, in your first sentence state HISTORY's typical length (roughly how "
    "many words), structural form (e.g. short imperative subject line, prefix-tag style, bullet "
    "list), casing convention (e.g. lowercase vs. capitalized start), and punctuation/mood "
    "convention (e.g. trailing punctuation, imperative vs. other mood) vs. CANDIDATE_MESSAGE's own "
    "length, form, casing, and punctuation/mood — do this explicitly before giving the coherence "
    "score, do not just assert a match. Cap coherence at 1-2 (not higher) whenever the candidate "
    "diverges from typical HISTORY entries on any of length/detail, structural form, casing, or "
    "punctuation/mood — even if individual words look plausible and even if the content is "
    "otherwise accurate. Judge coherence purely on this style/tone match vs. HISTORY, independently "
    "of correctness — do not let factual accuracy issues affect coherence_rationale or the "
    "coherence score.\n\n"
    "For fluency_rationale: is CANDIDATE_MESSAGE grammatically correct and natural to read?\n\n"
    "For correctness_rationale, in your first sentence state whether every specific claim in "
    "CANDIDATE_MESSAGE (ticket/issue numbers, named identifiers, external motivations) is directly "
    "evidenced in DIFF, and whether the message concretely conveys the main change rather than "
    "being too generic/vague — do this explicitly before giving the score, do not just assert "
    "accuracy. Cap correctness at 1-2 whenever any specific claim is not evidenced in DIFF "
    "(unverifiable/potentially hallucinated, regardless of plausibility) or the message is too "
    "vague to convey what changed; also score down for invented facts contradicting DIFF or "
    "omission of the main change. REFERENCE_MESSAGE is the original developer's message, given "
    "only to help judge correctness against the diff — do not penalize phrasing that merely "
    "differs from the reference."
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
