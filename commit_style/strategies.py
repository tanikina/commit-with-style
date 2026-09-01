from dataclasses import dataclass, field

from commit_style.data import Example, format_history
from commit_style.llm.base import LLMClient
from commit_style.schemas import ChangePlan, VerificationResult

_VANILLA_SYSTEM = (
    "You are completing a developer's in-progress git commit message for the given diff, "
    "using the repo's recent commit history as a style reference. Continue naturally from "
    "PREFIX. Do not repeat the prefix text. Output ONLY the continuation text, no markdown, "
    "no quotes, no explanation."
)

_PLAN_SYSTEM = (
    "You extract a compact structured record describing a code change from its diff and the "
    "repo's commit history. Only fill `supported_motivation` if the diff or history actually "
    "state a reason for the change; otherwise it MUST be null. Do not invent facts not visible "
    "in the diff or history."
)

_PLAN_GENERATE_SYSTEM = (
    "You write the continuation of an in-progress git commit message, using a structured change "
    "plan and the repo's recent commit history as a style reference. Continue naturally from "
    "PREFIX. Do not repeat the prefix text. Output ONLY the continuation text, no markdown, no "
    "quotes, no explanation."
)

_VERIFY_SYSTEM = (
    "You verify a candidate commit-message continuation against the diff, history, and prefix "
    "it was generated from. Check: (1) every claim is supported by the diff/history, (2) no "
    "important changes are omitted, (3) the prefix is preserved (the continuation doesn't "
    "repeat or contradict it), (4) the message matches the repo's style, (5) it is concise. If "
    "everything holds, set revision to null. Otherwise set revision to a corrected continuation "
    "text (still just the continuation, not the full message)."
)

_REWRITE_SYSTEM = (
    "You improve a developer's original commit message for clarity, faithfulness, style "
    "compatibility with the repo's history, or grammar. You may NOT add facts, motivations, or "
    "identifiers that are not supported by the diff or history. Output ONLY the rewritten full "
    "commit message, no markdown, no quotes, no explanation."
)


@dataclass
class Candidate:
    strategy: str
    continuation: str
    full_message: str
    artifacts: dict = field(default_factory=dict)


def run_vanilla(llm: LLMClient, example: Example) -> Candidate:
    user = (
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"DIFF:\n{example.diff}\n\n"
        f"PREFIX:\n{example.prefix}"
    )
    continuation = llm.complete(_VANILLA_SYSTEM, user)
    return Candidate(
        strategy="vanilla",
        continuation=continuation,
        full_message=example.prefix + continuation,
    )


def run_structured(llm: LLMClient, example: Example) -> Candidate:
    plan_user = f"DIFF:\n{example.diff}\n\nHISTORY:\n{format_history(example.history)}"
    plan: ChangePlan = llm.complete_json(_PLAN_SYSTEM, plan_user, ChangePlan)

    generate_user = (
        f"PLAN:\n{plan.model_dump_json()}\n\n"
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"PREFIX:\n{example.prefix}"
    )
    initial_continuation = llm.complete(_PLAN_GENERATE_SYSTEM, generate_user)

    verify_user = (
        f"DIFF:\n{example.diff}\n\n"
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"PREFIX:\n{example.prefix}\n\n"
        f"CANDIDATE_CONTINUATION:\n{initial_continuation}"
    )
    verification: VerificationResult = llm.complete_json(_VERIFY_SYSTEM, verify_user, VerificationResult)

    final_continuation = verification.revision if verification.revision else initial_continuation
    return Candidate(
        strategy="structured",
        continuation=final_continuation,
        full_message=example.prefix + final_continuation,
        artifacts={
            "plan": plan.model_dump(),
            "verification": verification.model_dump(),
            "initial_continuation": initial_continuation,
            "revised": verification.revision is not None,
        },
    )


def run_reference_guided(llm: LLMClient, example: Example) -> Candidate:
    user = (
        f"DIFF:\n{example.diff}\n\n"
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"ORIGINAL_MESSAGE:\n{example.reference_full_message}\n\n"
        f"PREFIX:\n{example.prefix}"
    )
    rewritten = llm.complete(_REWRITE_SYSTEM, user)

    if rewritten.startswith(example.prefix):
        continuation = rewritten[len(example.prefix):]
        prefix_stripped = True
    else:
        continuation = rewritten
        prefix_stripped = False

    return Candidate(
        strategy="reference_guided",
        continuation=continuation,
        full_message=rewritten if prefix_stripped else example.prefix + continuation,
        artifacts={"prefix_stripped": prefix_stripped, "raw_rewrite": rewritten},
    )


STRATEGIES = {
    "vanilla": run_vanilla,
    "structured": run_structured,
    "reference_guided": run_reference_guided,
}
