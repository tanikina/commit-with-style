import re

from pydantic import BaseModel

from commit_style.llm.base import LLMClient
from commit_style.schemas import ChangePlan, JudgeScores, StyleInfo, VerificationResult

_PREFIX_RE = re.compile(r"PREFIX:\n(.*?)(?:\n\n|\Z)", re.DOTALL)
_DIFF_RE = re.compile(r"DIFF:\n(.*?)(?:\n\nPREFIX|\Z)", re.DOTALL)


class MockLLMClient(LLMClient):
    """Deterministic, network-free stand-in used to validate pipeline wiring."""

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        prefix_match = _PREFIX_RE.search(user)
        prefix = prefix_match.group(1).strip() if prefix_match else ""
        diff_match = _DIFF_RE.search(user)
        n_files = diff_match.group(1).count("--- ") if diff_match else 0
        if prefix:
            return f"solved issue across {n_files} file(s) via mock generation"
        return f"Mock commit message touching {n_files} file(s)."

    def complete_json(
        self,
        system: str,
        user: str,
        model_cls: type[BaseModel],
        max_tokens: int = 500,
    ) -> BaseModel:
        if model_cls is ChangePlan:
            return ChangePlan(
                change_type="fix",
                component="mock-component",
                main_change="Mock-extracted description of the main change.",
                supported_motivation=None,
                important_identifiers=["mock_identifier"],
                style=StyleInfo(tense="imperative", conventional_commit=False),
            )
        if model_cls is VerificationResult:
            return VerificationResult(
                claims_supported=True,
                omitted_important_changes=False,
                prefix_preserved=True,
                style_match=True,
                concise=True,
                revision=None,
                rationale="Mock verification: candidate looks consistent with the diff.",
            )
        if model_cls is JudgeScores:
            return JudgeScores(
                coherence=4,
                fluency=4,
                correctness=4,
                rationale="Mock judge: plausible, on-topic, and reasonably fluent.",
            )
        raise NotImplementedError(f"MockLLMClient has no fake response for {model_cls!r}")
