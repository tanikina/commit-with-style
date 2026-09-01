from typing import Literal

from pydantic import BaseModel


class StyleInfo(BaseModel):
    tense: Literal["imperative", "past", "other"]
    conventional_commit: bool


class ChangePlan(BaseModel):
    change_type: Literal["fix", "feature", "refactor", "test", "docs", "build", "other"]
    component: str
    main_change: str
    supported_motivation: str | None
    important_identifiers: list[str]
    style: StyleInfo


class VerificationResult(BaseModel):
    claims_supported: bool
    omitted_important_changes: bool
    prefix_preserved: bool
    style_match: bool
    concise: bool
    revision: str | None
    rationale: str


class JudgeScores(BaseModel):
    coherence: int
    fluency: int
    correctness: int
    rationale: str
