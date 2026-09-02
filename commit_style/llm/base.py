import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def strip_thinking(text: str) -> str:
    """Remove a `<think>...</think>` reasoning block some models emit despite being asked
    not to. If `<think>` opens but never closes (the whole max_tokens budget was consumed by
    reasoning), everything from `<think>` onward is dropped rather than kept, since nothing
    after an incomplete thought is trustworthy. A no-op on text with no `<think>` tag —
    suppressing the reasoning at generation time (e.g. `enable_thinking=False`) is the real
    fix; this is just a defensive second layer.
    """
    stripped = _THINK_CLOSED_RE.sub("", text)
    if stripped == text:
        stripped = _THINK_UNCLOSED_RE.sub("", text)
    return stripped.strip()


def _extract_json_object(text: str) -> str:
    """Pull out the first balanced {...} block, tolerating surrounding commentary."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _describe_field(prop: dict, defs: dict) -> object:
    if "$ref" in prop:
        return _describe_object(defs.get(prop["$ref"].split("/")[-1], {}), defs)
    if "anyOf" in prop:
        options = [p for p in prop["anyOf"] if p.get("type") != "null"]
        desc = _describe_field(options[0], defs) if options else "any"
        if len(options) != len(prop["anyOf"]):
            desc = f"{desc} or null"
        return desc
    if "enum" in prop:
        return f"one of {prop['enum']}"
    t = prop.get("type")
    if t == "object":
        return _describe_object(prop, defs)
    if t == "array":
        return [_describe_field(prop.get("items", {}), defs)]
    return t or "any"


def _describe_object(schema: dict, defs: dict) -> dict:
    return {name: _describe_field(prop, defs) for name, prop in schema.get("properties", {}).items()}


class LLMClient(ABC):
    """`last_usage` is set as a side effect of `complete()` — read it immediately after a
    call (before any concurrent/overlapping call) to get that call's
    `{input_tokens, output_tokens, latency_s}`. Any field a backend can't determine (e.g.
    Grazie doesn't return raw token counts) is `None` rather than an estimate.
    """

    last_usage: dict[str, float | None] | None = None

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        ...

    def complete_json(
        self,
        system: str,
        user: str,
        model_cls: type[BaseModel],
        max_tokens: int = 1500,
    ) -> BaseModel:
        schema = model_cls.model_json_schema()
        field_desc = _describe_object(schema, schema.get("$defs", {}))
        json_instruction = (
            f"\n\nRespond with ONLY a JSON object containing real values for exactly these "
            f"fields (shown below as field: type/allowed-values — this is a description, not "
            f"the JSON to return) — no markdown fences, no commentary:\n{json.dumps(field_desc)}"
        )
        raw = self.complete(system, user + json_instruction, max_tokens=max_tokens)
        try:
            return model_cls.model_validate_json(_extract_json_object(_strip_fences(raw)))
        except (ValidationError, json.JSONDecodeError) as exc:
            repair_user = (
                f"{user}{json_instruction}\n\nYour previous response failed to parse: "
                f"{raw!r}\nError: {exc}\nReturn ONLY corrected valid JSON."
            )
            raw_retry = self.complete(system, repair_user, max_tokens=max_tokens)
            return model_cls.model_validate_json(_extract_json_object(_strip_fences(raw_retry)))
