from pathlib import Path

from commit_style.llm.base import LLMClient
from commit_style.llm.mock import MockLLMClient


def load_grazie_token(path: str) -> str:
    raw = Path(path).read_text().strip()
    if "=" in raw:
        # supports a shell-export style file, e.g. GRAZIE_JWT_TOKEN="..."
        raw = raw.split("=", 1)[1].strip()
    return raw.strip().strip('"').strip("'")


def build_llm(
    backend: str,
    model_name: str,
    grazie_token_file: str | None,
    adapter_path: str | None = None,
) -> LLMClient:
    if backend == "mock":
        return MockLLMClient()
    if backend == "grazie":
        from commit_style.llm.grazie_client import GrazieLLMClient

        token = load_grazie_token(grazie_token_file) if grazie_token_file else None
        return GrazieLLMClient(model_name=model_name, token=token)
    if backend == "local_hf":
        from commit_style.llm.local_hf import LocalHFClient

        return LocalHFClient(
            model_name=model_name or "Qwen/Qwen2.5-0.5B-Instruct",
            adapter_path=adapter_path,
        )
    if backend == "vllm":
        from commit_style.llm.vllm_client import VLLMClient

        return VLLMClient(
            model_name=model_name or "Qwen/Qwen2.5-0.5B-Instruct",
            adapter_path=adapter_path,
        )
    raise ValueError(f"Unknown backend: {backend}")


def judge_id(backend: str, model: str) -> str:
    return model or backend


def build_judges(
    judge_specs: list[str] | None,
    grazie_token_file: str | None,
    fallback_backend: str,
    fallback_model: str,
) -> dict[str, LLMClient]:
    """Parse repeatable `--judge BACKEND:MODEL` specs into {judge_id: LLMClient}.

    An empty/None `judge_specs` falls back to one judge built from
    `(fallback_backend, fallback_model)`, preserving each script's current default.
    """
    specs = judge_specs or [f"{fallback_backend}:{fallback_model}"]
    judges = {}
    for spec in specs:
        backend, _, model = spec.partition(":")
        judges[judge_id(backend, model)] = build_llm(backend, model, grazie_token_file)
    return judges
