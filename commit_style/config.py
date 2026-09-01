from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    backend: str  # "mock" | "grazie" | "local_hf"
    model_name: str = ""
    temperature: float = 0.0
    grazie_token: str | None = None


@dataclass
class PipelineConfig:
    data_path: str
    n_samples: int = 12
    seed: int = 0
    languages: list[str] | None = None
    max_history_messages: int = 5
    max_diff_chars: int = 4000
    strategies: list[str] = field(
        default_factory=lambda: ["vanilla", "structured", "reference_guided"]
    )
    output_path: str = "outputs/dry_run.jsonl"
    with_bertscore: bool = False
