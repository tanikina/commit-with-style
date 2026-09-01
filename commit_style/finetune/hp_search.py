import os

from commit_style.data import Example
from commit_style.finetune.train import LoraParams, TrainResult, train_one_config

HP_GRID = [
    LoraParams(r=8, alpha=16, dropout=0.05, lr=2e-4),
    LoraParams(r=16, alpha=32, dropout=0.05, lr=1e-4),
    LoraParams(r=8, alpha=16, dropout=0.1, lr=5e-5),
]


def run_hp_search(
    model_name: str,
    train_examples: list[Example],
    val_examples: list[Example],
    output_root: str,
    **train_kwargs,
) -> list[TrainResult]:
    results = []
    for i, lora_params in enumerate(HP_GRID):
        output_dir = os.path.join(output_root, f"config_{i}")
        result = train_one_config(
            model_name,
            lora_params,
            train_examples,
            val_examples,
            output_dir,
            **train_kwargs,
        )
        print(f"[hp_search] config {i} ({lora_params}) -> eval_loss={result.eval_loss:.4f}")
        results.append(result)
    return results


def pick_best(results: list[TrainResult]) -> TrainResult:
    return min(results, key=lambda r: r.eval_loss)
