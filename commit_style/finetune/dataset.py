from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from commit_style.data import Example, format_history
from commit_style.strategies import _VANILLA_SYSTEM


def _render_user(example: Example) -> str:
    return (
        f"HISTORY:\n{format_history(example.history)}\n\n"
        f"DIFF:\n{example.diff}\n\n"
        f"PREFIX:\n{example.prefix}"
    )


def build_sft_features(tokenizer, example: Example, max_length: int = 1536) -> dict:
    """Render the vanilla-strategy prompt + completion, masking the prompt in `labels`.

    If the combined length exceeds `max_length`, the *prompt* is truncated from the left
    (dropping the earliest diff/history content) so the completion is never cut.
    """
    user = _render_user(example)
    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _VANILLA_SYSTEM},
            {"role": "user", "content": user},
        ],
        add_generation_prompt=True,
        tokenize=True,
    )
    # In transformers 5.x+, apply_chat_template with tokenize=True returns a list directly
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    completion_ids = tokenizer(
        example.reference_completion, add_special_tokens=False
    )["input_ids"] + [tokenizer.eos_token_id]

    overflow = len(prompt_ids) + len(completion_ids) - max_length
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {"input_ids": input_ids, "labels": labels}


class CommitSFTDataset(Dataset):
    def __init__(self, tokenizer, examples: list[Example], max_length: int = 1536):
        self._tokenizer = tokenizer
        self._examples = examples
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        return build_sft_features(self._tokenizer, self._examples[idx], max_length=self._max_length)


def collate_batch(tokenizer, features: list[dict]) -> dict:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
