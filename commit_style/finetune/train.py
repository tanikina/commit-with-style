import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from commit_style.data import Example
from commit_style.finetune.dataset import CommitSFTDataset, collate_batch

_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoraParams:
    r: int
    alpha: int
    dropout: float
    lr: float


@dataclass
class TrainResult:
    adapter_path: str
    eval_loss: float
    lora_params: LoraParams


def write_lora_params(params: LoraParams, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=2)


def load_lora_params(path: str) -> LoraParams:
    with open(path) as f:
        return LoraParams(**json.load(f))


def train_one_config(
    model_name: str,
    lora_params: LoraParams,
    train_examples: list[Example],
    val_examples: list[Example],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_length: int = 768,
    load_in_4bit: bool = False,
) -> TrainResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        # Untested this session — for the real Qwen3.5-4B target on a bigger GPU VM.
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.alpha,
            lora_dropout=lora_params.dropout,
            target_modules=_TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )

    train_dataset = CommitSFTDataset(tokenizer, train_examples, max_length=max_seq_length)
    val_dataset = CommitSFTDataset(tokenizer, val_examples, max_length=max_seq_length)

    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lora_params.lr,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(collate_batch, tokenizer),
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    model.save_pretrained(output_dir)
    return TrainResult(
        adapter_path=output_dir,
        eval_loss=eval_metrics["eval_loss"],
        lora_params=lora_params,
    )
