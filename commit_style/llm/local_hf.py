"""Local backend for a small Qwen model, e.g. on a 4GB GPU.

Also doubles as the backend for a LoRA-fine-tuned adapter (`adapter_path`), see
`commit_style/finetune/`.
"""

import time

from commit_style.llm.base import LLMClient


class LocalHFClient(LLMClient):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        adapter_path: str | None = None,
    ):
        from transformers import pipeline

        self._model_name = model_name

        if adapter_path:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base_model, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        else:
            self._pipe = pipeline("text-generation", model=model_name, device=device)

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tokenizer = self._pipe.tokenizer
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        start = time.time()
        output = self._pipe(messages, max_new_tokens=max_tokens, do_sample=False)
        latency_s = time.time() - start

        generated = output[0]["generated_text"]
        last_message = generated[-1]
        content = last_message["content"] if isinstance(last_message, dict) else str(last_message)
        output_tokens = len(tokenizer(content, add_special_tokens=False)["input_ids"])
        self.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": latency_s,
        }
        return content
