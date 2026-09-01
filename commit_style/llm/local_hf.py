"""Local backend for a small Qwen model, e.g. on a 4GB GPU.

Also doubles as the backend for a LoRA-fine-tuned adapter (`adapter_path`), see
`commit_style/finetune/`.
"""

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
        output = self._pipe(messages, max_new_tokens=max_tokens, do_sample=False)
        generated = output[0]["generated_text"]
        last_message = generated[-1]
        return last_message["content"] if isinstance(last_message, dict) else str(last_message)
