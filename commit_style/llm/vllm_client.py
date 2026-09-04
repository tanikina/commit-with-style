"""vLLM-backed local generation backend — faster inference than LocalHFClient's plain
transformers.pipeline for anything beyond a tiny stand-in model.

Generation/inference only: vLLM has no gradient-based training capability, so it never
replaces the transformers.Trainer + peft LoRA training loop in commit_style/finetune/.
"""

import time

from commit_style.llm.base import LLMClient, strip_thinking


class VLLMClient(LLMClient):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path: str | None = None,
        gpu_memory_utilization: float = 0.6,
        max_model_len: int = 4096,
        max_lora_rank: int = 32,
        quantization: str | None = None,
    ):
        from vllm import LLM

        self._model_name = model_name
        llm_kwargs = dict(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=adapter_path is not None,
            max_lora_rank=max_lora_rank,
        )
        if quantization:
            # On-the-fly quantization of a full-precision checkpoint (e.g. "bitsandbytes"
            # 4-bit NF4) shrinks weight memory ~4x, which is what actually matters for OOMs
            # during weight loading — gpu_memory_utilization only budgets memory *after* the
            # weights are already resident, so it can't fix a checkpoint that doesn't fit at
            # all. bitsandbytes also requires load_format="bitsandbytes" to quantize while
            # loading rather than expect an already-quantized checkpoint.
            llm_kwargs["quantization"] = quantization
            if quantization == "bitsandbytes":
                llm_kwargs["load_format"] = "bitsandbytes"
        self._llm = LLM(**llm_kwargs)
        self._lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        from vllm import SamplingParams
        import inspect

        start = time.time()

        # vllm 0.7.x may not support chat_template_kwargs yet
        chat_kwargs = {
            "messages": [[{"role": "system", "content": system}, {"role": "user", "content": user}]],
            "sampling_params": SamplingParams(max_tokens=max_tokens, temperature=0.2),
            "lora_request": self._lora_request,
        }

        # Only add chat_template_kwargs if supported (vllm 0.8+)
        if "chat_template_kwargs" in inspect.signature(self._llm.chat).parameters:
            # Qwen3(.5)-family hybrid-reasoning models insert a chain-of-thought preamble
            # by default, which can consume the entire max_tokens budget before any answer
            # is produced (see README).
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        [output] = self._llm.chat(**chat_kwargs)
        latency_s = time.time() - start

        self.last_usage = {
            "input_tokens": len(output.prompt_token_ids),
            "output_tokens": len(output.outputs[0].token_ids),
            "latency_s": latency_s,
        }
        return strip_thinking(output.outputs[0].text)
