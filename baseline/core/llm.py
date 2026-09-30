from __future__ import annotations

import os
import random
import time
from typing import List, Optional

from .types import Generation, ModelConfig


class LLMClient:
    def __init__(
        self,
        model_config: ModelConfig,
        backend: str = "vllm",
        dtype: str = "auto",
        gpu: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        enforce_eager: bool = False,
    ) -> None:
        if gpu is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.model_config = model_config
        self.backend = backend
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size or model_config.tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len or model_config.max_model_len
        self.enforce_eager = enforce_eager
        self._load()

    def _load(self) -> None:
        if self.backend == "vllm":
            try:
                from transformers import AutoTokenizer
                from vllm import LLM
            except Exception as e:
                raise RuntimeError(
                    "vLLM backend requested but vllm/transformers is not importable. "
                    "Run with an environment that has vLLM, or pass --backend transformers."
                ) from e

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, trust_remote_code=True)
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.stop_token_ids = self._resolve_stop_token_ids()
            resolved_dtype = self.dtype
            if resolved_dtype == "auto":
                resolved_dtype = "bfloat16"
            self.model = LLM(
                model=self.model_config.path,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=resolved_dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enforce_eager=self.enforce_eager,
            )
            return

        if self.backend == "transformers":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as e:
                raise RuntimeError("Transformers backend requested but torch/transformers is not importable.") from e

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, trust_remote_code=True)
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.stop_token_ids = self._resolve_stop_token_ids()
            self.tokenizer.padding_side = "left"
            torch_dtype = torch.bfloat16 if self.dtype in {"auto", "bfloat16"} else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            self.model.eval()
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def _resolve_stop_token_ids(self) -> List[int]:
        stop_ids = set()
        for token_id in (getattr(self.tokenizer, "eos_token_id", None), getattr(self.tokenizer, "pad_token_id", None)):
            if isinstance(token_id, int) and token_id >= 0:
                stop_ids.add(token_id)
        for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0 and token_id != getattr(self.tokenizer, "unk_token_id", None):
                stop_ids.add(token_id)
        return sorted(stop_ids)

    def _chat_text(self, system_prompt: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            pass
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{system_prompt}\n\n{prompt}"

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(str(text), add_special_tokens=False))
        except Exception:
            return len(str(text).split())

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        seed: int,
        call_type: str,
        max_tokens: int,
        temperature: float = 0.1,
        top_p: float = 1.0,
    ) -> Generation:
        random.seed(seed)
        text_input = self._chat_text(system_prompt, prompt)
        input_tokens = self.count_tokens(text_input)
        start = time.time()
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                stop_token_ids=self.stop_token_ids or None,
            )
            outs = self.model.generate([text_input], params)
            raw = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            out_tokens = len(getattr(outs[0].outputs[0], "token_ids", []) or []) if outs and outs[0].outputs else self.count_tokens(raw)
        else:
            torch = self.torch
            encoded = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_ids = out[0][encoded["input_ids"].shape[1] :]
            raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            out_tokens = int(new_ids.numel())
        end = time.time()
        return Generation(
            text=str(raw).strip(),
            input_tokens=int(input_tokens),
            output_tokens=int(out_tokens),
            latency=float(end - start),
            seed=int(seed),
            call_type=call_type,
            start_time=float(start),
            end_time=float(end),
            prompt=prompt,
        )
