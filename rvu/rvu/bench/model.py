from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Protocol

from .config import ModelConfig


class Planner(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class HFPlanner:
    model_id: str
    revision: str
    dtype: str
    max_tokens: int

    def __post_init__(self) -> None:
        self._require("transformers")
        self._require("torch")
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        tokenizer_cls = transformers.AutoTokenizer
        model_cls = transformers.AutoModelForCausalLM
        torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        self._tokenizer = tokenizer_cls.from_pretrained(self.model_id, revision=self.revision)
        self._model = model_cls.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    @staticmethod
    def _require(module_name: str) -> None:
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(f"Missing dependency: {module_name}")

    def generate(self, prompt: str) -> str:
        importlib.import_module("torch")
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**encoded, max_new_tokens=self.max_tokens)
        return str(self._tokenizer.decode(output[0], skip_special_tokens=True))


@dataclass
class VLLMPlanner:
    endpoint: str
    model_id: str

    def __post_init__(self) -> None:
        self._require("openai")
        openai = importlib.import_module("openai")
        self._client = openai.OpenAI(base_url=self.endpoint, api_key="EMPTY")

    @staticmethod
    def _require(module_name: str) -> None:
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(f"Missing dependency: {module_name}")

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return str(content or "")


def build_planner(config: ModelConfig) -> Planner:
    if config.provider == "vllm":
        if config.endpoint is None:
            raise ValueError("vLLM provider requires model.endpoint")
        return VLLMPlanner(endpoint=config.endpoint, model_id=config.model_id)
    if config.provider == "transformers":
        return HFPlanner(
            model_id=config.model_id,
            revision=config.revision,
            dtype=config.dtype,
            max_tokens=config.max_tokens,
        )
    raise ValueError(f"Unsupported provider {config.provider}")
