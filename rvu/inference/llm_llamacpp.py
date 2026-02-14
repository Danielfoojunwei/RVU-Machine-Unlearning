"""llama.cpp CPU inference adapter using llama-cpp-python."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from llama_cpp import Llama


class LlamaCppLLM:
    """Wraps a GGUF model loaded via llama-cpp-python for deterministic CPU inference."""

    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        seed: int = 42,
    ) -> None:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"GGUF model file not found: {resolved}\n"
                "Ensure the model has been downloaded and the path is correct."
            )

        self.model_path = str(resolved)
        self.context_length = context_length
        self.seed = seed

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_length,
            seed=self.seed,
            n_threads=os.cpu_count() or 4,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Run text completion and return the generated string."""
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=stop or [],
            seed=self.seed,
        )
        return output["choices"][0]["text"]

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Run chat completion and return the assistant reply."""
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=stop or [],
            seed=self.seed,
        )
        return output["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# Config-driven factory
# ------------------------------------------------------------------


def load_model_from_config(
    model_key: str,
    config_path: str = "configs/models.yaml",
) -> LlamaCppLLM:
    """Instantiate a LlamaCppLLM from a models.yaml entry.

    Parameters
    ----------
    model_key:
        Key under the top-level ``llm`` mapping (e.g. ``"model_a"``).
    config_path:
        Path to the YAML config file, resolved relative to the project root.
    """
    cfg_file = Path(config_path).expanduser().resolve()
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    with open(cfg_file, "r") as fh:
        config = yaml.safe_load(fh)

    if "llm" not in config or model_key not in config["llm"]:
        available = list(config.get("llm", {}).keys())
        raise KeyError(
            f"Model key '{model_key}' not found under 'llm' in {cfg_file}. "
            f"Available keys: {available}"
        )

    entry = config["llm"][model_key]
    local_dir = Path(entry["local_dir"])
    gguf_file = entry["gguf_file"]
    model_path = local_dir / gguf_file

    # If the path is relative, resolve it against the config file's parent
    # (assumed to be the project root's configs/ directory).
    if not model_path.is_absolute():
        model_path = cfg_file.parent.parent / model_path

    sampling = config.get("sampling", {}).get("deterministic", {})
    context_length = entry.get("context_length", 4096)
    seed = sampling.get("seed", 42)

    return LlamaCppLLM(
        model_path=str(model_path),
        context_length=context_length,
        seed=seed,
    )
