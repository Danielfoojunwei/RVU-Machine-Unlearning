"""PromptGuard injection classifier using HuggingFace transformers on CPU."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PromptGuardClassifier:
    """Classifies prompts as benign or injection using a fine-tuned transformer."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"PromptGuard model directory not found: {resolved}\n"
                "Ensure the model has been downloaded and the path is correct."
            )

        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(str(resolved))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(resolved)
        ).to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_proba(self, texts: list[str]) -> torch.Tensor:
        """Return softmax probabilities for a batch of texts.

        Returns a tensor of shape ``(len(texts), num_labels)``.
        """
        encodings = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**encodings).logits

        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_injection(self, text: str) -> float:
        """Return the injection probability in [0, 1] for a single string."""
        probs = self._predict_proba([text])
        # The injection class is the last label (highest index).
        return probs[0, -1].item()

    def is_injection(self, text: str, threshold: float = 0.85) -> bool:
        """Return True if the injection score exceeds *threshold*."""
        return self.score_injection(text) >= threshold

    def score_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[float]:
        """Score a list of texts in mini-batches and return injection probabilities."""
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            probs = self._predict_proba(batch)
            scores.extend(probs[:, -1].tolist())
        return scores


# ------------------------------------------------------------------
# Config-driven factory
# ------------------------------------------------------------------


def load_from_config(
    config_path: str = "configs/models.yaml",
) -> PromptGuardClassifier:
    """Instantiate a PromptGuardClassifier from a models.yaml config."""
    cfg_file = Path(config_path).expanduser().resolve()
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    with open(cfg_file, "r") as fh:
        config = yaml.safe_load(fh)

    if "guard" not in config or "prompt_guard" not in config["guard"]:
        raise KeyError(
            f"'guard.prompt_guard' section not found in {cfg_file}."
        )

    entry = config["guard"]["prompt_guard"]
    model_dir = Path(entry["local_dir"])

    # Resolve relative paths against the project root (parent of configs/).
    if not model_dir.is_absolute():
        model_dir = cfg_file.parent.parent / model_dir

    return PromptGuardClassifier(model_path=str(model_dir))
