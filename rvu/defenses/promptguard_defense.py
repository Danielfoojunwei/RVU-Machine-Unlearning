"""PromptGuard-based defense using the real PromptGuardClassifier for tool output filtering.

Scores every tool output with a fine-tuned transformer injection classifier
(``rvu.inference.promptguard.PromptGuardClassifier``).  When the injection
probability exceeds a configurable threshold the output is either stripped,
blocked, or flagged depending on the configured action policy.
"""

from __future__ import annotations

from typing import Any

from rvu.defenses.base import BaseDefense, format_tool_outputs_block
from rvu.inference.promptguard import PromptGuardClassifier


class PromptGuardDefense(BaseDefense):
    """Filter tool outputs using a real PromptGuard injection classifier.

    Parameters
    ----------
    model_path:
        Path to the PromptGuard model directory (HuggingFace format).
        Passed directly to :class:`PromptGuardClassifier`.
    threshold:
        Injection probability in [0, 1] above which the defense acts.
    action:
        Policy when injection is detected:

        * ``"strip"``  -- replace the tool output with a safe placeholder.
        * ``"block"``  -- return an empty string as the output.
        * ``"flag"``   -- allow the output through but annotate metadata
          with ``flagged=True``.
    """

    name: str = "promptguard"

    _STRIP_PLACEHOLDER = "[CONTENT REMOVED: injection detected by PromptGuard]"

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.85,
        action: str = "strip",
    ) -> None:
        if action not in ("strip", "block", "flag"):
            raise ValueError(
                f"Invalid action '{action}'; expected 'strip', 'block', or 'flag'."
            )
        self.threshold = threshold
        self.action = action
        # Load the real PromptGuard classifier -- no mocks.
        self._classifier = PromptGuardClassifier(model_path=model_path)

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def wrap_prompt(
        self,
        system_prompt: str,
        user_input: str,
        tool_outputs: list[dict],
    ) -> str:
        """Assemble a standard prompt.

        PromptGuard is an input-filtering defense; prompt layout is the
        same as the vanilla baseline.  The real filtering happens in
        :meth:`filter_tool_output`.
        """
        parts: list[str] = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}\n")
        tool_block = format_tool_outputs_block(tool_outputs)
        if tool_block:
            parts.append(tool_block + "\n")
        if user_input:
            parts.append(f"[USER]\n{user_input}\n")
        return "\n".join(parts)

    def filter_tool_output(
        self,
        tool_name: str,
        output: str,
    ) -> tuple[str, dict]:
        """Score *output* with the PromptGuard classifier and act if above threshold.

        Returns
        -------
        tuple[str, dict]
            The (possibly modified) output and a metadata dict containing
            the injection score, detection flag, and action taken.
        """
        score: float = self._classifier.score_injection(output)
        detected: bool = score >= self.threshold

        metadata: dict[str, Any] = {
            "defense": self.name,
            "tool_name": tool_name,
            "injection_score": score,
            "threshold": self.threshold,
            "detected": detected,
        }

        if not detected:
            metadata["action"] = "allow"
            return output, metadata

        # Injection detected -- apply configured policy.
        if self.action == "block":
            metadata["action"] = "block"
            metadata["reason"] = "blocked by PromptGuard"
            return "", metadata

        if self.action == "strip":
            metadata["action"] = "strip"
            metadata["reason"] = "stripped by PromptGuard"
            return self._STRIP_PLACEHOLDER, metadata

        # action == "flag"
        metadata["action"] = "flag"
        metadata["reason"] = "flagged by PromptGuard (allowed through)"
        metadata["flagged"] = True
        return output, metadata

    def post_episode(self, episode_log: dict) -> dict:
        """Aggregate injection detection statistics for the episode.

        Iterates over the episode turns and counts how many tool outputs
        were scored and how many were flagged / blocked / stripped.
        """
        episode_id = episode_log.get("episode_id", "unknown")
        turns = episode_log.get("turns", [])

        total_outputs = 0
        detections = 0

        for turn in turns:
            tool_results = turn.get("tool_results", [])
            for result in tool_results:
                total_outputs += 1
                meta = result.get("defense_metadata", {})
                if meta.get("detected", False):
                    detections += 1

        return {
            "defense": self.name,
            "episode_id": episode_id,
            "total_tool_outputs_scanned": total_outputs,
            "injection_detections": detections,
            "detection_rate": detections / total_outputs if total_outputs > 0 else 0.0,
            "threshold": self.threshold,
            "action_policy": self.action,
        }
