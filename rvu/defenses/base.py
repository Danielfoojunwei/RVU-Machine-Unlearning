"""Abstract base class for all prompt-injection defenses in the RVU framework.

Every concrete defense must subclass :class:`BaseDefense` and implement the
three abstract methods that define the common defense interface:

* :meth:`wrap_prompt`        -- modify prompts / context per the defense strategy
* :meth:`filter_tool_output` -- inspect and possibly modify tool outputs
* :meth:`post_episode`       -- post-episode processing (cleanup, certificates, etc.)
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def sha256_hex(data: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 encoded string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def timestamp_now() -> float:
    """Return the current POSIX timestamp as a float."""
    return time.time()


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing side-channels."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a.encode("utf-8"), b.encode("utf-8")):
        result |= x ^ y
    return result == 0


def format_tool_outputs_block(tool_outputs: list[dict]) -> str:
    """Render a list of tool-output dicts into a human-readable text block.

    Each dict is expected to have at least ``"tool_name"`` and ``"output"``
    keys.  Additional keys are silently ignored.

    Parameters
    ----------
    tool_outputs:
        List of dicts, each with at least ``tool_name`` and ``output``.

    Returns
    -------
    str
        Formatted multi-line string with one section per tool output.
    """
    if not tool_outputs:
        return ""
    parts: list[str] = []
    for idx, entry in enumerate(tool_outputs, 1):
        tool_name = entry.get("tool_name", "unknown_tool")
        output = entry.get("output", "")
        parts.append(f"[TOOL OUTPUT {idx}: {tool_name}]\n{output}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseDefense(ABC):
    """Base class for all prompt-injection defenses.

    Every concrete defense must set a ``name`` class-level attribute and
    implement the three abstract methods below.
    """

    name: str

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def wrap_prompt(
        self,
        system_prompt: str,
        user_input: str,
        tool_outputs: list[dict],
    ) -> str:
        """Modify prompts / context according to the defense strategy.

        Parameters
        ----------
        system_prompt:
            The system-level instructions for the agent.
        user_input:
            The latest user message / turn.
        tool_outputs:
            Zero or more tool-output dicts.  Each dict should contain at
            least ``"tool_name"`` (str) and ``"output"`` (str).

        Returns
        -------
        str
            The fully assembled prompt string ready for the LLM.
        """

    @abstractmethod
    def filter_tool_output(
        self,
        tool_name: str,
        output: str,
    ) -> tuple[str, dict]:
        """Inspect and possibly modify a single tool output.

        Parameters
        ----------
        tool_name:
            Name of the tool that produced *output*.
        output:
            Raw text output from the tool.

        Returns
        -------
        tuple[str, dict]
            A 2-tuple of (possibly modified output, metadata dict).
            The metadata dict carries defense-specific details (scores,
            taint IDs, provenance IDs, etc.).
        """

    @abstractmethod
    def post_episode(self, episode_log: dict) -> dict:
        """Post-episode processing hook.

        Called after an episode (one full agent interaction) completes.
        Defenses can use this for cleanup, certificate emission,
        contamination detection, or statistics aggregation.

        Parameters
        ----------
        episode_log:
            A dict describing the completed episode.  Expected keys vary
            by harness but typically include ``"episode_id"``,
            ``"turns"`` (list of turn dicts), and ``"metadata"``.

        Returns
        -------
        dict
            Defense-specific metadata produced during post-processing.
        """
