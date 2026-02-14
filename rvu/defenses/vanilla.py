"""Vanilla (no-op) defense baseline -- passes everything through unchanged.

This serves as the control condition in defense evaluations: the agent
operates exactly as it would without any defense layer.  All three
interface methods are identity / passthrough operations.
"""

from __future__ import annotations

from rvu.defenses.base import BaseDefense, format_tool_outputs_block


class VanillaDefense(BaseDefense):
    """No-filtering baseline.  All content is allowed without modification."""

    name: str = "vanilla"

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def wrap_prompt(
        self,
        system_prompt: str,
        user_input: str,
        tool_outputs: list[dict],
    ) -> str:
        """Assemble a standard prompt without any defense-specific wrappers.

        The prompt is laid out as::

            [SYSTEM]
            <system_prompt>

            [TOOL OUTPUT 1: <tool_name>]
            <output>
            ...

            [USER]
            <user_input>
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
        """Return the output unchanged with minimal metadata."""
        metadata = {
            "defense": self.name,
            "tool_name": tool_name,
            "action": "allow",
        }
        return output, metadata

    def post_episode(self, episode_log: dict) -> dict:
        """No post-episode processing -- return a trivial summary."""
        return {
            "defense": self.name,
            "episode_id": episode_log.get("episode_id", "unknown"),
            "action": "none",
        }
