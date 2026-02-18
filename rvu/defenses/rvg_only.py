"""RVG-only defense -- Verifier-gated tool boundary with allowlists and taint tracking.

Loads a tool allowlist from YAML config.  Every tool output is checked
against the allowlist; outputs from disallowed tools are blocked.  Taint
propagation is tracked: if a tool output is tainted, downstream uses
inherit the taint.

This defense does NOT perform state recovery -- it is a pure gating layer.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from rvu.defenses.base import BaseDefense, format_tool_outputs_block, sha256_hex


class RVGOnlyDefense(BaseDefense):
    """Verifier-gated tool boundary with scope allowlists and taint propagation.

    Parameters
    ----------
    allowlist_path:
        Path to a YAML file that defines ``allowed_tools`` (categorised
        dict of lists) and ``blocked_patterns`` (list of strings).
        See ``configs/tool_allowlist.yaml``.
    taint_propagation:
        When ``True``, track taint labels through tool outputs and mark
        any content derived from a tainted source as tainted.
    """

    name: str = "rvg"

    _BLOCKED_PLACEHOLDER = "[TOOL OUTPUT BLOCKED: tool not on allowlist]"

    def __init__(
        self,
        allowlist_path: str,
        taint_propagation: bool = True,
    ) -> None:
        self.taint_propagation = taint_propagation

        # Load and parse the YAML allowlist.
        resolved = Path(allowlist_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Tool allowlist YAML not found: {resolved}"
            )
        with open(resolved, "r") as fh:
            config = yaml.safe_load(fh)

        # Build a flat set of allowed tool names from categorised config.
        self._allowed_tools: set[str] = set()
        for _category, tools in config.get("allowed_tools", {}).items():
            if isinstance(tools, list):
                self._allowed_tools.update(tools)

        # Blocked patterns -- regexes that always cause rejection.
        raw_patterns: list[str] = config.get("blocked_patterns", [])
        self._blocked_patterns: list[re.Pattern[str]] = [
            re.compile(re.escape(p), re.IGNORECASE) for p in raw_patterns
        ]

        # ----- Taint tracking state -----
        # Maps taint_id -> {source, snippet, content_hash}
        self._taint_registry: dict[str, dict[str, str]] = {}
        # Maps content-key (lowercase prefix) -> taint_id for fast lookup.
        self._taint_content_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_key(content: str) -> str:
        """Short deterministic key for content (first 200 chars, lowered)."""
        return content[:200].strip().lower()

    def _register_taint(self, tool_name: str, content: str) -> str:
        """Register a taint label for *content* from *tool_name*.

        Returns the newly created taint ID.
        """
        taint_id = f"taint-{uuid.uuid4().hex[:12]}"
        key = self._content_key(content)
        self._taint_registry[taint_id] = {
            "source": tool_name,
            "snippet": content[:200],
            "content_hash": sha256_hex(content),
        }
        self._taint_content_index[key] = taint_id
        return taint_id

    def _find_taints(self, text: str) -> list[str]:
        """Return taint IDs whose content keys appear in *text*."""
        hits: list[str] = []
        text_lower = text.lower()
        for key, taint_id in self._taint_content_index.items():
            if key and key in text_lower:
                hits.append(taint_id)
        return hits

    def _check_blocked_patterns(self, text: str) -> str | None:
        """Return the first blocked pattern found in *text*, or None."""
        for pattern in self._blocked_patterns:
            if pattern.search(text):
                return pattern.pattern
        return None

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def wrap_prompt(
        self,
        system_prompt: str,
        user_input: str,
        tool_outputs: list[dict],
    ) -> str:
        """Add a verifier preamble about tool restrictions to the system prompt."""
        verifier_preamble = (
            "TOOL USAGE RESTRICTIONS (enforced by verifier):\n"
            f"  Allowed tools: {sorted(self._allowed_tools)}\n"
            "  All tool calls are verified before execution.\n"
            "  Calls to tools not on the allowlist will be REJECTED.\n"
            "  Do NOT attempt to call tools outside the allowlist.\n"
        )

        parts: list[str] = []
        augmented_system = (
            f"{verifier_preamble}\n{system_prompt}"
            if system_prompt
            else verifier_preamble
        )
        parts.append(f"[SYSTEM]\n{augmented_system}\n")

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
        """Check *tool_name* against the allowlist and apply taint tracking.

        A tool output is **blocked** if:

        1. The tool name is not in the allowlist.
        2. The output matches a blocked pattern.

        If ``taint_propagation`` is enabled, outputs from external/untrusted
        tools receive a taint label.  Any downstream content containing
        tainted substrings is also marked tainted.
        """
        metadata: dict[str, Any] = {
            "defense": self.name,
            "tool_name": tool_name,
        }

        # 1. Allowlist check.
        if tool_name not in self._allowed_tools:
            metadata["action"] = "block"
            metadata["reason"] = f"tool '{tool_name}' not in allowlist"
            metadata["tainted"] = True
            # Still register taint so downstream propagation works.
            if self.taint_propagation:
                taint_id = self._register_taint(tool_name, output)
                metadata["taint_id"] = taint_id
            return self._BLOCKED_PLACEHOLDER, metadata

        # 2. Blocked-pattern check on the output content.
        blocked = self._check_blocked_patterns(output)
        if blocked:
            metadata["action"] = "block"
            metadata["reason"] = f"blocked pattern detected in output: {blocked}"
            metadata["tainted"] = True
            if self.taint_propagation:
                taint_id = self._register_taint(tool_name, output)
                metadata["taint_id"] = taint_id
            return self._BLOCKED_PLACEHOLDER, metadata

        # 3. Taint propagation check.
        taint_hits = self._find_taints(output) if self.taint_propagation else []
        if taint_hits:
            # Output contains content from a previously tainted source.
            metadata["action"] = "flag"
            metadata["tainted"] = True
            metadata["inherited_taint_ids"] = taint_hits
            metadata["reason"] = (
                f"output contains tainted content (taint IDs: {taint_hits})"
            )
            # Register new taint for this output as well.
            taint_id = self._register_taint(tool_name, output)
            metadata["taint_id"] = taint_id
            return output, metadata

        # 4. All clear.
        metadata["action"] = "allow"
        metadata["tainted"] = False
        return output, metadata

    def post_episode(self, episode_log: dict) -> dict:
        """Summarise gating and taint tracking stats, then clear taint state."""
        episode_id = episode_log.get("episode_id", "unknown")

        total_outputs = 0
        blocked_count = 0
        tainted_count = 0

        for turn in episode_log.get("turns", []):
            for result in turn.get("tool_results", []):
                total_outputs += 1
                meta = result.get("defense_metadata", {})
                if meta.get("action") == "block":
                    blocked_count += 1
                if meta.get("tainted", False):
                    tainted_count += 1

        summary = {
            "defense": self.name,
            "episode_id": episode_id,
            "total_tool_outputs": total_outputs,
            "blocked_outputs": blocked_count,
            "tainted_outputs": tainted_count,
            "taint_entries_cleared": len(self._taint_registry),
            "allowed_tools": sorted(self._allowed_tools),
        }

        # Clear taint state between episodes.
        self._taint_registry.clear()
        self._taint_content_index.clear()

        return summary

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def allowed_tools(self) -> set[str]:
        """Return a copy of the current tool allowlist."""
        return set(self._allowed_tools)

    @property
    def taint_registry(self) -> dict[str, dict[str, str]]:
        """Return a copy of the current taint registry."""
        return dict(self._taint_registry)

    def clear_taint(self) -> int:
        """Remove all taint labels.  Returns the count of cleared entries."""
        count = len(self._taint_registry)
        self._taint_registry.clear()
        self._taint_content_index.clear()
        return count
