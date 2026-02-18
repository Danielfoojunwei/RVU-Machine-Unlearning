"""Full RVU (Recovery and Verification Utility) defense implementation.

Combines:
* **Provenance tracking** -- SQLite database logging all tool I/O, memory
  entries, timestamps, and parent pointers.
* **Embedding-based retrieval** -- FAISS (faiss-cpu) index for similarity
  search over logged content.
* **Contamination detection** -- finds compromised entries via embedding
  similarity against incident indicators.
* **Provenance closure** -- follows parent pointers transitively to compute
  the full set of entries affected by contamination.
* **Purge / rollback** -- marks entries as purged in the DB and removes
  their vectors from FAISS.
* **Certificate emission / verification** -- creates a signed certificate
  (SHA-256 hash of purge manifest) that an auditor can verify.
* **Allowlist gating** -- inherits tool-allowlist logic from RVG.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from rvu.defenses.base import (
    BaseDefense,
    constant_time_compare,
    format_tool_outputs_block,
    sha256_hex,
    timestamp_now,
)

# ---------------------------------------------------------------------------
# SQL DDL -- provenance table per the spec
# ---------------------------------------------------------------------------

_DDL_PROVENANCE = """\
CREATE TABLE IF NOT EXISTS provenance (
    entry_id     TEXT PRIMARY KEY,
    episode_id   TEXT,
    action_type  TEXT,
    content      TEXT,
    content_hash TEXT,
    parent_id    TEXT,
    timestamp    REAL,
    tainted      INTEGER DEFAULT 0,
    purged       INTEGER DEFAULT 0
);
"""


class RVUDefense(BaseDefense):
    """Full Recovery and Verification Utility defense.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file for the provenance log.
    faiss_index_path:
        Path where the FAISS index is stored / loaded from.
    embedding_model_path:
        Name or local directory of a ``sentence-transformers`` model used
        to embed content for similarity-based retrieval.
    allowlist_path:
        Path to YAML tool allowlist (same format as RVG).  If ``None``,
        all tools are allowed.
    certificate_dir:
        Directory where emitted JSON certificates are saved.
    similarity_threshold:
        Cosine-similarity threshold for contamination detection.
        Entries with similarity >= this value to an indicator are
        considered contaminated.
    closure_max_depth:
        Default maximum BFS depth for :meth:`compute_closure`.
    """

    name: str = "rvu"

    def __init__(
        self,
        db_path: str,
        faiss_index_path: str,
        embedding_model_path: str,
        allowlist_path: str | None = None,
        certificate_dir: str = "artifacts/certificates",
        similarity_threshold: float = 0.85,
        closure_max_depth: int = 10,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.faiss_index_path = Path(faiss_index_path).expanduser().resolve()
        self.certificate_dir = Path(certificate_dir).expanduser().resolve()
        self.similarity_threshold = similarity_threshold
        self.closure_max_depth = closure_max_depth

        # Ensure directories exist.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.certificate_dir.mkdir(parents=True, exist_ok=True)

        # ---- SQLite setup ----
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_DDL_PROVENANCE)
        self._conn.commit()

        # ---- Embedding model (sentence-transformers) ----
        self._embedder = SentenceTransformer(embedding_model_path)
        self._embedding_dim: int = self._embedder.get_sentence_embedding_dimension()

        # ---- FAISS index (inner product on L2-normalised vectors) ----
        if self.faiss_index_path.is_file():
            self._index = faiss.read_index(str(self.faiss_index_path))
        else:
            self._index = faiss.IndexFlatIP(self._embedding_dim)

        # Maps FAISS row position -> provenance entry_id.
        self._faiss_id_map: list[str] = self._load_faiss_id_map()

        # ---- Tool allowlist (optional, RVG-style) ----
        self._allowed_tools: set[str] | None = None
        self._blocked_patterns: list[re.Pattern[str]] = []
        if allowlist_path is not None:
            self._load_allowlist(allowlist_path)

        # Current episode ID (set at the start of each episode).
        self._current_episode_id: str = f"ep-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Allowlist loading (RVG-inherited)
    # ------------------------------------------------------------------

    def _load_allowlist(self, allowlist_path: str) -> None:
        """Load tool allowlist and blocked patterns from a YAML file."""
        resolved = Path(allowlist_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Tool allowlist YAML not found: {resolved}"
            )
        with open(resolved, "r") as fh:
            config = yaml.safe_load(fh)

        self._allowed_tools = set()
        for _category, tools in config.get("allowed_tools", {}).items():
            if isinstance(tools, list):
                self._allowed_tools.update(tools)

        raw_patterns: list[str] = config.get("blocked_patterns", [])
        self._blocked_patterns = [
            re.compile(re.escape(p), re.IGNORECASE) for p in raw_patterns
        ]

    # ------------------------------------------------------------------
    # FAISS id-map persistence
    # ------------------------------------------------------------------

    @property
    def _faiss_id_map_path(self) -> Path:
        return self.faiss_index_path.with_suffix(".idmap.json")

    def _load_faiss_id_map(self) -> list[str]:
        if self._faiss_id_map_path.is_file():
            with open(self._faiss_id_map_path, "r") as fh:
                return json.load(fh)
        return []

    def _save_faiss_id_map(self) -> None:
        with open(self._faiss_id_map_path, "w") as fh:
            json.dump(self._faiss_id_map, fh)

    def _persist_faiss(self) -> None:
        """Write FAISS index and id-map to disk."""
        faiss.write_index(self._index, str(self.faiss_index_path))
        self._save_faiss_id_map()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Return L2-normalised embedding vector for *text*."""
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    # ------------------------------------------------------------------
    # Provenance DB operations
    # ------------------------------------------------------------------

    def record_action(
        self,
        action_type: str,
        content: str,
        parent_id: str | None = None,
    ) -> str:
        """Log an action to the provenance DB and FAISS index.

        Parameters
        ----------
        action_type:
            One of ``'tool_call'``, ``'tool_output'``, ``'memory_write'``,
            ``'retrieval'``.
        content:
            The text content to log.
        parent_id:
            Optional entry_id of the parent provenance record (for
            building the provenance DAG).

        Returns
        -------
        str
            The newly created ``entry_id``.
        """
        entry_id = str(uuid.uuid4())
        now = timestamp_now()
        content_hash = sha256_hex(content)

        self._conn.execute(
            "INSERT INTO provenance "
            "(entry_id, episode_id, action_type, content, content_hash, "
            " parent_id, timestamp, tainted, purged) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
            (
                entry_id,
                self._current_episode_id,
                action_type,
                content,
                content_hash,
                parent_id,
                now,
            ),
        )
        self._conn.commit()

        # Embed and add to FAISS.
        vec = self._embed(content)
        self._index.add(vec)
        self._faiss_id_map.append(entry_id)
        self._persist_faiss()

        return entry_id

    def _get_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Fetch a provenance row as a dict, or None if not found."""
        cur = self._conn.execute(
            "SELECT entry_id, episode_id, action_type, content, content_hash, "
            "       parent_id, timestamp, tainted, purged "
            "FROM provenance WHERE entry_id = ?",
            (entry_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "entry_id": row[0],
            "episode_id": row[1],
            "action_type": row[2],
            "content": row[3],
            "content_hash": row[4],
            "parent_id": row[5],
            "timestamp": row[6],
            "tainted": row[7],
            "purged": row[8],
        }

    def _get_children(self, entry_id: str) -> list[str]:
        """Return entry_ids that have *entry_id* as their parent_id."""
        cur = self._conn.execute(
            "SELECT entry_id FROM provenance WHERE parent_id = ? AND purged = 0",
            (entry_id,),
        )
        return [row[0] for row in cur]

    # ------------------------------------------------------------------
    # Contamination detection
    # ------------------------------------------------------------------

    def detect_contamination(self, indicators: list[str]) -> set[str]:
        """Find contaminated entry IDs via embedding similarity.

        Parameters
        ----------
        indicators:
            Strings representing known-compromised content (substrings,
            content hashes, or free-text descriptions of malicious payloads).

        Returns
        -------
        set[str]
            Entry IDs whose embeddings are above the similarity threshold
            to any indicator, plus direct content-hash and substring matches.
        """
        contaminated: set[str] = set()

        for indicator in indicators:
            # 1. Direct entry_id match.
            if self._get_entry(indicator) is not None:
                contaminated.add(indicator)
                continue

            # 2. Content-hash match.
            cur = self._conn.execute(
                "SELECT entry_id FROM provenance "
                "WHERE content_hash = ? AND purged = 0",
                (indicator,),
            )
            for row in cur:
                contaminated.add(row[0])

            # 3. Substring match in content.
            cur = self._conn.execute(
                "SELECT entry_id FROM provenance "
                "WHERE content LIKE ? AND purged = 0",
                (f"%{indicator}%",),
            )
            for row in cur:
                contaminated.add(row[0])

            # 4. Embedding similarity search via FAISS.
            if self._index.ntotal > 0:
                vec = self._embed(indicator)
                k = min(20, self._index.ntotal)
                distances, indices = self._index.search(vec, k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self._faiss_id_map):
                        continue
                    if dist >= self.similarity_threshold:
                        candidate_id = self._faiss_id_map[int(idx)]
                        entry = self._get_entry(candidate_id)
                        if entry and not entry["purged"]:
                            contaminated.add(candidate_id)

        return contaminated

    # ------------------------------------------------------------------
    # Provenance closure
    # ------------------------------------------------------------------

    def compute_closure(
        self,
        contaminated_ids: set[str],
        max_depth: int = 10,
    ) -> set[str]:
        """Follow parent pointers transitively to compute the full closure.

        Uses BFS in both directions (parent -> children AND child -> parent)
        up to *max_depth* hops.

        Parameters
        ----------
        contaminated_ids:
            Initial seed set of entry IDs.
        max_depth:
            Maximum BFS depth (default 10).

        Returns
        -------
        set[str]
            All entry IDs reachable from the contamination set.
        """
        closure: set[str] = set(contaminated_ids)
        queue: deque[tuple[str, int]] = deque(
            (eid, 0) for eid in contaminated_ids
        )

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Follow children (entries that cite current_id as parent).
            for child_id in self._get_children(current_id):
                if child_id not in closure:
                    closure.add(child_id)
                    queue.append((child_id, depth + 1))

            # Follow parent edge.
            entry = self._get_entry(current_id)
            if entry and entry["parent_id"] and entry["parent_id"] not in closure:
                parent_id = entry["parent_id"]
                closure.add(parent_id)
                queue.append((parent_id, depth + 1))

        return closure

    # ------------------------------------------------------------------
    # Purge / rollback
    # ------------------------------------------------------------------

    def purge(self, closure_ids: set[str]) -> dict:
        """Mark entries as purged in the DB and remove from FAISS.

        Parameters
        ----------
        closure_ids:
            Set of entry_ids to purge.

        Returns
        -------
        dict
            Purge log with counts and lists of affected IDs.
        """
        purge_log: dict[str, Any] = {
            "purged_entry_ids": [],
            "faiss_vectors_removed": 0,
            "timestamp": timestamp_now(),
        }

        for entry_id in closure_ids:
            entry = self._get_entry(entry_id)
            if entry is None or entry["purged"]:
                continue

            # Mark as purged and tainted in the DB.
            self._conn.execute(
                "UPDATE provenance SET purged = 1, tainted = 1 WHERE entry_id = ?",
                (entry_id,),
            )
            purge_log["purged_entry_ids"].append(entry_id)

        self._conn.commit()

        # Rebuild the FAISS index without the purged vectors.
        remaining_ids: list[str] = []
        remaining_vecs: list[np.ndarray] = []

        for idx, fid in enumerate(self._faiss_id_map):
            if fid not in closure_ids:
                vec = np.zeros((1, self._embedding_dim), dtype=np.float32)
                self._index.reconstruct(idx, vec.ravel())
                remaining_vecs.append(vec)
                remaining_ids.append(fid)
            else:
                purge_log["faiss_vectors_removed"] += 1

        new_index = faiss.IndexFlatIP(self._embedding_dim)
        if remaining_vecs:
            stacked = np.vstack(remaining_vecs)
            new_index.add(stacked)
        self._index = new_index
        self._faiss_id_map = remaining_ids
        self._persist_faiss()

        return purge_log

    # ------------------------------------------------------------------
    # Certificate emission
    # ------------------------------------------------------------------

    def emit_certificate(
        self,
        episode_id: str,
        closure_ids: set[str],
    ) -> dict:
        """Create a signed certificate with SHA-256 hash of the purge manifest.

        Parameters
        ----------
        episode_id:
            The episode that triggered the purge.
        closure_ids:
            The set of entry IDs that were purged.

        Returns
        -------
        dict
            The certificate dict, including ``certificate_id``,
            ``signature``, and ``certificate_path``.
        """
        cert_id = f"cert-{uuid.uuid4().hex[:16]}"
        now = timestamp_now()

        # Build the purge manifest.
        manifest_entries: list[dict[str, Any]] = []
        for eid in sorted(closure_ids):
            entry = self._get_entry(eid)
            if entry is not None:
                manifest_entries.append({
                    "entry_id": entry["entry_id"],
                    "content_hash": entry["content_hash"],
                    "action_type": entry["action_type"],
                    "purged": entry["purged"],
                })

        manifest = {
            "certificate_id": cert_id,
            "timestamp": now,
            "episode_id": episode_id,
            "closure_ids": sorted(closure_ids),
            "manifest_entries": manifest_entries,
            "total_purged": len(closure_ids),
        }

        # SHA-256 signature over the deterministic manifest.
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        signature = hashlib.sha256(manifest_bytes).hexdigest()

        certificate: dict[str, Any] = {
            **manifest,
            "signature": signature,
        }

        # Save to filesystem.
        cert_path = self.certificate_dir / f"{cert_id}.json"
        with open(cert_path, "w") as fh:
            json.dump(certificate, fh, indent=2, sort_keys=True)

        certificate["certificate_path"] = str(cert_path)
        return certificate

    # ------------------------------------------------------------------
    # Certificate verification
    # ------------------------------------------------------------------

    def verify_certificate(self, certificate: dict) -> bool:
        """Auditor verification of a purge certificate.

        Checks:

        1. Required fields are present.
        2. Signature integrity (recompute SHA-256 over the manifest).
        3. Purge completeness -- all entries in ``closure_ids`` should
           be marked as purged in the DB (or absent from it).

        Parameters
        ----------
        certificate:
            The certificate dict (as returned by :meth:`emit_certificate`
            or loaded from the JSON file).

        Returns
        -------
        bool
            ``True`` if the certificate passes all checks.
        """
        required_fields = {
            "certificate_id", "timestamp", "episode_id",
            "closure_ids", "manifest_entries", "total_purged", "signature",
        }
        if not required_fields.issubset(certificate.keys()):
            return False

        # 1. Recompute signature.
        stored_sig = certificate["signature"]
        manifest = {
            k: v for k, v in certificate.items()
            if k not in ("signature", "certificate_path")
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        expected_sig = hashlib.sha256(manifest_bytes).hexdigest()

        if not constant_time_compare(stored_sig, expected_sig):
            return False

        # 2. Verify purge completeness.
        for entry_id in certificate.get("closure_ids", []):
            entry = self._get_entry(entry_id)
            if entry is not None and not entry["purged"]:
                return False

        return True

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def wrap_prompt(
        self,
        system_prompt: str,
        user_input: str,
        tool_outputs: list[dict],
    ) -> str:
        """Add RVU provenance preamble and optional tool-restriction notice."""
        rvu_preamble = (
            "RVU PROVENANCE TRACKING ACTIVE.\n"
            "All inputs, tool calls, and outputs are logged with cryptographic "
            "hashes for auditability. Contamination detection and verifiable "
            "state recovery are enabled. Do not attempt to circumvent provenance "
            "tracking.\n"
        )

        if self._allowed_tools is not None:
            rvu_preamble += (
                f"\nTOOL RESTRICTIONS: Allowed tools: {sorted(self._allowed_tools)}\n"
                "Calls to tools not on the allowlist will be REJECTED.\n"
            )

        parts: list[str] = []
        augmented_system = (
            f"{rvu_preamble}\n{system_prompt}"
            if system_prompt
            else rvu_preamble
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
        """Record tool output to provenance and apply allowlist gating.

        All tool outputs are logged to the provenance DB and embedded
        into FAISS regardless of gating outcome.  If an allowlist is
        configured, disallowed tools are blocked.
        """
        metadata: dict[str, Any] = {
            "defense": self.name,
            "tool_name": tool_name,
        }

        # Record the tool output in provenance.
        entry_id = self.record_action(
            action_type="tool_output",
            content=output,
            parent_id=None,
        )
        metadata["provenance_entry_id"] = entry_id
        metadata["content_hash"] = sha256_hex(output)

        # Allowlist gating (inherited from RVG).
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            metadata["action"] = "block"
            metadata["reason"] = f"tool '{tool_name}' not in allowlist"
            metadata["tainted"] = True

            # Mark the provenance entry as tainted.
            self._conn.execute(
                "UPDATE provenance SET tainted = 1 WHERE entry_id = ?",
                (entry_id,),
            )
            self._conn.commit()

            return "[TOOL OUTPUT BLOCKED: tool not on allowlist]", metadata

        # Blocked-pattern check.
        for pattern in self._blocked_patterns:
            if pattern.search(output):
                metadata["action"] = "block"
                metadata["reason"] = f"blocked pattern detected: {pattern.pattern}"
                metadata["tainted"] = True
                self._conn.execute(
                    "UPDATE provenance SET tainted = 1 WHERE entry_id = ?",
                    (entry_id,),
                )
                self._conn.commit()
                return "[TOOL OUTPUT BLOCKED: blocked pattern detected]", metadata

        metadata["action"] = "allow"
        metadata["tainted"] = False
        return output, metadata

    def post_episode(self, episode_log: dict) -> dict:
        """Run contamination detection + closure + purge + certificate emission.

        Parameters
        ----------
        episode_log:
            Must contain ``"episode_id"`` (str).  May contain
            ``"contamination_indicators"`` (list[str]) for explicit
            contamination detection.

        Returns
        -------
        dict
            Defence metadata including contamination set, closure set,
            purge log, and certificate.
        """
        episode_id = episode_log.get("episode_id", self._current_episode_id)

        result: dict[str, Any] = {
            "defense": self.name,
            "episode_id": episode_id,
        }

        # Step 1: Contamination detection.
        indicators = episode_log.get("contamination_indicators", [])
        if not indicators:
            # If no explicit indicators, check for any tainted entries
            # from this episode.
            cur = self._conn.execute(
                "SELECT entry_id FROM provenance "
                "WHERE episode_id = ? AND tainted = 1 AND purged = 0",
                (episode_id,),
            )
            tainted_ids = {row[0] for row in cur}
            if tainted_ids:
                result["contaminated_ids"] = sorted(tainted_ids)
                contaminated = tainted_ids
            else:
                result["contaminated_ids"] = []
                result["closure_ids"] = []
                result["purge_log"] = {"purged_entry_ids": [], "faiss_vectors_removed": 0}
                result["certificate"] = None
                # Rotate episode ID for next episode.
                self._current_episode_id = f"ep-{uuid.uuid4().hex[:12]}"
                return result
        else:
            contaminated = self.detect_contamination(indicators)
            result["contaminated_ids"] = sorted(contaminated)

        if not contaminated:
            result["closure_ids"] = []
            result["purge_log"] = {"purged_entry_ids": [], "faiss_vectors_removed": 0}
            result["certificate"] = None
            self._current_episode_id = f"ep-{uuid.uuid4().hex[:12]}"
            return result

        # Step 2: Compute closure.
        closure = self.compute_closure(
            contaminated, max_depth=self.closure_max_depth
        )
        result["closure_ids"] = sorted(closure)

        # Step 3: Purge.
        purge_log = self.purge(closure)
        result["purge_log"] = purge_log

        # Step 4: Emit certificate.
        certificate = self.emit_certificate(episode_id, closure)
        result["certificate"] = certificate

        # Rotate episode ID for next episode.
        self._current_episode_id = f"ep-{uuid.uuid4().hex[:12]}"

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_episode_id(self, episode_id: str) -> None:
        """Set the current episode ID for provenance tracking."""
        self._current_episode_id = episode_id

    def close(self) -> None:
        """Flush and close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
