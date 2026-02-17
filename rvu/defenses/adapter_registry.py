"""Adapter Provenance Registry — LoRA lifecycle tracking as first-class provenance.

Extends the RVU provenance model to track adapter load, unload, swap, and
fusion events.  Every adapter gets a cryptographic hash (SHA-256 of the weight
file) and a risk score.  The registry links adapter lifecycle events into the
existing provenance DAG so that closure computation can follow
adapter → entry chains.

Theoretical basis:
    - NTU DTC "Certifying the Right to Be Forgotten" (arXiv 2512.23171):
      Verifiable removal requires an audit trail of what was present.
    - NTU DTC "Open Problems in MU for AI Safety" (arXiv 2501.04952):
      Adapter removal ≠ unlearning when fused; track fusion as irreversible.
    - Theorem 1 (Provenance Completeness): every adapter lifecycle event is
      recorded with parent pointers, preserving causal chains.

Benchmarks & libraries:
    - safetensors: secure weight loading and deterministic hashing
    - peft (HuggingFace): LoRA adapter management API surface
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rvu.defenses.base import sha256_hex, timestamp_now


# ---------------------------------------------------------------------------
# SQL DDL — adapter provenance table
# ---------------------------------------------------------------------------

_DDL_ADAPTER_PROVENANCE = """\
CREATE TABLE IF NOT EXISTS adapter_provenance (
    adapter_id        TEXT PRIMARY KEY,
    adapter_name      TEXT NOT NULL,
    adapter_hash      TEXT NOT NULL,
    source            TEXT DEFAULT 'local',
    loaded_at         REAL,
    unloaded_at       REAL,
    fused             INTEGER DEFAULT 0,
    risk_score        REAL DEFAULT 0.0,
    tainted           INTEGER DEFAULT 0,
    purged            INTEGER DEFAULT 0,
    parent_adapter_id TEXT,
    episode_id        TEXT
);
"""

# Migration: add active_adapter_id to the provenance table if absent.
_DDL_ADD_ADAPTER_COL = """\
ALTER TABLE provenance ADD COLUMN active_adapter_id TEXT DEFAULT NULL;
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AdapterRecord:
    """Immutable snapshot of an adapter provenance row."""

    adapter_id: str
    adapter_name: str
    adapter_hash: str
    source: str = "local"
    loaded_at: float | None = None
    unloaded_at: float | None = None
    fused: bool = False
    risk_score: float = 0.0
    tainted: bool = False
    purged: bool = False
    parent_adapter_id: str | None = None
    episode_id: str | None = None


@dataclass
class AdapterEvent:
    """A single adapter lifecycle event for audit logging."""

    event_type: str  # load | unload | fuse | swap | taint | purge
    adapter_id: str
    timestamp: float = field(default_factory=timestamp_now)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """Track LoRA adapter lifecycle in the provenance database.

    Parameters
    ----------
    conn:
        An open ``sqlite3.Connection`` (shared with ``RVUDefense``).
        The registry creates its own table but shares the connection
        so that adapter provenance and entry provenance live in the
        same database and can be joined.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_schema()
        self._event_log: list[AdapterEvent] = []

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the adapter_provenance table and migrate provenance table."""
        self._conn.executescript(_DDL_ADAPTER_PROVENANCE)
        # Add active_adapter_id column to provenance if it doesn't exist.
        try:
            self._conn.execute(_DDL_ADD_ADAPTER_COL)
        except sqlite3.OperationalError:
            pass  # Column already exists.
        self._conn.commit()

    # ------------------------------------------------------------------
    # Adapter hashing
    # ------------------------------------------------------------------

    @staticmethod
    def hash_adapter_file(path: str | Path) -> str:
        """Compute SHA-256 of an adapter weight file.

        For safetensors files this hashes the raw bytes.  For directories
        (HuggingFace format with multiple shards) it hashes each file
        in sorted order and then hashes the concatenated digests.

        This implements the cryptographic identity required by Theorem 4
        (Certified Recovery Integrity) — the certificate includes
        adapter_hash so an auditor can verify which adapter was present.
        """
        p = Path(path)
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        if p.is_dir():
            digests: list[str] = []
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix in (
                    ".safetensors", ".bin", ".pt", ".pth",
                ):
                    digests.append(AdapterRegistry.hash_adapter_file(child))
            if not digests:
                return sha256_hex("")
            return sha256_hex("".join(digests))

        return sha256_hex("")

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def register_adapter(
        self,
        adapter_name: str,
        adapter_hash: str,
        source: str = "local",
        parent_adapter_id: str | None = None,
        episode_id: str | None = None,
    ) -> str:
        """Register a new adapter in the provenance database.

        Returns the generated adapter_id.  Does NOT mark the adapter as
        loaded — call :meth:`record_load` after the adapter is actually
        attached to the model.
        """
        adapter_id = f"adapter-{uuid.uuid4().hex[:12]}"

        self._conn.execute(
            "INSERT INTO adapter_provenance "
            "(adapter_id, adapter_name, adapter_hash, source, "
            " parent_adapter_id, episode_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (adapter_id, adapter_name, adapter_hash, source,
             parent_adapter_id, episode_id),
        )
        self._conn.commit()

        self._event_log.append(AdapterEvent(
            event_type="register",
            adapter_id=adapter_id,
            details={"name": adapter_name, "source": source},
        ))
        return adapter_id

    def record_load(self, adapter_id: str, episode_id: str | None = None) -> None:
        """Record that *adapter_id* was loaded into the serving runtime."""
        now = timestamp_now()
        self._conn.execute(
            "UPDATE adapter_provenance SET loaded_at = ?, episode_id = ? "
            "WHERE adapter_id = ?",
            (now, episode_id, adapter_id),
        )
        self._conn.commit()
        self._event_log.append(AdapterEvent(
            event_type="load", adapter_id=adapter_id, timestamp=now,
        ))

    def record_unload(self, adapter_id: str) -> None:
        """Record that *adapter_id* was removed from the serving runtime."""
        now = timestamp_now()
        self._conn.execute(
            "UPDATE adapter_provenance SET unloaded_at = ? WHERE adapter_id = ?",
            (now, adapter_id),
        )
        self._conn.commit()
        self._event_log.append(AdapterEvent(
            event_type="unload", adapter_id=adapter_id, timestamp=now,
        ))

    def record_fuse(self, adapter_id: str) -> None:
        """Record that *adapter_id* was fused (merged) into base weights.

        Per NTU DTC "Open Problems in MU for AI Safety": fusion is
        IRREVERSIBLE.  Floating-point reversal leaves residual traces.
        The certificate must honestly report this (Theorem 4, Limitation).
        """
        now = timestamp_now()
        self._conn.execute(
            "UPDATE adapter_provenance SET fused = 1 WHERE adapter_id = ?",
            (adapter_id,),
        )
        self._conn.commit()
        self._event_log.append(AdapterEvent(
            event_type="fuse", adapter_id=adapter_id, timestamp=now,
            details={"irreversible": True},
        ))

    def mark_tainted(self, adapter_id: str) -> None:
        """Mark an adapter as contaminated."""
        self._conn.execute(
            "UPDATE adapter_provenance SET tainted = 1 WHERE adapter_id = ?",
            (adapter_id,),
        )
        self._conn.commit()
        self._event_log.append(AdapterEvent(
            event_type="taint", adapter_id=adapter_id,
        ))

    def mark_purged(self, adapter_id: str) -> None:
        """Mark an adapter as purged (evicted + flagged for audit)."""
        self._conn.execute(
            "UPDATE adapter_provenance SET purged = 1 WHERE adapter_id = ?",
            (adapter_id,),
        )
        self._conn.commit()
        self._event_log.append(AdapterEvent(
            event_type="purge", adapter_id=adapter_id,
        ))

    def set_risk_score(self, adapter_id: str, score: float) -> None:
        """Update the risk score for an adapter (FROC R(e) value)."""
        self._conn.execute(
            "UPDATE adapter_provenance SET risk_score = ? WHERE adapter_id = ?",
            (score, adapter_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_adapter(self, adapter_id: str) -> AdapterRecord | None:
        """Fetch a single adapter record by ID."""
        cur = self._conn.execute(
            "SELECT adapter_id, adapter_name, adapter_hash, source, "
            "       loaded_at, unloaded_at, fused, risk_score, "
            "       tainted, purged, parent_adapter_id, episode_id "
            "FROM adapter_provenance WHERE adapter_id = ?",
            (adapter_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return AdapterRecord(
            adapter_id=row[0], adapter_name=row[1], adapter_hash=row[2],
            source=row[3], loaded_at=row[4], unloaded_at=row[5],
            fused=bool(row[6]), risk_score=row[7], tainted=bool(row[8]),
            purged=bool(row[9]), parent_adapter_id=row[10], episode_id=row[11],
        )

    def get_active_adapters(self) -> list[AdapterRecord]:
        """Return all adapters that are loaded but not yet unloaded."""
        cur = self._conn.execute(
            "SELECT adapter_id, adapter_name, adapter_hash, source, "
            "       loaded_at, unloaded_at, fused, risk_score, "
            "       tainted, purged, parent_adapter_id, episode_id "
            "FROM adapter_provenance "
            "WHERE loaded_at IS NOT NULL AND unloaded_at IS NULL AND purged = 0",
        )
        return [
            AdapterRecord(
                adapter_id=r[0], adapter_name=r[1], adapter_hash=r[2],
                source=r[3], loaded_at=r[4], unloaded_at=r[5],
                fused=bool(r[6]), risk_score=r[7], tainted=bool(r[8]),
                purged=bool(r[9]), parent_adapter_id=r[10], episode_id=r[11],
            )
            for r in cur
        ]

    def get_adapter_by_hash(self, adapter_hash: str) -> AdapterRecord | None:
        """Look up an adapter by its weight-file hash."""
        cur = self._conn.execute(
            "SELECT adapter_id FROM adapter_provenance WHERE adapter_hash = ?",
            (adapter_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self.get_adapter(row[0])

    def get_adapters_for_episode(self, episode_id: str) -> list[AdapterRecord]:
        """Return all adapters associated with a specific episode."""
        cur = self._conn.execute(
            "SELECT adapter_id FROM adapter_provenance WHERE episode_id = ?",
            (episode_id,),
        )
        return [self.get_adapter(row[0]) for row in cur if self.get_adapter(row[0]) is not None]

    def is_fused(self, adapter_id: str) -> bool:
        """Return True if the adapter has been fused into base weights."""
        rec = self.get_adapter(adapter_id)
        return rec is not None and rec.fused

    def get_entries_for_adapter(self, adapter_id: str) -> list[str]:
        """Return entry_ids from provenance table where this adapter was active.

        This is the key link between adapter provenance and entry provenance,
        enabling adapter-aware closure computation (Theorem 2, Corollary 1.1).
        """
        cur = self._conn.execute(
            "SELECT entry_id FROM provenance "
            "WHERE active_adapter_id = ? AND purged = 0",
            (adapter_id,),
        )
        return [row[0] for row in cur]

    def get_contaminated_adapters(self) -> list[AdapterRecord]:
        """Return all adapters that are tainted but not yet purged."""
        cur = self._conn.execute(
            "SELECT adapter_id FROM adapter_provenance "
            "WHERE tainted = 1 AND purged = 0",
        )
        return [self.get_adapter(row[0]) for row in cur
                if self.get_adapter(row[0]) is not None]

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> list[AdapterEvent]:
        """Return the in-memory event log for the current session."""
        return list(self._event_log)

    def clear_event_log(self) -> int:
        """Clear the in-memory event log.  Returns count of cleared events."""
        count = len(self._event_log)
        self._event_log.clear()
        return count

    def get_full_history(self, adapter_id: str) -> dict[str, Any]:
        """Return complete audit history for an adapter.

        Includes the adapter record, all provenance entries produced while
        it was active, and all lifecycle events.
        """
        record = self.get_adapter(adapter_id)
        if record is None:
            return {"error": f"adapter {adapter_id} not found"}

        entries = self.get_entries_for_adapter(adapter_id)
        events = [e for e in self._event_log if e.adapter_id == adapter_id]

        return {
            "adapter": record,
            "provenance_entry_ids": entries,
            "lifecycle_events": events,
            "entry_count": len(entries),
            "is_fused": record.fused,
            "is_tainted": record.tainted,
            "is_purged": record.purged,
        }
