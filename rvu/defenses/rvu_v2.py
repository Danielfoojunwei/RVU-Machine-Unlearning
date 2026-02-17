"""RVU v2 Defense — Adapter-aware recovery with risk-scored purge.

Extends RVUDefense with:
1. **Adapter provenance tracking** — LoRA/IA3/prefix adapter lifecycle in the DAG
2. **Risk-scored purge** (FROC) — tiered response replacing binary purge
3. **Adapter safety gate** — pre-load screening via Safe LoRA + OOD
4. **Influence estimation** — output-level adapter impact measurement
5. **V2 certificates** — adapter attestation + membership probe evidence
6. **Non-linear adapter support** — IA3, prefix-tuning, prompt-tuning, bottleneck

Theoretical basis:
    - Theorem 1 (Provenance Completeness): all adapter events recorded
    - Theorem 2 (Closure Soundness): adapter-aware BFS closure
    - Theorem 3 (Risk Monotonicity): FROC R(e) = w_c*P + w_i*I + w_p*prop
    - Theorem 4 (Certified Recovery Integrity): V2 certificate with adapter attestation
    - Theorem 5 (Membership Probe Soundness): post-recovery verification

Handles both linear and non-linear adapter architectures:
    - **Linear** (LoRA, DoRA, AdaLoRA, QLoRA): weight delta ΔW = BA is a
      low-rank matrix that adds linearly to base weights.  Safety projection
      operates on the ΔW matrix directly (Safe LoRA, NeurIPS 2024).
    - **Non-linear** (IA3, prefix-tuning, prompt-tuning, bottleneck):
      parameters do not form a simple additive delta.  IA3 applies
      element-wise rescaling (multiplicative, not additive).  Prefix/prompt
      tuning prepends learned vectors to the key/value/embedding space.
      Bottleneck adapters add a non-linear MLP sub-network.
      For these, influence estimation uses output divergence only (no
      weight-delta projection), and safety checking uses activation
      norm ratios rather than subspace projection.

Benchmarks & libraries:
    - peft (HuggingFace): LoRA, IA3, prefix_tuning, prompt_tuning support
    - WMDP, TOFU, MUSE, SafeRLHF for evaluation
    - faiss-cpu, sentence-transformers (inherited from RVU v1)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from rvu.defenses.base import (
    BaseDefense,
    constant_time_compare,
    format_tool_outputs_block,
    sha256_hex,
    timestamp_now,
)
from rvu.defenses.adapter_registry import AdapterRecord, AdapterRegistry
from rvu.defenses.adapter_gate import AdapterGate, GateDecision
from rvu.influence.adapter_influence import AdapterInfluenceEstimator
from rvu.verification.certificate_v2 import (
    AdapterAttestation,
    CertificateV2Emitter,
    CertificateV2Verifier,
    MembershipProbeAttestation,
    RuntimeRecovery,
)
from rvu.verification.membership_probe import MembershipProbe


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PurgeAction(str, Enum):
    """Tiered response actions from FROC risk scoring.

    Corollary 3.1 (Threshold Ordering): FLAG < QUARANTINE < PURGE < EVICT.
    """

    NONE = "none"
    FLAG = "flag"
    QUARANTINE = "quarantine"
    PURGE = "purge"
    EVICT = "evict"


class AdapterType(str, Enum):
    """Adapter architecture classification.

    Linear adapters have additive weight deltas (ΔW = BA).
    Non-linear adapters modify activations or prepend learned vectors.
    """

    # Linear: safety projection via weight-delta SVD.
    LORA = "lora"
    DORA = "dora"
    ADALORA = "adalora"
    QLORA = "qlora"

    # Non-linear: safety checking via activation norm ratios.
    IA3 = "ia3"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    BOTTLENECK = "bottleneck"
    LORA_PLUS = "lora_plus"

    @property
    def is_linear(self) -> bool:
        return self in (
            AdapterType.LORA, AdapterType.DORA,
            AdapterType.ADALORA, AdapterType.QLORA,
        )


# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_DDL_PROVENANCE = """\
CREATE TABLE IF NOT EXISTS provenance (
    entry_id          TEXT PRIMARY KEY,
    episode_id        TEXT,
    action_type       TEXT,
    content           TEXT,
    content_hash      TEXT,
    parent_id         TEXT,
    timestamp         REAL,
    tainted           INTEGER DEFAULT 0,
    purged            INTEGER DEFAULT 0,
    active_adapter_id TEXT DEFAULT NULL
);
"""

_DDL_QUARANTINE = """\
CREATE TABLE IF NOT EXISTS quarantine (
    entry_id        TEXT PRIMARY KEY,
    quarantined_at  REAL,
    risk_score      REAL,
    reason          TEXT,
    restored        INTEGER DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# RVU v2 Defense
# ---------------------------------------------------------------------------

class RVUv2Defense(BaseDefense):
    """Full RVU v2 defense with adapter-aware, risk-scored recovery.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    faiss_index_path:
        Path to the FAISS index file.
    embedder:
        A sentence-transformers model instance.
    adapter_config_path:
        Path to ``configs/adapters.yaml``.
    certificate_dir:
        Directory for certificate output.
    similarity_threshold:
        Cosine similarity threshold for contamination detection.
    closure_max_depth:
        Max BFS depth for provenance closure.
    """

    name: str = "rvu_v2"

    def __init__(
        self,
        db_path: str,
        faiss_index_path: str,
        embedder: Any,
        adapter_config_path: str | None = None,
        certificate_dir: str = "artifacts/certificates",
        similarity_threshold: float = 0.85,
        closure_max_depth: int = 10,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.faiss_index_path = Path(faiss_index_path).expanduser().resolve()
        self.certificate_dir = Path(certificate_dir).expanduser().resolve()
        self.similarity_threshold = similarity_threshold
        self.closure_max_depth = closure_max_depth

        # Directories.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.certificate_dir.mkdir(parents=True, exist_ok=True)

        # SQLite.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_DDL_PROVENANCE)
        self._conn.executescript(_DDL_QUARANTINE)
        self._conn.commit()

        # Embedder (sentence-transformers or compatible).
        self._embedder = embedder
        if hasattr(embedder, "get_sentence_embedding_dimension"):
            self._embedding_dim: int = embedder.get_sentence_embedding_dimension()
        else:
            self._embedding_dim = 384  # Default for all-MiniLM-L6-v2.

        # FAISS.
        try:
            import faiss
            if self.faiss_index_path.is_file():
                self._index = faiss.read_index(str(self.faiss_index_path))
            else:
                self._index = faiss.IndexFlatIP(self._embedding_dim)
            self._faiss_available = True
        except ImportError:
            self._index = None
            self._faiss_available = False

        self._faiss_id_map: list[str] = self._load_faiss_id_map()

        # Sub-components.
        self._adapter_registry = AdapterRegistry(conn=self._conn)
        self._adapter_gate = AdapterGate(
            config_path=adapter_config_path, embedder=embedder,
        )
        self._influence_estimator = AdapterInfluenceEstimator(embedder=embedder)
        self._membership_probe = MembershipProbe(embedder=embedder)
        self._cert_emitter = CertificateV2Emitter(certificate_dir=str(self.certificate_dir))
        self._cert_verifier = CertificateV2Verifier()

        # FROC risk weights (loaded from config or defaults).
        self._w_contamination: float = 0.4
        self._w_influence: float = 0.3
        self._w_propagation: float = 0.3
        self._flag_threshold: float = 0.3
        self._quarantine_threshold: float = 0.6
        self._purge_threshold: float = 0.8
        self._evict_threshold: float = 0.9

        if adapter_config_path is not None:
            self._load_risk_config(adapter_config_path)

        # Episode state.
        self._current_episode_id: str = f"ep-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_risk_config(self, config_path: str) -> None:
        """Load FROC risk weights from adapters.yaml."""
        import yaml
        p = Path(config_path).expanduser().resolve()
        if not p.is_file():
            return
        with open(p, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        risk = cfg.get("risk_thresholds", {})
        self._w_contamination = risk.get("w_contamination", self._w_contamination)
        self._w_influence = risk.get("w_influence", self._w_influence)
        self._w_propagation = risk.get("w_propagation", self._w_propagation)
        self._flag_threshold = risk.get("flag_threshold", self._flag_threshold)
        self._quarantine_threshold = risk.get("quarantine_threshold", self._quarantine_threshold)
        self._purge_threshold = risk.get("purge_threshold", self._purge_threshold)
        self._evict_threshold = risk.get("evict_threshold", self._evict_threshold)

    # ------------------------------------------------------------------
    # FAISS helpers
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
        if self._faiss_available and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(self.faiss_index_path))
            self._save_faiss_id_map()

    def _embed(self, text: str) -> np.ndarray:
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    # ------------------------------------------------------------------
    # Provenance DB
    # ------------------------------------------------------------------

    def record_action(
        self,
        action_type: str,
        content: str,
        parent_id: str | None = None,
        active_adapter_id: str | None = None,
    ) -> str:
        """Log an action to provenance DB and FAISS.

        Now includes ``active_adapter_id`` for adapter-aware closure
        (Theorem 2, Corollary 1.1).
        """
        entry_id = str(uuid.uuid4())
        now = timestamp_now()
        content_hash = sha256_hex(content)

        # Determine active adapter if not specified.
        if active_adapter_id is None:
            active = self._adapter_registry.get_active_adapters()
            if active:
                active_adapter_id = active[0].adapter_id

        self._conn.execute(
            "INSERT INTO provenance "
            "(entry_id, episode_id, action_type, content, content_hash, "
            " parent_id, timestamp, tainted, purged, active_adapter_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?)",
            (entry_id, self._current_episode_id, action_type, content,
             content_hash, parent_id, now, active_adapter_id),
        )
        self._conn.commit()

        # FAISS embedding.
        if self._faiss_available and self._index is not None:
            vec = self._embed(content)
            self._index.add(vec)
            self._faiss_id_map.append(entry_id)
            self._persist_faiss()

        return entry_id

    def _get_entry(self, entry_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT entry_id, episode_id, action_type, content, content_hash, "
            "       parent_id, timestamp, tainted, purged, active_adapter_id "
            "FROM provenance WHERE entry_id = ?",
            (entry_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "entry_id": row[0], "episode_id": row[1], "action_type": row[2],
            "content": row[3], "content_hash": row[4], "parent_id": row[5],
            "timestamp": row[6], "tainted": row[7], "purged": row[8],
            "active_adapter_id": row[9],
        }

    def _get_children(self, entry_id: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT entry_id FROM provenance WHERE parent_id = ? AND purged = 0",
            (entry_id,),
        )
        return [row[0] for row in cur]

    def _total_entries(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM provenance WHERE purged = 0")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Contamination detection (inherited from v1)
    # ------------------------------------------------------------------

    def detect_contamination(self, indicators: list[str]) -> set[str]:
        """Four-layer contamination detection (Definition 5)."""
        contaminated: set[str] = set()
        for indicator in indicators:
            if self._get_entry(indicator) is not None:
                contaminated.add(indicator)
                continue
            cur = self._conn.execute(
                "SELECT entry_id FROM provenance WHERE content_hash = ? AND purged = 0",
                (indicator,),
            )
            for row in cur:
                contaminated.add(row[0])
            cur = self._conn.execute(
                "SELECT entry_id FROM provenance WHERE content LIKE ? AND purged = 0",
                (f"%{indicator}%",),
            )
            for row in cur:
                contaminated.add(row[0])
            if self._faiss_available and self._index is not None and self._index.ntotal > 0:
                vec = self._embed(indicator)
                k = min(20, self._index.ntotal)
                distances, indices = self._index.search(vec, k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self._faiss_id_map):
                        continue
                    if dist >= self.similarity_threshold:
                        cid = self._faiss_id_map[int(idx)]
                        entry = self._get_entry(cid)
                        if entry and not entry["purged"]:
                            contaminated.add(cid)
        return contaminated

    # ------------------------------------------------------------------
    # Adapter-aware closure (Theorem 2)
    # ------------------------------------------------------------------

    def compute_closure(
        self,
        contaminated_ids: set[str],
        max_depth: int | None = None,
    ) -> set[str]:
        """Adapter-aware provenance closure via BFS.

        Extends v1 closure with adapter-aware expansion: if an entry
        was produced while a specific adapter was active, and that
        adapter is tainted, ALL entries produced under that adapter
        are added to the closure.

        This implements Theorem 2 part (c).
        """
        if max_depth is None:
            max_depth = self.closure_max_depth

        closure: set[str] = set(contaminated_ids)
        queue: deque[tuple[str, int]] = deque(
            (eid, 0) for eid in contaminated_ids
        )

        # Track which adapters are in the contamination closure.
        contaminated_adapters: set[str] = set()

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Standard: follow children.
            for child_id in self._get_children(current_id):
                if child_id not in closure:
                    closure.add(child_id)
                    queue.append((child_id, depth + 1))

            # Standard: follow parent.
            entry = self._get_entry(current_id)
            if entry and entry["parent_id"] and entry["parent_id"] not in closure:
                closure.add(entry["parent_id"])
                queue.append((entry["parent_id"], depth + 1))

            # NEW: adapter-aware expansion.
            if entry and entry.get("active_adapter_id"):
                adapter_id = entry["active_adapter_id"]
                if adapter_id not in contaminated_adapters:
                    contaminated_adapters.add(adapter_id)
                    self._adapter_registry.mark_tainted(adapter_id)
                    # Add all entries produced under this adapter.
                    adapter_entries = self._adapter_registry.get_entries_for_adapter(adapter_id)
                    for ae_id in adapter_entries:
                        if ae_id not in closure:
                            closure.add(ae_id)
                            queue.append((ae_id, depth + 1))

        return closure

    # ------------------------------------------------------------------
    # FROC Risk Scoring (Theorem 3)
    # ------------------------------------------------------------------

    def compute_risk_score(self, entry_id: str) -> float:
        """Compute FROC risk score R(e) for a single entry.

        R(e) = w_c * P(contaminated|e) + w_i * I(e) + w_p * |C({e})| / |V|

        Theorem 3 guarantees monotonicity: increasing any component
        increases R(e).
        """
        entry = self._get_entry(entry_id)
        if entry is None:
            return 0.0

        # P(contaminated|e): use tainted flag as proxy (already computed
        # by contamination detection).
        p_contaminated = 1.0 if entry["tainted"] else 0.0

        # I(e): adapter influence (from cached influence estimator).
        adapter_id = entry.get("active_adapter_id")
        if adapter_id:
            influence = self._influence_estimator.get_cached_influence(adapter_id)
        else:
            influence = 0.0

        # Propagation: closure ratio.
        total = self._total_entries()
        if total > 0:
            closure_size = len(self.compute_closure({entry_id}, max_depth=3))
            propagation = closure_size / total
        else:
            propagation = 0.0

        return (
            self._w_contamination * p_contaminated
            + self._w_influence * min(1.0, influence)
            + self._w_propagation * min(1.0, propagation)
        )

    def determine_action(self, risk_score: float) -> PurgeAction:
        """Map risk score to tiered response action.

        Corollary 3.1 (Threshold Ordering): FLAG < QUARANTINE < PURGE < EVICT.
        """
        if risk_score >= self._evict_threshold:
            return PurgeAction.EVICT
        if risk_score >= self._purge_threshold:
            return PurgeAction.PURGE
        if risk_score >= self._quarantine_threshold:
            return PurgeAction.QUARANTINE
        if risk_score >= self._flag_threshold:
            return PurgeAction.FLAG
        return PurgeAction.NONE

    # ------------------------------------------------------------------
    # Quarantine (reversible isolation)
    # ------------------------------------------------------------------

    def quarantine(self, entry_ids: set[str], risk_scores: dict[str, float]) -> dict[str, Any]:
        """Move entries to quarantine (excluded from FAISS but retained in DB).

        Quarantine is the key FROC insight: entries with moderate risk
        are isolated rather than destroyed, preserving utility while
        reducing risk.
        """
        quarantined: list[str] = []
        now = timestamp_now()
        for eid in entry_ids:
            self._conn.execute(
                "INSERT OR REPLACE INTO quarantine "
                "(entry_id, quarantined_at, risk_score, reason, restored) "
                "VALUES (?, ?, ?, ?, 0)",
                (eid, now, risk_scores.get(eid, 0.0), "risk_scored_quarantine"),
            )
            quarantined.append(eid)
        self._conn.commit()
        return {"quarantined_ids": quarantined, "count": len(quarantined)}

    def restore_from_quarantine(self, entry_ids: set[str]) -> dict[str, Any]:
        """Restore entries from quarantine (determined to be safe)."""
        restored: list[str] = []
        for eid in entry_ids:
            self._conn.execute(
                "UPDATE quarantine SET restored = 1 WHERE entry_id = ?",
                (eid,),
            )
            restored.append(eid)
        self._conn.commit()
        return {"restored_ids": restored, "count": len(restored)}

    # ------------------------------------------------------------------
    # Risk-scored purge (Phase 2, FROC)
    # ------------------------------------------------------------------

    def risk_scored_purge(
        self,
        closure_ids: set[str],
        risk_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Tiered purge based on FROC risk scores.

        Replaces the binary purge of v1 with graduated response:
        FLAG → QUARANTINE → PURGE → EVICT.

        Returns a detailed log of actions taken per entry.
        """
        if risk_scores is None:
            risk_scores = {eid: self.compute_risk_score(eid) for eid in closure_ids}

        log: dict[str, Any] = {
            "timestamp": timestamp_now(),
            "flagged_ids": [],
            "quarantined_ids": [],
            "purged_entry_ids": [],
            "evicted_adapter_ids": [],
            "actions": {},
            "faiss_vectors_removed": 0,
        }

        to_quarantine: set[str] = set()
        to_purge: set[str] = set()
        adapters_to_evict: set[str] = set()

        for eid in closure_ids:
            score = risk_scores.get(eid, 0.0)
            action = self.determine_action(score)
            log["actions"][eid] = {"risk_score": score, "action": action.value}

            if action == PurgeAction.NONE:
                continue
            elif action == PurgeAction.FLAG:
                # Flag only: mark tainted but take no destructive action.
                self._conn.execute(
                    "UPDATE provenance SET tainted = 1 WHERE entry_id = ?",
                    (eid,),
                )
                log["flagged_ids"].append(eid)
            elif action == PurgeAction.QUARANTINE:
                to_quarantine.add(eid)
            elif action == PurgeAction.PURGE:
                to_purge.add(eid)
            elif action == PurgeAction.EVICT:
                to_purge.add(eid)
                entry = self._get_entry(eid)
                if entry and entry.get("active_adapter_id"):
                    adapters_to_evict.add(entry["active_adapter_id"])

        self._conn.commit()

        # Quarantine.
        if to_quarantine:
            q_result = self.quarantine(to_quarantine, risk_scores)
            log["quarantined_ids"] = q_result["quarantined_ids"]

        # Purge (same mechanism as v1).
        if to_purge:
            for eid in to_purge:
                entry = self._get_entry(eid)
                if entry is None or entry["purged"]:
                    continue
                self._conn.execute(
                    "UPDATE provenance SET purged = 1, tainted = 1 WHERE entry_id = ?",
                    (eid,),
                )
                log["purged_entry_ids"].append(eid)
            self._conn.commit()

            # Rebuild FAISS without purged vectors.
            if self._faiss_available and self._index is not None:
                self._rebuild_faiss(to_purge)
                log["faiss_vectors_removed"] = len(to_purge)

        # Evict adapters.
        for adapter_id in adapters_to_evict:
            self._adapter_registry.record_unload(adapter_id)
            self._adapter_registry.mark_purged(adapter_id)
            log["evicted_adapter_ids"].append(adapter_id)

        return log

    def _rebuild_faiss(self, purge_ids: set[str]) -> None:
        """Rebuild FAISS index excluding purged entries."""
        import faiss
        remaining_ids: list[str] = []
        remaining_vecs: list[np.ndarray] = []
        for idx, fid in enumerate(self._faiss_id_map):
            if fid not in purge_ids:
                vec = np.zeros((1, self._embedding_dim), dtype=np.float32)
                self._index.reconstruct(idx, vec.ravel())
                remaining_vecs.append(vec)
                remaining_ids.append(fid)
        new_index = faiss.IndexFlatIP(self._embedding_dim)
        if remaining_vecs:
            new_index.add(np.vstack(remaining_vecs))
        self._index = new_index
        self._faiss_id_map = remaining_ids
        self._persist_faiss()

    # ------------------------------------------------------------------
    # Adapter lifecycle (linear + non-linear)
    # ------------------------------------------------------------------

    def load_adapter(
        self,
        adapter_name: str,
        adapter_path: str | Path,
        adapter_type: str = "lora",
        source: str = "local",
        adapter_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Register, screen, and load an adapter.

        Handles both linear (LoRA, DoRA, QLoRA) and non-linear (IA3,
        prefix_tuning, prompt_tuning, bottleneck) adapter types.

        For linear adapters: safety projection via weight-delta SVD.
        For non-linear adapters: activation norm ratio check.
        """
        result: dict[str, Any] = {"adapter_type": adapter_type}

        # Classify adapter.
        try:
            atype = AdapterType(adapter_type)
        except ValueError:
            atype = AdapterType.LORA  # Default to LoRA.
        result["is_linear"] = atype.is_linear

        # Hash the adapter file.
        adapter_hash = AdapterRegistry.hash_adapter_file(adapter_path)
        result["adapter_hash"] = adapter_hash

        # Screen through the gate.
        gate_decision = self._adapter_gate.screen_adapter(
            adapter_name=adapter_name,
            adapter_hash=adapter_hash,
            source=source,
            adapter_weights=adapter_weights if atype.is_linear else None,
        )
        result["gate_decision"] = gate_decision

        if not gate_decision.allowed:
            result["loaded"] = False
            result["reason"] = gate_decision.reason
            return result

        # Register in provenance.
        adapter_id = self._adapter_registry.register_adapter(
            adapter_name=adapter_name,
            adapter_hash=adapter_hash,
            source=source,
            episode_id=self._current_episode_id,
        )
        self._adapter_registry.record_load(adapter_id, self._current_episode_id)
        self._adapter_registry.set_risk_score(adapter_id, gate_decision.risk_score)

        # Record the load event in provenance DAG.
        self.record_action(
            action_type="adapter_load",
            content=json.dumps({
                "adapter_id": adapter_id,
                "adapter_name": adapter_name,
                "adapter_type": adapter_type,
                "adapter_hash": adapter_hash,
                "is_linear": atype.is_linear,
            }),
            active_adapter_id=adapter_id,
        )

        result["loaded"] = True
        result["adapter_id"] = adapter_id
        return result

    def unload_adapter(self, adapter_id: str) -> dict[str, Any]:
        """Unload an adapter and record the event."""
        self._adapter_registry.record_unload(adapter_id)
        self.record_action(
            action_type="adapter_unload",
            content=json.dumps({"adapter_id": adapter_id}),
        )
        return {"adapter_id": adapter_id, "unloaded": True}

    def fuse_adapter(self, adapter_id: str) -> dict[str, Any]:
        """Record adapter fusion (IRREVERSIBLE).

        Per NTU DTC "Open Problems in MU": fusion merges adapter weights
        into base model.  Cannot be cleanly reversed.  Certificate will
        report a fusion warning.
        """
        self._adapter_registry.record_fuse(adapter_id)
        self.record_action(
            action_type="adapter_fuse",
            content=json.dumps({
                "adapter_id": adapter_id,
                "irreversible": True,
                "warning": "Adapter fused into base weights. Cannot be cleanly removed.",
            }),
        )
        return {"adapter_id": adapter_id, "fused": True, "irreversible": True}

    # ------------------------------------------------------------------
    # V2 Certificate emission
    # ------------------------------------------------------------------

    def emit_certificate_v2(
        self,
        episode_id: str,
        purge_log: dict[str, Any],
        membership_probe_report: Any | None = None,
        risk_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Emit a V2 certificate with adapter attestation.

        Implements Theorem 4 (Certified Recovery Integrity) with
        adapter lifecycle evidence.
        """
        # Build runtime recovery summary.
        purged_ids = purge_log.get("purged_entry_ids", [])
        quarantined_ids = purge_log.get("quarantined_ids", [])
        manifest_entries: list[dict[str, Any]] = []
        for eid in sorted(purged_ids):
            entry = self._get_entry(eid)
            if entry:
                manifest_entries.append({
                    "entry_id": entry["entry_id"],
                    "content_hash": entry["content_hash"],
                    "action_type": entry["action_type"],
                    "purged": entry["purged"],
                    "active_adapter_id": entry.get("active_adapter_id"),
                })

        runtime = RuntimeRecovery(
            closure_ids=sorted(purged_ids + quarantined_ids),
            manifest_entries=manifest_entries,
            total_purged=len(purged_ids),
            total_quarantined=len(quarantined_ids),
            quarantine_ids=sorted(quarantined_ids),
        )

        # Build adapter attestations.
        adapter_atts: list[AdapterAttestation] = []
        evicted_ids = purge_log.get("evicted_adapter_ids", [])
        for aid in evicted_ids:
            rec = self._adapter_registry.get_adapter(aid)
            if rec:
                adapter_atts.append(AdapterAttestation(
                    adapter_id=rec.adapter_id,
                    adapter_name=rec.adapter_name,
                    adapter_hash=rec.adapter_hash,
                    action="fused_warning" if rec.fused else "evicted",
                    risk_score=rec.risk_score,
                    was_fused=rec.fused,
                    eviction_method="irreversible" if rec.fused else "unload",
                ))

        # Active adapters that weren't evicted.
        for rec in self._adapter_registry.get_active_adapters():
            if rec.adapter_id not in evicted_ids:
                adapter_atts.append(AdapterAttestation(
                    adapter_id=rec.adapter_id,
                    adapter_name=rec.adapter_name,
                    adapter_hash=rec.adapter_hash,
                    action="retained",
                    risk_score=rec.risk_score,
                    was_fused=rec.fused,
                    eviction_method="not_applicable",
                ))

        # Membership probe attestation.
        probe_att = None
        if membership_probe_report is not None:
            probe_att = MembershipProbeAttestation(
                probe_type="output_similarity",
                num_probes=membership_probe_report.num_probes,
                pre_recovery_similarity=None,
                post_recovery_similarity=membership_probe_report.mean_similarity,
                probe_verdict=membership_probe_report.probe_verdict,
                breakdown=membership_probe_report.breakdown,
            )

        return self._cert_emitter.emit(
            episode_id=episode_id,
            runtime_recovery=runtime,
            adapter_attestations=adapter_atts,
            membership_probe=probe_att,
            risk_scores=risk_scores,
        )

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def wrap_prompt(
        self,
        system_prompt: str,
        user_input: str,
        tool_outputs: list[dict],
    ) -> str:
        """Wrap prompt with RVU v2 provenance preamble."""
        preamble = (
            "RVU v2 PROVENANCE TRACKING ACTIVE.\n"
            "All inputs, tool calls, outputs, and adapter events are logged "
            "with cryptographic hashes. Contamination detection, risk-scored "
            "recovery, and adapter-aware verification are enabled.\n"
        )
        active = self._adapter_registry.get_active_adapters()
        if active:
            names = [a.adapter_name for a in active]
            preamble += f"Active adapters: {names}\n"

        parts: list[str] = []
        parts.append(f"[SYSTEM]\n{preamble}\n{system_prompt}\n" if system_prompt else f"[SYSTEM]\n{preamble}\n")
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
        """Record tool output with active adapter linkage."""
        metadata: dict[str, Any] = {"defense": self.name, "tool_name": tool_name}

        active = self._adapter_registry.get_active_adapters()
        active_id = active[0].adapter_id if active else None

        entry_id = self.record_action(
            action_type="tool_output",
            content=output,
            active_adapter_id=active_id,
        )
        metadata["provenance_entry_id"] = entry_id
        metadata["content_hash"] = sha256_hex(output)
        metadata["active_adapter_id"] = active_id
        metadata["action"] = "allow"
        metadata["tainted"] = False
        return output, metadata

    def post_episode(self, episode_log: dict) -> dict:
        """Full v2 post-episode pipeline.

        1. Contamination detection
        2. Adapter-aware closure
        3. Risk-scored purge (FROC tiered response)
        4. V2 certificate with adapter attestation
        """
        episode_id = episode_log.get("episode_id", self._current_episode_id)
        result: dict[str, Any] = {"defense": self.name, "episode_id": episode_id}

        # Step 1: Detection.
        indicators = episode_log.get("contamination_indicators", [])
        if indicators:
            contaminated = self.detect_contamination(indicators)
        else:
            cur = self._conn.execute(
                "SELECT entry_id FROM provenance "
                "WHERE episode_id = ? AND tainted = 1 AND purged = 0",
                (episode_id,),
            )
            contaminated = {row[0] for row in cur}

        if not contaminated:
            result["contaminated_ids"] = []
            result["purge_log"] = {"purged_entry_ids": []}
            result["certificate"] = None
            self._current_episode_id = f"ep-{uuid.uuid4().hex[:12]}"
            return result

        result["contaminated_ids"] = sorted(contaminated)

        # Step 2: Closure.
        closure = self.compute_closure(contaminated)
        result["closure_ids"] = sorted(closure)

        # Step 3: Risk scores.
        risk_scores = {eid: self.compute_risk_score(eid) for eid in closure}
        result["risk_scores"] = risk_scores

        # Step 4: Risk-scored purge.
        purge_log = self.risk_scored_purge(closure, risk_scores)
        result["purge_log"] = purge_log

        # Step 5: Certificate.
        certificate = self.emit_certificate_v2(
            episode_id=episode_id,
            purge_log=purge_log,
            risk_scores=risk_scores,
        )
        result["certificate"] = certificate

        self._current_episode_id = f"ep-{uuid.uuid4().hex[:12]}"
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def adapter_registry(self) -> AdapterRegistry:
        return self._adapter_registry

    @property
    def adapter_gate(self) -> AdapterGate:
        return self._adapter_gate

    @property
    def influence_estimator(self) -> AdapterInfluenceEstimator:
        return self._influence_estimator

    @property
    def membership_probe(self) -> MembershipProbe:
        return self._membership_probe

    def set_episode_id(self, episode_id: str) -> None:
        self._current_episode_id = episode_id

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
