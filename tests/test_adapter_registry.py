"""Tests for the Adapter Provenance Registry.

Validates Phase 1: adapter lifecycle tracking, schema integrity,
provenance DAG linkage, and fusion detection.

Theorem references:
    - Theorem 1 (Provenance Completeness): every lifecycle event recorded
    - Theorem 4 (Certified Recovery Integrity): adapter_hash for auditing
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from rvu.defenses.adapter_registry import AdapterRegistry, AdapterRecord


@pytest.fixture
def registry(tmp_path: Path):
    """Create an AdapterRegistry backed by a temp SQLite database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    # Create the provenance table (needed for get_entries_for_adapter).
    conn.executescript("""
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
    """)
    conn.commit()
    reg = AdapterRegistry(conn=conn)
    yield reg
    conn.close()


class TestAdapterRegistration:
    """Test adapter registration and basic CRUD."""

    def test_register_creates_record(self, registry: AdapterRegistry) -> None:
        adapter_id = registry.register_adapter(
            adapter_name="test-lora",
            adapter_hash="abc123",
            source="local",
        )
        assert adapter_id.startswith("adapter-")
        record = registry.get_adapter(adapter_id)
        assert record is not None
        assert record.adapter_name == "test-lora"
        assert record.adapter_hash == "abc123"
        assert record.source == "local"
        assert not record.fused
        assert not record.tainted
        assert not record.purged

    def test_register_with_parent(self, registry: AdapterRegistry) -> None:
        parent_id = registry.register_adapter("parent-lora", "hash1")
        child_id = registry.register_adapter(
            "child-lora", "hash2", parent_adapter_id=parent_id,
        )
        child = registry.get_adapter(child_id)
        assert child is not None
        assert child.parent_adapter_id == parent_id

    def test_get_nonexistent_returns_none(self, registry: AdapterRegistry) -> None:
        assert registry.get_adapter("nonexistent") is None

    def test_get_by_hash(self, registry: AdapterRegistry) -> None:
        registry.register_adapter("my-adapter", "unique-hash-xyz")
        record = registry.get_adapter_by_hash("unique-hash-xyz")
        assert record is not None
        assert record.adapter_name == "my-adapter"

    def test_get_by_hash_not_found(self, registry: AdapterRegistry) -> None:
        assert registry.get_adapter_by_hash("nonexistent-hash") is None


class TestAdapterLifecycle:
    """Test load, unload, fuse lifecycle events."""

    def test_load_sets_timestamp(self, registry: AdapterRegistry) -> None:
        aid = registry.register_adapter("test", "hash")
        registry.record_load(aid, episode_id="ep-001")
        record = registry.get_adapter(aid)
        assert record is not None
        assert record.loaded_at is not None
        assert record.unloaded_at is None

    def test_unload_sets_timestamp(self, registry: AdapterRegistry) -> None:
        aid = registry.register_adapter("test", "hash")
        registry.record_load(aid)
        registry.record_unload(aid)
        record = registry.get_adapter(aid)
        assert record is not None
        assert record.unloaded_at is not None

    def test_fuse_sets_irreversible_flag(self, registry: AdapterRegistry) -> None:
        """Per NTU DTC 'Open Problems': fusion is irreversible."""
        aid = registry.register_adapter("test", "hash")
        registry.record_load(aid)
        registry.record_fuse(aid)
        assert registry.is_fused(aid) is True
        record = registry.get_adapter(aid)
        assert record is not None
        assert record.fused is True

    def test_active_adapters(self, registry: AdapterRegistry) -> None:
        a1 = registry.register_adapter("active1", "h1")
        a2 = registry.register_adapter("active2", "h2")
        a3 = registry.register_adapter("unloaded", "h3")
        registry.record_load(a1)
        registry.record_load(a2)
        registry.record_load(a3)
        registry.record_unload(a3)

        active = registry.get_active_adapters()
        active_ids = {a.adapter_id for a in active}
        assert a1 in active_ids
        assert a2 in active_ids
        assert a3 not in active_ids

    def test_taint_and_purge(self, registry: AdapterRegistry) -> None:
        aid = registry.register_adapter("test", "hash")
        registry.mark_tainted(aid)
        record = registry.get_adapter(aid)
        assert record is not None
        assert record.tainted is True

        registry.mark_purged(aid)
        record = registry.get_adapter(aid)
        assert record is not None
        assert record.purged is True


class TestAdapterEventLog:
    """Test the in-memory event log for audit."""

    def test_events_recorded(self, registry: AdapterRegistry) -> None:
        aid = registry.register_adapter("test", "hash")
        registry.record_load(aid)
        registry.record_unload(aid)

        events = registry.event_log
        assert len(events) >= 3  # register, load, unload
        types = [e.event_type for e in events]
        assert "register" in types
        assert "load" in types
        assert "unload" in types

    def test_clear_event_log(self, registry: AdapterRegistry) -> None:
        registry.register_adapter("test", "hash")
        count = registry.clear_event_log()
        assert count >= 1
        assert len(registry.event_log) == 0


class TestAdapterFileHashing:
    """Test adapter file hashing for Theorem 4 (certificate integrity)."""

    def test_hash_file(self, tmp_path: Path) -> None:
        f = tmp_path / "adapter.safetensors"
        f.write_bytes(b"fake adapter weights")
        h = AdapterRegistry.hash_adapter_file(str(f))
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "adapter_dir"
        d.mkdir()
        (d / "layer1.safetensors").write_bytes(b"layer1")
        (d / "layer2.safetensors").write_bytes(b"layer2")
        h = AdapterRegistry.hash_adapter_file(str(d))
        assert len(h) == 64

    def test_hash_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "adapter.bin"
        f.write_bytes(b"deterministic content")
        h1 = AdapterRegistry.hash_adapter_file(str(f))
        h2 = AdapterRegistry.hash_adapter_file(str(f))
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert AdapterRegistry.hash_adapter_file(str(f1)) != AdapterRegistry.hash_adapter_file(str(f2))


class TestAdapterProvenance:
    """Test provenance linkage between adapters and entries."""

    def test_contaminated_adapters_query(self, registry: AdapterRegistry) -> None:
        a1 = registry.register_adapter("clean", "h1")
        a2 = registry.register_adapter("tainted", "h2")
        registry.mark_tainted(a2)

        contaminated = registry.get_contaminated_adapters()
        ids = {a.adapter_id for a in contaminated}
        assert a2 in ids
        assert a1 not in ids

    def test_full_history(self, registry: AdapterRegistry) -> None:
        aid = registry.register_adapter("test", "hash")
        registry.record_load(aid)
        registry.mark_tainted(aid)

        history = registry.get_full_history(aid)
        assert history["is_tainted"] is True
        assert history["adapter"].adapter_name == "test"

    def test_history_nonexistent(self, registry: AdapterRegistry) -> None:
        history = registry.get_full_history("nonexistent")
        assert "error" in history

    def test_risk_score(self, registry: AdapterRegistry) -> None:
        aid = registry.register_adapter("test", "hash")
        registry.set_risk_score(aid, 0.75)
        record = registry.get_adapter(aid)
        assert record is not None
        assert abs(record.risk_score - 0.75) < 1e-6
