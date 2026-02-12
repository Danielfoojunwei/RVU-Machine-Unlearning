from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryRecord:
    id: str
    text: str


@dataclass
class MemoryStore:
    records: dict[str, MemoryRecord] = field(default_factory=dict)

    def upsert(self, rec: MemoryRecord) -> None:
        self.records[rec.id] = rec

    def purge_ids(self, ids: set[str]) -> None:
        for rid in ids:
            self.records.pop(rid, None)

    def export(self) -> dict[str, str]:
        return {k: v.text for k, v in sorted(self.records.items())}
