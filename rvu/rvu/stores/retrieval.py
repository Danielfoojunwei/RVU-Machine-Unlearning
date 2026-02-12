from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalRecord:
    id: str
    payload: str


@dataclass
class RetrievalStore:
    records: dict[str, RetrievalRecord] = field(default_factory=dict)
    rebuilt_index: list[str] = field(default_factory=list)

    def upsert(self, rec: RetrievalRecord) -> None:
        self.records[rec.id] = rec
        self.rebuild_index()

    def purge_ids(self, ids: set[str]) -> None:
        for rid in ids:
            self.records.pop(rid, None)
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self.rebuilt_index = sorted(self.records.keys())

    def export(self) -> dict[str, str]:
        return {k: v.payload for k, v in sorted(self.records.items())}
