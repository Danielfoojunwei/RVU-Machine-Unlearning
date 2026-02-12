from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouterRule:
    id: str
    pattern: str
    score: float


@dataclass
class RouterState:
    rules: dict[str, RouterRule] = field(default_factory=dict)

    def upsert(self, rule: RouterRule) -> None:
        self.rules[rule.id] = rule

    def purge_ids(self, ids: set[str]) -> None:
        for rid in ids:
            self.rules.pop(rid, None)

    def export(self) -> dict[str, tuple[str, float]]:
        return {k: (v.pattern, v.score) for k, v in sorted(self.rules.items())}
