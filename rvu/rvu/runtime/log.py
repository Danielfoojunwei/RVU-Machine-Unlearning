from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rvu.runtime.privilege import PrivAction, PrivKind


@dataclass
class StepRecord:
    t: int
    inputs: list[str]
    artifacts_created: list[str]
    edges_added: list[tuple[str, str, str]]
    priv_actions: list[PrivAction]
    tool_outputs: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        raw_priv = data.get("priv_actions")
        priv_list = raw_priv if isinstance(raw_priv, list) else []
        priv = [
            PrivAction(
                kind=PrivKind(p["kind"]),
                scope=str(p["scope"]),
                args_hash=str(p["args_hash"]),
                deps=set(p.get("deps", [])),
            )
            for p in priv_list
            if isinstance(p, dict)
        ]

        raw_edges = data.get("edges_added")
        edge_list = raw_edges if isinstance(raw_edges, list) else []
        edges = [tuple(e) for e in edge_list if isinstance(e, (list, tuple)) and len(e) == 3]

        raw_inputs = data.get("inputs")
        raw_created = data.get("artifacts_created")
        raw_tool = data.get("tool_outputs")
        return cls(
            t=int(data.get("t", 0)),
            inputs=[str(v) for v in (raw_inputs if isinstance(raw_inputs, list) else [])],
            artifacts_created=[
                str(v) for v in (raw_created if isinstance(raw_created, list) else [])
            ],
            edges_added=[(str(a), str(b), str(c)) for a, b, c in edges],
            priv_actions=priv,
            tool_outputs=[str(v) for v in (raw_tool if isinstance(raw_tool, list) else [])],
        )


@dataclass
class RuntimeLog:
    records: list[StepRecord] = field(default_factory=list)

    def append(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def export(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for r in self.records:
            row = asdict(r)
            row["priv_actions"] = [
                {
                    "kind": p.kind.value,
                    "scope": p.scope,
                    "args_hash": p.args_hash,
                    "deps": sorted(p.deps),
                }
                for p in r.priv_actions
            ]
            rows.append(row)
        return rows
