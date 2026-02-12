from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArtifactKind(str, Enum):
    INPUT = "INPUT"
    UNTRUSTED_SPAN = "UNTRUSTED_SPAN"
    CANDIDATE_FACT = "CANDIDATE_FACT"
    VERIFIED_FACT = "VERIFIED_FACT"
    MEMORY_REC = "MEMORY_REC"
    VECTOR_REC = "VECTOR_REC"
    TOOL_CALL = "TOOL_CALL"
    TOOL_OUT = "TOOL_OUT"
    ROUTER_RULE = "ROUTER_RULE"
    POLICY_ITEM = "POLICY_ITEM"
    REPAIRED = "REPAIRED"


class Principal(str, Enum):
    SYS = "SYS"
    USER = "USER"
    WEB = "WEB"
    TOOL = "TOOL"
    SKILL = "SKILL"


class EdgeKind(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    SUMMARIZED_FROM = "SUMMARIZED_FROM"
    EMBEDDED_FROM = "EMBEDDED_FROM"
    ROUTED_FROM = "ROUTED_FROM"
    WROTE_TO = "WROTE_TO"


@dataclass(frozen=True)
class ArtifactNode:
    id: str
    kind: ArtifactKind
    provenance: Principal
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src_id: str
    dst_id: str
    edge_kind: EdgeKind


@dataclass
class ProvenanceDAG:
    nodes: dict[str, ArtifactNode] = field(default_factory=dict)
    outgoing: dict[str, set[str]] = field(default_factory=dict)
    incoming: dict[str, set[str]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: ArtifactNode) -> None:
        self.nodes[node.id] = node
        self.outgoing.setdefault(node.id, set())
        self.incoming.setdefault(node.id, set())

    def add_edge(self, edge: Edge) -> None:
        if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
            raise KeyError("Edge references unknown node")
        self.edges.append(edge)
        self.outgoing.setdefault(edge.src_id, set()).add(edge.dst_id)
        self.incoming.setdefault(edge.dst_id, set()).add(edge.src_id)

    def descendants(self, seeds: set[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(seeds)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.outgoing.get(node, set()) - seen)
        return seen

    def copy(self) -> ProvenanceDAG:
        return ProvenanceDAG(
            nodes=dict(self.nodes),
            outgoing={k: set(v) for k, v in self.outgoing.items()},
            incoming={k: set(v) for k, v in self.incoming.items()},
            edges=list(self.edges),
        )
