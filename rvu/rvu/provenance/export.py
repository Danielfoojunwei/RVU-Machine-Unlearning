from __future__ import annotations

from .dag import ProvenanceDAG


def export_dag(dag: ProvenanceDAG) -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind.value,
                "provenance": n.provenance.value,
                "metadata": n.metadata,
            }
            for n in dag.nodes.values()
        ],
        "edges": [
            {"src": e.src_id, "dst": e.dst_id, "edge_kind": e.edge_kind.value} for e in dag.edges
        ],
    }
