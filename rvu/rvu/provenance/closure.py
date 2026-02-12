from __future__ import annotations

from .dag import ProvenanceDAG


def contamination_closure(dag: ProvenanceDAG, k_set: set[str]) -> set[str]:
    return dag.descendants(k_set)
