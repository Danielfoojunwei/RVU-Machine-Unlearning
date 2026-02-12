from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rvu.hashing import stable_hash_obj
from rvu.provenance.dag import ProvenanceDAG

if TYPE_CHECKING:
    from .control_plane import ControlPlane
    from .memory import MemoryStore
    from .retrieval import RetrievalStore
    from .router import RouterState


@dataclass
class SystemState:
    control_plane: ControlPlane
    memory: MemoryStore
    retrieval: RetrievalStore
    router: RouterState
    provenance_dag: ProvenanceDAG
    state_hash: str


def compute_state_hash(
    control_plane: ControlPlane,
    memory: MemoryStore,
    retrieval: RetrievalStore,
    router: RouterState,
) -> str:
    return stable_hash_obj(
        {
            "control": control_plane.export(),
            "memory": memory.export(),
            "retrieval": retrieval.export(),
            "router": router.export(),
        }
    )
