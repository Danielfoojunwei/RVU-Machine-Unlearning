from __future__ import annotations

from dataclasses import dataclass

from rvu.provenance.closure import contamination_closure
from rvu.provenance.dag import ArtifactKind
from rvu.runtime.log import RuntimeLog
from rvu.runtime.privilege import PrivAction
from rvu.runtime.repair import repair_artifact
from rvu.stores.base import SystemState, compute_state_hash
from rvu.stores.memory import MemoryRecord
from rvu.stores.retrieval import RetrievalRecord


@dataclass
class ReplayResult:
    state: SystemState
    replay_log: RuntimeLog
    priv_actions: list[PrivAction]


def replay_without_k(state: SystemState, runtime_log: RuntimeLog, k_set: set[str]) -> ReplayResult:
    new_state = SystemState(
        control_plane=state.control_plane,
        memory=type(state.memory)(),
        retrieval=type(state.retrieval)(),
        router=type(state.router)(),
        provenance_dag=state.provenance_dag.copy(),
        state_hash=state.state_hash,
    )
    closure = contamination_closure(new_state.provenance_dag, k_set)
    replayed = RuntimeLog()
    priv_actions: list[PrivAction] = []
    for rec in runtime_log.records:
        depends = bool(set(rec.artifacts_created) & closure)
        if depends:
            rep = repair_artifact(f"repaired-{rec.t}")
            new_state.provenance_dag.add_node(rep.artifact)
            priv_actions.append(rep.priv_action)
        else:
            for aid in rec.artifacts_created:
                node = new_state.provenance_dag.nodes.get(aid)
                if node is None:
                    continue
                if node.kind == ArtifactKind.MEMORY_REC:
                    new_state.memory.upsert(MemoryRecord(aid, node.metadata.get("summary", "")))
                if node.kind == ArtifactKind.VECTOR_REC:
                    new_state.retrieval.upsert(
                        RetrievalRecord(aid, node.metadata.get("payload", ""))
                    )
            priv_actions.extend(rec.priv_actions)
        replayed.append(rec)
    new_state.memory.purge_ids(closure)
    new_state.retrieval.purge_ids(closure)
    # Remove closure nodes from DAG live view.
    for cid in closure:
        new_state.provenance_dag.nodes.pop(cid, None)
    new_state.state_hash = compute_state_hash(
        new_state.control_plane, new_state.memory, new_state.retrieval, new_state.router
    )
    return ReplayResult(new_state, replayed, priv_actions)
