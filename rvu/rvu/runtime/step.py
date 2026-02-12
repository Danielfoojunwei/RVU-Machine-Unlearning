from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rvu.hashing import sha256_text
from rvu.ids import IdGenerator
from rvu.provenance.dag import ArtifactKind, ArtifactNode, Edge, EdgeKind, Principal
from rvu.runtime.log import RuntimeLog, StepRecord
from rvu.runtime.privilege import PrivAction, PrivKind
from rvu.stores.base import SystemState, compute_state_hash
from rvu.stores.memory import MemoryRecord
from rvu.stores.retrieval import RetrievalRecord


class InputType(str, Enum):
    USER = "USER"
    UNTRUSTED_DOC = "UNTRUSTED_DOC"
    TOOL_OUTPUT = "TOOL_OUTPUT"


@dataclass(frozen=True)
class InputEvent:
    kind: InputType
    text: str
    source: str = "USER"


@dataclass
class Simulator:
    id_gen: IdGenerator
    unsafe_mode: bool = False

    def run_step(
        self, state: SystemState, runtime_log: RuntimeLog, event: InputEvent
    ) -> tuple[SystemState, StepRecord]:
        dag = state.provenance_dag.copy()
        created: list[str] = []
        edges: list[tuple[str, str, str]] = []
        priv_actions: list[PrivAction] = []

        principal = Principal.USER if event.kind == InputType.USER else Principal.WEB
        if event.kind == InputType.TOOL_OUTPUT:
            principal = Principal.TOOL
        node_kind = (
            ArtifactKind.INPUT if event.kind == InputType.USER else ArtifactKind.UNTRUSTED_SPAN
        )
        input_id = self.id_gen.new_id()
        dag.add_node(ArtifactNode(input_id, node_kind, principal, {"text": event.text}))
        created.append(input_id)

        lowered = event.text.lower()
        should_write = any(p in lowered for p in ["save", "write", "store"])
        is_untrusted = event.kind != InputType.USER
        tainted = is_untrusted

        if should_write and (self.unsafe_mode or not tainted):
            mem_id = self.id_gen.new_id()
            dag.add_node(
                ArtifactNode(
                    mem_id, ArtifactKind.MEMORY_REC, Principal.SYS, {"summary": event.text[:30]}
                )
            )
            dag.add_edge(Edge(input_id, mem_id, EdgeKind.SUMMARIZED_FROM))
            created.append(mem_id)
            edges.append((input_id, mem_id, EdgeKind.SUMMARIZED_FROM.value))

            state.memory.upsert(MemoryRecord(mem_id, event.text[:120]))
            mem_act = PrivAction(
                PrivKind.MEMORY_WRITE, "memory", sha256_text(event.text), {input_id}
            )
            priv_actions.append(mem_act)

            if "forbidden" in lowered or "/etc/passwd" in lowered:
                scope = "/tmp/forbidden"
            else:
                scope = f"{state.control_plane.sandbox_root}/notes.txt"
            file_act = PrivAction(PrivKind.FILE_WRITE, scope, sha256_text(event.text), {input_id})
            priv_actions.append(file_act)

            vec_id = self.id_gen.new_id()
            dag.add_node(
                ArtifactNode(
                    vec_id, ArtifactKind.VECTOR_REC, Principal.SYS, {"payload": event.text[:30]}
                )
            )
            dag.add_edge(Edge(mem_id, vec_id, EdgeKind.EMBEDDED_FROM))
            created.append(vec_id)
            edges.append((mem_id, vec_id, EdgeKind.EMBEDDED_FROM.value))
            state.retrieval.upsert(RetrievalRecord(vec_id, event.text[:120]))

        new_hash = compute_state_hash(
            state.control_plane, state.memory, state.retrieval, state.router
        )
        new_state = SystemState(
            control_plane=state.control_plane,
            memory=state.memory,
            retrieval=state.retrieval,
            router=state.router,
            provenance_dag=dag,
            state_hash=new_hash,
        )
        rec = StepRecord(
            t=len(runtime_log.records),
            inputs=[event.text],
            artifacts_created=created,
            edges_added=edges,
            priv_actions=priv_actions,
            tool_outputs=[],
        )
        runtime_log.append(rec)
        return new_state, rec
