from __future__ import annotations

from dataclasses import dataclass

from rvu.provenance.dag import ArtifactKind, ArtifactNode, Principal
from rvu.runtime.privilege import PrivAction, PrivKind


@dataclass(frozen=True)
class RepairResult:
    artifact: ArtifactNode
    priv_action: PrivAction


def repair_artifact(artifact_id: str) -> RepairResult:
    node = ArtifactNode(artifact_id, ArtifactKind.REPAIRED, Principal.SYS, {"status": "repaired"})
    action = PrivAction(PrivKind.MEMORY_WRITE, "noop", "repaired", set())
    return RepairResult(node, action)
