from __future__ import annotations

from dataclasses import dataclass

from rvu.runtime.privilege import PrivAction


@dataclass(frozen=True)
class DistanceWeights:
    kind: float = 0.35
    scope: float = 0.25
    args_hash: float = 0.25
    count: float = 0.15


def trace_delta(
    left: list[PrivAction], right: list[PrivAction], w: DistanceWeights | None = None
) -> float:
    weights = w or DistanceWeights()
    max_len = max(len(left), len(right), 1)
    penalty = 0.0
    for i in range(min(len(left), len(right))):
        if left[i].kind != right[i].kind:
            penalty += weights.kind
        if left[i].scope != right[i].scope:
            penalty += weights.scope
        if left[i].args_hash != right[i].args_hash:
            penalty += weights.args_hash
    if len(left) != len(right):
        penalty += weights.count * abs(len(left) - len(right))
    return penalty / max_len


def state_distance(traces: list[tuple[list[PrivAction], list[PrivAction]]]) -> float:
    if not traces:
        return 0.0
    return sum(trace_delta(a, b) for a, b in traces) / len(traces)
