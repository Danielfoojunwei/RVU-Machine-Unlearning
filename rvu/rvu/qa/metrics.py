from __future__ import annotations

from rvu.runtime.log import RuntimeLog
from rvu.rvu_core.distance import state_distance


def privileged_trace_distance(log_a: RuntimeLog, log_b: RuntimeLog) -> float:
    pairs = [
        ([*a.priv_actions], [*b.priv_actions])
        for a, b in zip(log_a.records, log_b.records, strict=False)
    ]
    return state_distance(pairs)
