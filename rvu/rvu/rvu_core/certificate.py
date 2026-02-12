from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class Certificate:
    cert_id: str
    timestamp: str
    k_hashes: list[str]
    closure_hash: str
    ops: list[str]
    postchecks: dict[str, bool]
    epsilon: float
    final_state_hash: str
    baseline_state_hash: str
    mode: str = "APPLIED"
    signatures: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        cert_id: str,
        k_hashes: list[str],
        closure_hash: str,
        ops: list[str],
        postchecks: dict[str, bool],
        epsilon: float,
        final_state_hash: str,
        baseline_state_hash: str,
        mode: str = "APPLIED",
    ) -> Certificate:
        return cls(
            cert_id=cert_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            k_hashes=k_hashes,
            closure_hash=closure_hash,
            ops=ops,
            postchecks=postchecks,
            epsilon=epsilon,
            final_state_hash=final_state_hash,
            baseline_state_hash=baseline_state_hash,
            mode=mode,
        )

    def export(self) -> dict[str, object]:
        return asdict(self)
