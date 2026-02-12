from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rvu.hashing import stable_hash_obj
from rvu.provenance.closure import contamination_closure
from rvu.qa.scenarios import initial_state
from rvu.runtime.log import RuntimeLog
from rvu.rvu_core.certificate import Certificate
from rvu.rvu_core.operator import load_runtime_log, run_rvu
from rvu.stores.base import SystemState
from rvu.stores.control_plane import Policy


@dataclass
class AuditReport:
    ok: bool
    mismatches: list[str]


def verify_cert(
    cert: Certificate, runtime_log: RuntimeLog, state_snapshot: SystemState, k_set: set[str]
) -> AuditReport:
    expected_closure_hash = stable_hash_obj(
        sorted(contamination_closure(state_snapshot.provenance_dag, k_set))
    )
    rerun = run_rvu(state_snapshot, runtime_log, k_set, Policy(), requester="admin")
    mismatches: list[str] = []
    if cert.closure_hash != expected_closure_hash:
        mismatches.append("closure_hash")
    if cert.final_state_hash != rerun.certificate.final_state_hash:
        mismatches.append("final_state_hash")
    if cert.epsilon != rerun.certificate.epsilon:
        mismatches.append("epsilon")
    return AuditReport(ok=not mismatches, mismatches=mismatches)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--cert", required=True)
    verify.add_argument("--log", required=True)
    verify.add_argument("--K", required=False, default="[]")
    args = parser.parse_args()

    cert_data = json.loads(Path(args.cert).read_text(encoding="utf-8"))
    cert = Certificate(**cert_data)
    runtime_log = load_runtime_log(Path(args.log))
    k_set = set(json.loads(args.K))
    report = verify_cert(cert, runtime_log, initial_state(), k_set)
    print(json.dumps({"ok": report.ok, "mismatches": report.mismatches}))


if __name__ == "__main__":
    main()
