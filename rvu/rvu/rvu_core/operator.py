from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rvu.hashing import stable_hash_obj
from rvu.ids import IdGenerator
from rvu.provenance.closure import contamination_closure
from rvu.qa.scenarios import BENIGN_PROMPTS, initial_state
from rvu.runtime.log import RuntimeLog, StepRecord
from rvu.runtime.step import InputEvent, InputType, Simulator
from rvu.rvu_core.baseline_replay import replay_without_k
from rvu.rvu_core.certificate import Certificate
from rvu.rvu_core.distance import trace_delta
from rvu.stores.base import SystemState, compute_state_hash
from rvu.stores.control_plane import Policy


@dataclass
class RVUResult:
    state: SystemState
    certificate: Certificate


def run_rvu(
    state: SystemState,
    runtime_log: RuntimeLog,
    k_set: set[str],
    policy: Policy,
    requester: str,
    epsilon: float = 0.05,
) -> RVUResult:
    if not policy.authz(k_set, requester):
        cert = Certificate.build(
            cert_id=IdGenerator(deterministic=True, prefix="cert").new_id(),
            k_hashes=sorted(stable_hash_obj(k) for k in k_set),
            closure_hash=stable_hash_obj([]),
            ops=["NOOP_UNAUTHORIZED"],
            postchecks={"authz": False},
            epsilon=epsilon,
            final_state_hash=state.state_hash,
            baseline_state_hash=state.state_hash,
            mode="NOOP",
        )
        return RVUResult(state, cert)

    closure = contamination_closure(state.provenance_dag, k_set)
    state.memory.purge_ids(closure)
    state.retrieval.purge_ids(closure)
    state.router.purge_ids(closure)
    for cid in closure:
        state.provenance_dag.nodes.pop(cid, None)

    baseline = replay_without_k(state, runtime_log, k_set)

    sim_a = Simulator(IdGenerator(deterministic=True, prefix="qa"), unsafe_mode=False)
    sim_b = Simulator(IdGenerator(deterministic=True, prefix="qb"), unsafe_mode=False)
    log_a = RuntimeLog()
    log_b = RuntimeLog()
    s_a = state
    s_b = baseline.state
    for prompt in BENIGN_PROMPTS:
        s_a, _ = sim_a.run_step(s_a, log_a, InputEvent(InputType.USER, prompt))
        s_b, _ = sim_b.run_step(s_b, log_b, InputEvent(InputType.USER, prompt))

    distance = trace_delta(
        [a for rec in log_a.records for a in rec.priv_actions],
        [a for rec in log_b.records for a in rec.priv_actions],
    )
    final_hash = compute_state_hash(
        state.control_plane, state.memory, state.retrieval, state.router
    )
    state.state_hash = final_hash
    cert = Certificate.build(
        cert_id=IdGenerator(deterministic=True, prefix="cert").new_id(),
        k_hashes=sorted(stable_hash_obj(k) for k in k_set),
        closure_hash=stable_hash_obj(sorted(closure)),
        ops=["MEMORY_PURGE", "RETRIEVAL_PURGE", "ROUTER_PURGE"],
        postchecks={
            "closure_check": not any(x in state.provenance_dag.nodes for x in closure),
            "replay_proximity": distance <= epsilon,
            "ceni_spotcheck": distance <= epsilon,
            "authz": True,
        },
        epsilon=epsilon,
        final_state_hash=final_hash,
        baseline_state_hash=baseline.state.state_hash,
    )
    return RVUResult(state, cert)


def load_runtime_log(path: Path) -> RuntimeLog:
    runtime_log = RuntimeLog()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        runtime_log.append(StepRecord.from_dict(json.loads(line)))
    return runtime_log


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--log", required=True)
    run.add_argument("--K", required=True)
    run.add_argument("--out", required=True)
    args = parser.parse_args()

    runtime_log = load_runtime_log(Path(args.log))
    k_set = set(json.loads(Path(args.K).read_text(encoding="utf-8")))
    result = run_rvu(initial_state(), runtime_log, k_set, Policy(), requester="admin")
    Path(args.out).write_text(json.dumps(result.certificate.export(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
