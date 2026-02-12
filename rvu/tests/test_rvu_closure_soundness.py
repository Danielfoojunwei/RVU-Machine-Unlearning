from rvu.ids import IdGenerator
from rvu.qa.scenarios import initial_state
from rvu.runtime.log import RuntimeLog
from rvu.runtime.step import InputEvent, InputType, Simulator
from rvu.rvu_core.operator import run_rvu
from rvu.stores.control_plane import Policy


def test_t1_closure_soundness() -> None:
    state = initial_state()
    log = RuntimeLog()
    sim = Simulator(IdGenerator(deterministic=True), unsafe_mode=True)
    state, rec = sim.run_step(
        state, log, InputEvent(InputType.UNTRUSTED_DOC, "summarize and write")
    )
    k_set = {rec.artifacts_created[0]}
    result = run_rvu(state, log, k_set, Policy(), requester="admin")
    for rid in k_set:
        assert rid not in result.state.provenance_dag.nodes
    assert all(k not in result.state.memory.records for k in k_set)
    assert all(k not in result.state.retrieval.records for k in k_set)
