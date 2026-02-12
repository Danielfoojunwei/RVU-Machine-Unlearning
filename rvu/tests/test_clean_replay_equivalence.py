from rvu.ids import IdGenerator
from rvu.qa.scenarios import initial_state
from rvu.runtime.log import RuntimeLog
from rvu.runtime.step import InputEvent, InputType, Simulator
from rvu.rvu_core.baseline_replay import replay_without_k


def test_t3_replay_baseline_runs() -> None:
    state = initial_state()
    log = RuntimeLog()
    sim = Simulator(IdGenerator(deterministic=True), unsafe_mode=True)
    state, rec = sim.run_step(state, log, InputEvent(InputType.UNTRUSTED_DOC, "write to forbidden"))
    out = replay_without_k(state, log, {rec.artifacts_created[0]})
    assert out.state.state_hash
    assert len(out.priv_actions) >= 1
