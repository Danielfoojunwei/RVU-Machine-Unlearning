from rvu.ids import IdGenerator
from rvu.qa.scenarios import initial_state
from rvu.runtime.log import RuntimeLog
from rvu.runtime.step import InputEvent, InputType, Simulator
from rvu.rvu_core.operator import run_rvu
from rvu.stores.control_plane import Policy


def test_t4_authorized_only_and_control_plane_immutable() -> None:
    state = initial_state()
    control_before = state.control_plane.export()
    old_hash = state.state_hash
    log = RuntimeLog()
    sim = Simulator(IdGenerator(deterministic=True), unsafe_mode=True)
    state, rec = sim.run_step(state, log, InputEvent(InputType.UNTRUSTED_DOC, "write to forbidden"))
    denied = run_rvu(state, log, {rec.artifacts_created[0]}, Policy(), requester="guest")
    assert denied.certificate.mode == "NOOP"
    assert denied.state.state_hash == state.state_hash
    assert denied.state.control_plane.export() == control_before

    allowed = run_rvu(state, log, {rec.artifacts_created[0]}, Policy(), requester="admin")
    assert allowed.state.control_plane.export() == control_before
    assert old_hash != allowed.state.state_hash
