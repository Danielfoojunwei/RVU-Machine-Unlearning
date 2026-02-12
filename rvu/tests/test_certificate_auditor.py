import copy

from rvu.ids import IdGenerator
from rvu.qa.scenarios import initial_state
from rvu.runtime.log import RuntimeLog
from rvu.runtime.step import InputEvent, InputType, Simulator
from rvu.rvu_core.auditor import verify_cert
from rvu.rvu_core.certificate import Certificate
from rvu.rvu_core.operator import run_rvu
from rvu.stores.control_plane import Policy


def test_t5_certificate_verifiability() -> None:
    state = initial_state()
    log = RuntimeLog()
    sim = Simulator(IdGenerator(deterministic=True), unsafe_mode=True)
    state, rec = sim.run_step(state, log, InputEvent(InputType.UNTRUSTED_DOC, "save this"))
    snapshot = copy.deepcopy(state)
    result = run_rvu(state, log, {rec.artifacts_created[0]}, Policy(), requester="admin")
    report = verify_cert(result.certificate, log, snapshot, {rec.artifacts_created[0]})
    assert report.ok

    bad = Certificate(**{**result.certificate.export(), "epsilon": 0.0001})
    report_bad = verify_cert(bad, log, snapshot, {rec.artifacts_created[0]})
    assert not report_bad.ok
