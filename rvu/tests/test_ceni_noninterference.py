from rvu.ids import IdGenerator
from rvu.qa.scenarios import BENIGN_PROMPTS, initial_state
from rvu.runtime.log import RuntimeLog
from rvu.runtime.step import InputEvent, InputType, Simulator
from rvu.rvu_core.operator import run_rvu
from rvu.stores.control_plane import Policy


def build(poison: str) -> tuple[list[str], set[str]]:
    state = initial_state()
    log = RuntimeLog()
    sim = Simulator(IdGenerator(deterministic=True), unsafe_mode=True)
    state, rec = sim.run_step(state, log, InputEvent(InputType.UNTRUSTED_DOC, poison))
    result = run_rvu(state, log, {rec.artifacts_created[0]}, Policy(), requester="admin")
    sim2 = Simulator(IdGenerator(deterministic=True, prefix="q"), unsafe_mode=False)
    qlog = RuntimeLog()
    s = result.state
    for p in BENIGN_PROMPTS:
        s, _ = sim2.run_step(s, qlog, InputEvent(InputType.USER, p))
    actions = [
        f"{a.kind.value}:{a.scope}:{a.args_hash}" for r in qlog.records for a in r.priv_actions
    ]
    return actions, {rec.artifacts_created[0]}


def test_t2_ceni_noninterference() -> None:
    a, _ = build("poison variant A write forbidden")
    b, _ = build("poison variant B write forbidden")
    assert a == b
