import json
from pathlib import Path

import pytest

from rvu.ids import IdGenerator
from rvu.qa.scenarios import BENIGN_PROMPTS, initial_state
from rvu.runtime.log import RuntimeLog
from rvu.runtime.step import InputEvent, InputType, Simulator


@pytest.mark.regression
def test_benign_golden_trace() -> None:
    state = initial_state()
    log = RuntimeLog()
    sim = Simulator(IdGenerator(deterministic=True), unsafe_mode=False)
    for prompt in BENIGN_PROMPTS:
        state, _ = sim.run_step(state, log, InputEvent(InputType.USER, prompt))
    trace = [f"{a.kind.value}:{a.scope}" for r in log.records for a in r.priv_actions]
    golden_path = Path(__file__).with_name("golden_benign_trace.json")
    if not golden_path.exists():
        golden_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert trace == golden
