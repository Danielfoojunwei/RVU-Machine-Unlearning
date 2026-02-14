# Makefile -- RVU Machine Unlearning project targets
#
# Variables (override on the command line):
#   MODEL   -- model key from configs/models.yaml  (default: model_a)
#   DEFENSE -- defense key from configs/defenses.yaml (default: rvu)

MODEL   ?= model_a
DEFENSE ?= rvu

.PHONY: test lint typecheck \
        bench_smoke_cpu bench_full_cpu \
        agentdojo_smoke agentdojo_full \
        injecagent_smoke injecagent_full \
        bipia_smoke bipia_full \
        aggregate regression

# ---------------------------------------------------------------------------
# Testing & quality
# ---------------------------------------------------------------------------

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m ruff check rvu/ tests/ scripts/

typecheck:
	python3 -m mypy rvu/ tests/

# ---------------------------------------------------------------------------
# Individual benchmark targets
# ---------------------------------------------------------------------------

agentdojo_smoke:
	python3 -m rvu.harness.agentdojo_runner \
		--model $(MODEL) --defense $(DEFENSE) --mode smoke

agentdojo_full:
	python3 -m rvu.harness.agentdojo_runner \
		--model $(MODEL) --defense $(DEFENSE) --mode full

injecagent_smoke:
	python3 -m rvu.harness.injecagent_runner \
		--model $(MODEL) --defense $(DEFENSE) --mode smoke

injecagent_full:
	python3 -m rvu.harness.injecagent_runner \
		--model $(MODEL) --defense $(DEFENSE) --mode full

bipia_smoke:
	python3 -m rvu.harness.bipia_runner \
		--model $(MODEL) --defense $(DEFENSE) --mode smoke

bipia_full:
	python3 -m rvu.harness.bipia_runner \
		--model $(MODEL) --defense $(DEFENSE) --mode full

# ---------------------------------------------------------------------------
# Batch targets
# ---------------------------------------------------------------------------

bench_smoke_cpu: agentdojo_smoke injecagent_smoke bipia_smoke

bench_full_cpu: agentdojo_full injecagent_full bipia_full

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

aggregate:
	python3 scripts/aggregate_results.py

regression:
	python3 scripts/regression_compare.py
