.PHONY: test regression audit-demo lint typecheck bench_smoke bench_agentdojo_full bench_injecagent_full bench_bipia_full report

lint:
	ruff check rvu tests scripts
	black --check rvu tests scripts

typecheck:
	mypy rvu

test:
	python -m pytest -q

regression:
	python -m pytest -q -m regression

bench_smoke:
	python -m pytest -q -m bench_smoke
	python -m rvu.qa.run_benchmarks --config docker/bench.config.json --smoke

bench_agentdojo_full:
	python -m rvu.qa.benchmark_agentdojo --mode full --out artifacts/runs/manual

bench_injecagent_full:
	python -m rvu.qa.benchmark_injecagent --mode full --out artifacts/runs/manual

bench_bipia_full:
	python -m rvu.qa.benchmark_bipia --mode full --out artifacts/runs/manual

report:
	python scripts/regression_report.py --metrics artifacts/runs/current/metrics.json --baseline artifacts/runs/baseline/metrics.json --out artifacts/reports

audit-demo:
	python -m rvu.qa.scenarios run_incident --scenario doc_poison --out artifacts
	python -c "import json;from pathlib import Path;Path('artifacts/K.json').write_text(json.dumps(['a-00000001']))"
	python -m rvu.rvu_core.operator run --log artifacts/log.jsonl --K artifacts/K.json --out artifacts/cert.json
	python -m rvu.rvu_core.auditor verify --cert artifacts/cert.json --log artifacts/log.jsonl --K '["a-00000001"]'
