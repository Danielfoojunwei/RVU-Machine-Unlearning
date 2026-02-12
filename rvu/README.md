# RVU: Recursive Verified Unlearning (Empirical Runtime)

This repository implements RVU as a runtime defense layer for tool-using agents with:
- provenance DAG + contamination closure,
- unlearning (memory/vector purge + rebuild),
- clean replay baseline,
- certificates + auditor.

It now includes **empirical benchmark runners** for canonical indirect prompt-injection evaluations:
- InjecAgent,
- BIPIA,
- AgentDojo,
with open-weight models (vLLM or HuggingFace Transformers).

## Threat model
- Attackers can inject indirect instructions through untrusted documents, tool outputs, web pages, or skill text.
- The planner might propagate these strings into privileged tool calls unless runtime guards are enforced.
- Persistence risk is memory/retrieval contamination that survives after one compromised episode.
- RVU is designed to recover state and certify erasure after incidents.

## Quickstart (Docker + GPU)
```bash
cd rvu
docker build -f docker/Dockerfile -t rvu-empirical .

docker run --gpus all --rm -it \
  -v $PWD:/workspace \
  -w /workspace rvu-empirical bash
```

Download open weights (example Qwen2.5-7B):
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --revision main
```

Start vLLM server (preferred):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 8000
```

Run smoke benchmark orchestration:
```bash
make bench_smoke
```

## Benchmark execution
Set benchmark CLIs in environment so RVU adapters can run official harnesses:
```bash
export AGENTDOJO_CLI=/path/to/agentdojo
export INJECAGENT_CLI=/path/to/injecagent
export BIPIA_CLI=/path/to/bipia
```

Reproduce main tables:
```bash
make bench_agentdojo_full
make bench_injecagent_full
make bench_bipia_full
make report
```

Expected artifacts:
- `artifacts/runs/<timestamp>/metrics.json`
- `artifacts/runs/<timestamp>/*_metrics.json`
- `artifacts/reports/regression_report.md`
- `artifacts/runs/<timestamp>/certificates/*.json` (when benchmark harness exports certs)

## RVU flow
```bash
python -m rvu.qa.scenarios run_incident --scenario doc_poison --out artifacts/
python -m rvu.rvu_core.operator run --log artifacts/log.jsonl --K artifacts/K.json --out artifacts/rvu_out.json
python -m rvu.rvu_core.auditor verify --cert artifacts/cert.json --log artifacts/log.jsonl --K '["a-00000001"]'
```

## QA and regression
```bash
make test
make regression
pytest -m bench_smoke
pytest -m bench_full
```

## Safety note
- We do not ship exploit payloads; benchmark-defined attack strings remain in external benchmark repos.
- Tool execution is mediated by `ToolGateway` allowlists (paths, hosts, tool names).

## References
- InjecAgent benchmark and paper.
- BIPIA benchmark and paper.
- AgentDojo benchmark and paper.
- Fides/IFC planner paper (optional comparative baseline).
