# RVU-Machine-Unlearning

Empirical evaluation of **RVU (Recovery-based Verifiable Unlearning)** against
SOTA prompt-injection defenses on canonical benchmarks. CPU-only, open weights,
real benchmarks, no mocks.

## Defenses evaluated

| Defense | Description |
|---------|-------------|
| **Vanilla** | Standard ReAct agent, no defense applied |
| **PromptGuard** | Filter untrusted content via [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) classifier |
| **FATH** | Formatting + Authentication Tags + Hash verification ([paper](https://github.com/Jayfeather1024/FATH)) |
| **RVG-only** | Verifier-gated tool boundary with scope allowlists and taint propagation; no unlearning |
| **RVU** | Full recovery-based verifiable unlearning: provenance tracking (SQLite + FAISS), contamination closure, purge/rollback, certificate emission, auditor verification |

## Benchmarks

| Benchmark | Source | Scope |
|-----------|--------|-------|
| **AgentDojo** | [usnistgov/agentdojo-inspect](https://github.com/usnistgov/agentdojo-inspect) | Agentic task suites with injection attacks |
| **InjecAgent** | [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | 1,054 tool-augmented agent injection cases |
| **BIPIA** | [microsoft/BIPIA](https://github.com/microsoft/BIPIA) | Indirect prompt injection in contextual data |

## Models

| Model | Quantization | HuggingFace Repo |
|-------|-------------|------------------|
| Qwen2.5-1.5B-Instruct | GGUF Q4_K_M | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| Llama-3.2-1B-Instruct | GGUF Q4_K_M | `bartowski/Llama-3.2-1B-Instruct-GGUF` |

Inference via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) on CPU. Deterministic
generation: `temperature=0, top_p=1, seed=42`.

---

## A) CPU-only Quickstart

```bash
# 1. Set up the Python environment and install all dependencies
./scripts/setup_cpu.sh

# 2. Download models (GGUF LLMs, PromptGuard, embedding model)
./scripts/download_models.sh

# 3. Run smoke tests across all benchmarks
make bench_smoke_cpu
```

## B) Run individual benchmarks

```bash
# AgentDojo
make agentdojo_smoke MODEL=model_a DEFENSE=rvu
make agentdojo_smoke MODEL=model_b DEFENSE=vanilla

# InjecAgent (supports FATH)
make injecagent_smoke MODEL=model_a DEFENSE=fath
make injecagent_smoke MODEL=model_a DEFENSE=rvu

# BIPIA
make bipia_smoke MODEL=model_b DEFENSE=promptguard
make bipia_smoke MODEL=model_a DEFENSE=rvg
```

Or use the shell scripts directly:

```bash
MODEL=model_a DEFENSE=rvu bash scripts/run_agentdojo_smoke.sh
MODEL=model_a DEFENSE=fath bash scripts/run_injecagent_smoke.sh
MODEL=model_b DEFENSE=promptguard bash scripts/run_bipia_smoke.sh
```

## C) Full paper runs

```bash
# Run all benchmark x model x defense combinations
make bench_full_cpu

# Aggregate results into tables and plots
python scripts/aggregate_results.py --in artifacts/runs --out artifacts/report
```

Output:
- `artifacts/report/summary_table.csv` / `.md`
- `artifacts/report/asr_vs_utility.png`
- `artifacts/report/overhead_vs_defense.png`
- `artifacts/report/rvu_certificate_pass.png`
- `artifacts/report/report_meta.json` (includes CPU info, model revisions, git hash)

## D) QA and regression

```bash
# Run unit tests
make test

# Run smoke benchmark tests (requires downloaded models)
pytest -m bench_smoke

# Compare current metrics against golden baselines
python scripts/regression_compare.py \
    --golden artifacts/golden \
    --current artifacts/runs
```

### Regression tolerances

| Metric category | Tolerance |
|-----------------|-----------|
| Utility (success rate) | +/- 5 percentage points |
| Security (ASR, violation) | +/- 3 percentage points |
| Latency / overhead | +/- 50% relative (CPU variance) |

Golden baselines are stored under `artifacts/golden/<benchmark>/<model>/<defense>/metrics.json`.

## E) Reproducibility notes

- **Deterministic generation**: `temperature=0, top_p=1, seed=42, repeat_penalty=1.0`
- **Pinned dependencies**: all versions locked in `scripts/setup_cpu.sh`
- **Explicit benchmark versions**: vendored under `vendor/` with pinned commits
- **Model revision hashes**: recorded in `artifacts/model_manifest.json` by `download_models.sh`
- **CPU-only**: `n_gpu_layers=0` for llama.cpp; PyTorch CPU wheels

## Project structure

```
rvu/
  inference/
    llm_llamacpp.py          # llama.cpp CPU adapter (GGUF models)
    promptguard.py            # PromptGuard-2-86M classifier
  defenses/
    base.py                   # Abstract BaseDefense interface
    vanilla.py                # No-op baseline
    promptguard_defense.py    # PromptGuard filter defense
    fath_adapter.py           # FATH authentication tags + hash
    rvg_only.py               # Verifier-gated tool boundary
    rvu.py                    # Full RVU (provenance + closure + purge + certificate)
  harness/
    agentdojo_runner.py       # AgentDojo benchmark harness
    injecagent_runner.py      # InjecAgent benchmark harness
    bipia_runner.py           # BIPIA benchmark harness
  reports/
    schema.json               # JSON Schema for metrics output
    tables.py                 # Table generation utilities
    plots.py                  # Matplotlib plot generation
configs/
  models.yaml                # LLM and auxiliary model configs
  agentdojo.yaml              # AgentDojo benchmark config
  injecagent.yaml             # InjecAgent benchmark config
  bipia.yaml                  # BIPIA benchmark config
  defenses.yaml               # Defense configurations
  tool_allowlist.yaml         # Tool allowlist for RVG/RVU
scripts/
  setup_cpu.sh                # Environment setup (venv + deps + benchmarks)
  download_models.sh          # Download all model weights
  aggregate_results.py        # Aggregate runs into report
  regression_compare.py       # Golden-vs-current regression check
  run_*.sh                    # Benchmark run wrapper scripts
tests/
  test_smoke_agentdojo.py     # AgentDojo smoke test
  test_smoke_injecagent.py    # InjecAgent smoke test
  test_smoke_bipia.py         # BIPIA smoke test
  test_regression_metrics.py  # Regression comparison tests
  test_certificate_auditor_realruns.py  # RVU certificate E2E test
```

## License

MIT
