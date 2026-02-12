from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .benchmarks import BenchmarkCommand, BenchmarkResult, run_command
from .config import BenchConfig, RunMetadata


def collect_metadata() -> RunMetadata:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0]
    return RunMetadata(commit_hash=commit, gpu_name=gpu_name, machine=platform.platform())


def run_benchmarks(config: BenchConfig, smoke: bool) -> Path:
    run_dir = config.output_root / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata()

    bench_commands: list[BenchmarkCommand] = []
    mode = "smoke" if smoke else "full"
    if config.run_agentdojo:
        bench_commands.append(
            BenchmarkCommand(
                name="agentdojo",
                cmd=[
                    "python",
                    "-m",
                    "rvu.qa.benchmark_agentdojo",
                    "--mode",
                    mode,
                    "--out",
                    str(run_dir),
                ],
            )
        )
    if config.run_injecagent:
        bench_commands.append(
            BenchmarkCommand(
                name="injecagent",
                cmd=[
                    "python",
                    "-m",
                    "rvu.qa.benchmark_injecagent",
                    "--mode",
                    mode,
                    "--out",
                    str(run_dir),
                ],
            )
        )
    if config.run_bipia:
        bench_commands.append(
            BenchmarkCommand(
                name="bipia",
                cmd=[
                    "python",
                    "-m",
                    "rvu.qa.benchmark_bipia",
                    "--mode",
                    mode,
                    "--out",
                    str(run_dir),
                ],
            )
        )

    results: list[BenchmarkResult] = []
    for bench_command in bench_commands:
        results.append(run_command(bench_command, run_dir))

    metrics_payload = {
        "metadata": asdict(metadata),
        "model": asdict(config.model),
        "results": [asdict(result) for result in results],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return run_dir
