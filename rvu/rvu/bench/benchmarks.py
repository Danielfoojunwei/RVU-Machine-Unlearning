from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    utility: float
    asr: float
    latency_s: float
    certificate_pass_rate: float


@dataclass(frozen=True)
class BenchmarkCommand:
    name: str
    cmd: list[str]


def run_command(command: BenchmarkCommand, output_dir: Path) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{command.name}_stdout.log"
    with output_file.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command.cmd, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({command.name}): {' '.join(command.cmd)}")

    metrics_path = output_dir / f"{command.name}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected benchmark metrics at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    return BenchmarkResult(
        name=command.name,
        utility=float(metrics["utility"]),
        asr=float(metrics["asr"]),
        latency_s=float(metrics["latency_s"]),
        certificate_pass_rate=float(metrics["certificate_pass_rate"]),
    )
