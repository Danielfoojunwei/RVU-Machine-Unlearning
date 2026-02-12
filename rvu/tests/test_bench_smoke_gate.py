from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.bench_smoke
def test_benchmark_cli_env_is_declared() -> None:
    required = ["AGENTDOJO_CLI", "INJECAGENT_CLI", "BIPIA_CLI"]
    missing = [name for name in required if name not in os.environ]
    if missing:
        pytest.skip(f"Missing benchmark CLIs: {', '.join(missing)}")

    subprocess.run([os.environ["AGENTDOJO_CLI"], "--help"], check=True)
    subprocess.run([os.environ["INJECAGENT_CLI"], "--help"], check=True)
    subprocess.run([os.environ["BIPIA_CLI"], "--help"], check=True)
