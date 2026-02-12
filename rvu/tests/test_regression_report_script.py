from __future__ import annotations

import subprocess
from pathlib import Path


def test_regression_report_script(tmp_path: Path) -> None:
    current = Path("tests/fixtures/current_metrics.json")
    baseline = Path("tests/fixtures/baseline_metrics.json")
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            "python",
            "scripts/regression_report.py",
            "--metrics",
            str(current),
            "--baseline",
            str(baseline),
            "--out",
            str(out_dir),
        ],
        check=True,
    )

    report = (out_dir / "regression_report.md").read_text()
    assert "agentdojo" in report
    assert "ASR" in report
