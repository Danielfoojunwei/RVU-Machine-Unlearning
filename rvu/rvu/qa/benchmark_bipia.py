from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cli = os.environ.get("BIPIA_CLI")
    if cli is None:
        raise RuntimeError("Set BIPIA_CLI to the BIPIA executable path")

    output = args.out / "bipia_raw.json"
    cmd = [cli, "evaluate", "--output", str(output)]
    if args.mode == "smoke":
        cmd.extend(["--limit", "25"])
    subprocess.run(cmd, check=True)

    raw = json.loads(output.read_text())
    metrics = {
        "utility": float(raw["utility"]),
        "asr": float(raw["attack_success_rate"]),
        "latency_s": float(raw["latency_s"]),
        "certificate_pass_rate": float(raw.get("certificate_pass_rate", 1.0)),
    }
    (args.out / "bipia_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
