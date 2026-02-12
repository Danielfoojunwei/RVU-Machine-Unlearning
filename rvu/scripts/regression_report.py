from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    current = json.loads(args.metrics.read_text())
    baseline = json.loads(args.baseline.read_text())

    by_name_baseline = {entry["name"]: entry for entry in baseline["results"]}
    report_rows: list[str] = [
        "| Benchmark | Utility Δ | ASR Δ | Latency Δ (s) | Cert pass Δ |",
        "|---|---:|---:|---:|---:|",
    ]

    for result in current["results"]:
        previous = by_name_baseline[result["name"]]
        report_rows.append(
            "| {name} | {u:+.4f} | {a:+.4f} | {l:+.4f} | {c:+.4f} |".format(
                name=result["name"],
                u=result["utility"] - previous["utility"],
                a=result["asr"] - previous["asr"],
                l=result["latency_s"] - previous["latency_s"],
                c=result["certificate_pass_rate"] - previous["certificate_pass_rate"],
            )
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "regression_report.md").write_text("\n".join(report_rows), encoding="utf-8")
    (args.out / "regression_report.json").write_text(
        json.dumps({"current": current, "baseline": baseline}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
