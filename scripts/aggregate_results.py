#!/usr/bin/env python3
"""aggregate_results.py -- Aggregate benchmark run metrics into reports.

Reads JSON metrics files from a runs directory and produces:
  - summary_table.csv
  - summary_table.md  (markdown table)
  - asr_vs_utility.png
  - overhead_vs_defense.png
  - rvu_certificate_pass.png
  - report_meta.json

Usage:
    python scripts/aggregate_results.py \
        --in artifacts/runs \
        --out artifacts/report
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must be set before pyplot import.
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_metrics_files(runs_dir: Path) -> list[Path]:
    """Recursively find all JSON files under *runs_dir*."""
    files: list[Path] = []
    for p in sorted(runs_dir.rglob("*.json")):
        if p.is_file():
            files.append(p)
    return files


def _load_all_metrics(files: list[Path]) -> list[dict[str, Any]]:
    """Load and return parsed JSON dicts, skipping unparseable files."""
    records: list[dict[str, Any]] = []
    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data["_source_file"] = str(f)
                records.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item["_source_file"] = str(f)
                        records.append(item)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] skipping {f}: {exc}", file=sys.stderr)
    return records


def _collect_system_info(repo_root: Path) -> dict[str, Any]:
    """Gather system metadata for the report."""
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }

    # lscpu
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10
        )
        info["lscpu"] = result.stdout.strip()
    except Exception:
        info["lscpu"] = "unavailable"

    # RAM
    try:
        result = subprocess.run(
            ["free", "-h"], capture_output=True, text=True, timeout=10
        )
        info["ram"] = result.stdout.strip()
    except Exception:
        info["ram"] = "unavailable"

    # Git commit hash
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        info["repo_commit"] = result.stdout.strip()
    except Exception:
        info["repo_commit"] = "unknown"

    # Model manifest
    manifest_path = repo_root / "artifacts" / "model_manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r") as fh:
                info["model_manifest"] = json.load(fh)
        except Exception:
            info["model_manifest"] = "unreadable"
    else:
        info["model_manifest"] = "not found"

    return info


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

# Canonical column ordering (present columns are shown in this order).
_PREFERRED_COLS = [
    "benchmark", "model", "defense", "mode",
    "utility_success_rate", "utility_score",
    "attack_success_rate", "robustness_rate", "violation_rate",
    "mean_steps", "mean_tokens", "mean_wall_time_s",
    "certificates_generated", "auditor_pass_rate",
]


def _build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalise heterogeneous records into a single DataFrame."""
    df = pd.json_normalize(records, sep="_")

    # Drop internal helper columns.
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Reorder columns: preferred first, then remainder alphabetically.
    present_preferred = [c for c in _PREFERRED_COLS if c in df.columns]
    remaining = sorted(set(df.columns) - set(present_preferred))
    df = df[present_preferred + remaining]

    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows)")


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    md = df.to_markdown(index=False)
    with open(path, "w") as fh:
        fh.write(md)
        fh.write("\n")
    print(f"  -> {path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, filling NaN with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _plot_asr_vs_utility(df: pd.DataFrame, path: Path) -> None:
    """Scatter: Attack Success Rate vs Utility, coloured by defense."""
    # Determine which utility column is available.
    util_col = None
    for candidate in ("utility_success_rate", "utility_score"):
        if candidate in df.columns:
            util_col = candidate
            break
    asr_col = "attack_success_rate"
    if util_col is None or asr_col not in df.columns:
        print(f"  [skip] asr_vs_utility.png -- missing columns ({util_col}, {asr_col})")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    defense_col = "defense" if "defense" in df.columns else None

    if defense_col:
        for name, group in df.groupby(defense_col):
            ax.scatter(
                _safe_numeric(group[util_col]),
                _safe_numeric(group[asr_col]),
                label=str(name), s=80, alpha=0.8,
            )
        ax.legend(title="Defense", loc="best")
    else:
        ax.scatter(
            _safe_numeric(df[util_col]),
            _safe_numeric(df[asr_col]),
            s=80, alpha=0.8,
        )

    ax.set_xlabel(util_col.replace("_", " ").title())
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate vs Utility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def _plot_overhead_vs_defense(df: pd.DataFrame, path: Path) -> None:
    """Bar chart: mean wall-time per defense."""
    time_col = "mean_wall_time_s"
    defense_col = "defense"
    if time_col not in df.columns or defense_col not in df.columns:
        print(f"  [skip] overhead_vs_defense.png -- missing columns")
        return

    grouped = df.groupby(defense_col)[time_col].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(grouped.index.astype(str), grouped.values, color="steelblue", alpha=0.85)
    ax.set_xlabel("Mean Wall Time (s)")
    ax.set_title("Overhead by Defense")
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bar values.
    for bar, val in zip(bars, grouped.values):
        ax.text(
            bar.get_width() + 0.01 * grouped.values.max(),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s", va="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def _plot_rvu_certificate(df: pd.DataFrame, path: Path) -> None:
    """Bar chart: RVU certificate pass rates by benchmark/model."""
    cert_col = "auditor_pass_rate"
    defense_col = "defense"
    if cert_col not in df.columns or defense_col not in df.columns:
        print(f"  [skip] rvu_certificate_pass.png -- missing columns")
        return

    rvu_df = df[df[defense_col].str.lower() == "rvu"].copy()
    if rvu_df.empty:
        print(f"  [skip] rvu_certificate_pass.png -- no RVU rows")
        return

    # Build a label from benchmark + model if available.
    label_parts = []
    for col in ("benchmark", "model", "mode"):
        if col in rvu_df.columns:
            label_parts.append(col)
    if label_parts:
        rvu_df["_label"] = rvu_df[label_parts].astype(str).agg(" / ".join, axis=1)
    else:
        rvu_df["_label"] = [f"run_{i}" for i in range(len(rvu_df))]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(rvu_df))))
    vals = _safe_numeric(rvu_df[cert_col])
    bars = ax.barh(rvu_df["_label"], vals, color="forestgreen", alpha=0.85)
    ax.set_xlabel("Auditor Pass Rate")
    ax.set_xlim(0, 1.05)
    ax.set_title("RVU Certificate Verification Pass Rate")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2%}", va="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RVU benchmark run metrics into a report."
    )
    parser.add_argument(
        "--in", dest="in_dir", default="artifacts/runs",
        help="Input directory containing JSON metrics files (default: artifacts/runs)",
    )
    parser.add_argument(
        "--out", dest="out_dir", default="artifacts/report",
        help="Output directory for the generated report (default: artifacts/report)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    in_dir = Path(args.in_dir)
    if not in_dir.is_absolute():
        in_dir = repo_root / in_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {in_dir}")
    print(f"Output : {out_dir}")

    # ---- Load metrics ----
    metrics_files = _find_metrics_files(in_dir)
    if not metrics_files:
        print(f"No JSON metrics files found under {in_dir}. Nothing to aggregate.")
        sys.exit(0)

    print(f"Found {len(metrics_files)} metrics file(s).")
    records = _load_all_metrics(metrics_files)
    if not records:
        print("No parseable records found. Exiting.")
        sys.exit(0)

    print(f"Loaded {len(records)} record(s).")

    # ---- Build DataFrame ----
    df = _build_dataframe(records)

    # ---- Write tables ----
    print("\nGenerating tables...")
    _write_csv(df, out_dir / "summary_table.csv")
    _write_markdown(df, out_dir / "summary_table.md")

    # ---- Generate plots ----
    print("\nGenerating plots...")
    _plot_asr_vs_utility(df, out_dir / "asr_vs_utility.png")
    _plot_overhead_vs_defense(df, out_dir / "overhead_vs_defense.png")
    _plot_rvu_certificate(df, out_dir / "rvu_certificate_pass.png")

    # ---- System info & meta report ----
    print("\nCollecting system info...")
    system_info = _collect_system_info(repo_root)

    report_meta: dict[str, Any] = {
        "generated_at": system_info["timestamp"],
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "metrics_files_count": len(metrics_files),
        "records_count": len(records),
        "columns": list(df.columns),
        "system": system_info,
        "artifacts": {
            "summary_table_csv": str(out_dir / "summary_table.csv"),
            "summary_table_md": str(out_dir / "summary_table.md"),
            "asr_vs_utility_png": str(out_dir / "asr_vs_utility.png"),
            "overhead_vs_defense_png": str(out_dir / "overhead_vs_defense.png"),
            "rvu_certificate_pass_png": str(out_dir / "rvu_certificate_pass.png"),
        },
    }

    meta_path = out_dir / "report_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(report_meta, fh, indent=2, default=str)
    print(f"\n  -> {meta_path}")

    print("\nDone. Report written to", out_dir)


if __name__ == "__main__":
    main()
