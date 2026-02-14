#!/usr/bin/env python3
"""regression_compare.py -- Compare current metrics against golden baselines.

Loads golden and current JSON metrics files, compares each metric within
configurable tolerances, and prints a pass/fail table. Exits with code 1 if
any metric is outside tolerance.

Usage:
    python scripts/regression_compare.py \
        --golden artifacts/golden \
        --current artifacts/runs \
        [--tolerance-file configs/tolerances.yaml]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Default tolerances
# ---------------------------------------------------------------------------

_DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    # Utility metrics: +/- 5 percentage points
    "utility_success_rate": {"abs": 0.05, "rel": None},
    "utility_score":        {"abs": 0.05, "rel": None},
    "robustness_rate":      {"abs": 0.05, "rel": None},

    # Security metrics: +/- 3 percentage points
    "attack_success_rate":  {"abs": 0.03, "rel": None},
    "violation_rate":       {"abs": 0.03, "rel": None},
    "auditor_pass_rate":    {"abs": 0.03, "rel": None},

    # Latency / overhead: +/- 50% relative (wide for CPU variance)
    "mean_wall_time_s":     {"abs": None, "rel": 0.50},
    "mean_tokens":          {"abs": None, "rel": 0.50},
    "mean_steps":           {"abs": None, "rel": 0.50},

    # Certificate counts: +/- 50% relative
    "certificates_generated": {"abs": None, "rel": 0.50},
}

# Fallback tolerance for any metric not listed above.
_FALLBACK_TOLERANCE: dict[str, float | None] = {"abs": 0.05, "rel": 0.50}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_dir(directory: Path) -> dict[str, Any]:
    """Load and merge all JSON files in *directory* into a flat metric dict.

    If multiple files are found, they are merged (last-write-wins for
    duplicate keys). Nested dicts are flattened with dot separators.
    """
    merged: dict[str, Any] = {}
    files = sorted(directory.rglob("*.json"))
    if not files:
        print(f"  [warn] No JSON files found in {directory}", file=sys.stderr)
        return merged

    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                _flatten_into(merged, data, prefix="")
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        _flatten_into(merged, item, prefix="")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] skipping {f}: {exc}", file=sys.stderr)

    return merged


def _flatten_into(
    target: dict[str, Any],
    source: dict[str, Any],
    prefix: str,
) -> None:
    """Recursively flatten *source* into *target* with dot-separated keys."""
    for key, value in source.items():
        full_key = f"{prefix}.{key}" if prefix else key
        # Skip internal/metadata keys.
        if full_key.startswith("_"):
            continue
        if isinstance(value, dict):
            _flatten_into(target, value, full_key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[full_key] = value


def _load_tolerances(path: Path | None) -> dict[str, dict[str, float | None]]:
    """Load tolerance overrides from a YAML file, falling back to defaults."""
    tolerances = dict(_DEFAULT_TOLERANCES)

    if path is None or not path.is_file():
        return tolerances

    if not _HAS_YAML:
        print(
            "  [warn] pyyaml not installed -- ignoring tolerance file",
            file=sys.stderr,
        )
        return tolerances

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        return tolerances

    for metric_name, spec in raw.items():
        if isinstance(spec, dict):
            tolerances[metric_name] = {
                "abs": spec.get("abs"),
                "rel": spec.get("rel"),
            }
        elif isinstance(spec, (int, float)):
            # Treat a bare number as an absolute tolerance.
            tolerances[metric_name] = {"abs": float(spec), "rel": None}

    return tolerances


def _get_tolerance(
    metric: str,
    tolerances: dict[str, dict[str, float | None]],
) -> dict[str, float | None]:
    """Return tolerance spec for a metric, walking dot-prefix hierarchy."""
    # Exact match.
    if metric in tolerances:
        return tolerances[metric]

    # Try the last segment (e.g., "agentdojo.attack_success_rate" -> "attack_success_rate").
    last_segment = metric.rsplit(".", 1)[-1]
    if last_segment in tolerances:
        return tolerances[last_segment]

    return dict(_FALLBACK_TOLERANCE)


def _within_tolerance(
    golden_val: float,
    current_val: float,
    tol: dict[str, float | None],
) -> tuple[bool, float, str]:
    """Check whether the delta between golden and current is within tolerance.

    Returns (pass_bool, delta_value, tolerance_description).
    """
    delta = current_val - golden_val

    abs_tol = tol.get("abs")
    rel_tol = tol.get("rel")

    # If both are specified, the metric passes if it satisfies EITHER.
    checks: list[tuple[bool, str]] = []

    if abs_tol is not None:
        ok = abs(delta) <= abs_tol
        checks.append((ok, f"abs<={abs_tol}"))

    if rel_tol is not None:
        if golden_val == 0:
            # Avoid division by zero; pass only if current is also zero.
            rel_ok = current_val == 0
        else:
            rel_ok = abs(delta / golden_val) <= rel_tol
        checks.append((rel_ok, f"rel<={rel_tol:.0%}"))

    if not checks:
        # No tolerance defined -- always pass.
        return True, delta, "none"

    passed = any(ok for ok, _ in checks)
    desc = " | ".join(label for _, label in checks)
    return passed, delta, desc


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

class ComparisonResult:
    """Container for one metric comparison."""

    def __init__(
        self,
        metric: str,
        golden: float,
        current: float,
        delta: float,
        tolerance_desc: str,
        passed: bool,
    ) -> None:
        self.metric = metric
        self.golden = golden
        self.current = current
        self.delta = delta
        self.tolerance_desc = tolerance_desc
        self.passed = passed

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def compare_metrics(
    golden: dict[str, float],
    current: dict[str, float],
    tolerances: dict[str, dict[str, float | None]],
) -> list[ComparisonResult]:
    """Compare every metric present in *golden* against *current*."""
    results: list[ComparisonResult] = []

    # Union of keys; metrics only in golden are flagged as missing.
    all_keys = sorted(set(golden.keys()) | set(current.keys()))

    for key in all_keys:
        g = golden.get(key)
        c = current.get(key)

        if g is None:
            # Metric only in current -- informational, always pass.
            if c is not None:
                results.append(ComparisonResult(
                    metric=key, golden=math.nan, current=c,
                    delta=math.nan, tolerance_desc="new metric", passed=True,
                ))
            continue

        if c is None:
            # Metric missing from current -- FAIL.
            results.append(ComparisonResult(
                metric=key, golden=g, current=math.nan,
                delta=math.nan, tolerance_desc="MISSING", passed=False,
            ))
            continue

        tol = _get_tolerance(key, tolerances)
        passed, delta, desc = _within_tolerance(g, c, tol)
        results.append(ComparisonResult(
            metric=key, golden=g, current=c,
            delta=delta, tolerance_desc=desc, passed=passed,
        ))

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_table(results: list[ComparisonResult]) -> None:
    """Print a human-readable comparison table to stdout."""
    # Column headers.
    headers = ["Metric", "Golden", "Current", "Delta", "Tolerance", "Status"]
    rows: list[list[str]] = []

    for r in results:
        golden_str = f"{r.golden:.6g}" if not math.isnan(r.golden) else "---"
        current_str = f"{r.current:.6g}" if not math.isnan(r.current) else "---"
        delta_str = f"{r.delta:+.6g}" if not math.isnan(r.delta) else "---"
        rows.append([r.metric, golden_str, current_str, delta_str, r.tolerance_desc, r.status])

    # Compute column widths.
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current benchmark metrics against golden baselines."
    )
    parser.add_argument(
        "--golden", required=True,
        help="Directory containing golden baseline JSON metrics.",
    )
    parser.add_argument(
        "--current", required=True,
        help="Directory containing current run JSON metrics.",
    )
    parser.add_argument(
        "--tolerance-file", default=None,
        help="Optional YAML file with per-metric tolerance overrides.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    golden_dir = Path(args.golden)
    if not golden_dir.is_absolute():
        golden_dir = repo_root / golden_dir
    current_dir = Path(args.current)
    if not current_dir.is_absolute():
        current_dir = repo_root / current_dir
    tol_file = Path(args.tolerance_file) if args.tolerance_file else None
    if tol_file and not tol_file.is_absolute():
        tol_file = repo_root / tol_file

    print(f"Golden  : {golden_dir}")
    print(f"Current : {current_dir}")
    if tol_file:
        print(f"Tol file: {tol_file}")
    print()

    # ---- Load data ----
    golden_metrics = _load_json_dir(golden_dir)
    current_metrics = _load_json_dir(current_dir)

    if not golden_metrics:
        print("ERROR: No numeric metrics found in golden directory.", file=sys.stderr)
        sys.exit(1)

    if not current_metrics:
        print("ERROR: No numeric metrics found in current directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Golden metrics : {len(golden_metrics)} numeric values")
    print(f"Current metrics: {len(current_metrics)} numeric values")
    print()

    # ---- Load tolerances ----
    tolerances = _load_tolerances(tol_file)

    # ---- Compare ----
    results = compare_metrics(golden_metrics, current_metrics, tolerances)

    # ---- Print table ----
    _print_table(results)

    # ---- Summary ----
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print()
    print(f"Total: {total}   Passed: {passed}   Failed: {failed}")

    if failed > 0:
        print("\nREGRESSION DETECTED -- exiting with code 1")
        sys.exit(1)
    else:
        print("\nAll metrics within tolerance.")
        sys.exit(0)


if __name__ == "__main__":
    main()
