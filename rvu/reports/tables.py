"""Table generation utilities for RVU benchmark reports.

Provides functions to normalise raw metrics dicts into flat table rows,
build summary DataFrames, and render them as Markdown or CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# Canonical column ordering -- present columns appear in this order first,
# then any remaining columns are appended alphabetically.
_PREFERRED_COLS = [
    "benchmark",
    "model",
    "defense",
    "mode",
    "utility_success_rate",
    "utility_score",
    "attack_success_rate",
    "robustness_rate",
    "violation_rate",
    "mean_steps",
    "mean_tokens",
    "mean_wall_time_s",
    "certificates_generated",
    "auditor_pass_rate",
    "n_episodes",
    "n_cases",
    "n_samples",
]


def metrics_to_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalise a metrics dict into a flat table row.

    Nested dicts are flattened with underscore separators.  Keys starting
    with ``_`` are dropped.  Only scalar values (str, int, float, bool)
    are kept.

    Parameters
    ----------
    metrics:
        Raw metrics dict as produced by a benchmark runner.

    Returns
    -------
    dict[str, Any]
        Flat dict suitable for insertion into a DataFrame row.
    """
    row: dict[str, Any] = {}
    _flatten(row, metrics, prefix="")
    return row


def _flatten(target: dict[str, Any], source: dict[str, Any], prefix: str) -> None:
    """Recursively flatten *source* into *target* with underscore-separated keys."""
    for key, value in source.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if full_key.startswith("_"):
            continue
        if isinstance(value, dict):
            _flatten(target, value, full_key)
        elif isinstance(value, (str, int, float, bool)):
            target[full_key] = value


def build_summary_table(metrics_list: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a summary DataFrame from a list of metrics dicts.

    Parameters
    ----------
    metrics_list:
        List of raw metrics dicts (one per benchmark run).

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per metrics dict, columns ordered
        according to :data:`_PREFERRED_COLS`.
    """
    rows = [metrics_to_row(m) for m in metrics_list]
    df = pd.DataFrame(rows)

    # Reorder columns: preferred first, then remainder alphabetically.
    present_preferred = [c for c in _PREFERRED_COLS if c in df.columns]
    remaining = sorted(set(df.columns) - set(present_preferred))
    df = df[present_preferred + remaining]

    return df


def render_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table string.

    Parameters
    ----------
    df:
        The DataFrame to render.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    return df.to_markdown(index=False)


def render_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to a CSV file.

    Parameters
    ----------
    df:
        The DataFrame to write.
    path:
        Destination file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
