"""Plot generation utilities for RVU benchmark reports.

Uses matplotlib with the Agg (non-interactive) backend and saves all
figures as PNG files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, filling NaN with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def plot_asr_vs_utility(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot: Attack Success Rate vs Utility, coloured by defense.

    Parameters
    ----------
    df:
        Summary DataFrame with at least ``attack_success_rate`` and one
        of ``utility_success_rate`` or ``utility_score``.
    output_path:
        Destination PNG file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pick the best available utility column.
    util_col = None
    for candidate in ("utility_success_rate", "utility_score"):
        if candidate in df.columns:
            util_col = candidate
            break

    asr_col = "attack_success_rate"
    if util_col is None or asr_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    defense_col = "defense" if "defense" in df.columns else None

    if defense_col:
        for name, group in df.groupby(defense_col):
            ax.scatter(
                _safe_numeric(group[util_col]),
                _safe_numeric(group[asr_col]),
                label=str(name),
                s=80,
                alpha=0.8,
            )
        ax.legend(title="Defense", loc="best")
    else:
        ax.scatter(
            _safe_numeric(df[util_col]),
            _safe_numeric(df[asr_col]),
            s=80,
            alpha=0.8,
        )

    ax.set_xlabel(util_col.replace("_", " ").title())
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate vs Utility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def plot_overhead_by_defense(df: pd.DataFrame, output_path: str | Path) -> None:
    """Horizontal bar chart: mean wall-time per defense.

    Parameters
    ----------
    df:
        Summary DataFrame with ``mean_wall_time_s`` and ``defense`` columns.
    output_path:
        Destination PNG file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    time_col = "mean_wall_time_s"
    defense_col = "defense"
    if time_col not in df.columns or defense_col not in df.columns:
        return

    grouped = df.groupby(defense_col)[time_col].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        grouped.index.astype(str), grouped.values, color="steelblue", alpha=0.85
    )
    ax.set_xlabel("Mean Wall Time (s)")
    ax.set_title("Overhead by Defense")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, grouped.values):
        ax.text(
            bar.get_width() + 0.01 * max(grouped.values.max(), 1),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def plot_certificate_pass_rate(df: pd.DataFrame, output_path: str | Path) -> None:
    """Horizontal bar chart: RVU certificate auditor pass rates.

    Parameters
    ----------
    df:
        Summary DataFrame with ``auditor_pass_rate`` and ``defense`` columns.
    output_path:
        Destination PNG file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cert_col = "auditor_pass_rate"
    defense_col = "defense"
    if cert_col not in df.columns or defense_col not in df.columns:
        return

    rvu_df = df[df[defense_col].str.lower() == "rvu"].copy()
    if rvu_df.empty:
        return

    # Build label from available columns.
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
            f"{val:.2%}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
