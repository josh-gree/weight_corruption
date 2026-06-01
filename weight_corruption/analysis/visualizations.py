from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_layer_sensitivity_heatmap(
    df: pd.DataFrame,
    metric: str = "delta_loss",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Layer × Matrix-Type Sensitivity",
) -> plt.Figure:
    """Heatmap: rows = layers, columns = matrix types, values = mean metric."""
    pivot = df.pivot_table(
        index="layer_idx",
        columns="matrix_type",
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.4))
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        annot=True,
        fmt=".3f",
        linewidths=0.4,
        cbar_kws={"label": metric},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Matrix type")
    ax.set_ylabel("Layer index")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_category_comparison(
    df: pd.DataFrame,
    metric: str = "delta_loss",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Bar chart: mean metric by matrix category."""
    agg = df.groupby("category")[metric].mean().sort_values(ascending=False)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(agg)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(8, 5))
    agg.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(f"Mean {metric} by matrix category")
    ax.set_xlabel("Category")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_layer_sensitivity_line(
    df: pd.DataFrame,
    metric: str = "delta_loss",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Line plot: metric vs layer index, one line per category."""
    layer_df = df[df["layer_idx"] >= 0]

    fig, ax = plt.subplots(figsize=(12, 5))
    for category, group in layer_df.groupby("category"):
        agg = group.groupby("layer_idx")[metric].mean()
        ax.plot(agg.index, agg.values, marker="o", label=category)

    ax.set_title(f"{metric} vs layer index")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ranked_importance(
    df: pd.DataFrame,
    metric: str = "delta_loss",
    top_n: int = 30,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N most sensitive matrices."""
    agg = df.groupby("matrix_name")[metric].mean().nlargest(top_n)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(agg)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    agg.plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_title(f"Top {top_n} most sensitive matrices ({metric})")
    ax.set_xlabel(metric)
    ax.invert_yaxis()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_robustness_curves(
    dfs: dict[float, pd.DataFrame],
    matrix_names: list[str],
    metric: str = "delta_loss",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Robustness curve: metric vs corruption fraction for selected matrices.

    *dfs* maps fraction → DataFrame (one DataFrame per fraction level).
    """
    fractions = sorted(dfs.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for mname in matrix_names:
        values = []
        for frac in fractions:
            row = dfs[frac][dfs[frac]["matrix_name"] == mname]
            values.append(row[metric].iloc[0] if len(row) > 0 else float("nan"))
        label = ".".join(mname.split(".")[-2:])
        ax.plot(fractions, values, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_title("Robustness curves")
    ax.set_xlabel("Corruption fraction")
    ax.set_ylabel(metric)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_kl_sensitivity_map(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """KL-divergence version of the layer × matrix-type heatmap."""
    return plot_layer_sensitivity_heatmap(
        df,
        metric="kl_divergence",
        output_path=output_path,
        title="KL Divergence Sensitivity Map (Layer × Matrix Type)",
    )
