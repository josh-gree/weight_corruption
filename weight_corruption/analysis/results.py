from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import pandas as pd

from ..experiments.runner import ExperimentResult


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "matrix_name": r.matrix_name,
                "layer_idx": r.layer_idx,
                "matrix_type": r.matrix_type,
                "category": r.category,
                "matrix_shape": str(r.matrix_shape),
                "corruption_method": r.corruption_method,
                "corruption_fraction": r.corruption_fraction,
                "baseline_loss": r.baseline_loss,
                "corrupted_loss": r.corrupted_loss,
                "delta_loss": r.delta_loss,
                "baseline_perplexity": r.baseline_perplexity,
                "corrupted_perplexity": r.corrupted_perplexity,
                "kl_divergence": r.kl_divergence,
                "top1_agreement": r.top1_agreement,
                "top5_agreement": r.top5_agreement,
            }
        )
    return pd.DataFrame(rows)


def save_results(
    results: list[ExperimentResult],
    output_dir: Union[str, Path],
    prefix: str = "results",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    df.to_csv(output_dir / f"{prefix}.csv", index=False)

    with open(output_dir / f"{prefix}.json", "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2, default=str)

    print(f"Saved {len(results)} results to {output_dir / prefix}.[csv|json]")


def load_results(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, metric: str = "delta_loss") -> None:
    print(f"\n── Top 10 most sensitive matrices ({metric}) ──")
    print(df.groupby("matrix_name")[metric].mean().nlargest(10).to_string())

    print(f"\n── Sensitivity by category ({metric}) ──")
    print(
        df.groupby("category")[metric]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )

    print(f"\n── Sensitivity by matrix type ({metric}) ──")
    print(
        df.groupby("matrix_type")[metric]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )
