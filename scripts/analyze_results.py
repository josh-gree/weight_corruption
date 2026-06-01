#!/usr/bin/env python3
"""Analyze and visualize corruption experiment results.

Example
-------
python scripts/analyze_results.py results/phase1/EleutherAI_pythia-70m_random_0.01.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weight_corruption.analysis.results import load_results, summarize
from weight_corruption.analysis.visualizations import (
    plot_category_comparison,
    plot_kl_sensitivity_map,
    plot_layer_sensitivity_heatmap,
    plot_layer_sensitivity_line,
    plot_ranked_importance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize weight corruption experiment results")
    p.add_argument("results_csv", help="Path to a results CSV file")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--metric", default="delta_loss")
    p.add_argument("--top-n", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_results(args.results_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summarize(df, metric=args.metric)

    plots = [
        (plot_layer_sensitivity_heatmap, "heatmap_layer_mattype.png", {}),
        (plot_category_comparison,       "category_comparison.png",   {}),
        (plot_layer_sensitivity_line,    "layer_sensitivity_line.png",{}),
        (plot_kl_sensitivity_map,        "kl_sensitivity_map.png",    {}),
    ]

    for fn, fname, kwargs in plots:
        fn(df, output_path=out / fname, **kwargs)
        print(f"Saved {out / fname}")

    plot_ranked_importance(df, metric=args.metric, top_n=args.top_n, output_path=out / "ranked_importance.png")
    print(f"Saved {out / 'ranked_importance.png'}")

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
