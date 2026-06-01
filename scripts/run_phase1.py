#!/usr/bin/env python3
"""Run Phase 1 (single-matrix corruption) experiments.

Example
-------
python scripts/run_phase1.py \\
    --model EleutherAI/pythia-70m \\
    --method random \\
    --fraction 0.01 \\
    --output-dir results/phase1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weight_corruption.analysis.results import save_results
from weight_corruption.experiments.runner import CorruptionConfig, Phase1Runner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: corrupt each weight matrix in turn and evaluate"
    )
    p.add_argument("--model", default="EleutherAI/pythia-70m")
    p.add_argument(
        "--method",
        default="random",
        choices=["random", "zeroing", "shuffle", "gaussian", "quantize"],
    )
    p.add_argument("--fraction", type=float, default=0.01, help="Fraction of weights to corrupt")
    p.add_argument("--sigma", type=float, default=0.01, help="Sigma for Gaussian noise")
    p.add_argument("--bits", type=int, default=8, choices=[3, 4, 8], help="Bits for quantization")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/phase1")
    p.add_argument("--lm-samples", type=int, default=50)
    p.add_argument("--behavioral-samples", type=int, default=30)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--categories",
        nargs="+",
        help="Restrict to these categories: attention mlp embedding output",
    )
    p.add_argument("--no-behavioral", action="store_true", help="Skip behavioural divergence")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.model} …")
    runner = Phase1Runner(
        model_name=args.model,
        device=args.device,
        lm_samples=args.lm_samples,
        behavioral_samples=args.behavioral_samples,
        max_length=args.max_length,
    )

    config = CorruptionConfig(
        method=args.method,
        fraction=args.fraction,
        sigma=args.sigma,
        bits=args.bits,
        seed=args.seed,
    )

    cat_filter = None
    if args.categories:
        cats = set(args.categories)
        cat_filter = lambda m: m.category in cats  # noqa: E731

    results = runner.run_all(
        config,
        matrix_filter=cat_filter,
        include_behavioral=not args.no_behavioral,
    )

    safe_model = args.model.replace("/", "_")
    prefix = f"{safe_model}_{args.method}_{args.fraction}"
    save_results(results, args.output_dir, prefix=prefix)
    print(f"Done — {len(results)} experiments saved to {args.output_dir}/{prefix}.*")


if __name__ == "__main__":
    main()
