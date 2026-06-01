"""Modal job: baseline evaluation of EleutherAI/pythia-70m (no corruption).

Run with:
    doppler run --project claude-mobilr --config prd -- modal run scripts/modal_baseline.py
"""

import json

import modal

MODEL_NAME = "EleutherAI/pythia-70m"

# Tier 1 + Tier 2 tasks (wikitext gives perplexity; rest are accuracy benchmarks)
TASKS = [
    "wikitext",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
]

app = modal.App("pythia-70m-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=2.14",
        "accelerate>=0.24",
        "lm-eval>=0.4.3",
        "numpy",
        "pandas",
    )
)

volume = modal.Volume.from_name("weight-corruption-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="t4",
    timeout=3600,
    volumes={"/results": volume},
)
def run_baseline() -> dict:
    import json as _json

    import lm_eval

    print(f"Model : {MODEL_NAME}")
    print(f"Tasks : {TASKS}\n")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={MODEL_NAME},dtype=float32",
        tasks=TASKS,
        batch_size=32,
        log_samples=False,
    )

    # Pull out the metrics we care about (drop _stderr entries for readability)
    summary: dict = {}
    for task, metrics in results["results"].items():
        summary[task] = {k: v for k, v in metrics.items() if not k.endswith(",none")}

    # Pretty-print
    print("\n" + "=" * 60)
    print(f"BASELINE RESULTS — {MODEL_NAME}")
    print("=" * 60)
    for task, metrics in summary.items():
        print(f"\n{task}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:40s} {v:.4f}")
            else:
                print(f"  {k:40s} {v}")

    # Persist to volume
    out_path = f"/results/baseline_{MODEL_NAME.replace('/', '_')}.json"
    with open(out_path, "w") as fh:
        _json.dump({"model": MODEL_NAME, "tasks": TASKS, "results": summary}, fh, indent=2)
    volume.commit()
    print(f"\nSaved to {out_path}")

    return summary


@app.local_entrypoint()
def main():
    summary = run_baseline.remote()
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
