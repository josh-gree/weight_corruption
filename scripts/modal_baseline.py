"""Modal job: baseline evaluation of Pythia models (no corruption).

Run with:
    doppler run --project claude-mobilr --config prd -- modal run scripts/modal_baseline.py
"""

import json

import modal

TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
]

# Models to evaluate in parallel
MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]

app = modal.App("pythia-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=2.14",
        "accelerate>=0.24",
        "lm-eval>=0.4.3",
        "numpy",
    )
)

volume = modal.Volume.from_name("weight-corruption-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="t4",
    timeout=7200,
    volumes={"/results": volume},
)
def run_baseline(model_name: str) -> dict:
    import json as _json

    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Model : {model_name}")
    print(f"Tasks : {TASKS}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to("cuda").eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    results = lm_eval.simple_evaluate(model=lm, tasks=TASKS, log_samples=False)

    summary: dict = {}
    for task, metrics in results["results"].items():
        summary[task] = {
            k.replace(",none", ""): v
            for k, v in metrics.items()
            if "stderr" not in k and k not in ("alias",)
        }

    print("\n" + "=" * 60)
    print(f"BASELINE RESULTS — {model_name}")
    print("=" * 60)
    for task, metrics in summary.items():
        print(f"\n{task}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:40s} {v:.4f}")
            else:
                print(f"  {k:40s} {v}")

    out_path = f"/results/baseline_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as fh:
        _json.dump({"model": model_name, "tasks": TASKS, "results": summary}, fh, indent=2)
    volume.commit()
    print(f"\nSaved to {out_path}")

    return {"model": model_name, "results": summary}


@app.local_entrypoint()
def main():
    all_results = list(run_baseline.map(MODELS))

    print("\n" + "=" * 60)
    print("ALL RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))
