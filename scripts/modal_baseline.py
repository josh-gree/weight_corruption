"""Modal job: baseline evaluation of EleutherAI/pythia-70m (no corruption).

Run with:
    doppler run --project claude-mobilr --config prd -- modal run scripts/modal_baseline.py
"""

import json

import modal

MODEL_NAME = "EleutherAI/pythia-70m"

TASKS = [
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
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Model : {MODEL_NAME}")
    print(f"Tasks : {TASKS}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
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
    print(f"BASELINE RESULTS — {MODEL_NAME}")
    print("=" * 60)
    for task, metrics in summary.items():
        print(f"\n{task}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:40s} {v:.4f}")
            else:
                print(f"  {k:40s} {v}")

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
