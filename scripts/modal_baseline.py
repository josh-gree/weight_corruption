"""Modal job: baseline evaluation of EleutherAI/pythia-70m (no corruption).

Run with:
    doppler run --project claude-mobilr --config prd -- modal run scripts/modal_baseline.py
"""

import json

import modal

MODEL_NAME = "EleutherAI/pythia-70m"

# Accuracy benchmarks via lm_eval.
# wikitext is computed separately with a sliding-window approach — lm_eval's
# rolling log-likelihood over the full concatenated test set causes OOM on T4
# when the vocab projection is materialised for the entire sequence at once.
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
        "pandas",
    )
)

volume = modal.Volume.from_name("weight-corruption-results", create_if_missing=True)


def _wikitext2_perplexity(model, tokenizer, device, max_length=512, stride=256) -> dict:
    """Sliding-window perplexity on WikiText-2 test split."""
    import math
    import torch
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.shape[1]

    nlls, prev_end = [], 0
    model.eval()
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end
            chunk = input_ids[:, begin:end].to(device)
            labels = chunk.clone()
            labels[:, : chunk.shape[1] - target_len] = -100
            loss = model(chunk, labels=labels).loss
            nlls.append(loss * target_len)
            prev_end = end
            if end == seq_len:
                break

    mean_loss = (torch.stack(nlls).sum() / seq_len).item()
    return {"word_perplexity": math.exp(mean_loss), "bits_per_byte": mean_loss / math.log(2)}


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    print(f"Model : {MODEL_NAME}")

    # Load once; pass to lm_eval via pretrained= so it doesn't reload
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(device).eval()

    # ── Tier 1: WikiText-2 perplexity ────────────────────────────────────────
    print("Computing WikiText-2 perplexity...")
    wt2 = _wikitext2_perplexity(model, tokenizer, device)
    print(f"  word_perplexity : {wt2['word_perplexity']:.2f}")
    print(f"  bits_per_byte   : {wt2['bits_per_byte']:.4f}")

    # ── Tier 2: accuracy benchmarks ──────────────────────────────────────────
    print(f"\nRunning lm_eval tasks: {TASKS}")
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    results = lm_eval.simple_evaluate(model=lm, tasks=TASKS, log_samples=False)

    summary: dict = {"wikitext": wt2}
    for task, metrics in results["results"].items():
        summary[task] = {k: v for k, v in metrics.items() if not k.endswith(",none")}

    # ── Print ─────────────────────────────────────────────────────────────────
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

    # ── Save to volume ────────────────────────────────────────────────────────
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
