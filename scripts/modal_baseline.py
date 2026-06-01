"""Modal job: baseline evaluation + residual stream capture for Pythia models.

For each model and each eval task we:
  1. Run lm_eval harness tasks to get accuracy scores.
  2. Re-run the eval questions through the model with output_hidden_states=True
     and save the last-token residual vector at every layer depth.

Activations saved as .npz:
  activations  : float16 (n_examples, n_layers+1, hidden_dim)
                 layer 0 = embedding output, layer k = output of block k
  labels       : int32   (n_examples,)  ground-truth answer indices

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

MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]

# How to load each task's raw questions and labels from HF datasets.
# (dataset_path, config, split, text_fn, label_fn)
TASK_CONFIG = {
    "hellaswag": (
        "Rowan/hellaswag", None, "validation",
        lambda ex: ex["ctx"],
        lambda ex: int(ex["label"]),
    ),
    "arc_easy": (
        "allenai/ai2_arc", "ARC-Easy", "test",
        lambda ex: ex["question"],
        lambda ex: ["A", "B", "C", "D"].index(ex["answerKey"]) if ex["answerKey"] in "ABCD" else 0,
    ),
    "arc_challenge": (
        "allenai/ai2_arc", "ARC-Challenge", "test",
        lambda ex: ex["question"],
        lambda ex: ["A", "B", "C", "D"].index(ex["answerKey"]) if ex["answerKey"] in "ABCD" else 0,
    ),
    "piqa": (
        "ybisk/piqa", None, "validation",
        lambda ex: ex["goal"],
        lambda ex: int(ex["label"]),
    ),
    "winogrande": (
        "allenai/winogrande", "winogrande_xl", "validation",
        lambda ex: ex["sentence"],
        lambda ex: int(ex["answer"]) - 1,
    ),
}

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


def _capture_residual_stream(model, tokenizer, texts, device, batch_size=32, max_length=256):
    """
    Forward-pass each text and return last-token hidden states at every depth.

    Returns np.ndarray of shape (n_texts, n_layers+1, hidden_dim) in float16.
    Layer 0 is the embedding output; layer k is the output of transformer block k.
    """
    import numpy as np
    import torch

    all_reps = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # last non-padding token index per example
        last_idx = attention_mask.sum(dim=1) - 1  # (batch,)

        for j in range(len(batch)):
            idx = last_idx[j].item()
            # stack across layers: (n_layers+1, hidden_dim)
            rep = np.stack([
                h[j, idx, :].float().cpu().numpy()
                for h in out.hidden_states
            ]).astype(np.float16)
            all_reps.append(rep)

    return np.stack(all_reps)  # (n_texts, n_layers+1, hidden_dim)


@app.function(
    image=image,
    gpu="t4",
    timeout=7200,
    volumes={"/results": volume},
)
def run_baseline(model_name: str) -> dict:
    import json as _json

    import lm_eval
    import numpy as np
    import torch
    from datasets import load_dataset
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to("cuda").eval()

    # ── 1. lm_eval accuracy scores ────────────────────────────────────────────
    print("\nRunning lm_eval...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    results = lm_eval.simple_evaluate(model=lm, tasks=TASKS, log_samples=False)

    summary: dict = {}
    for task, metrics in results["results"].items():
        summary[task] = {
            k.replace(",none", ""): v
            for k, v in metrics.items()
            if "stderr" not in k and k not in ("alias",)
        }

    print(f"\nScores:")
    for task, metrics in summary.items():
        acc = metrics.get("acc_norm", metrics.get("acc", "?"))
        print(f"  {task:20s} acc_norm={acc:.4f}" if isinstance(acc, float) else f"  {task}")

    # ── 2. Residual stream capture ────────────────────────────────────────────
    print("\nCapturing residual stream activations...")
    safe_model = model_name.replace("/", "_")

    for task, (ds_path, ds_cfg, split, text_fn, label_fn) in TASK_CONFIG.items():
        print(f"  {task}...")
        ds = load_dataset(ds_path, ds_cfg, split=split) if ds_cfg else load_dataset(ds_path, split=split)

        texts, labels = [], []
        for ex in ds:
            try:
                texts.append(text_fn(ex))
                labels.append(label_fn(ex))
            except Exception:
                continue

        acts = _capture_residual_stream(model, tokenizer, texts, "cuda")
        out_path = f"/results/activations_{safe_model}_{task}.npz"
        np.savez_compressed(
            out_path,
            activations=acts,
            labels=np.array(labels, dtype=np.int32),
        )
        print(f"    saved {acts.shape} → {out_path}")

    volume.commit()

    # ── 3. Save scores ────────────────────────────────────────────────────────
    scores_path = f"/results/baseline_{safe_model}.json"
    with open(scores_path, "w") as fh:
        _json.dump({"model": model_name, "tasks": TASKS, "results": summary}, fh, indent=2)
    volume.commit()

    return {"model": model_name, "results": summary}


@app.local_entrypoint()
def main():
    all_results = list(run_baseline.map(MODELS))

    print("\n" + "=" * 60)
    print("ALL RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))
