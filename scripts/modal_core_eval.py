"""Modal job: DCLM CORE score evaluation for Pythia models.

Implements the exact CORE eval from https://github.com/karpathy/nanochat,
adapted for HuggingFace models. Downloads Karpathy's public eval bundle
(same data used in the DCLM paper benchmark) and evaluates each of the
22 CORE tasks using the same scoring logic:

  - multiple_choice : pick the option with lowest mean NLL
  - schema          : pick the context with lowest mean NLL for a fixed continuation
  - language_modeling: greedy next-token exact match accuracy

CORE score = mean((acc - 0.01*rand_baseline) / (1 - 0.01*rand_baseline))
             over all 22 tasks.

Results saved to Modal volume as core_eval_{safe_model}.json.

Run with:
    doppler run --project claude-mobilr --config prd -- modal run scripts/modal_core_eval.py
"""

import json

import modal

MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

app = modal.App("pythia-core-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.24",
        "jinja2",
        "pyyaml",
        "numpy",
        "requests",
    )
)

volume = modal.Volume.from_name("weight-corruption-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Prompt rendering (verbatim from nanochat/core_eval.py)
# ---------------------------------------------------------------------------

def _render_mc(item, delim, fewshot):
    from jinja2 import Template
    tmpl = Template(
        "{%- for e in fewshot -%}{{ e.query }}{{ delim }}{{ e.choices[e.gold] }}\n\n{% endfor -%}"
        "{{ item.query }}{{ delim }}{{ choice }}"
    )
    return [tmpl.render(fewshot=fewshot, delim=delim, item=item, choice=c)
            for c in item["choices"]]


def _render_schema(item, delim, fewshot):
    from jinja2 import Template
    tmpl = Template(
        "{%- for e in fewshot -%}{{ e.context_options[e.gold] }}{{ delim }}{{ e.continuation }}\n\n{% endfor -%}"
        "{{ ctx }}{{ delim }}{{ item.continuation }}"
    )
    return [tmpl.render(fewshot=fewshot, delim=delim, item=item, ctx=ctx)
            for ctx in item["context_options"]]


def _render_lm(item, delim, fewshot):
    from jinja2 import Template
    tmpl = Template(
        "{%- for e in fewshot -%}{{ e.context | trim }}{{ delim }}{{ e.continuation }}\n\n{% endfor -%}"
        "{{ item.context | trim }}{{ delim }}{% if with_cont %}{{ item.continuation }}{% endif %}"
    )
    without = tmpl.render(fewshot=fewshot, delim=delim, item=item, with_cont=False).strip()
    with_ = tmpl.render(fewshot=fewshot, delim=delim, item=item, with_cont=True)
    return [without, with_]


# ---------------------------------------------------------------------------
# Tokenisation helpers (adapted for HuggingFace tokenizers)
# ---------------------------------------------------------------------------

def _tokenize(tokenizer, text):
    """Encode text, prepending BOS, without adding HF special tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    if bos is not None:
        ids = [bos] + ids
    return ids


def _find_common_len(seqs, direction="left"):
    min_len = min(len(s) for s in seqs)
    idxs = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(idxs):
        if not all(s[idx] == seqs[0][idx] for s in seqs):
            return i
    return min_len


def _stack(seqs, pad_id):
    import torch
    bsz = len(seqs)
    max_len = max(len(s) for s in seqs)
    t = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return t


def _batch_mc(tokenizer, prompts):
    tokens = [_tokenize(tokenizer, p) for p in prompts]
    start = _find_common_len(tokens, "left")
    ends = [len(t) for t in tokens]
    return tokens, [start] * len(prompts), ends


def _batch_schema(tokenizer, prompts):
    tokens = [_tokenize(tokenizer, p) for p in prompts]
    suffix = _find_common_len(tokens, "right")
    ends = [len(t) for t in tokens]
    starts = [e - suffix for e in ends]
    return tokens, starts, ends


def _batch_lm(tokenizer, prompts):
    without, with_ = [_tokenize(tokenizer, p) for p in prompts]
    start, end = len(without), len(with_)
    assert without == with_[:start], "LM prompt_without must be a prefix of prompt_with"
    return [with_], [start], [end]


# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------

def _forward(model, input_ids_tensor):
    """Returns (losses, predictions) shaped (B, T)."""
    import torch
    B, T = input_ids_tensor.shape
    logits = model(input_ids=input_ids_tensor).logits  # (B, T, V)
    targets = torch.roll(input_ids_tensor, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        logits.view(B * T, -1),
        targets.view(B * T),
        reduction="none",
    ).view(B, T)
    losses[:, -1] = float("nan")
    preds = logits.argmax(dim=-1)
    return losses, preds


# ---------------------------------------------------------------------------
# Batched task evaluation
# ---------------------------------------------------------------------------

def _prepare_example(idx, data, tokenizer, task_meta, max_seq_len):
    """Tokenize one example; return (flat_tokens, flat_starts, flat_ends, gold, task_type)."""
    import random
    item = data[idx]
    task_type = task_meta["task_type"]
    delim = task_meta["continuation_delimiter"]
    num_fewshot = task_meta["num_fewshot"]

    fewshot = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        pool = [i for i in range(len(data)) if i != idx]
        fewshot = [data[i] for i in rng.sample(pool, num_fewshot)]

    if task_type == "multiple_choice":
        prompts = _render_mc(item, delim, fewshot)
        tokens, starts, ends = _batch_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = _render_schema(item, delim, fewshot)
        tokens, starts, ends = _batch_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = _render_lm(item, delim, fewshot)
        tokens, starts, ends = _batch_lm(tokenizer, prompts)
    else:
        raise ValueError(task_type)

    if max_seq_len is not None:
        new_tokens, new_starts, new_ends = [], [], []
        for t, s, e in zip(tokens, starts, ends):
            if len(t) > max_seq_len:
                crop = len(t) - max_seq_len
                t = t[-max_seq_len:]
                s = max(0, s - crop)
                e = e - crop
            new_tokens.append(t)
            new_starts.append(s)
            new_ends.append(e)
        tokens, starts, ends = new_tokens, new_starts, new_ends

    gold = item.get("gold")  # None for language_modeling tasks
    return tokens, starts, ends, gold, task_type


def _evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len=None,
                   example_batch=64, max_fwd_seqs=16):
    """Evaluate all examples.

    example_batch: number of examples to tokenise together (reduces Python overhead).
    max_fwd_seqs:  max sequences per GPU forward pass (controls peak VRAM use).
                   With 4 choices and 2048-token 10-shot prompts, each sequence
                   uses ~800 tokens × 50k vocab × fp16 ≈ 80MB of logit memory,
                   so 16 seqs ≈ 1.3 GB — safe on a T4.
    """
    import torch

    pad_id = tokenizer.bos_token_id or 0
    correct = 0

    for batch_start in range(0, len(data), example_batch):
        batch_indices = range(batch_start, min(batch_start + example_batch, len(data)))

        # CPU work: tokenise all examples in this batch
        all_tokens, all_starts, all_ends = [], [], []
        example_meta = []

        for idx in batch_indices:
            tokens, starts, ends, gold, task_type = _prepare_example(
                idx, data, tokenizer, task_meta, max_seq_len
            )
            seq_offset = len(all_tokens)
            all_tokens.extend(tokens)
            all_starts.extend(starts)
            all_ends.extend(ends)
            example_meta.append((seq_offset, len(tokens), gold, task_type))

        # GPU work: forward in sub-batches capped at max_fwd_seqs
        all_losses_list, all_preds_list = [], []
        for fwd_start in range(0, len(all_tokens), max_fwd_seqs):
            chunk = all_tokens[fwd_start : fwd_start + max_fwd_seqs]
            input_chunk = _stack(chunk, pad_id).to(device)
            with torch.no_grad():
                l, p = _forward(model, input_chunk)
            all_losses_list.append(l.cpu())
            all_preds_list.append(p.cpu())

        # Pad sub-batch tensors to the same T before catting (each chunk was
        # padded to its own max length, so T may differ across chunks).
        max_t = max(l.size(1) for l in all_losses_list)
        all_losses = torch.cat(
            [torch.nn.functional.pad(l, (0, max_t - l.size(1)), value=float("nan"))
             for l in all_losses_list], dim=0)
        all_preds = torch.cat(
            [torch.nn.functional.pad(p, (0, max_t - p.size(1)), value=0)
             for p in all_preds_list], dim=0)
        all_input  = _stack(all_tokens, pad_id)  # CPU, for indexing

        # Score each example
        for seq_offset, n_seqs, gold, task_type in example_meta:
            if task_type == "language_modeling":
                row = seq_offset
                si, ei = all_starts[row], all_ends[row]
                predicted = all_preds[row, si - 1 : ei - 1]
                actual    = all_input[row, si : ei]
                correct += float(torch.all(predicted == actual).item())
            else:
                mean_losses = [
                    all_losses[seq_offset + i,
                               all_starts[seq_offset + i] - 1 :
                               all_ends[seq_offset + i]   - 1].mean().item()
                    for i in range(n_seqs)
                ]
                correct += float(mean_losses.index(min(mean_losses)) == gold)

    return correct / len(data)


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="t4",
    timeout=10800,
    volumes={"/results": volume},
)
def run_core_eval(model_name: str, max_per_task: int = -1) -> dict:
    import csv, io, json as _json, os, random, requests, shutil, tempfile, time, zipfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"DCLM CORE eval: {model_name}")
    print(f"{'='*60}")

    # ── Download eval bundle ──────────────────────────────────────────────────
    bundle_dir = "/tmp/eval_bundle"
    if not os.path.exists(bundle_dir):
        print("Downloading eval bundle from S3...")
        resp = requests.get(EVAL_BUNDLE_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
            zip_path = f.name
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/tmp")
        os.unlink(zip_path)
        print(f"Eval bundle extracted to {bundle_dir}")

    import yaml
    with open(os.path.join(bundle_dir, "core.yaml")) as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(os.path.join(bundle_dir, "eval_meta_data.csv"), newline="") as f:
        for row in csv.DictReader(f):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    print(f"max_seq_len={max_seq_len}")

    # ── Evaluate each task ────────────────────────────────────────────────────
    results = {}
    centered_results = {}
    for task in tasks:
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        data_path = os.path.join(bundle_dir, "eval_data", task["dataset_uri"])
        with open(data_path, encoding="utf-8") as f:
            data = [_json.loads(line) for line in f if line.strip()]

        # Consistent shuffle (matches nanochat)
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        t0 = time.time()
        print(f"  {label} ({task_meta['num_fewshot']}-shot, {task_meta['task_type']}, n={len(data)})...", flush=True)
        acc = _evaluate_task(model, tokenizer, data, "cuda", task_meta, max_seq_len)
        rand = random_baselines.get(label, 0.0)
        centered = (acc - 0.01 * rand) / (1.0 - 0.01 * rand)
        results[label] = acc
        centered_results[label] = centered
        print(f"    acc={acc:.4f}  centered={centered:.4f}  ({time.time()-t0:.0f}s)")

    core_score = sum(centered_results.values()) / len(centered_results)
    print(f"\nCORE score: {core_score:.6f}")

    out = {
        "model": model_name,
        "max_per_task": max_per_task,
        "core_score": core_score,
        "results": results,
        "centered_results": centered_results,
    }
    safe_model = model_name.replace("/", "_")
    suffix = f"_n{max_per_task}" if max_per_task > 0 else ""
    out_path = f"/results/core_eval_{safe_model}{suffix}.json"
    with open(out_path, "w") as fh:
        _json.dump(out, fh, indent=2)
    volume.commit()
    print(f"Saved → {out_path}")
    return out


@app.local_entrypoint()
def main(model: str = "", max_per_task: int = -1):
    targets = [model] if model else MODELS
    all_results = list(run_core_eval.map(targets, kwargs={"max_per_task": max_per_task}))
    print("\n" + "=" * 60)
    print("DCLM CORE SCORES")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['model']:35s}  CORE={r['core_score']:.6f}")
    print(json.dumps(all_results, indent=2))
