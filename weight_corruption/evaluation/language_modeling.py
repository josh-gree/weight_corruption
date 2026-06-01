import math
from typing import Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    n_samples: int = 100,
    max_length: int = 512,
    stride: int = 256,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Compute LM loss and perplexity on a sample of a HuggingFace dataset.

    Uses a sliding window to handle sequences longer than `max_length`.
    Returns a dict with keys ``loss`` and ``perplexity``.
    """
    if device is None:
        device = next(model.parameters()).device

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    texts = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        texts.append(item["text"])

    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids  # [1, total_tokens]
    seq_len = input_ids.shape[1]

    nlls: list[torch.Tensor] = []
    prev_end = 0
    model.eval()

    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end

            chunk = input_ids[:, begin:end].to(device)
            labels = chunk.clone()
            # Mask tokens that are context-only from the previous window.
            labels[:, : chunk.shape[1] - target_len] = -100

            outputs = model(chunk, labels=labels)
            nlls.append(outputs.loss * target_len)

            prev_end = end
            if end == seq_len:
                break

    mean_loss = (torch.stack(nlls).sum() / seq_len).item()
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
