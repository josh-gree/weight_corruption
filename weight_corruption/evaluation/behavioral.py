from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _load_eval_texts(
    dataset_name: str,
    dataset_config: str,
    split: str,
    n_samples: int,
    min_length: int = 50,
) -> list[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    texts: list[str] = []
    for item in dataset:
        if len(item["text"].strip()) >= min_length:
            texts.append(item["text"])
        if len(texts) >= n_samples:
            break
    return texts


def precompute_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = 256,
    device: Optional[torch.device] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (logits_list, input_ids_list) for each text, stored on CPU."""
    if device is None:
        device = next(model.parameters()).device

    logits_list: list[torch.Tensor] = []
    input_ids_list: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            input_ids = enc.input_ids.to(device)
            if input_ids.shape[1] < 2:
                continue
            logits = model(input_ids).logits  # [1, T, V]
            logits_list.append(logits.cpu())
            input_ids_list.append(input_ids.cpu())

    return logits_list, input_ids_list


def compute_divergence_from_logits(
    baseline_logits: list[torch.Tensor],
    corrupted_logits: list[torch.Tensor],
) -> dict[str, float]:
    """Compare pre-computed logit lists and return per-token divergence metrics.

    KL divergence is KL(baseline || corrupted), averaged per token.
    Top-1/top-5 agreement: how often the corrupted model's prediction matches
    the baseline's top prediction.
    """
    total_kl = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0

    for b_logits, c_logits in zip(baseline_logits, corrupted_logits):
        b = b_logits[:, :-1, :].float()   # [1, T-1, V]
        c = c_logits[:, :-1, :].float()

        b_probs = F.softmax(b, dim=-1)
        c_log_probs = F.log_softmax(c, dim=-1)

        kl = F.kl_div(c_log_probs, b_probs, reduction="sum", log_target=False)
        total_kl += kl.item()

        b_top1 = b.argmax(dim=-1)          # [1, T-1]
        c_top1 = c.argmax(dim=-1)
        total_top1 += int((b_top1 == c_top1).sum().item())

        c_top5 = c.topk(5, dim=-1).indices  # [1, T-1, 5]
        total_top5 += int(
            (b_top1.unsqueeze(-1) == c_top5).any(dim=-1).sum().item()
        )

        total_tokens += b.shape[1]

    if total_tokens == 0:
        nan = float("nan")
        return {"kl_divergence": nan, "top1_agreement": nan, "top5_agreement": nan}

    return {
        "kl_divergence": total_kl / total_tokens,
        "top1_agreement": total_top1 / total_tokens,
        "top5_agreement": total_top5 / total_tokens,
    }


def compute_behavioral_divergence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    baseline_logits: list[torch.Tensor],
    texts: list[str],
    max_length: int = 256,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Compute divergence between pre-computed baseline logits and the current
    (already-corrupted) state of *model*.
    """
    corrupted_logits, _ = precompute_logits(model, tokenizer, texts, max_length, device)
    return compute_divergence_from_logits(baseline_logits, corrupted_logits)
