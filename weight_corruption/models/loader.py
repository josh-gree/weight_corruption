from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer.

    When *device* is ``"auto"``, CUDA is used if available, otherwise CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs: dict = {"torch_dtype": dtype}
    if device != "cpu":
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device == "cpu":
        model = model.to("cpu")

    model.eval()
    return model, tokenizer
