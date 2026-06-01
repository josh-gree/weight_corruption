import warnings

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Known Pythia model IDs for reference; not an exhaustive allow-list.
PYTHIA_MODEL_IDS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-12b-deduped",
]


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Pythia (GPT-NeoX) causal LM and its tokenizer.

    Warns if the loaded model is not a GPT-NeoX architecture.
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

    arch = getattr(model.config, "model_type", "unknown")
    if arch != "gpt_neox":
        warnings.warn(
            f"Expected a GPT-NeoX (Pythia) model but got model_type={arch!r}. "
            "The matrix registry only supports Pythia architectures.",
            stacklevel=2,
        )

    model.eval()
    return model, tokenizer
