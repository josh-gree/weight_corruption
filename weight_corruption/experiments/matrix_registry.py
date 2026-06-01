import re
from dataclasses import dataclass

from transformers import PreTrainedModel


@dataclass(frozen=True)
class MatrixInfo:
    name: str            # full parameter name
    layer_idx: int       # layer index; -1 for global params (embed, lm_head)
    matrix_type: str     # e.g. "qkv_proj", "mlp_up", "token_embed", "lm_head"
    category: str        # "attention" | "mlp" | "embedding" | "output" | "normalization"
    shape: tuple[int, ...]


# Pythia / GPT-NeoX parameter patterns.
# Each entry: (regex, fixed_layer_or_None, matrix_type, category)
# Trailing "\\." matches both ".weight" and ".bias" suffixes.
_PATTERNS: list[tuple[str, int | None, str, str]] = [
    (r"^gpt_neox\.embed_in\.weight$",                                -1,   "token_embed", "embedding"),
    (r"^embed_out\.weight$",                                         -1,   "lm_head",     "output"),
    (r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.",      None, "qkv_proj",    "attention"),
    (r"^gpt_neox\.layers\.(\d+)\.attention\.dense\.",                None, "o_proj",      "attention"),
    (r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_h_to_4h\.",             None, "mlp_up",      "mlp"),
    (r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.",             None, "mlp_down",    "mlp"),
    (r"^gpt_neox\.layers\.(\d+)\.input_layernorm\.",                 None, "layernorm_1", "normalization"),
    (r"^gpt_neox\.layers\.(\d+)\.post_attention_layernorm\.",        None, "layernorm_2", "normalization"),
    (r"^gpt_neox\.final_layer_norm\.",                               -1,   "layernorm_f", "normalization"),
    # rotary embeddings appear as a parameter in some transformers versions
    (r"\.rotary_emb\.",                                              None, "rotary_emb",  "other"),
]


def get_matrix_registry(model: PreTrainedModel) -> list[MatrixInfo]:
    """Return a ``MatrixInfo`` for every parameter in *model*.

    Raises ``ValueError`` for parameters that do not match any known
    Pythia / GPT-NeoX pattern.
    """
    registry: list[MatrixInfo] = []

    for name, param in model.named_parameters():
        if param.dim() == 0:
            continue

        matched = False
        for pattern, fixed_layer, mat_type, category in _PATTERNS:
            m = re.search(pattern, name)
            if m is None:
                continue
            if fixed_layer is not None:
                layer_idx = fixed_layer
            else:
                layer_idx = int(m.group(1)) if m.lastindex and m.lastindex >= 1 else -1

            registry.append(
                MatrixInfo(
                    name=name,
                    layer_idx=layer_idx,
                    matrix_type=mat_type,
                    category=category,
                    shape=tuple(param.shape),
                )
            )
            matched = True
            break

        if not matched:
            raise ValueError(
                f"Unrecognised Pythia parameter: {name!r}. "
                "Only GPT-NeoX (Pythia) architectures are supported."
            )

    return registry
