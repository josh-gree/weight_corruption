import re
from dataclasses import dataclass

from transformers import PreTrainedModel


@dataclass(frozen=True)
class MatrixInfo:
    name: str            # full parameter name
    layer_idx: int       # layer index; -1 for global params (embed, lm_head)
    matrix_type: str     # e.g. "q_proj", "mlp_up", "token_embed", "lm_head"
    category: str        # "attention" | "mlp" | "embedding" | "output" | "normalization"
    shape: tuple[int, ...]


# Each entry: (regex, fixed_layer_or_None, matrix_type, category)
# When fixed_layer is None the first capture group gives the layer index.
_PATTERNS: list[tuple[str, int | None, str, str]] = [
    # ── GPT-2 ───────────────────────────────────────────────────────────────
    (r"^transformer\.wte\.weight$",                    -1,   "token_embed",   "embedding"),
    (r"^transformer\.wpe\.weight$",                    -1,   "pos_embed",     "embedding"),
    (r"^lm_head\.weight$",                             -1,   "lm_head",       "output"),
    (r"^transformer\.h\.(\d+)\.attn\.c_attn\.weight$", None, "qkv_proj",      "attention"),
    (r"^transformer\.h\.(\d+)\.attn\.c_proj\.weight$", None, "o_proj",        "attention"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_fc\.weight$",    None, "mlp_up",        "mlp"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_proj\.weight$",  None, "mlp_down",      "mlp"),
    (r"^transformer\.h\.(\d+)\.ln_\d+\.",              None, "layernorm",     "normalization"),
    (r"^transformer\.ln_f\.",                          -1,   "layernorm_f",   "normalization"),
    # ── Pythia / GPT-NeoX ───────────────────────────────────────────────────
    (r"^gpt_neox\.embed_in\.weight$",                                          -1,   "token_embed", "embedding"),
    (r"^embed_out\.weight$",                                                   -1,   "lm_head",     "output"),
    (r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$",         None, "qkv_proj",    "attention"),
    (r"^gpt_neox\.layers\.(\d+)\.attention\.dense\.weight$",                   None, "o_proj",      "attention"),
    (r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_h_to_4h\.weight$",                None, "mlp_up",      "mlp"),
    (r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.weight$",                None, "mlp_down",    "mlp"),
    (r"^gpt_neox\.layers\.(\d+)\.input_layernorm\.",                           None, "layernorm_1", "normalization"),
    (r"^gpt_neox\.layers\.(\d+)\.post_attention_layernorm\.",                  None, "layernorm_2", "normalization"),
    (r"^gpt_neox\.final_layer_norm\.",                                         -1,   "layernorm_f", "normalization"),
    # ── LLaMA / Mistral / TinyLlama ─────────────────────────────────────────
    (r"^model\.embed_tokens\.weight$",                                         -1,   "token_embed",  "embedding"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",                     None, "q_proj",       "attention"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",                     None, "k_proj",       "attention"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",                     None, "v_proj",       "attention"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",                     None, "o_proj",       "attention"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",                          None, "mlp_up",       "mlp"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",                        None, "mlp_gate",     "mlp"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",                        None, "mlp_down",     "mlp"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.",                               None, "layernorm_1",  "normalization"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.",                      None, "layernorm_2",  "normalization"),
    (r"^model\.norm\.",                                                         -1,   "layernorm_f",  "normalization"),
]


def _fallback_category(name: str) -> tuple[str, str]:
    n = name.lower()
    if "embed" in n and "token" in n:
        return "embedding", "token_embed"
    if "lm_head" in n or "embed_out" in n:
        return "output", "lm_head"
    if any(x in n for x in ("ln", "norm", "layer_norm")):
        return "normalization", "layernorm"
    if any(x in n for x in ("q_proj", "k_proj", "v_proj", "attn", "attention")):
        return "attention", "attn"
    if any(x in n for x in ("mlp", "ffn", "fc", "dense")):
        return "mlp", "mlp"
    return "other", "other"


def get_matrix_registry(model: PreTrainedModel) -> list[MatrixInfo]:
    """Return a ``MatrixInfo`` for every parameter in *model*."""
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
            category, mat_type = _fallback_category(name)
            nums = re.findall(r"\d+", name)
            layer_idx = int(nums[0]) if nums else -1
            registry.append(
                MatrixInfo(
                    name=name,
                    layer_idx=layer_idx,
                    matrix_type=mat_type,
                    category=category,
                    shape=tuple(param.shape),
                )
            )

    return registry
