import pytest
import torch

from weight_corruption.experiments.matrix_registry import MatrixInfo, get_matrix_registry


def _make_gpt2_tiny():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        vocab_size=256,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    return GPT2LMHeadModel(cfg)


def test_registry_returns_list_of_matrix_info():
    model = _make_gpt2_tiny()
    registry = get_matrix_registry(model)
    assert isinstance(registry, list)
    assert len(registry) > 0
    assert all(isinstance(m, MatrixInfo) for m in registry)


def test_registry_names_match_model_parameters():
    model = _make_gpt2_tiny()
    param_names = {n for n, _ in model.named_parameters()}
    for info in get_matrix_registry(model):
        assert info.name in param_names


def test_registry_covers_all_parameters():
    model = _make_gpt2_tiny()
    registry_names = {m.name for m in get_matrix_registry(model)}
    param_names = {n for n, _ in model.named_parameters()}
    assert registry_names == param_names, (
        f"Missing: {param_names - registry_names}\nExtra: {registry_names - param_names}"
    )


def test_registry_categories_are_valid():
    valid = {"attention", "mlp", "embedding", "output", "normalization", "other"}
    model = _make_gpt2_tiny()
    for info in get_matrix_registry(model):
        assert info.category in valid


def test_embedding_and_output_have_layer_minus_one():
    model = _make_gpt2_tiny()
    for info in get_matrix_registry(model):
        if info.category in ("embedding", "output"):
            assert info.layer_idx == -1, f"{info.name}: expected -1, got {info.layer_idx}"


def test_attention_matrices_have_non_negative_layer_idx():
    model = _make_gpt2_tiny()
    for info in get_matrix_registry(model):
        if info.category == "attention":
            assert info.layer_idx >= 0, f"{info.name}: expected >= 0"


def test_shapes_match_parameter_shapes():
    model = _make_gpt2_tiny()
    params = dict(model.named_parameters())
    for info in get_matrix_registry(model):
        assert info.shape == tuple(params[info.name].shape)
