"""Lightweight runner tests that do not require downloading models."""

import torch

from weight_corruption.experiments.runner import (
    CorruptionConfig,
    apply_corruption,
    restore_parameter,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)


def test_apply_corruption_returns_original():
    model = _TinyModel()
    original_weight = model.linear.weight.data.clone()
    config = CorruptionConfig(method="zeroing", fraction=1.0)

    returned = apply_corruption(model, "linear.weight", config)
    assert torch.allclose(returned, original_weight)
    # After full zeroing the weight should be all zeros
    assert torch.all(model.linear.weight.data == 0)


def test_restore_parameter():
    model = _TinyModel()
    original = model.linear.weight.data.clone()
    config = CorruptionConfig(method="zeroing", fraction=1.0)

    saved = apply_corruption(model, "linear.weight", config)
    restore_parameter(model, "linear.weight", saved)

    assert torch.allclose(model.linear.weight.data, original)


def test_apply_and_restore_all_methods():
    for method in ("random", "zeroing", "shuffle", "gaussian", "quantize"):
        model = _TinyModel()
        original = model.linear.weight.data.clone()
        config = CorruptionConfig(method=method, fraction=0.5, sigma=0.1, bits=8)
        saved = apply_corruption(model, "linear.weight", config)
        restore_parameter(model, "linear.weight", saved)
        assert torch.allclose(model.linear.weight.data, original), (
            f"Restore failed for method={method}"
        )


def test_corruption_config_defaults():
    cfg = CorruptionConfig(method="random")
    assert cfg.fraction == 0.01
    assert cfg.sigma == 0.01
    assert cfg.bits == 8
    assert cfg.seed == 42
