import torch
import pytest

from weight_corruption.corruption.methods import (
    corrupt_gaussian_noise,
    corrupt_quantize,
    corrupt_random_replacement,
    corrupt_shuffle,
    corrupt_zeroing,
)


def _weight(shape=(200, 200), seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(shape, generator=g)


def _gen(seed: int = 42) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class TestRandomReplacement:
    def test_shape_preserved(self):
        w = _weight()
        assert corrupt_random_replacement(w, 0.1).shape == w.shape

    def test_no_corruption_at_zero_fraction(self):
        w = _weight()
        assert torch.allclose(corrupt_random_replacement(w, 0.0, _gen()), w)

    def test_approximate_fraction(self):
        w = _weight((5000,))
        c = corrupt_random_replacement(w, 0.1, _gen())
        changed = (c != w).float().mean().item()
        assert 0.05 < changed < 0.20

    def test_original_not_mutated(self):
        w = _weight()
        original = w.clone()
        corrupt_random_replacement(w, 0.5)
        assert torch.allclose(w, original)


class TestZeroing:
    def test_shape_preserved(self):
        w = _weight()
        assert corrupt_zeroing(w, 0.1).shape == w.shape

    def test_full_fraction_all_zero(self):
        w = _weight()
        assert torch.all(corrupt_zeroing(w, 1.0, _gen()) == 0)

    def test_zero_fraction_no_change(self):
        w = _weight()
        assert torch.allclose(corrupt_zeroing(w, 0.0, _gen()), w)

    def test_original_not_mutated(self):
        w = _weight()
        original = w.clone()
        corrupt_zeroing(w, 0.5)
        assert torch.allclose(w, original)


class TestShuffle:
    def test_shape_preserved(self):
        w = _weight()
        assert corrupt_shuffle(w, 0.5).shape == w.shape

    def test_distribution_preserved(self):
        w = _weight((1000,))
        c = corrupt_shuffle(w, 0.5, _gen())
        assert abs(c.mean().item() - w.mean().item()) < 0.1
        assert abs(c.std().item() - w.std().item()) < 0.1

    def test_values_are_subset_of_original(self):
        w = _weight((50,))
        c = corrupt_shuffle(w, 1.0, _gen())
        assert torch.allclose(w.sort().values, c.sort().values)

    def test_original_not_mutated(self):
        w = _weight()
        original = w.clone()
        corrupt_shuffle(w, 0.5)
        assert torch.allclose(w, original)


class TestGaussianNoise:
    def test_shape_preserved(self):
        w = _weight()
        assert corrupt_gaussian_noise(w, 0.01).shape == w.shape

    def test_zero_sigma_no_change(self):
        w = _weight()
        assert torch.allclose(corrupt_gaussian_noise(w, 0.0, _gen()), w)

    def test_original_not_mutated(self):
        w = _weight()
        original = w.clone()
        corrupt_gaussian_noise(w, 1.0)
        assert torch.allclose(w, original)

    def test_large_sigma_changes_values(self):
        w = _weight()
        c = corrupt_gaussian_noise(w, 100.0, _gen())
        assert not torch.allclose(w, c)


class TestQuantize:
    def test_shape_preserved(self):
        w = _weight()
        assert corrupt_quantize(w, 8).shape == w.shape

    def test_range_preserved(self):
        w = _weight()
        c = corrupt_quantize(w, 8)
        assert c.min().item() >= w.min().item() - 1e-4
        assert c.max().item() <= w.max().item() + 1e-4

    def test_fewer_unique_values_at_lower_bits(self):
        w = _weight((1000,))
        n8 = corrupt_quantize(w, 8).unique().numel()
        n4 = corrupt_quantize(w, 4).unique().numel()
        n3 = corrupt_quantize(w, 3).unique().numel()
        assert n8 >= n4 >= n3

    def test_constant_tensor_unchanged(self):
        w = torch.ones(10, 10)
        assert torch.allclose(corrupt_quantize(w, 8), w)
