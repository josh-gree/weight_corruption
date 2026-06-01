from typing import Literal

import torch


def corrupt_random_replacement(
    weight: torch.Tensor,
    fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace `fraction` of weights with samples from N(mean(w), std(w))."""
    corrupted = weight.clone()
    mask = torch.rand(weight.shape, generator=generator, device=weight.device) < fraction
    mean = weight.mean().item()
    std = weight.std().item()
    noise = torch.randn(weight.shape, generator=generator, device=weight.device) * std + mean
    corrupted[mask] = noise[mask]
    return corrupted


def corrupt_zeroing(
    weight: torch.Tensor,
    fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Set `fraction` of weights to zero."""
    corrupted = weight.clone()
    mask = torch.rand(weight.shape, generator=generator, device=weight.device) < fraction
    corrupted[mask] = 0.0
    return corrupted


def corrupt_shuffle(
    weight: torch.Tensor,
    fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly permute `fraction` of weights, preserving the marginal distribution."""
    corrupted = weight.clone()
    flat = corrupted.view(-1)
    n = flat.shape[0]
    n_corrupt = max(1, int(n * fraction))

    if generator is not None:
        indices = torch.randperm(n, generator=generator, device=weight.device)[:n_corrupt]
        shuffle_order = torch.randperm(n_corrupt, generator=generator, device=weight.device)
    else:
        indices = torch.randperm(n, device=weight.device)[:n_corrupt]
        shuffle_order = torch.randperm(n_corrupt, device=weight.device)

    flat[indices] = flat[indices[shuffle_order]]
    return corrupted


def corrupt_gaussian_noise(
    weight: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise: w += sigma * N(0, 1)."""
    noise = torch.randn(weight.shape, generator=generator, device=weight.device) * sigma
    return weight + noise


def corrupt_quantize(
    weight: torch.Tensor,
    bits: Literal[3, 4, 8],
) -> torch.Tensor:
    """Simulate symmetric uniform quantization to the given bit width."""
    n_levels = 2 ** bits
    w_min = weight.min().item()
    w_max = weight.max().item()
    scale = (w_max - w_min) / (n_levels - 1)
    if scale == 0:
        return weight.clone()
    quantized = torch.round((weight - w_min) / scale) * scale + w_min
    return quantized
