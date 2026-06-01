from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from tqdm import tqdm

from ..corruption.methods import (
    corrupt_gaussian_noise,
    corrupt_quantize,
    corrupt_random_replacement,
    corrupt_shuffle,
    corrupt_zeroing,
)
from ..evaluation.behavioral import (
    _load_eval_texts,
    compute_behavioral_divergence,
    precompute_logits,
)
from ..evaluation.language_modeling import compute_perplexity
from ..models.loader import load_model_and_tokenizer
from .matrix_registry import MatrixInfo, get_matrix_registry


@dataclass
class CorruptionConfig:
    method: str        # "random" | "zeroing" | "shuffle" | "gaussian" | "quantize"
    fraction: float = 0.01
    sigma: float = 0.01
    bits: int = 8
    seed: int = 42


@dataclass
class ExperimentResult:
    # Identity
    matrix_name: str
    layer_idx: int
    matrix_type: str
    category: str
    matrix_shape: tuple[int, ...]

    # Corruption config
    corruption_method: str
    corruption_fraction: float

    # Language modelling
    baseline_loss: float
    corrupted_loss: float
    delta_loss: float
    baseline_perplexity: float
    corrupted_perplexity: float

    # Behavioural divergence
    kl_divergence: float
    top1_agreement: float
    top5_agreement: float

    # Optional per-capability deltas (benchmark name → delta score)
    capabilities: dict[str, float] = field(default_factory=dict)


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def apply_corruption(
    model: torch.nn.Module,
    name: str,
    config: CorruptionConfig,
) -> torch.Tensor:
    """Corrupt parameter *name* in-place and return the original tensor."""
    param = dict(model.named_parameters())[name]
    original = param.data.clone()
    g = _make_generator(config.seed, param.device)

    if config.method == "random":
        corrupted = corrupt_random_replacement(param.data, config.fraction, g)
    elif config.method == "zeroing":
        corrupted = corrupt_zeroing(param.data, config.fraction, g)
    elif config.method == "shuffle":
        corrupted = corrupt_shuffle(param.data, config.fraction, g)
    elif config.method == "gaussian":
        corrupted = corrupt_gaussian_noise(param.data, config.sigma, g)
    elif config.method == "quantize":
        corrupted = corrupt_quantize(param.data, config.bits)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown corruption method: {config.method!r}")

    param.data.copy_(corrupted)
    return original


def restore_parameter(model: torch.nn.Module, name: str, original: torch.Tensor) -> None:
    """Restore parameter *name* to *original*."""
    param = dict(model.named_parameters())[name]
    param.data.copy_(original)


class Phase1Runner:
    """Iterate over every weight matrix, corrupt it, evaluate, restore."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        lm_samples: int = 100,
        behavioral_samples: int = 50,
        max_length: int = 512,
        behavioral_max_length: int = 256,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
    ) -> None:
        self.model, self.tokenizer = load_model_and_tokenizer(model_name, device)
        self.device: torch.device = next(self.model.parameters()).device
        self.lm_samples = lm_samples
        self.behavioral_samples = behavioral_samples
        self.max_length = max_length
        self.behavioral_max_length = behavioral_max_length
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

        self.registry = get_matrix_registry(self.model)

        # Lazy baseline caches
        self._baseline_lm: dict[str, float] | None = None
        self._baseline_logits: list[torch.Tensor] | None = None
        self._eval_texts: list[str] | None = None

    # ── Baseline helpers ────────────────────────────────────────────────────

    def _get_baseline_lm(self) -> dict[str, float]:
        if self._baseline_lm is None:
            self._baseline_lm = compute_perplexity(
                self.model,
                self.tokenizer,
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                n_samples=self.lm_samples,
                max_length=self.max_length,
                device=self.device,
            )
        return self._baseline_lm

    def _get_eval_texts(self) -> list[str]:
        if self._eval_texts is None:
            self._eval_texts = _load_eval_texts(
                self.dataset_name,
                self.dataset_config,
                split="test",
                n_samples=self.behavioral_samples,
            )
        return self._eval_texts

    def _get_baseline_logits(self) -> list[torch.Tensor]:
        if self._baseline_logits is None:
            texts = self._get_eval_texts()
            self._baseline_logits, _ = precompute_logits(
                self.model,
                self.tokenizer,
                texts,
                max_length=self.behavioral_max_length,
                device=self.device,
            )
        return self._baseline_logits

    # ── Single experiment ───────────────────────────────────────────────────

    def run_single(
        self,
        matrix_info: MatrixInfo,
        config: CorruptionConfig,
        include_behavioral: bool = True,
    ) -> ExperimentResult:
        baseline_lm = self._get_baseline_lm()
        baseline_logits = self._get_baseline_logits() if include_behavioral else []
        texts = self._get_eval_texts() if include_behavioral else []

        original = apply_corruption(self.model, matrix_info.name, config)
        try:
            corrupted_lm = compute_perplexity(
                self.model,
                self.tokenizer,
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                n_samples=self.lm_samples,
                max_length=self.max_length,
                device=self.device,
            )

            if include_behavioral and baseline_logits:
                div = compute_behavioral_divergence(
                    self.model,
                    self.tokenizer,
                    baseline_logits,
                    texts,
                    max_length=self.behavioral_max_length,
                    device=self.device,
                )
            else:
                nan = float("nan")
                div = {"kl_divergence": nan, "top1_agreement": nan, "top5_agreement": nan}
        finally:
            restore_parameter(self.model, matrix_info.name, original)

        corruption_fraction = (
            config.fraction if config.method != "quantize" else 1.0
        )

        return ExperimentResult(
            matrix_name=matrix_info.name,
            layer_idx=matrix_info.layer_idx,
            matrix_type=matrix_info.matrix_type,
            category=matrix_info.category,
            matrix_shape=matrix_info.shape,
            corruption_method=config.method,
            corruption_fraction=corruption_fraction,
            baseline_loss=baseline_lm["loss"],
            corrupted_loss=corrupted_lm["loss"],
            delta_loss=corrupted_lm["loss"] - baseline_lm["loss"],
            baseline_perplexity=baseline_lm["perplexity"],
            corrupted_perplexity=corrupted_lm["perplexity"],
            kl_divergence=div["kl_divergence"],
            top1_agreement=div["top1_agreement"],
            top5_agreement=div["top5_agreement"],
        )

    # ── Full sweep ──────────────────────────────────────────────────────────

    def run_all(
        self,
        config: CorruptionConfig,
        matrix_filter: Optional[Callable[[MatrixInfo], bool]] = None,
        skip_normalization: bool = True,
        include_behavioral: bool = True,
    ) -> list[ExperimentResult]:
        matrices = [
            m for m in self.registry
            if not (skip_normalization and m.category == "normalization")
            and (matrix_filter is None or matrix_filter(m))
        ]

        results: list[ExperimentResult] = []
        for matrix_info in tqdm(matrices, desc="Corrupting matrices"):
            try:
                result = self.run_single(matrix_info, config, include_behavioral)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Skipping {matrix_info.name}: {exc}")

        return results
