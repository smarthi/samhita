"""`lloyd_max`: torch mirror of `crates/core/src/stages/lloyd_max.rs`.

Uses `torch.special.ndtri` (exact to float64 precision) for the inverse
normal CDF rather than Rust's Acklam rational approximation
(~1e-9 relative error) — see docs/open-questions.md for why this is an
accepted, documented source of tiny (not bit-exact) divergence between the
two languages' codebooks.
"""

from __future__ import annotations

import torch

from .base import Codes, ShapeCtx, Quantizer

N_SAMPLES = 20_000
ITERS = 60


def _gaussian_quantile_grid(n: int) -> torch.Tensor:
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.special.ndtri(p)


def _build_gaussian_codebook(levels: int) -> torch.Tensor:
    samples = _gaussian_quantile_grid(N_SAMPLES)  # sorted ascending
    p_init = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    centroids = torch.special.ndtri(p_init)

    for _ in range(ITERS):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        idx = torch.bucketize(samples, boundaries)
        sums = torch.zeros(levels, dtype=torch.float64).scatter_add_(0, idx, samples)
        counts = torch.zeros(levels, dtype=torch.float64).scatter_add_(
            0, idx, torch.ones_like(samples)
        )
        nonzero = counts > 0
        centroids = torch.where(nonzero, sums / counts.clamp_min(1), centroids)

    return centroids.sort().values.to(torch.float32)


class LloydMax(Quantizer):
    name = "lloyd_max"

    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.codebook = _build_gaussian_codebook(1 << bits)

    def state_bytes(self, ctx: ShapeCtx) -> int:  # noqa: ARG002
        return self.codebook.numel() * 4

    def per_token_meta_bytes(self, ctx: ShapeCtx) -> float:  # noqa: ARG002
        return 4.0

    def encode(self, x: torch.Tensor) -> Codes:
        # Matches samhita-core's lloyd_max exactly: `std` is the row's
        # standard deviation (variance taken around the row mean), but the
        # per-element normalization divides the raw value by `std` — it
        # does not re-center by subtracting the mean first.
        std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-8)
        normalized = x / std
        boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
        idx = torch.bucketize(normalized, boundaries)

        return Codes(
            kind="codebook", rows=x.shape[0], cols=x.shape[1], bits=self.bits,
            payload=idx, row_scale=std.squeeze(-1),
        )

    def decode(self, c: Codes) -> torch.Tensor:
        assert c.payload is not None and c.row_scale is not None
        values = self.codebook[c.payload]
        return values * c.row_scale.unsqueeze(-1)

    def bits_per_element(self) -> float:
        return float(self.bits)
