"""`random_orthogonal`: torch mirror of
`crates/core/src/stages/random_orthogonal.rs` (Gram-Schmidt on a
SplitMix64+inverse-normal-CDF seeded Gaussian matrix)."""

from __future__ import annotations

import torch

from ..prng import SplitMix64
from .base import ShapeCtx, SideInfo, Transform


class RandomOrthogonal(Transform):
    name = "random_orthogonal"

    def __init__(self, dim: int, seed: int) -> None:
        self.dim = dim
        rng = SplitMix64(seed)
        rows = [[rng.next_gaussian() for _ in range(dim)] for _ in range(dim)]
        q = torch.tensor(rows, dtype=torch.float64)

        for i in range(dim):
            for j in range(i):
                dot = torch.dot(q[i], q[j])
                q[i] = q[i] - dot * q[j]
            norm = q[i].norm().clamp_min(1e-12)
            q[i] = q[i] / norm

        self.q = q.to(torch.float32)

    def state_bytes(self, ctx: ShapeCtx) -> int:  # noqa: ARG002
        return self.dim * self.dim * 4

    def forward(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor:  # noqa: ARG002
        return x @ self.q.T

    def inverse(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor:  # noqa: ARG002
        return x @ self.q
