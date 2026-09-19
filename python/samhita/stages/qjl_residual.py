"""`qjl_1bit`: torch mirror of `crates/core/src/stages/qjl_residual.rs`.
See that module's docstring for the formula and its provenance caveat
(SPEC.md §3 principle 7 — extracted via an automated paper-fetch pass, not
a hand read of the source; re-verify before quoting parity numbers)."""

from __future__ import annotations

import math

import torch

from ..prng import SplitMix64
from .base import Residual, ResidualCodes, ShapeCtx


class QjlResidual(Residual):
    name = "qjl_1bit"

    def __init__(self, dim: int, seed: int) -> None:
        self.dim = dim
        rng = SplitMix64(seed)
        s = [rng.next_gaussian() for _ in range(dim * dim)]
        self.s = torch.tensor(s, dtype=torch.float32).reshape(dim, dim)

    def state_bytes(self, ctx: ShapeCtx) -> int:  # noqa: ARG002
        return self.dim * self.dim * 4

    def per_token_meta_bytes(self, ctx: ShapeCtx) -> float:  # noqa: ARG002
        return 4.0

    def encode(self, x: torch.Tensor, recon: torch.Tensor) -> ResidualCodes:
        residual = x - recon
        norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        unit = residual / norm
        projected = unit @ self.s.T  # (rows, dim)
        sign_packed = projected >= 0.0
        return ResidualCodes(
            kind="qjl_1bit", rows=x.shape[0], proj_dim=self.dim,
            sign_packed=sign_packed, row_norm=norm.squeeze(-1),
        )

    def apply(self, recon: torch.Tensor, r: ResidualCodes) -> torch.Tensor:  # noqa: ARG002
        # No-op by design: see crates/core/src/stages/qjl_residual.rs docs.
        return recon

    def score_correction(self, q: torch.Tensor, r: ResidualCodes) -> torch.Tensor:
        assert r.kind == "qjl_1bit"
        scale = math.sqrt(math.pi / 2.0) / self.dim
        s_q = q @ self.s.T  # (n_q, dim)
        signs = torch.where(r.sign_packed, 1.0, -1.0)  # (n_k, dim)
        dot = s_q @ signs.T  # (n_q, n_k)
        return r.row_norm.unsqueeze(0) * scale * dot
