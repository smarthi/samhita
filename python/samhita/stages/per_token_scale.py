"""`per_token_scale`: torch mirror of
`crates/core/src/stages/per_token_scale.rs`."""

from __future__ import annotations

import torch

from .base import ShapeCtx, SideInfo, Transform


class PerTokenScale(Transform):
    name = "per_token_scale"

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def per_token_meta_bytes(self, ctx: ShapeCtx) -> float:  # noqa: ARG002
        return 4.0

    def forward(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor:
        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        side.row_scale = max_abs.squeeze(-1)
        return x / max_abs

    def inverse(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor:
        assert side.row_scale is not None
        return x * side.row_scale.unsqueeze(-1)
