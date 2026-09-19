"""`group_rtn`: torch mirror of `crates/core/src/stages/group_rtn.rs`
(KIVI-style per-channel or per-token symmetric uniform quantization)."""

from __future__ import annotations

import torch

from .base import Codes, ShapeCtx, Quantizer


class GroupRtn(Quantizer):
    name = "group_rtn"

    def __init__(self, bits: int, axis: str) -> None:
        assert axis in ("per_token", "per_channel")
        self.bits = bits
        self.axis = axis

    def _qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1

    def per_token_meta_bytes(self, ctx: ShapeCtx) -> float:
        if self.axis == "per_token":
            return 4.0
        return 0.0 if ctx.n_tokens == 0 else (ctx.head_dim * 4.0) / ctx.n_tokens

    def encode(self, x: torch.Tensor) -> Codes:
        qmax = self._qmax()
        if self.axis == "per_token":
            amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = amax / qmax
            payload = torch.round(x / scale).clamp(-qmax - 1, qmax)
            return Codes(
                kind="uniform", rows=x.shape[0], cols=x.shape[1], bits=self.bits,
                payload=payload, row_scale=scale.squeeze(-1), axis="per_token",
            )
        amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        payload = torch.round(x / scale).clamp(-qmax - 1, qmax)
        return Codes(
            kind="uniform", rows=x.shape[0], cols=x.shape[1], bits=self.bits,
            payload=payload, row_scale=scale.squeeze(0), axis="per_channel",
        )

    def decode(self, c: Codes) -> torch.Tensor:
        assert c.payload is not None and c.row_scale is not None
        if c.axis == "per_token":
            return c.payload * c.row_scale.unsqueeze(-1)
        return c.payload * c.row_scale.unsqueeze(0)

    def bits_per_element(self) -> float:
        return float(self.bits)
