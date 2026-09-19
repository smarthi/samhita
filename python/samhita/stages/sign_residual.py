"""`sign_residual`: torch mirror of
`crates/core/src/stages/sign_residual.rs`, itself ported from
`dhurandhar/src/dhurandhar/turboquant.py::TurboQuantCodec` (arXiv:2504.19874
Stage 2)."""

from __future__ import annotations

import torch

from .base import Codes, ShapeCtx, Quantizer


class SignResidual(Quantizer):
    name = "sign_residual"

    def __init__(self, residual_bits: int) -> None:
        self.residual_bits = residual_bits

    def _qmax(self) -> int:
        return (1 << (self.residual_bits - 1)) - 1

    def per_token_meta_bytes(self, ctx: ShapeCtx) -> float:  # noqa: ARG002
        return 8.0

    def encode(self, x: torch.Tensor) -> Codes:
        d = x.shape[-1]
        sqrt_d = d**0.5
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sign_bits = x >= 0
        recon_sign = torch.where(sign_bits, 1.0, -1.0) * (norm / sqrt_d)
        residual = x - recon_sign

        qmax = self._qmax()
        amax = residual.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        residual_payload = torch.round(residual / scale).clamp(-qmax - 1, qmax)

        return Codes(
            kind="sign_residual",
            rows=x.shape[0],
            cols=x.shape[1],
            bits=1 + self.residual_bits,
            sign_packed=sign_bits,
            row_norm=norm.squeeze(-1),
            residual_bits=self.residual_bits,
            residual_payload=residual_payload,
            residual_scale=scale.squeeze(-1),
        )

    def decode(self, c: Codes) -> torch.Tensor:
        assert c.sign_packed is not None
        sqrt_d = c.cols**0.5
        recon_sign = torch.where(c.sign_packed, 1.0, -1.0) * (c.row_norm.unsqueeze(-1) / sqrt_d)
        residual = c.residual_payload * c.residual_scale.unsqueeze(-1)
        return recon_sign + residual

    def bits_per_element(self) -> float:
        return 1.0 + self.residual_bits
