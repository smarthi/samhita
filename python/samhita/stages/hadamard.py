"""`hadamard`: torch mirror of `crates/core/src/stages/hadamard.rs`, same
FWHT butterfly and the same SplitMix64-derived signs, so Rust and Python
rotate a given vector identically (see `python/samhita/prng.py`).
"""

from __future__ import annotations

import torch

from ..prng import SplitMix64
from .base import ShapeCtx, SideInfo, Stage, Transform


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht_raw(x: torch.Tensor) -> torch.Tensor:
    """In-place-semantics (returns a new tensor) fast Walsh-Hadamard
    butterfly along the last dim. Unnormalized: applying twice scales by n."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, "fwht requires a power-of-two last dim"
    out = x.clone()
    length = 1
    while length < n:
        out = out.view(*out.shape[:-1], n // (2 * length), 2, length)
        u = out[..., 0, :].clone()
        v = out[..., 1, :].clone()
        out[..., 0, :] = u + v
        out[..., 1, :] = u - v
        out = out.view(*out.shape[:-3], n)
        length *= 2
    return out


def _apply_h(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    return _fwht_raw(x) / (n**0.5)


class Hadamard(Transform):
    name = "hadamard"

    def __init__(self, head_dim: int, seed: int) -> None:
        self.orig_dim = head_dim
        self.padded_dim = _next_pow2(head_dim)
        rng = SplitMix64(seed)
        signs = [rng.next_sign() for _ in range(self.padded_dim)]
        self.signs = torch.tensor(signs, dtype=torch.float32)

    def state_bytes(self, ctx: ShapeCtx) -> int:  # noqa: ARG002
        return (self.padded_dim + 7) // 8

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.padded_dim == self.orig_dim:
            return x
        pad = self.padded_dim - self.orig_dim
        return torch.nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor:  # noqa: ARG002
        padded = self._pad(x)
        signed = padded * self.signs
        return _apply_h(signed)

    def inverse(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor:  # noqa: ARG002
        unrotated = _apply_h(x) * self.signs
        if self.padded_dim != self.orig_dim:
            unrotated = unrotated[..., : self.orig_dim]
        return unrotated
