"""Stage/Transform/Quantizer ABCs mirroring `crates/core/src/traits.rs`.

This is the torch reference side of the pipeline (SPEC.md §7: "Python owns:
torch reference"). It is a *diagnostic-path* implementation only — encode
followed immediately by decode, both in float32 — never a claim about
memory savings (SPEC.md §5.1, §10 "fake-quant trap"). The Rust core is the
serving-path oracle; `tests/test_bytes_parity_with_rust.py` checks the two
agree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SideInfo:
    row_scale: torch.Tensor | None = None


@dataclass
class ShapeCtx:
    n_tokens: int
    head_dim: int


@dataclass
class Codes:
    """Loose torch-side mirror of `samhita_core::codes::Codes`. Only the
    fields a given quantizer populates are meaningful."""

    kind: str
    rows: int
    cols: int
    bits: int
    payload: torch.Tensor | None = None
    row_scale: torch.Tensor | None = None
    axis: str = "per_token"
    sign_packed: torch.Tensor | None = None
    row_norm: torch.Tensor | None = None
    residual_bits: int = 0
    residual_payload: torch.Tensor | None = None
    residual_scale: torch.Tensor | None = None


class Stage(ABC):
    name: str = "stage"

    def fit(self, calib: list[torch.Tensor] | None, seed: int) -> None:  # noqa: ARG002
        return None

    def state_bytes(self, ctx: ShapeCtx) -> int:  # noqa: ARG002
        return 0

    def per_token_meta_bytes(self, ctx: ShapeCtx) -> float:  # noqa: ARG002
        return 0.0


class Transform(Stage):
    @abstractmethod
    def forward(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor: ...

    @abstractmethod
    def inverse(self, x: torch.Tensor, side: SideInfo) -> torch.Tensor: ...


class Quantizer(Stage):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Codes: ...

    @abstractmethod
    def decode(self, c: Codes) -> torch.Tensor: ...

    @abstractmethod
    def bits_per_element(self) -> float: ...
