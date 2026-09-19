"""`Pipeline`: torch mirror of `crates/core/src/pipeline.rs`.

Diagnostic path only (SPEC.md §5.1): `encode` immediately followed by
`decode` round-trips through float32 the whole way, which is exactly the
fake-quant pattern the framework must never report as a memory saving.
Real memory figures come from the Rust serving path
(`crates/cli`/`compute_bytes`), which this module's callers (`harness.py`,
`cache.py`) invoke separately — see `docs/design.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .stages import (
    Codes,
    GroupRtn,
    Hadamard,
    LloydMax,
    PerTokenScale,
    QjlResidual,
    RandomOrthogonal,
    ResidualCodes,
    SideInfo,
    SignResidual,
    WindowPolicy,
    WindowSplit,
)


@dataclass
class Packed:
    window_split: WindowSplit
    high_prec: torch.Tensor
    side: SideInfo
    codes: Codes
    residual_codes: ResidualCodes


class Pipeline:
    """Built from one `[k]`/`[v]` table of a preset TOML (see `presets.py`)."""

    def __init__(self, cfg: dict, head_dim: int, seed: int) -> None:
        self.head_dim = head_dim
        self.path = "diagnostic"

        rotation_kind = cfg.get("rotation", "none")
        if rotation_kind == "none":
            self.rotation = None
        elif rotation_kind == "hadamard":
            self.rotation = Hadamard(head_dim, seed)
        elif rotation_kind == "random_orthogonal":
            self.rotation = RandomOrthogonal(head_dim, seed)
        else:
            raise ValueError(f"unknown rotation kind: {rotation_kind}")

        scale_kind = cfg.get("scale", "none")
        if scale_kind == "none":
            self.scale = None
        elif scale_kind == "per_token":
            self.scale = PerTokenScale()
        else:
            raise ValueError(f"unknown scale kind: {scale_kind}")

        q = cfg["quantizer"]
        kind = q["kind"]
        if kind == "group_rtn":
            self.quantizer = GroupRtn(bits=q["bits"], axis=q["axis"])
        elif kind == "lloyd_max":
            self.quantizer = LloydMax(bits=q["bits"])
        elif kind == "sign_residual":
            self.quantizer = SignResidual(residual_bits=q["bits"])
        else:
            raise ValueError(f"unknown quantizer kind: {kind}")

        residual_kind = cfg.get("residual", "none")
        if residual_kind == "none":
            self.residual = None
        elif residual_kind == "qjl_1bit":
            if scale_kind != "none":
                raise ValueError(
                    "qjl_1bit residual + a `scale` stage isn't supported yet: `scale` is "
                    "per-token/data-dependent on K, so there's no shared transform score() "
                    "could apply to `q` to keep it in the same space as the scaled K "
                    "reconstruction. See docs/open-questions.md."
                )
            # Working (post-rotation) dim, matching Rust's `working_dim`.
            working_dim = self.rotation.padded_dim if isinstance(self.rotation, Hadamard) else head_dim
            self.residual = QjlResidual(working_dim, seed + 0x5115_5115)
        else:
            raise ValueError(f"unknown residual kind: {residual_kind}")

        w = cfg["window"]
        self.window = WindowPolicy(sink=w["sink"], recent=w["recent"], high_prec_dtype_bytes=w["dtype_bytes"])

    def encode(self, x: torch.Tensor) -> Packed:
        assert x.shape[-1] == self.head_dim
        seq_len = x.shape[0]
        split = self.window.split(seq_len)

        high_prec = torch.cat([x[: split.sink], x[seq_len - split.recent :]], dim=0)
        middle = x[split.sink : split.sink + split.middle]

        side = SideInfo()
        if self.rotation is not None:
            middle = self.rotation.forward(middle, side)
        if self.scale is not None:
            middle = self.scale.forward(middle, side)
        codes = self.quantizer.encode(middle)
        if self.residual is not None:
            stage1_recon = self.quantizer.decode(codes)
            residual_codes = self.residual.encode(middle, stage1_recon)
        else:
            residual_codes = ResidualCodes()

        return Packed(window_split=split, high_prec=high_prec, side=side, codes=codes, residual_codes=residual_codes)

    def decode(self, p: Packed) -> torch.Tensor:
        middle = self.quantizer.decode(p.codes)
        if self.residual is not None:
            middle = self.residual.apply(middle, p.residual_codes)
        if self.scale is not None:
            middle = self.scale.inverse(middle, p.side)
        if self.rotation is not None:
            middle = self.rotation.inverse(middle, p.side)

        sink_part = p.high_prec[: p.window_split.sink]
        recent_part = p.high_prec[p.window_split.sink :]
        return torch.cat([sink_part, middle, recent_part], dim=0)

    def score(self, q: torch.Tensor, p: Packed) -> torch.Tensor:
        """Estimate `q @ k^T` directly from packed keys. See
        `crates/core/src/pipeline.rs::score` for the full design note —
        this mirrors it exactly (and shares its `qjl_1bit`-only divergence
        from decode+dot)."""
        if self.residual is None:
            k = self.decode(p)
            return q @ k.T

        # Sink/recent window tokens are exact (never rotated/quantized):
        # score them by a plain dot product against the original q, same
        # as decode()+dot would give for those columns.
        sink = p.high_prec[: p.window_split.sink]
        recent = p.high_prec[p.window_split.sink :]
        sink_scores = q @ sink.T
        recent_scores = q @ recent.T

        q_rot = q
        if self.rotation is not None:
            q_rot = self.rotation.forward(q_rot, SideInfo())
        k_stage1 = self.quantizer.decode(p.codes)  # still in rotated space
        middle_scores = q_rot @ k_stage1.T
        middle_scores = middle_scores + self.residual.score_correction(q_rot, p.residual_codes)

        return torch.cat([sink_scores, middle_scores, recent_scores], dim=-1)

    def round_trip_mse(self, x: torch.Tensor) -> float:
        """Convenience for the diagnostic path: encode, decode, MSE."""
        packed = self.encode(x)
        recon = self.decode(packed)
        return torch.nn.functional.mse_loss(recon, x).item()
