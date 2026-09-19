"""A minimal, explicitly diagnostic-path KV cache wrapper.

SPEC.md §5.1 / §10 ("fake-quant trap"): round-tripping through `Pipeline`
per call is a *diagnostic* operation (encode then immediately decode back
to float), useful for quick ad hoc "what does this preset do to my
activations" checks — it is NOT a serving integration and must never be
read as a memory saving. It is intentionally not wired into a live HF
`generate()` loop for M1; that belongs to the CPU/edge serving path in
SPEC.md M4, which reuses the Rust serving path (`crates/core`), not this
class.
"""

from __future__ import annotations

import torch

from .pipeline import Pipeline


class DiagnosticKVCache:
    """Per-layer fake-quant round trip for K and V, for quick preset
    exploration. `path` is always `"diagnostic"` — see module docstring."""

    path = "diagnostic"

    def __init__(self, k_pipeline: Pipeline, v_pipeline: Pipeline) -> None:
        self.k_pipeline = k_pipeline
        self.v_pipeline = v_pipeline

    def round_trip(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`key_states`/`value_states`: `(seq_len, head_dim)`, one KV head.
        Returns the fake-quantized reconstruction of each, same shape."""
        k_packed = self.k_pipeline.encode(key_states)
        v_packed = self.v_pipeline.encode(value_states)
        return self.k_pipeline.decode(k_packed), self.v_pipeline.decode(v_packed)
