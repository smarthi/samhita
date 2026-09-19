from __future__ import annotations

import torch

from samhita.cache import DiagnosticKVCache
from samhita.presets import load_preset


def test_diagnostic_cache_round_trip_shapes():
    preset = load_preset("kivi")
    head_dim = 32
    k_pipe = preset.build_k(head_dim=head_dim)
    v_pipe = preset.build_v(head_dim=head_dim)
    cache = DiagnosticKVCache(k_pipe, v_pipe)
    assert cache.path == "diagnostic"

    k = torch.randn(200, head_dim)
    v = torch.randn(200, head_dim)
    k_hat, v_hat = cache.round_trip(k, v)
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape
