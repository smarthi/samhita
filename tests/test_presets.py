from __future__ import annotations

import torch

from samhita.presets import load_preset


def test_kivi_preset_loads_and_round_trips():
    preset = load_preset("kivi")
    assert preset.name == "kivi"
    assert preset.evidence is not None
    assert preset.evidence["source"] == "arXiv:2402.02750"

    x = torch.randn(256, 32)
    k_pipe = preset.build_k(head_dim=32)
    v_pipe = preset.build_v(head_dim=32)
    assert k_pipe.round_trip_mse(x) >= 0.0
    assert v_pipe.round_trip_mse(x) >= 0.0


def test_turboquant_mse_preset_loads_and_round_trips():
    preset = load_preset("turboquant_mse")
    assert preset.name == "turboquant_mse"
    assert preset.evidence["source"] == "arXiv:2504.19874"

    x = torch.randn(256, 64)
    k_pipe = preset.build_k(head_dim=64)
    assert k_pipe.round_trip_mse(x) < 0.01
