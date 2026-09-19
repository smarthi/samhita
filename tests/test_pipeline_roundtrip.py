from __future__ import annotations

import torch

from samhita.io import read_smhf
from samhita.presets import load_preset


def test_window_tokens_are_exact_on_short_sequence(fixtures_dir):
    """A sequence shorter than sink+recent never reaches the quantizer at
    all, so the round trip must be exact (SPEC.md §4.1 layout stage)."""
    arr = read_smhf(fixtures_dir / "small_8x16.smhf")
    x = torch.tensor(arr)
    preset = load_preset("turboquant_mse")  # window sink=4, recent=128 > 8 rows
    pipe = preset.build_k(head_dim=x.shape[1])
    packed = pipe.encode(x)
    assert packed.window_split.middle == 0
    recon = pipe.decode(packed)
    assert torch.allclose(recon, x, atol=1e-6)


def test_long_sequence_exercises_quantizer(fixtures_dir):
    arr = read_smhf(fixtures_dir / "long_256x64.smhf")
    x = torch.tensor(arr)
    preset = load_preset("kivi")
    pipe = preset.build_v(head_dim=x.shape[1])
    packed = pipe.encode(x)
    assert packed.window_split.middle > 0
    recon = pipe.decode(packed)
    mse = torch.nn.functional.mse_loss(recon, x).item()
    assert 0.0 < mse < 1.0
