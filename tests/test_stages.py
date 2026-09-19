from __future__ import annotations

import torch

from samhita.stages import GroupRtn, Hadamard, LloydMax, PerTokenScale, RandomOrthogonal, SignResidual
from samhita.stages.base import SideInfo


def test_hadamard_round_trip_power_of_two():
    h = Hadamard(8, seed=42)
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5, -0.25, 4.0, -1.5, 2.5]])
    side = SideInfo()
    y = h.forward(x, side)
    assert y.shape == x.shape
    recon = h.inverse(y, side)
    assert torch.allclose(x, recon, atol=1e-4)


def test_hadamard_round_trip_pads_non_power_of_two():
    h = Hadamard(5, seed=7)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    side = SideInfo()
    y = h.forward(x, side)
    assert y.shape[-1] == 8
    recon = h.inverse(y, side)
    assert recon.shape[-1] == 5
    assert torch.allclose(x, recon, atol=1e-4)


def test_random_orthogonal_round_trip():
    ro = RandomOrthogonal(6, seed=123)
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5, -0.25, 4.0]])
    side = SideInfo()
    y = ro.forward(x, side)
    recon = ro.inverse(y, side)
    assert torch.allclose(x, recon, atol=1e-3)


def test_per_token_scale_round_trip():
    stage = PerTokenScale()
    x = torch.tensor([[1.0, -4.0, 2.0], [0.0, 0.0, 0.0]])
    side = SideInfo()
    y = stage.forward(x, side)
    assert torch.allclose(y[0].abs().max(), torch.tensor(1.0))
    recon = stage.inverse(y, side)
    assert torch.allclose(x, recon, atol=1e-5)


def test_group_rtn_per_token_round_trip_is_close():
    q = GroupRtn(bits=4, axis="per_token")
    x = torch.tensor([[1.0, -2.0, 3.0, -0.5], [0.1, 0.2, -0.3, 0.05]])
    codes = q.encode(x)
    recon = q.decode(codes)
    assert torch.nn.functional.mse_loss(recon, x).item() < 0.05


def test_group_rtn_per_channel_round_trip_is_close():
    q = GroupRtn(bits=4, axis="per_channel")
    x = torch.tensor([[1.0, -20.0, 3.0, -0.5], [0.9, -18.0, 2.8, -0.4], [1.1, -19.0, 3.2, -0.6]])
    codes = q.encode(x)
    recon = q.decode(codes)
    assert torch.nn.functional.mse_loss(recon, x).item() < 1.0


def test_sign_residual_round_trip_is_close():
    q = SignResidual(residual_bits=4)
    x = torch.tensor([[1.0, -2.0, 3.0, -0.5, 0.25, -4.0, 1.5, -2.5]])
    codes = q.encode(x)
    recon = q.decode(codes)
    assert torch.nn.functional.mse_loss(recon, x).item() < 0.2


def test_lloyd_max_codebook_sorted_and_symmetric():
    lm = LloydMax(bits=3)
    assert lm.codebook.numel() == 8
    assert torch.all(lm.codebook[1:] > lm.codebook[:-1])
    assert lm.codebook.sum().abs().item() < 0.05


def test_lloyd_max_round_trip_reasonable():
    lm = LloydMax(bits=4)
    row = torch.tensor(
        [[-2.0, -1.5, -1.0, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, -0.5, 0.5, 0.05, -0.05]]
    )
    codes = lm.encode(row)
    recon = lm.decode(codes)
    assert torch.nn.functional.mse_loss(recon, row).item() < 0.2
