"""Python-side mirror of crates/core/src/stages/qjl_residual.rs's own
tests, plus a `Pipeline`-level check that mirrors
crates/core/tests/qjl_score_vs_decode.rs. See both Rust files' docstrings
for why the "does qjl_1bit improve score()" question is deliberately NOT
asserted here on synthetic data (SPEC.md §2/§10)."""

from __future__ import annotations

import torch

from samhita.presets import load_preset
from samhita.stages.qjl_residual import QjlResidual


def test_unbiased_over_resampled_projection_matrix():
    torch.manual_seed(1)
    dim = 32
    r = torch.randn(dim)
    q = torch.randn(dim)
    true_ip = (q @ r).item()

    n_seeds = 3000
    ests = []
    x = r.unsqueeze(0)
    zero = torch.zeros(1, dim)
    q_t = q.unsqueeze(0)
    for seed in range(n_seeds):
        qjl = QjlResidual(dim, seed=1000 + seed)
        codes = qjl.encode(x, zero)
        correction = qjl.score_correction(q_t, codes)
        ests.append(correction[0, 0].item())

    ests_t = torch.tensor(ests)
    mean_est = ests_t.mean().item()
    stderr = (ests_t.var(unbiased=False) / n_seeds).sqrt().item()

    assert abs(mean_est - true_ip) < 6 * stderr, f"mean_est={mean_est} true_ip={true_ip} stderr={stderr}"


def test_pipeline_decode_unchanged_score_well_defined():
    preset = load_preset("turboquant_prod")
    head_dim = 32

    # Zero out the window so all 64 test tokens actually reach the
    # quantizer/residual — turboquant_prod's real window (sink=4,
    # recent=128) would swallow a 64-token test sequence whole.
    qjl_cfg = dict(preset.raw["k"])
    qjl_cfg["window"] = {"sink": 0, "recent": 0, "dtype_bytes": 2}
    no_residual_cfg = dict(qjl_cfg)
    no_residual_cfg["residual"] = "none"

    from samhita.pipeline import Pipeline

    k_pipe = Pipeline(qjl_cfg, head_dim, seed=1)
    baseline_pipe = Pipeline(no_residual_cfg, head_dim, seed=1)

    torch.manual_seed(0)
    k = torch.randn(64, head_dim)
    q = torch.randn(16, head_dim)

    qjl_packed = k_pipe.encode(k)
    baseline_packed = baseline_pipe.encode(k)

    qjl_recon = k_pipe.decode(qjl_packed)
    baseline_recon = baseline_pipe.decode(baseline_packed)
    assert torch.allclose(qjl_recon, baseline_recon, atol=1e-5)

    qjl_score = k_pipe.score(q, qjl_packed)
    baseline_score = baseline_pipe.score(q, baseline_packed)
    assert qjl_score.shape == (q.shape[0], k.shape[0])
    assert baseline_score.shape == (q.shape[0], k.shape[0])
    assert not torch.allclose(qjl_score, baseline_score), "qjl_1bit should change score()"


def test_score_covers_full_sequence_with_a_real_window():
    """Regression test mirroring
    crates/core/tests/qjl_score_vs_decode.rs::score_shape_and_window_columns_are_correct_with_a_real_window:
    an earlier version of score()'s residual branch only covered the
    "middle" tokens, silently truncating the score matrix whenever the
    window (sink/recent) is non-empty."""
    from samhita.pipeline import Pipeline

    head_dim = 16
    cfg = {
        "rotation": "hadamard",
        "scale": "none",
        "quantizer": {"kind": "sign_residual", "bits": 3},
        "residual": "qjl_1bit",
        "window": {"sink": 2, "recent": 3, "dtype_bytes": 2},
    }
    pipe = Pipeline(cfg, head_dim, seed=7)

    torch.manual_seed(11)
    seq_len = 20
    k = torch.randn(seq_len, head_dim)
    q = torch.randn(4, head_dim)

    packed = pipe.encode(k)
    score = pipe.score(q, packed)
    assert score.shape == (q.shape[0], seq_len)

    window_rows = list(range(2)) + list(range(seq_len - 3, seq_len))
    expected = q @ k[window_rows].T
    assert torch.allclose(score[:, window_rows], expected, atol=1e-3)
