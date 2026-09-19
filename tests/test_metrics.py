from __future__ import annotations

import torch

from samhita.metrics import compute_attention_metrics
from samhita.presets import load_preset


def test_metrics_are_near_zero_for_a_near_lossless_config():
    """A high-bit, no-window group_rtn pipeline should reconstruct K/V
    almost exactly, so all three metrics should be small."""
    torch.manual_seed(0)
    head_dim = 16
    seq_len = 40

    cfg = {
        "rotation": "none",
        "scale": "none",
        "quantizer": {"kind": "group_rtn", "bits": 8, "axis": "per_token"},
        "residual": "none",
        "window": {"sink": 0, "recent": 0, "dtype_bytes": 2},
    }
    from samhita.pipeline import Pipeline

    k_pipe = Pipeline(cfg, head_dim, seed=1)
    v_pipe = Pipeline(cfg, head_dim, seed=2)

    q = torch.randn(seq_len, head_dim)
    k = torch.randn(seq_len, head_dim)
    v = torch.randn(seq_len, head_dim)

    metrics = compute_attention_metrics(q, k, v, k_pipe, v_pipe)
    assert metrics.qkt_logit_error < 0.05
    assert metrics.attention_output_error < 0.05
    assert metrics.v_reconstruction_error < 0.05


def test_metrics_are_worse_for_a_coarser_config():
    """Sanity: a much coarser (2-bit) config should score strictly worse
    than the 8-bit config above on the same data, on all three metrics."""
    torch.manual_seed(0)
    head_dim = 16
    seq_len = 40

    def build(bits):
        cfg = {
            "rotation": "none",
            "scale": "none",
            "quantizer": {"kind": "group_rtn", "bits": bits, "axis": "per_token"},
            "residual": "none",
            "window": {"sink": 0, "recent": 0, "dtype_bytes": 2},
        }
        from samhita.pipeline import Pipeline

        return Pipeline(cfg, head_dim, seed=1), Pipeline(cfg, head_dim, seed=2)

    q = torch.randn(seq_len, head_dim)
    k = torch.randn(seq_len, head_dim)
    v = torch.randn(seq_len, head_dim)

    k8, v8 = build(8)
    k2, v2 = build(2)

    fine = compute_attention_metrics(q, k, v, k8, v8)
    coarse = compute_attention_metrics(q, k, v, k2, v2)

    assert coarse.qkt_logit_error > fine.qkt_logit_error
    assert coarse.attention_output_error > fine.attention_output_error
    assert coarse.v_reconstruction_error > fine.v_reconstruction_error


def test_metrics_run_end_to_end_on_a_real_captured_shard(repo_root):
    """Uses the real Qwen2.5-0.5B-Instruct shard captured during M1
    (reports/activations/qwen2.5-0.5b-instruct), if present, to prove the
    metric stack runs on real activations end-to-end — the actual point
    of this module (SPEC.md §2/§10: no synthetic-only validation)."""
    import pytest

    from samhita.harness import load_shard

    shard_dir = repo_root / "reports" / "activations" / "qwen2.5-0.5b-instruct"
    if not (shard_dir / "activations.safetensors").exists():
        pytest.skip("no captured shard on disk; run `python -m samhita.cli capture` first")

    manifest, tensors = load_shard(shard_dir)
    q = torch.tensor(tensors["layer0.q"][0])  # (seq_len, head_dim), head 0
    k = torch.tensor(tensors["layer0.k"][0])
    v = torch.tensor(tensors["layer0.v"][0])
    head_dim = manifest["head_dim"]

    preset = load_preset("kivi")
    k_pipe = preset.build_k(head_dim=head_dim, seed=1)
    v_pipe = preset.build_v(head_dim=head_dim, seed=2)

    metrics = compute_attention_metrics(q, k, v, k_pipe, v_pipe)
    assert metrics.qkt_logit_error >= 0.0
    assert metrics.attention_output_error >= 0.0
    assert metrics.v_reconstruction_error >= 0.0
