"""The M2 metric stack, in SPEC.md §6.2 priority order:

1. Relative QKᵀ logit error (per layer/head).
2. Attention-output error (softmax(QKᵀ/√d)·V vs reference).
3. V reconstruction error.

"MSE on K/V alone is a secondary diagnostic, not a headline metric" — that
secondary diagnostic is `Pipeline.round_trip_mse` (M1); this module is the
real thing SPEC.md §6.2 asks for. All three take real captured `q, k, v`
(the *reference*, unquantized, post-RoPE activations for one layer/head)
plus the K/V `Pipeline`s under test, and compute error relative to the
reference attention computation — never against another quantized variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .pipeline import Pipeline


def _relative_frobenius_error(est: torch.Tensor, truth: torch.Tensor) -> float:
    num = torch.sum((est - truth) ** 2)
    den = torch.sum(truth**2).clamp_min(1e-12)
    return (num / den).sqrt().item()


@dataclass
class AttentionMetrics:
    qkt_logit_error: float
    attention_output_error: float
    v_reconstruction_error: float


def compute_attention_metrics(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_pipeline: Pipeline, v_pipeline: Pipeline
) -> AttentionMetrics:
    """`q, k, v`: `(seq_len, head_dim)` reference (unquantized) activations
    for one layer/head. Encodes `k`/`v` through the given pipelines and
    scores/reconstructs via each pipeline's own `score`/`decode` — i.e. the
    K path uses `score()` (which is decode+dot unless a residual stage
    makes it diverge, e.g. `qjl_1bit`), and the V path uses `decode()`
    (reconstruction fidelity, per SPEC.md's K/V asymmetry, §4.1).
    """
    head_dim = q.shape[-1]
    scale = head_dim**-0.5

    k_packed = k_pipeline.encode(k)
    v_packed = v_pipeline.encode(v)
    v_hat = v_pipeline.decode(v_packed)

    logits_ref = (q @ k.T) * scale
    logits_hat = k_pipeline.score(q, k_packed) * scale
    qkt_err = _relative_frobenius_error(logits_hat, logits_ref)

    attn_ref = torch.softmax(logits_ref, dim=-1)
    attn_hat = torch.softmax(logits_hat, dim=-1)
    out_ref = attn_ref @ v
    out_hat = attn_hat @ v_hat
    attn_out_err = _relative_frobenius_error(out_hat, out_ref)

    v_err = _relative_frobenius_error(v_hat, v)

    return AttentionMetrics(
        qkt_logit_error=qkt_err, attention_output_error=attn_out_err, v_reconstruction_error=v_err
    )
