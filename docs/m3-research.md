# M3 prep: OSCAR, OScaR, KVarN — paper IDs, licenses, and formulas

Written up per the user's direction (2026-09-18 conversation): rather than
implement these ahead of the milestone plan, do the research now so M3
can start from primary sources instead of re-deriving this from scratch.
SPEC.md §12.3 ("do not trust paraphrases... math and hyperparameters come
from the original papers") — everything below is from the papers'/repos'
own text (fetched directly), not from SPEC.md's summaries of them. All
three are 2026 papers; SPEC.md's own naming/notes about them turn out to
be accurate. Nothing in this doc has been implemented or parity-tested —
it's a reading list plus a mapping onto samhita's stage architecture.

## OSCAR (offline, covariance-calibrated)

- **Paper:** arXiv:2605.17757, "OSCAR: Offline Spectral Covariance-Aware
  Rotation for 2-bit KV Cache Quantization" (May 2026).
- **Official code:** github.com/FutureMLS-Lab/OSCAR — **MIT license**.
- **Claims:** ~2.28 effective bits/element at near-BF16 accuracy, ~8x KV
  memory reduction, up to ~7x serving throughput (their numbers, unverified
  here).

**Mechanism** (per-layer, per-head, computed once offline from calibration
activations):

- Two *different* covariance targets for K and V — not naive `K^T K` /
  `V^T V`:
  - K's rotation target: `C_Q = (1/N) Σ_n q_n^T q_n` — the **query**
    covariance, because key quantization error propagates through the
    logits (`~ Q^T Q`-weighted), not through K's own marginal distribution.
  - V's rotation target: `C_S = (1/N) V^T S^T S V`, where
    `S = softmax_row(QK^T/√d)` — the attention-*weighted* value
    covariance, because value error propagates through the attention-
    weighted aggregation.
- Eigendecompose each: `C_Q = U_Q Λ_Q U_Q^T`, `C_S = U_S Λ_S U_S^T`.
- Final composed rotations: `R_K = U_Q · H_had · P_br`,
  `R_V = U_S · H_had · P_br` — eigenbasis first, then a Hadamard transform
  (spreads variance uniformly across the now-importance-ordered channels),
  then a bit-reversal permutation (interleaves large/small-variance
  channels so adjacent channels have similar dynamic range for
  group-wise quantization).
- **Calibration:** one forward pass, ~8,000-8,878 tokens from a GPQA-
  Diamond-style set, per model — not task-specific, reused across
  downstream benchmarks. This is the first real user of the `fit(&mut
  self, calib: &CalibData, seed)` hook every M1 `Stage` already declares
  as a no-op.

**Mapping onto samhita's stage architecture:** this needs one new
`Transform` stage, `eigenbasis`, whose `fit()` does the covariance
estimation + eigendecomposition (K and V need *different* fit logic, since
their covariance targets differ — K's needs Q, V's needs the full
attention matrix S, which is more calibration plumbing than any M1 stage
needed), plus a `composed` rotation that chains `eigenbasis → hadamard →
bit_reversal` (bit-reversal permutation is a new, trivial stage: a fixed
index permutation, effectively free to state/byte-account for). No changes
needed to `Pipeline`'s stage ordering (rotation still comes before
normalization/quantizer).

## OScaR (calibration-free)

- **Paper:** arXiv:2605.19660, "OScaR: The Occam's Razor for Extreme KV
  Cache Quantization in LLMs and Beyond" (May 2026).
- **Official code:** github.com/ZunhaiSu/OScaR-KV-Quant — **MIT license**.
  (This is the upstream repo `oscar-granite-kv-quant`'s
  `kv_cache_compression.quarot_utils` wraps — its classes being named
  `QuaRot*` is presumably just upstream's internal naming choice, not a
  sign OScaR *is* QuaRot; SPEC.md §10's naming-collision warning was about
  OSCAR-vs-OScaR, and that distinction holds up: these are two genuinely
  different papers, confirmed above.)
- **Claims:** near-lossless at INT2; vs BF16 FlashDecoding-v2, up to 3.0x
  decode speedup, 5.3x memory reduction, 4.1x throughput (unverified here).

**Mechanism:**

- Names the problem **Token Norm Imbalance (TNI)**: reconstruction error in
  a per-block quantizer is governed by the *range* of token norms within
  the block, and a sparse-but-consistent subset of tokens (attention-sink
  tokens) have **markedly reduced norms** across Q, K, and V. This
  resolves SPEC.md §10's own flagged ambiguity ("verify... attention-sink
  tokens are described as anomalously *low* norm in secondary summaries;
  confirm in the paper itself") — confirmed: low, not high.
- **Canalized Rotation** = a Fast Hadamard Transform on K (and a matching
  FHT on Q, applied so it cancels out of `QK^T` at attention time) — named
  "canalized" for what it enables next, not a distinct rotation math from
  plain Hadamard.
- **Omni-Token Scaling** = per-token scaling *after* the Hadamard
  rotation, by that token's L2 norm across all heads (not per-head
  max-abs, which is what samhita's current `per_token_scale` computes).
  Applying the scaling *before* rotation would amplify low-norm outlier
  tokens uniformly across all channels first ("Scaling-Induced Outlier
  Artifact") — the paper is explicit that the Hadamard step has to come
  first specifically to make the subsequent scaling safe.
- **Order confirmed: rotation, then scaling, then quantizer** — exactly
  samhita's existing `Pipeline` stage order (`rotation -> scale ->
  quantizer`). SPEC.md's M4 headline experiment H1 ("does per-token
  scaling add to sign+residual at matched bytes") is *this exact
  combination*, minus OScaR's own quantizer.
- **Calibration-free** — no `fit()` needed, unlike OSCAR.

**Mapping onto samhita's stage architecture:** `hadamard` already exists
and needs no changes. `per_token_scale` needs a new variant (or a sibling
stage, `omni_token_scale`) that computes the L2 norm rather than max-abs —
a small, well-scoped change. Concretely, `presets/oscar_scaled.toml` would
plausibly be `rotation = hadamard`, `scale = omni_token` (new), `quantizer
= sign_residual` or `group_rtn` at 2 bits — i.e., the H1 hybrid experiment
SPEC.md already anticipated is close to a real preset away.

## KVarN (calibration-free, decode-time)

- **Paper:** arXiv:2606.03458, "KVarN: Variance-Normalized KV-Cache
  Quantization Mitigates Error Accumulation in Reasoning Tasks" (June
  2026).
- **Official code:** github.com/huawei-csl/KVarN (vLLM v0.23.0 integration)
  — **Apache-2.0 license**.
- **Claims:** new SOTA at 2-bit on MATH500/AIME24/HumanEval-style
  reasoning benchmarks by targeting *error accumulation across decode
  steps*, not just single-step reconstruction error (unverified here).

**Mechanism:**

- Adapts "the log-domain standard-deviation-scaling implementation of
  SINQ" (their citation) to KV caches: an iterative, Sinkhorn-Knopp-style
  alternating normalization across **both** the channel and token
  dimensions (prior methods normalize only one axis). ~8 iterations in
  their experiments; each iteration recomputes row (token) and column
  (channel) log-scale factors and clamps for stability.
- **Order:** Hadamard rotation (channel dimension) first, once per token
  as it's generated; then, once a block of tokens (e.g. 128, i.e. the
  same order of magnitude as samhita's `recent`/`sink` window sizes)
  accumulates, that block is jointly variance-normalized across both axes;
  then round-to-nearest quantization.
- Calibration-free; the dual-axis normalization is computed purely from
  the KV cache's own current block, online.

**Mapping onto samhita's stage architecture:** needs one new
`normalization`-slot stage, `dual_axis_varnorm`, implementing the
iterative row/column log-scale Sinkhorn procedure (a genuinely new
algorithm, not a small variant of `per_token_scale`). Its natural home in
`Pipeline` is the same `scale` slot `per_token_scale` occupies today, but
it needs to operate over a whole block of tokens at once (not row-
independently), and — unlike everything M1 shipped — its `per_token_meta_bytes`
declaration needs to account for iteration count / convergence bytes if
any auxiliary state is kept, which none of M1's stages needed to think
about.

## Net effect on the M1 architecture

Nothing above required changing `Stage`/`Transform`/`Quantizer`/`Pipeline`
as built for M1. The one design choice validated by all three papers: the
`rotation -> normalization -> quantizer` stage order SPEC.md §4.1 lays out
(and `crates/core/src/pipeline.rs` implements) is the order all three
papers independently converge on. Three new things M1 genuinely doesn't
have yet, needed before any of these three presets exist for real:

1. A real `fit()` implementation using `CalibData` (OSCAR only) — the
   trait hook already exists; nothing uses it yet.
2. A `bit_reversal` permutation stage and a `composed` rotation that
   chains stages (OSCAR).
3. An L2-norm-based token-scaling variant (OScaR) and a genuinely new
   iterative dual-axis Sinkhorn stage (KVarN) — the first M1 stage
   ("stages operate row-independently") assumption to break, since KVarN's
   normalization needs the whole block at once.

None of this is implemented yet — this document is reading-list-plus-
mapping, not a plan of record.
