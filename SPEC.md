# SPEC: Composable KV-cache codec framework (working name: `samhita`)

Status: pre-implementation. Private repo until IP clearance is complete (see §11).
Audience: Claude Code / contributors. Read this fully before writing code. Build **Milestone 1 only** unless told otherwise.

---

## 1. Thesis

KV-cache quantization methods (KIVI, TurboQuant, OSCAR, OScaR, KVarN, and others) are largely different choices along the same small set of axes. A library organized around *stages* rather than *named codecs* lets us:

1. Reproduce each published method as a config.
2. Ablate one stage at a time under identical conditions.
3. Build hybrids as one-line config changes (e.g. OScaR per-token scaling followed by TurboQuant sign+residual).
4. Evaluate everything on **real model activations at matched measured bytes**, which is the thing existing wrappers mostly do not do.

The deliverable is a framework plus an honest evaluation harness. It is not a catalog of ports.

## 2. Non-goals

- No GPU serving kernels. Official kernels in SGLang/vLLM/llama.cpp will beat anything written here. We interoperate, we do not compete.
- No eviction/pruning methods (H2O, SnapKV, etc.). NVIDIA's `kvpress` covers that; interoperate if needed later.
- No full parity-tested ports of methods without live community adoption (RotorQuant, SpectralQuant, BlockQuant). Their *ideas* are welcome as stages (§4.3); their codecs are not v0 scope.
- No performance claims against vendor kernels.
- No synthetic-data-only validation of any codec. Synthetic tensors are permitted only as unit-test fixtures.

## 3. Design principles

1. **Stages, not codecs.** A codec is a pipeline of stages. Named methods are presets.
2. **Measured bytes only.** Never report `bits / 8`. Every memory figure is computed by the byte-accounting function (§5.3).
3. **Diagnostic vs serving is explicit.** A fake-quant path (round then dequantize back to float) must never be reported as memory savings. Every codec declares which path an implementation is.
4. **Real activations.** Metrics come from captured post-RoPE Q/K/V of real models.
5. **Matched-bytes comparison.** Compare codecs at equal effective bytes including all metadata, not equal nominal bits. (Example: a sign+residual scheme at nominal N bits is roughly N+1 effective bits; comparing it against plain N-bit INT is not an equal-bits comparison.)
6. **Evidence is metadata.** Every codec preset carries machine-readable evidence fields (§9).
7. **Do not trust paraphrases.** Math and hyperparameters come from the original papers and official repos, not from summaries, slides, or this document.

## 4. Architecture

### 4.1 Pipeline model

K and V have independent pipelines (they have different error sensitivities: K errors pass through softmax; V errors are linear in the output). A pipeline is an ordered composition:

```
input tensor (per layer, per KV head, tokens × head_dim)
  → layout/policy   (grouping axis; sink + recent-token windows kept high precision)
  → rotation        (none | random_orthogonal | hadamard | block_diagonal | eigenbasis | composed)
  → normalization   (none | per_channel | per_token | dual_axis_varnorm | omni_token)
  → allocation      (uniform | asymmetric_kv | mixed | water_filling)
  → quantizer       (uniform_rtn | lloyd_max | sign_residual | codebook)
  → residual        (none | qjl_1bit | error_correction)
encode → packed bytes + metadata;  decode / score(q, codes) → estimate
```

Every stage implements `fit` (optional, calibration), `encode`, `decode`, and declares `metadata_bytes()`. Stages that change the geometry of the data (rotations) must be exactly invertible or declare their reconstruction error.

### 4.2 Rust trait sketch (starting point, refine as needed)

```rust
pub trait Stage {
    fn name(&self) -> &'static str;
    /// Optional calibration from captured activations. Must be deterministic given a seed.
    fn fit(&mut self, calib: &CalibData, seed: u64) -> Result<()> { Ok(()) }
    /// Bytes of persistent state this stage needs (rotation matrices, codebooks, scales schema).
    fn state_bytes(&self, ctx: &ShapeCtx) -> usize;
    /// Bytes of per-token / per-group metadata emitted at encode time.
    fn per_token_meta_bytes(&self, ctx: &ShapeCtx) -> f64;
}

pub trait Transform: Stage {            // rotation, normalization
    fn forward(&self, x: &mut Tensor2<f32>, side: &mut SideInfo);
    fn inverse(&self, x: &mut Tensor2<f32>, side: &SideInfo);
}

pub trait Quantizer: Stage {            // rounder
    fn encode(&self, x: &Tensor2<f32>) -> Codes;
    fn decode(&self, c: &Codes) -> Tensor2<f32>;
    fn bits_per_element(&self) -> f32;  // payload only; metadata is counted separately
}

pub trait Residual: Stage {             // qjl, error correction
    fn encode(&self, x: &Tensor2<f32>, recon: &Tensor2<f32>) -> ResidualCodes;
    fn apply(&self, recon: &mut Tensor2<f32>, r: &ResidualCodes);
    /// Score-side estimator, needed for unbiased inner-product variants.
    fn score_correction(&self, q: &Tensor2<f32>, r: &ResidualCodes) -> Tensor2<f32>;
}

pub struct Pipeline { /* layout, transforms, allocation, quantizer, residual */ }
impl Pipeline {
    pub fn encode(&self, x: &Tensor2<f32>) -> Packed;
    pub fn decode(&self, p: &Packed) -> Tensor2<f32>;
    /// Estimate q·k logits directly from packed keys (may equal decode+dot in the reference path).
    pub fn score(&self, q: &Tensor2<f32>, p: &Packed) -> Tensor2<f32>;
}
```

Design note: keep `score` and `decode` as separate operations. Keys need score fidelity; values need reconstruction fidelity. This split is what makes the K/V asymmetry expressible.

### 4.3 Stage inventory

Milestone 1 stages: `hadamard` (fast Walsh–Hadamard), `random_orthogonal`, `group_rtn` (per-channel / per-token), `per_token_scale`, `lloyd_max` (Beta/Gaussian marginal codebook), `sign_residual`, `recent_window` / `sink_window`.

Later stages, with their origin (verify against the papers before implementing):

| Stage | Origin idea | Notes |
|---|---|---|
| `qjl_1bit` residual | TurboQuant (Prod variant) | unbiased inner products |
| `eigenbasis` rotation (offline covariance-aware) | OSCAR; also SpectralQuant's per-head eigenbasis | which covariance (QᵀQ vs KᵀK) is a documented design decision in OSCAR; check the paper |
| `composed` rotation (eigenbasis · Hadamard · bit-reversal) | OSCAR | |
| `canalized` Hadamard + `omni_token` scaling | OScaR | order of the two ops matters; check the paper |
| `dual_axis_varnorm` (Sinkhorn-style) | KVarN | |
| `water_filling` allocation | SpectralQuant | closed-form relaxation then integer projection |
| `block_diagonal` rotation (2D/3D/4D blocks) | RotorQuant family | O(d) rotation cost |
| `block_marginal` codebook | BlockQuant | possibly more useful for retrieval than KV |

### 4.4 Presets (reproductions of published methods)

Presets are TOML files under `presets/`. Each maps a published method onto stages. Initial set: `kivi`, `turboquant_mse`, `turboquant_prod`, then `oscar`, `oscar_scaled` (OScaR; note the name collision, see §10), `kvarn`.

Config sketch (hybrid experiment H1, see §8):

```toml
name = "oscar_scaling_plus_tq_signresidual"

[k]
group      = "channel"
rotation   = { kind = "hadamard" }
scale      = { kind = "per_token" }
quantizer  = { kind = "sign_residual", bits = 2 }
window     = { sink = 4, recent = 128, dtype = "bf16" }

[v]
group      = "token"
rotation   = { kind = "hadamard" }
scale      = { kind = "per_token" }
quantizer  = { kind = "uniform_rtn", bits = 2 }
window     = { sink = 4, recent = 128, dtype = "bf16" }
```

Preset values above are illustrative. Real values must be taken from the papers/official configs.

## 5. Execution paths and byte accounting

### 5.1 Diagnostic path
Fake-quant: encode then immediately decode to float. Used for error analysis only. Results tagged `path = "diagnostic"`. Never used to report memory.

### 5.2 Serving path
Stores packed bytes (real bit-packing). The Rust reference implements this on CPU. Results tagged `path = "serving"`.

### 5.3 Byte accounting (single source of truth)
One function, heavily tested, computes for a given (model config, context length, batch, pipeline):

- payload bytes (packed codes)
- per-token/per-group scales and zero-points
- codebooks and rotation matrices (persistent state, counted once per layer/head, plus reported amortized per token)
- residual/sign bits
- high-precision window tokens (sink + recent) at their true dtype
- peak transient workspace during encode/decode/score

Report: total bytes at context length S, and effective bits per element. Property tests must show accounting matches actual allocated buffer sizes in the serving path.

## 6. Evaluation harness

### 6.1 Activation capture
- Tool captures post-RoPE Q, K, V per layer/head from a real model on real text, saved as safetensors shards with a manifest (model id and revision, dtype, sequence length, dataset, prompt ids).
- M1 models: a small one (Llama-3.2-1B or Qwen2.5-0.5B). Later: Llama-3.1-8B and one model with a different head_dim / GQA ratio.
- Datasets: WikiText-2 for PPL; one long-context set for NIAH/RULER-style tasks.

### 6.2 Metrics, in priority order
1. Relative QKᵀ logit error (per layer/head).
2. Attention-output error (softmax(QKᵀ/√d)·V vs reference).
3. V reconstruction error.
4. Model-level: perplexity, NIAH, RULER/LongBench subset (later milestones).

MSE on K/V alone is a secondary diagnostic, not a headline metric.

### 6.3 Protocol
- Compare at matched measured bytes (interpolate or sweep the bit budget; plot error vs bytes, not error vs nominal bits).
- Multiple seeds for anything randomized (rotations, calibration subsampling); report bootstrap CIs over prompts.
- Fixed, versioned result schema (JSON) plus a viewer script.

## 7. Repository layout

```
/crates/core        Rust: stages, pipeline, packing, FWHT, codebook fitting, calibration, byte accounting
/crates/py          PyO3 bindings
/python/samhita  torch reference implementation, HF cache class, capture tool, harness
/presets            TOML presets for published methods
/fixtures           small tensors for bit-exact tests (synthetic allowed here only)
/reports            generated benchmark reports
/docs               design notes, per-codec evidence sheets
```

Rust owns: FWHT, bit packing, Lloyd–Max/codebook fitting, covariance/eigen calibration, bit-exact CPU reference (the oracle and the future edge path). Python owns: torch reference, HF integration, capture, harness. SGLang/vLLM adapters come later and stay thin.

## 8. Milestones

**M1: traits and two baselines with bit-exact tests**
- Stage traits and `Pipeline`; byte-accounting function with property tests.
- Stages: `hadamard`, `random_orthogonal`, `group_rtn`, `per_token_scale`, `lloyd_max`, `sign_residual`, `recent_window`/`sink_window`.
- Presets: `kivi`, `turboquant_mse`.
- Bit-exact (or documented tolerance) agreement between the Rust reference and a torch reference on fixtures.
- Activation capture tool working end-to-end on one small model.
- Start by extracting existing codec code from the author's `dhurandhar` repo into these traits rather than rewriting from scratch.
- Acceptance: `cargo test` and `pytest` green; one command produces a captured-activation shard and an error-vs-bytes plot for the two presets.

**M2: harness and first report**
- Full metric stack (§6.2 items 1–3), matched-bytes sweeps, seeds/CIs, result schema.
- First report on real activations supersedes earlier synthetic benchmarks.
- Add `turboquant_prod` (QJL).

**M3: live-adoption codecs with parity tests**
- `oscar` (offline calibration), `oscar_scaled` (OScaR), `kvarn`.
- Parity tests against official repos. **Check each upstream license before porting or copying any code.** Prefer clean-room implementation from the papers with parity checked on outputs.

**M4: ablations, hybrids, edge path**
- Stage ablations and hybrids. **Headline experiment H1:** does per-token (omni-token) scaling add to sign+residual at matched bytes, and on which models?
- Model-level metrics (PPL, NIAH, RULER/LongBench subset).
- CPU/edge serving path.

Definition of "done" for any milestone: tests pass, byte accounting verified against allocated buffers, results reproducible from a seed and a manifest.

## 9. Codec evidence metadata

Each preset ships `docs/codecs/<name>.md` and a TOML block:

```toml
[evidence]
source           = "arXiv:XXXX.XXXXX"      # fill from the paper
calibration      = "none | offline_small | per_model"
paper_claims     = ["..."]                  # what the paper claims
not_claimed      = ["..."]                  # e.g. single seed, single model, no kernel latency
seeds_in_paper   = 1
models_in_paper  = ["..."]
our_status       = "diagnostic | serving"
parity_checked   = false
license_of_reference_impl = "..."
```

## 10. Known pitfalls (from prior work; verify, do not assume)

- **Name collision:** OSCAR (Offline Spectral Covariance-Aware Rotation, calibrated) and OScaR (Omni-Scaled Canalized Rotation, calibration-free) are different methods. Use unambiguous internal names (`oscar_cov`, `oscar_scaled`) and cite arXiv IDs everywhere.
- **Fake-quant trap:** a runtime that rounds then dequantizes to BF16 before caching does not reduce memory. Also, for eager attention the O(seq²) score matrix can dominate peak memory and mask any KV savings; memory benchmarks need flash-style attention and cache-dominated regimes (long context, batched).
- **Synthetic-regime validity:** if a synthetic "token norm imbalance" regime is ever used in tests, verify its sign and shape against the OScaR paper (attention-sink tokens are described as anomalously *low* norm in secondary summaries; confirm in the paper itself).
- **Equal-bits confounds:** always compare at matched effective bytes.
- **Naming:** check PyPI and crates.io for the final project name; many `*quant` and `kvq-*` names are taken.

## 11. Process and IP

- Keep the repo **private** until IP/open-source clearance is confirmed with all relevant employers and patent counsel. Public GitHub, LinkedIn, and conference disclosures may already affect patent options in some jurisdictions; this is a question for counsel, not this repo.
- Use no confidential employer data, internal model weights, or internal benchmarks. Public models and datasets only.
- Include upstream attribution and license notices for anything derived from other repos.

## 12. Instructions for Claude Code

1. Read this file fully. Implement **M1 only**. Do not scaffold later milestones.
2. Before coding, propose the crate/package layout and the exact trait signatures; refine §4.2 if needed and record changes in `docs/design.md`.
3. Read the primary papers and official repos for any formula, constant, or default you implement; cite arXiv IDs in code comments. If a paper is ambiguous, write the ambiguity in `docs/open-questions.md` rather than guessing.
4. Write the byte-accounting function and its property tests first.
5. Keep diagnostic and serving paths separate at the type level.
6. Ask before adding dependencies with restrictive licenses or heavy build requirements.
7. End each session with a short status note: what passes, what is unverified, what is next.
