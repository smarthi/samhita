# Open questions / known ambiguities (M1 + M2)

Per SPEC.md §12.3: write the ambiguity down rather than guess. None of
these block M1 (traits + two baselines + byte accounting + capture tool)
or M2 (metric stack + report); items 1-4 gate the M3 parity pass and
should be resolved against the primary papers/official repos before
quoting any parity numbers.

**OSCAR/OScaR/KVarN (M3 codecs):** not summarized here as open questions —
they now have their own primary-source writeup, with real arXiv IDs,
license checks, and equations pulled from the papers directly, at
[docs/m3-research.md](m3-research.md). That includes the answer to
SPEC.md §10's own flagged ambiguity about attention-sink token norm
direction (confirmed: low, not high).

## 1. KIVI's per-channel grouping is windowed, M1's is whole-sequence

arXiv:2402.02750 quantizes K per-channel in fixed windows of `group_size`
tokens (32 in the paper's reference implementation), recomputing a scale
every `group_size` tokens as the sequence grows — this bounds how far a
single scale has to represent the channel's value range. `group_rtn`'s
`per_channel` axis (`crates/core/src/stages/group_rtn.rs`) instead computes
one scale per channel over the *entire* post-window sequence in a single
`encode` call. This is fine for M1's byte-accounting and stage-composition
goals (the `per_token_meta_bytes` amortization formula is already written
to generalize to windowed grouping — `head_dim * 4 bytes` per window
instead of per whole sequence — but the quantizer doesn't chunk yet).
**Before any KIVI parity claim (M3):** implement fixed-size windowed
grouping and re-check reconstruction error against the paper's own
ablations.

## 2. `lloyd_max` codebook: Acklam approximation (Rust) vs `ndtri` (Python)

`crates/core/src/normal.rs` implements Acklam's rational approximation to
the inverse normal CDF (~1e-9 relative error, no external dependency).
`python/samhita/stages/lloyd_max.py` uses `torch.special.ndtri` (exact to
float64 precision) instead, for simplicity on the Python side. The two
codebooks agree to roughly 1e-9 per centroid — negligible next to
quantization error itself — but it means `lloyd_max` round trips are not
bit-exact across languages, only close. If a future milestone needs true
bit-exact `lloyd_max` parity, port Acklam's approximation into
`prng.py` too (the constants are already there for `SplitMix64`-derived
Gaussians; only the codebook-construction grid would need to switch).

## 3. `crates/py` (PyO3 bindings) doesn't link out of the box

Building `samhita-py` on this machine (macOS/arm64, Homebrew rustc 1.98,
pyo3 0.22, `extension-module` feature) fails at the link step: undefined
Python C-API symbols (`_PyBool_Type`, `_Py_InitializeEx`, ...), meaning
pyo3-build-config isn't emitting the `-undefined dynamic_lookup` linker
arg it normally does for extension modules on macOS. It's excluded from
the workspace's `default-members` (see root `Cargo.toml`) so `cargo
build`/`cargo test` stay green without chasing this. SPEC.md's own M1
acceptance criteria don't require this binding — the Rust<->Python parity
check goes through `crates/cli` via subprocess instead (see
docs/design.md). Likely fixes to try when this becomes load-bearing (M2+,
if the subprocess bridge becomes a performance bottleneck for a larger
harness): building via `maturin develop` in a real virtualenv rather than
plain `cargo build` (maturin sets up the linker environment pyo3-build-config
expects), or pinning `pyo3` to a version tested against this exact
Python 3.12 framework build.

## 4. TurboQuant `rotate_post_rope` / per-head vs per-token rotation

`dhurandhar/src/dhurandhar/turboquant.py`'s `TurboQuantConfig` has
`per_head: bool` and `rotate_post_rope: bool` flags that this port doesn't
carry over — M1's `hadamard` stage always rotates whatever `Tensor2` it's
given (which, by construction, is one KV head's activations already), and
the activation-capture tool (`python/samhita/capture.py`) always captures
post-RoPE K/V by construction (reading them off the model's cache, which
is populated after `apply_rotary_pos_emb`). So the *behavior* these flags
selected for is already M1's only behavior; the flags themselves weren't
worth porting. Flagging in case a later milestone needs the pre-RoPE path
for some ablation — that would need a real capture-tool change, not just a
config flag.

## 5. `score()` is reference-only (decode + dot) for both K and V in M1

SPEC.md §4.2's design note calls for `score` and `decode` to stay separate
operations so an edge path can estimate `q·k` logits directly from packed
keys without a full dequantize. `Pipeline::score` in
`crates/core/src/pipeline.rs` currently *is* decode-then-dot (explicitly
documented as such) — no codec in the M1 stage inventory has a cheaper
estimator. **Resolved in M2:** `qjl_1bit` (TurboQuant Prod) is the first
codec where `score` genuinely diverges from `decode`+dot — see
docs/design.md's "M2: the metric stack, `qjl_1bit`, and the first real
report" section.

## 6. Is `turboquant_prod` actually worth shipping as a preset?

The M2 report (`reports/m2_report.md`, real Qwen2.5-0.5B-Instruct
activations, layer 0 head 0, 5 WikiText-2 prompts, 3 rotation seeds) found
`turboquant_prod` costs strictly *more* measured bytes than
`turboquant_mse` at every matched bit setting (the extra QJL bit) while
having *worse* QKᵀ logit error and worse attention-output error at every
one of them — on this one layer/head. That's a real, reproducible result
on real data, not a synthetic artifact (see docs/codecs/turboquant_prod.md
for the synthetic-data version of the same finding). It is *not* proof the
preset is useless in general: one layer, one head, one small model. Before
M3 (or before recommending `turboquant_prod` over `turboquant_mse` in any
doc), extend `report.py`'s sweep across multiple layers/heads and at least
one more model to see whether this holds up or was specific to layer-0's
well-known attention-sink-dominated behavior.
