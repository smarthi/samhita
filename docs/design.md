# Design notes (M1)

Per SPEC.md §12.2: this records the crate/package layout and trait
signatures actually implemented, and where they deviate from the
illustrative sketch in SPEC.md §4.2.

## Crate/package layout

```
crates/core   samhita-core: traits, M1 stage inventory, Pipeline, byte accounting
crates/cli    samhita-cli: `samhita` binary — runs one preset/side/fixture
              through the Rust reference and prints MSE + byte report as
              JSON. This is also the Rust half of the cross-language
              parity check (see "Two references, one bridge" below).
crates/py     samhita-py: PyO3 bindings (scaffolded, not in default
              workspace members — see docs/open-questions.md #3).
python/samhita  torch reference, presets loader, HF-adjacent diagnostic
              cache, activation capture, harness.
presets/      kivi.toml, turboquant_mse.toml (M1's two presets)
fixtures/     small deterministic synthetic tensors + generate.py
docs/         this file, open-questions.md, codecs/*.md evidence sheets
reports/      generated: captured activation shards, error-vs-bytes plots
```

## Trait signatures actually implemented

`crates/core/src/traits.rs` follows SPEC.md §4.2 closely, with one
mechanical change: `Transform`/`Quantizer`/`Residual` take `&Tensor2`
(a small row-major `f32` matrix defined in `tensor.rs`) rather than a
generic `Tensor2<f32>` — M1 has no need for a generic element type, and
keeping it concrete avoids threading a type parameter through every stage
and the `Pipeline`. `ShapeCtx` is `{ n_tokens, head_dim }`: it describes the
*post-window* shape a stage actually sees, since the sink/recent tokens
never reach the rotation/normalization/quantizer chain at all.

`Codes` is an enum (`Uniform`, `SignResidual`, `Codebook`) rather than a
single struct, because `uniform_rtn`/`group_rtn`, `sign_residual`, and
`lloyd_max` genuinely serialize differently (a per-group scale; a packed
sign + residual; a codebook index). Each variant implements
`packed_size_bytes()` — the actual bit-packed serving-path byte count —
which is what the byte-accounting property tests check against (see
"Byte accounting" below), not a formula re-derived by hand.

## Byte accounting: ground-truth-by-construction

SPEC.md §5.3 requires the accounting function's numbers to match "actual
allocated buffer sizes in the serving path," checked by property tests.
Rather than have `compute_bytes` re-derive payload/metadata size from each
stage's declared `per_token_meta_bytes()` in isolation — which could
silently drift from what `encode()` actually produces — `compute_bytes`
(`crates/core/src/bytes.rs`) runs a real `Pipeline::encode` on the given
sample and reads `Codes::packed_size_bytes()` off the result. Persistent
per-(layer, head) state (rotation matrices, codebooks) isn't observable
from a single `encode` call, so that component still comes from each
stage's declared `state_bytes()`. `bytes.rs`'s property tests
(`payload_matches_real_packed_codes`, `scales_linearly_with_instance_count`,
`fewer_bits_never_costs_more`, `window_bytes_scale_with_dtype_and_count`)
cover linearity in instance count, monotonicity in bits, and exact
agreement with the real packed buffer.

## Two references, one bridge (Rust <-> Python parity)

SPEC.md M1 acceptance asks for "bit-exact (or documented tolerance)
agreement between the Rust reference and a torch reference on fixtures."
Both languages implement the *same* deterministic PRNG
(`crates/core/src/rng.rs` + `normal.rs` vs `python/samhita/prng.py`:
SplitMix64 + Acklam's inverse-normal-CDF approximation), so rotation signs
and (nearly) codebook centroids match exactly across languages — verified
directly (`Hadamard::new(8, 42).signs` in Rust equals
`SplitMix64(42).next_sign()` x8 in Python, bit for bit).

Given that, `tests/test_bytes_parity_with_rust.py` builds `crates/cli` and
shells out to it (`python/samhita/rust_bridge.py`) rather than
re-implementing quantization math a third time in Python for the test
alone. Measured MSE between the Rust serving path and the Python
diagnostic-path pipeline agrees to ~1e-8 on `fixtures/long_256x64.smhf`
for both presets/sides (see docs/open-questions.md #2 for the one
intentional numerical divergence: `lloyd_max`'s codebook).

The M1 harness (`python/samhita/harness.py`) takes this further: it never
re-derives byte counts in Python at all. It calls the real `samhita` CLI
for every point in the bit-budget sweep and plots the JSON it gets back.
This is the literal reading of SPEC.md §7 ("Rust owns... the bit-exact CPU
reference, the oracle... Python owns... the harness").

## Presets: what M1 simplifies

Both `presets/kivi.toml` and `presets/turboquant_mse.toml` carry an
`[evidence]` block per SPEC.md §9 with the specific simplifications each
one makes relative to its paper — see the presets themselves and
docs/open-questions.md #1. Neither is parity-checked yet (`parity_checked
= false`); that's M3 scope.
