# Design notes (M1 + M2)

Per SPEC.md §12.2: this records the crate/package layout and trait
signatures actually implemented, and where they deviate from the
illustrative sketch in SPEC.md §4.2.

## Crate/package layout

```
crates/core   samhita-core: traits, stage inventory, Pipeline, byte accounting
crates/cli    samhita-cli: `samhita` binary — runs one preset/side/fixture
              through the Rust reference and prints MSE + byte report as
              JSON. This is also the Rust half of the cross-language
              parity check (see "Two references, one bridge" below).
crates/py     samhita-py: PyO3 bindings (scaffolded, not in default
              workspace members — see docs/open-questions.md #3).
python/samhita  torch reference, presets loader, HF-adjacent diagnostic
              cache, activation capture, metrics, harness/report.
presets/      kivi.toml, turboquant_mse.toml, turboquant_prod.toml
fixtures/     small deterministic synthetic tensors + generate.py
docs/         this file, open-questions.md, codecs/*.md evidence sheets,
              m3-research.md (OSCAR/OScaR/KVarN prep, not implemented)
reports/      generated: captured activation shards, error-vs-bytes plots,
              the M2 report (m2_report.{json,md,png})
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

`presets/kivi.toml`, `presets/turboquant_mse.toml`, and
`presets/turboquant_prod.toml` all carry an `[evidence]` block per
SPEC.md §9 with the specific simplifications each one makes relative to
its paper — see the presets themselves and docs/open-questions.md #1.
None are parity-checked yet (`parity_checked = false`); that's M3 scope.

## M2: the metric stack, `qjl_1bit`, and the first real report

`qjl_1bit` (TurboQuant "Prod", arXiv:2504.19874 building on QJL,
arXiv:2406.03482) is the first real `Residual`. It's score-only by
construction — `Residual::apply` is a documented no-op, only
`score_correction` does anything, because a 1-bit sign projection of a
residual doesn't invert to a vector correction, only to an *unbiased
inner-product* correction. Wiring it in required two real design changes
neither M1 codec needed:

- `Pipeline::score` now has to rotate `q` the same way `k` was rotated
  before dotting with the stage-1 (still-rotated-space) reconstruction —
  the first pipeline where `score()` and `decode()` genuinely diverge, not
  just "may equal decode+dot" as SPEC.md §4.2 allows for the reference
  path.
- `score()`'s residual branch has to score the sink/recent window columns
  by a plain dot product against the original `q` and concatenate them
  back in around the "middle" columns' stage-1+correction scores —
  an earlier version of this forgot the window entirely and silently
  returned a truncated (middle-only) score matrix whenever sink/recent > 0
  (i.e. for every real preset). Caught by
  `crates/core/tests/qjl_score_vs_decode.rs`'s
  `score_shape_and_window_columns_are_correct_with_a_real_window` and its
  Python mirror in `tests/test_qjl_residual.py` — both written *because*
  the original tests all used `sink=0, recent=0` to isolate the residual
  behavior and so never exercised this path.

**A real finding worth keeping, not just a caveat:** an early version of
the Rust integration test tried to assert "`qjl_1bit` makes `score()`
beat plain decode+dot," on synthetic iid Gaussian Q/K. It doesn't —
unbiased and lower-error are different properties, and for independent
random data the residual's true contribution to `q·k` already averages to
~0 by symmetry, so there's no systematic bias for the correction to
remove, only its own variance to add. The M2 report on real captured
Qwen2.5-0.5B-Instruct activations (layer 0, head 0, 5 WikiText-2 prompts,
`reports/m2_report.md`) confirms this isn't just a synthetic-data
artifact: `turboquant_prod` costs strictly *more* measured bytes than
`turboquant_mse` at every matched bit setting (the extra QJL bit) while
having *worse* QKᵀ logit error and attention-output error at every one of
them, on this layer/head. Neither result should be over-generalized from
one layer/head/model — but it's exactly the kind of honest, unflattering
number SPEC.md's "no synthetic-only validation" and "measured bytes only"
principles exist to surface, and it argues for using `turboquant_mse` over
`turboquant_prod` as the default TurboQuant preset until a broader sweep
(more layers/heads/models) says otherwise.

The metric stack itself (`python/samhita/metrics.py`) implements SPEC.md
§6.2 items 1-3 (QKᵀ logit error, attention-output error, V reconstruction
error) using real captured Q alongside K/V — K's path uses `Pipeline.score`
(so `qjl_1bit`'s divergence from decode+dot actually shows up in the
report), V's path uses `Pipeline.decode` (reconstruction fidelity), per
the K/V asymmetry in SPEC.md §4.1. `python/samhita/report.py` is the M2
harness proper: real WikiText-2 prompts (`Salesforce/wikitext`, the
`wikitext` loading-script path being broken under current
`datasets`/`huggingface_hub` — see `capture.load_wikitext2_prompts`'s
docstring), multiple rotation seeds for presets that have one, a
percentile bootstrap CI over (seed x prompt) trials, and a versioned JSON
schema (`schema_version: "2.0"`). Bytes still never come from a
Python-side formula: `report.run_trial` calls the real Rust CLI for every
trial, same as the M1 harness.
