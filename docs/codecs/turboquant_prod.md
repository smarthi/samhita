# turboquant_prod

**Source:** arXiv:2504.19874 (TurboQuant, the "Prod" / inner-product
variant) building on arXiv:2406.03482 ("QJL: 1-Bit Quantized JL Transform
for KV Cache Quantization with Zero Overhead", Zandieh/Daliri/Han).

**Stages:** same as `turboquant_mse` on both K and V (Hadamard rotation,
`sign_residual`), plus a `qjl_1bit` residual on K only
(`presets/turboquant_prod.toml`).

## What `qjl_1bit` actually does

Given a stage-1 (`sign_residual`) reconstruction of a rotated key vector,
let `r` be the leftover residual and `r̂ = r/‖r‖` its unit direction. A
fixed random matrix `S` (iid `N(0,1)` entries, shared across the whole
pipeline instance) is used to compute `sign(S·r̂)` — one bit per channel,
stored alongside `‖r‖`. At score time, for a query `q` (rotated the same
way `k` was — see `Pipeline::score`'s handling of this, the first place in
the codebase `score` and `decode` genuinely diverge), the correction
`‖r‖ · sqrt(π/2)/d · (S·q)ᵀ · sign(S·r̂)` is an estimator of `q·r` — proven
unbiased via the sign-correlation identity for jointly Gaussian variables
(Price's theorem / Stein's lemma), and independently confirmed by a
Monte-Carlo check over many resampled `S` matrices
(`crates/core/src/stages/qjl_residual.rs`'s own unit test).

`decode()` (full reconstruction) is unaffected — a 1-bit sign projection
doesn't invert to a vector correction, only to this inner-product
correction. That split is exactly why `Residual::apply` and
`Residual::score_correction` are separate trait methods.

## A real finding, not a caveat to skip past

An earlier version of the Rust integration test tried to assert
"`qjl_1bit`'s `score()` beats plain decode+dot" on synthetic iid Gaussian
Q/K data, averaged over 40 independent trials of 1024 (q,k) pairs each.
**It didn't** — `qjl_1bit`'s mean score error was consistently a bit
*higher* than the baseline's.

This isn't a bug. "Unbiased" and "lower-error" are different properties:
the correction removes the *systematic* bias from ignoring the residual
entirely, but replaces it with the correction's own estimation variance.
For iid random Q/K, the true residual contribution `q·r` already averages
to ~0 by symmetry (independent random vectors have zero expected dot
product) — there's no systematic bias to remove in the first place, so the
correction's variance is pure downside on this kind of data.

Real attention Q/K are not independent — that's the entire point of
attention. Whether the bias/variance tradeoff resolves the other way on
real activations, where a genuine, correctable bias might exist, is
precisely the kind of question SPEC.md §2/§10 says a synthetic tensor is
not allowed to answer ("No synthetic-data-only validation of any codec").
It's an open, empirical question for the M2 harness on real captured
activations — not resolved by this preset shipping, and not something to
assume in either direction until that report exists.

**Status:** `diagnostic`. `parity_checked = false`. See
`presets/turboquant_prod.toml`'s `[evidence]` block for the provenance
caveat on the formula itself (extracted via an automated paper-fetch pass,
not a hand read of the source).
