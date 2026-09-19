# turboquant_mse

**Source:** arXiv:2504.19874 — "TurboQuant: Online Vector Quantization via
Randomized Rotations."

**Stages:** `hadamard` rotation then `sign_residual` quantization
(`presets/turboquant_mse.toml`), on both K and V.

- Rotation: randomized Hadamard (`diag(signs) @ H`), deterministic signs
  from a seeded SplitMix64 stream, fast Walsh-Hadamard butterfly
  (`crates/core/src/stages/hadamard.rs`).
- Quantizer: `sign_residual`, `bits = 4` (1 sign bit + 4-bit residual per
  channel, both scaled per row) — matches the residual_bits=4 default in
  the author's prior reference, `dhurandhar/src/dhurandhar/turboquant.py`.
- Both sides: `sink = 4`, `recent = 128` tokens kept at full precision.

This preset is a direct port of `dhurandhar`'s existing `TurboQuantCodec`
reference (itself citing the same arXiv ID) into the stage/trait framework
here, per SPEC.md M1 instruction to extract from that repo rather than
rewrite from scratch — the Hadamard-rotation math and the sign+residual
scheme are unchanged; only the O(d^2) reference matmul rotation was
replaced with an O(d log d) fast Walsh-Hadamard butterfly, and the
sign/residual packing was made bit-exact for real byte accounting instead
of dhurandhar's illustrative `_pack_bits`/`_unpack_bits` helpers.

**Paper claims:** ~3.5 effective bits/channel (1 sign bit + n-bit residual
after randomized Hadamard rotation) with <1% downstream perplexity
degradation on KV cache compression at that budget (re-check the exact
number and setting in the paper before quoting).

**Not claimed:** no kernel latency comparisons (SPEC.md non-goal); the
paper's own approximate effective-bits formula is not used anywhere here —
SPEC.md §5.3's measured-bytes accounting is used instead, and the two can
legitimately differ (see the M1 harness output, where measured
`effective_bits_per_element` includes the per-row norm/scale metadata the
paper's back-of-envelope formula approximates).

**Status:** `diagnostic`. `parity_checked = false` (no official TurboQuant
repo has been diffed against this implementation yet — it's ported from
the author's own prior work, not from the paper's official code release).

**License of reference code used:** Apache-2.0
(`dhurandhar/src/dhurandhar/turboquant.py`, the author's own repository).
