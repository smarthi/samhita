# kivi

**Source:** arXiv:2402.02750 — "KIVI: A Tuning-Free Asymmetric 2bit
Quantization for KV Cache Compression."

**Stages:** `group_rtn` only, no rotation, no normalization stage
(`presets/kivi.toml`).

- K: `group_rtn`, `axis = per_channel`, 2-bit, symmetric* per-channel scale.
- V: `group_rtn`, `axis = per_token`, 2-bit, symmetric* per-token scale.
- Both sides: `recent = 128` tokens kept at full precision (the paper's
  "residual length" R), `sink = 0` (KIVI has no attention-sink concept).

\* the paper uses *asymmetric* (zero-point) affine quantization; M1's
`group_rtn` is symmetric-only (no zero-point). This is a real deviation,
tracked alongside the windowed-grouping deviation in
docs/open-questions.md #1 — both must be closed before any parity claim.

**Paper claims:** 2-bit KV cache quantization with roughly ≤1 point
perplexity/accuracy degradation vs fp16 across LLaMA/Falcon/Mistral; up to
2.6x peak-memory reduction and higher throughput at long context / large
batch (their Table/Figure results — re-check exact numbers in the paper
before quoting).

**Not claimed by the paper (and not attempted here for M1):** anything
about matched-bytes comparison against rotation-based codecs; the paper's
own group_size=32 fixed-window grouping, which M1 doesn't reproduce yet.

**Status:** `diagnostic` (fake-quant round trip only — no serving-path
integration in M1). `parity_checked = false`.

**License of any reference code used:** none — this is a clean-room
implementation from the paper description, no code ported from KIVI's own
repository.
