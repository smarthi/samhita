# samhita — संहिता

> *put together, joined into one* — the Sanskrit term for a compiled,
> canonical collection (e.g. a Veda-*samhita*).

A composable KV-cache codec framework: **stages, not codecs**. KIVI,
TurboQuant, OSCAR, OScaR, KVarN and friends are largely different choices
along the same small set of axes (layout → rotation → normalization →
allocation → quantizer → residual). A codec here is a pipeline of stages;
named methods are presets over that pipeline. See [SPEC.md](SPEC.md) for
the full design document and [docs/design.md](docs/design.md) for what M1
actually implements (and where it deviates from the spec's illustrative
sketch).

**Status:** M1 + M2 (traits, byte accounting, real activation capture,
full attention-error metric stack, matched-bytes multi-seed/prompt report
with bootstrap CIs). **Private** — no public disclosure until IP/open-source
clearance is complete (SPEC.md §11).

## Layout

```
crates/core     Rust: stage traits, stage inventory, Pipeline, byte accounting
crates/cli      `samhita` binary: run one preset/side/fixture through the
                Rust reference, print MSE + byte report as JSON
crates/py       PyO3 bindings (scaffolded; see docs/open-questions.md #3)
python/samhita  torch reference, presets loader, diagnostic KV cache,
                activation capture, attention-error metrics, harness/report
presets/        kivi.toml, turboquant_mse.toml, turboquant_prod.toml
fixtures/       small deterministic tensors for bit-exact tests
docs/           design notes, open questions, per-codec evidence sheets,
                m3-research.md (OSCAR/OScaR/KVarN prep, not implemented)
reports/        generated: captured activation shards, error-vs-bytes
                plots, the M2 report (m2_report.{json,md,png})
```

## Quickstart

Rust:

```bash
cargo test
```

Python (from repo root; `pyproject.toml` puts `python/` on `pythonpath`):

```bash
pip install -e ".[capture,harness,dev]"
pytest
```

Run one pipeline over a fixture and see the byte report:

```bash
cargo run -p samhita-cli -- --preset presets/turboquant_mse.toml \
    --fixture fixtures/long_256x64.smhf --side k --layers 32 --heads 8
```

The M1 "one command" demo — capture real post-RoPE activations from
Qwen2.5-0.5B-Instruct (cached after the first run) and sweep both presets'
bit budgets into an error-vs-bytes plot:

```bash
python -m samhita.cli demo
```

writes `reports/error_vs_bytes.json` and `reports/error_vs_bytes.png`.

The M2 report — real WikiText-2 prompts, all three presets, the full
attention-error metric stack, multiple rotation seeds, bootstrap CIs:

```bash
pip install -e ".[capture,harness]"
python -m samhita.cli report
```

writes `reports/m2_report.{json,md,png}`.

## Publishing

This repo includes a GitHub Actions workflow at
`.github/workflows/publish.yml` that publishes:

- `samhita-core` to crates.io
- `samhita` to PyPI

The workflow runs on `v*` tag pushes (and can also be started manually).
Before using it, configure these repository secrets:

- `CARGO_REGISTRY_TOKEN` — crates.io API token
- `PYPI_API_TOKEN` — PyPI API token

## Why "measured bytes only"

Every byte figure anywhere in this repo traces back to one function,
`crates/core/src/bytes.rs`, which runs a real `Pipeline::encode` and reads
the actual packed buffer size rather than computing `bits / 8` — see
docs/design.md for why, and its property tests for what's checked.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Code ported from the author's own
prior work (`dhurandhar`) carries the same license and is attributed at
each port site — see [docs/codecs/turboquant_mse.md](docs/codecs/turboquant_mse.md).
