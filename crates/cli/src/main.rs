//! `samhita` CLI: runs one K- or V-side pipeline (from a preset TOML) over
//! a fixture tensor and prints reconstruction MSE + the byte-accounting
//! report as JSON.
//!
//! This exists mainly as the Rust half of the cross-language parity check
//! described in SPEC.md M1 acceptance criteria ("bit-exact (or documented
//! tolerance) agreement between the Rust reference and a torch reference on
//! fixtures"): `tests/test_bytes_parity_with_rust.py` shells out to this
//! binary and compares its output to the Python reference.
//!
//! Fixture format (`.smhf`, written by `fixtures/generate.py`): u32 rows
//! (LE), u32 cols (LE), then `rows*cols` little-endian f32 values,
//! row-major.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use samhita_core::{compute_bytes, AccountingInput, Pipeline, Preset, Tensor2};

struct Args {
    preset: PathBuf,
    fixture: PathBuf,
    side: String, // "k" or "v"
    seed: u64,
    num_layers: usize,
    num_kv_heads: usize,
    batch: usize,
}

fn parse_args() -> Result<Args> {
    let mut preset = None;
    let mut fixture = None;
    let mut side = "k".to_string();
    let mut seed = 1u64;
    let mut num_layers = 1usize;
    let mut num_kv_heads = 1usize;
    let mut batch = 1usize;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().context("missing value for flag");
        match arg.as_str() {
            "--preset" => preset = Some(PathBuf::from(val()?)),
            "--fixture" => fixture = Some(PathBuf::from(val()?)),
            "--side" => side = val()?,
            "--seed" => seed = val()?.parse()?,
            "--layers" => num_layers = val()?.parse()?,
            "--heads" => num_kv_heads = val()?.parse()?,
            "--batch" => batch = val()?.parse()?,
            other => bail!("unknown flag: {other}"),
        }
    }

    Ok(Args {
        preset: preset.context("--preset is required")?,
        fixture: fixture.context("--fixture is required")?,
        side,
        seed,
        num_layers,
        num_kv_heads,
        batch,
    })
}

fn read_fixture(path: &PathBuf) -> Result<Tensor2> {
    let bytes = fs::read(path).with_context(|| format!("reading fixture {path:?}"))?;
    if bytes.len() < 8 {
        bail!("fixture too small: {}", bytes.len());
    }
    let rows = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let expected = 8 + rows * cols * 4;
    if bytes.len() != expected {
        bail!("fixture size mismatch: expected {expected}, got {}", bytes.len());
    }
    let mut data = Vec::with_capacity(rows * cols);
    for chunk in bytes[8..].chunks_exact(4) {
        data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(Tensor2 { rows, cols, data })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let preset = Preset::from_file(&args.preset)?;
    let cfg = match args.side.as_str() {
        "k" => &preset.k,
        "v" => &preset.v,
        other => bail!("--side must be 'k' or 'v', got {other}"),
    };

    let sample = read_fixture(&args.fixture)?;
    let pipeline = Pipeline::build(cfg, sample.cols, args.seed);

    let packed = pipeline.encode(&sample);
    let recon = pipeline.decode(&packed);
    let mse = sample.mse(&recon);

    let report = compute_bytes(
        &pipeline,
        &sample,
        &AccountingInput {
            seq_len: sample.rows,
            num_layers: args.num_layers,
            num_kv_heads: args.num_kv_heads,
            batch: args.batch,
        },
    );

    let out = serde_json::json!({
        "preset": preset.name,
        "side": args.side,
        "seq_len": sample.rows,
        "head_dim": sample.cols,
        "mse": mse,
        "byte_report": {
            "payload_bytes": report.payload_bytes,
            "window_bytes": report.window_bytes,
            "persistent_state_bytes": report.persistent_state_bytes,
            "persistent_state_amortized_per_token_bytes": report.persistent_state_amortized_per_token_bytes,
            "peak_transient_bytes": report.peak_transient_bytes,
            "total_resident_bytes": report.total_resident_bytes,
            "effective_bits_per_element": report.effective_bits_per_element,
        }
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
