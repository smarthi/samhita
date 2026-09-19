//! PyO3 bindings for `samhita-core`. Deliberately minimal for M1 (the repo
//! layout in SPEC.md §7 calls for this crate, but the M1 acceptance
//! criteria only require Rust/Python parity on fixtures, which
//! `crates/cli` + `tests/test_bytes_parity_with_rust.py` already cover via
//! a subprocess bridge). This crate exposes the same encode/decode/byte-
//! accounting path so later milestones can drop the subprocess bridge in
//! favor of an in-process binding; wiring it into `python/samhita` (via
//! `maturin develop`) is tracked in `docs/open-questions.md`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use samhita_core::{compute_bytes, AccountingInput, Pipeline, Preset, Tensor2};

/// Encode + decode `data` (row-major, `rows x cols`) through the named side
/// (`"k"` or `"v"`) of `preset_toml`, returning `(mse, effective_bits_per_element)`.
#[pyfunction]
fn encode_decode_mse(
    preset_toml: &str,
    side: &str,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    seed: u64,
) -> PyResult<(f64, f64)> {
    let preset = Preset::from_toml_str(preset_toml).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cfg = match side {
        "k" => &preset.k,
        "v" => &preset.v,
        other => return Err(PyValueError::new_err(format!("side must be 'k' or 'v', got {other}"))),
    };
    if data.len() != rows * cols {
        return Err(PyValueError::new_err("data length != rows*cols"));
    }

    let sample = Tensor2 { rows, cols, data };
    let pipeline = Pipeline::build(cfg, cols, seed);
    let packed = pipeline.encode(&sample);
    let recon = pipeline.decode(&packed);
    let mse = sample.mse(&recon);

    let report = compute_bytes(
        &pipeline,
        &sample,
        &AccountingInput { seq_len: rows, num_layers: 1, num_kv_heads: 1, batch: 1 },
    );

    Ok((mse, report.effective_bits_per_element))
}

#[pymodule]
fn _samhita_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_decode_mse, m)?)?;
    Ok(())
}
