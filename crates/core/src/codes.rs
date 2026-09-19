//! Quantizer outputs and the per-transform inversion state.
//!
//! `Codes::packed_size_bytes` is the actual serving-path allocation a real
//! bit-packed buffer would need. The byte-accounting property tests (see
//! `bytes.rs`) assert the accounting function's prediction matches this
//! number exactly, per SPEC §5.3.

/// State a `Transform` needs at `inverse` time. Only the fields a given
/// transform populates are `Some`; e.g. `hadamard` needs neither.
#[derive(Clone, Debug, Default)]
pub struct SideInfo {
    /// Per-row (per-token) scale used by `per_token_scale`.
    pub row_scale: Option<Vec<f32>>,
}

impl SideInfo {
    /// Metadata bytes this side info would cost per token in the serving
    /// path (f32 scale per row).
    pub fn per_token_bytes(&self) -> f64 {
        let mut b = 0.0;
        if self.row_scale.is_some() {
            b += 4.0;
        }
        b
    }
}

/// Which axis a quantizer's per-group scale is indexed by. `PerToken`
/// scale[i] applies to row i (all channels of one token); `PerChannel`
/// scale[j] applies to column j (one channel across all tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupAxis {
    PerToken,
    PerChannel,
}

#[derive(Clone, Debug)]
pub enum Codes {
    /// `uniform_rtn` / `group_rtn`: symmetric per-group linear quantization.
    Uniform {
        bits: u8,
        rows: usize,
        cols: usize,
        axis: GroupAxis,
        /// Row-major signed integer codes in `[-(2^(bits-1)), 2^(bits-1)-1]`.
        payload: Vec<i32>,
        /// One scale per group (see `axis`): `x ≈ payload * group_scale[g]`.
        row_scale: Vec<f32>,
    },
    /// `sign_residual` (TurboQuant-style): 1 sign bit per element plus an
    /// n-bit residual, both scaled per row. See `stages::sign_residual`.
    SignResidual {
        rows: usize,
        cols: usize,
        /// `ceil(cols/8)` sign bytes per row, packed LSB-first.
        sign_packed: Vec<u8>,
        /// Per-row L2 norm of the (rotated) vector, used to rescale the
        /// sign reconstruction.
        row_norm: Vec<f32>,
        residual_bits: u8,
        residual_payload: Vec<i32>,
        residual_scale: Vec<f32>,
    },
    /// `lloyd_max`: fixed non-uniform codebook, indices only (the codebook
    /// itself is persistent per-pipeline state, not per-encode payload).
    Codebook {
        bits: u8,
        rows: usize,
        cols: usize,
        payload: Vec<i32>,
        /// One scale per row (maps the unit-variance codebook onto the
        /// row's actual scale).
        row_scale: Vec<f32>,
    },
}

impl Codes {
    pub fn rows(&self) -> usize {
        match self {
            Codes::Uniform { rows, .. } => *rows,
            Codes::SignResidual { rows, .. } => *rows,
            Codes::Codebook { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Codes::Uniform { cols, .. } => *cols,
            Codes::SignResidual { cols, .. } => *cols,
            Codes::Codebook { cols, .. } => *cols,
        }
    }

    /// Real, bit-exact serving-path byte count for this codes buffer:
    /// payload (bit-packed) + per-row scale/zero-point metadata. This is
    /// the ground truth the accounting function in `bytes.rs` is checked
    /// against.
    pub fn packed_size_bytes(&self) -> f64 {
        match self {
            Codes::Uniform { bits, rows, cols, row_scale, .. } => {
                let payload_bits = (*bits as u64) * (*rows as u64) * (*cols as u64);
                let payload_bytes = payload_bits.div_ceil(8) as f64;
                let meta_bytes = row_scale.len() as f64 * 4.0; // f32 scale/row
                payload_bytes + meta_bytes
            }
            Codes::SignResidual { rows, cols, residual_bits, row_norm, .. } => {
                let sign_bytes = (*rows as u64) * (*cols as u64).div_ceil(8);
                let residual_bits_total = (*residual_bits as u64) * (*rows as u64) * (*cols as u64);
                let residual_bytes = residual_bits_total.div_ceil(8) as f64;
                // per-row: L2 norm (f32) + residual scale (f32)
                let meta_bytes = row_norm.len() as f64 * (4.0 + 4.0);
                sign_bytes as f64 + residual_bytes + meta_bytes
            }
            Codes::Codebook { bits, rows, cols, row_scale, .. } => {
                let payload_bits = (*bits as u64) * (*rows as u64) * (*cols as u64);
                let payload_bytes = payload_bits.div_ceil(8) as f64;
                let meta_bytes = row_scale.len() as f64 * 4.0;
                payload_bytes + meta_bytes
            }
        }
    }
}
