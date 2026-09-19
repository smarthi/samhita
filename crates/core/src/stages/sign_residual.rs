//! `sign_residual`: TurboQuant-style 1-bit sign + n-bit residual quantizer.
//! Ported from `dhurandhar/src/dhurandhar/turboquant.py::TurboQuantCodec`
//! (arXiv:2504.19874), which itself cites the paper's Stage 2. Reference
//! implementation, `path = "diagnostic"` unless wired through the packed
//! serving representation (`Codes::SignResidual` already is bit-packed,
//! see `packed_size_bytes`).
//!
//! At nominal `residual_bits` this is *not* an apples-to-apples comparison
//! against a plain `residual_bits`-bit uniform quantizer — SPEC §3 principle
//! 5 flags exactly this trap: effective bits are `1 + residual_bits` (plus
//! the per-row norm/scale overhead), not `residual_bits`.

use crate::codes::Codes;
use crate::shape::ShapeCtx;
use crate::tensor::Tensor2;
use crate::traits::{Quantizer, Stage};

pub struct SignResidual {
    pub residual_bits: u8,
}

impl SignResidual {
    fn qmax(&self) -> i32 {
        (1i32 << (self.residual_bits - 1)) - 1
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn unpack_bits(packed: &[u8], n: usize) -> Vec<bool> {
    (0..n).map(|i| (packed[i / 8] >> (i % 8)) & 1 == 1).collect()
}

impl Stage for SignResidual {
    fn name(&self) -> &'static str {
        "sign_residual"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        0
    }

    fn per_token_meta_bytes(&self, ctx: &ShapeCtx) -> f64 {
        // per row: L2 norm (f32) + residual scale (f32)
        let _ = ctx;
        8.0
    }
}

impl Quantizer for SignResidual {
    fn encode(&self, x: &Tensor2) -> Codes {
        let d = x.cols as f32;
        let sqrt_d = d.sqrt();
        let qmax = self.qmax();

        let mut sign_bits = Vec::with_capacity(x.rows * x.cols);
        let mut row_norm = Vec::with_capacity(x.rows);
        let mut residual_payload = vec![0i32; x.rows * x.cols];
        let mut residual_scale = Vec::with_capacity(x.rows);

        for r in 0..x.rows {
            let row = x.row(r);
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            let mut residual = vec![0.0f32; x.cols];
            for (c, &v) in row.iter().enumerate() {
                let bit = v >= 0.0;
                sign_bits.push(bit);
                let recon_sign = if bit { 1.0 } else { -1.0 } * (norm / sqrt_d);
                residual[c] = v - recon_sign;
            }
            let amax = residual.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-8);
            let scale = amax / qmax as f32;
            for (c, &rv) in residual.iter().enumerate() {
                let q = (rv / scale).round().clamp(-(qmax as f32) - 1.0, qmax as f32);
                residual_payload[r * x.cols + c] = q as i32;
            }
            row_norm.push(norm);
            residual_scale.push(scale);
        }

        Codes::SignResidual {
            rows: x.rows,
            cols: x.cols,
            sign_packed: pack_bits(&sign_bits),
            row_norm,
            residual_bits: self.residual_bits,
            residual_payload,
            residual_scale,
        }
    }

    fn decode(&self, c: &Codes) -> Tensor2 {
        let Codes::SignResidual { rows, cols, sign_packed, row_norm, residual_payload, residual_scale, .. } = c
        else {
            panic!("sign_residual::decode called with non-SignResidual codes");
        };
        let sqrt_d = (*cols as f32).sqrt();
        let sign_bits = unpack_bits(sign_packed, rows * cols);
        let mut out = Tensor2::zeros(*rows, *cols);
        for r in 0..*rows {
            let norm = row_norm[r];
            let scale = residual_scale[r];
            for ci in 0..*cols {
                let bit = sign_bits[r * cols + ci];
                let recon_sign = if bit { 1.0 } else { -1.0 } * (norm / sqrt_d);
                let residual = residual_payload[r * cols + ci] as f32 * scale;
                out.row_mut(r)[ci] = recon_sign + residual;
            }
        }
        out
    }

    fn bits_per_element(&self) -> f32 {
        1.0 + self.residual_bits as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_close() {
        let q = SignResidual { residual_bits: 4 };
        let x = Tensor2::from_rows(&[vec![1.0, -2.0, 3.0, -0.5, 0.25, -4.0, 1.5, -2.5]]);
        let codes = q.encode(&x);
        let recon = q.decode(&codes);
        assert!(x.mse(&recon) < 0.2, "mse={}", x.mse(&recon));
    }

    #[test]
    fn packed_bytes_are_bit_exact() {
        let q = SignResidual { residual_bits: 4 };
        let x = Tensor2::from_rows(&[vec![1.0; 16]]);
        let codes = q.encode(&x);
        // 16 sign bits -> 2 bytes; 16 * 4-bit residual -> 8 bytes; +8 bytes meta (norm+scale)
        assert_eq!(codes.packed_size_bytes(), 2.0 + 8.0 + 8.0);
    }
}
