//! `group_rtn`: symmetric round-to-nearest uniform quantization, grouped
//! either per-token (one scale per row) or per-channel (one scale per
//! column). This is KIVI's actual scheme (arXiv:2402.02750): keys are
//! quantized per-channel, values per-token — see `presets/kivi.toml`.

use crate::codes::{Codes, GroupAxis};
use crate::shape::ShapeCtx;
use crate::tensor::Tensor2;
use crate::traits::{Quantizer, Stage};

pub struct GroupRtn {
    pub bits: u8,
    pub axis: GroupAxis,
}

impl GroupRtn {
    fn qmax(&self) -> i32 {
        (1i32 << (self.bits - 1)) - 1
    }
}

impl Stage for GroupRtn {
    fn name(&self) -> &'static str {
        "group_rtn"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        0
    }

    fn per_token_meta_bytes(&self, ctx: &ShapeCtx) -> f64 {
        match self.axis {
            // One f32 scale per token.
            GroupAxis::PerToken => 4.0,
            // One f32 scale per channel, amortized over all tokens in this
            // encode call (a per-channel scale is shared across the whole
            // sequence, so its per-token cost shrinks as the context grows).
            GroupAxis::PerChannel => {
                if ctx.n_tokens == 0 {
                    0.0
                } else {
                    (ctx.head_dim as f64 * 4.0) / ctx.n_tokens as f64
                }
            }
        }
    }
}

impl Quantizer for GroupRtn {
    fn encode(&self, x: &Tensor2) -> Codes {
        let qmax = self.qmax();
        let mut payload = vec![0i32; x.rows * x.cols];

        match self.axis {
            GroupAxis::PerToken => {
                let mut row_scale = Vec::with_capacity(x.rows);
                for r in 0..x.rows {
                    let row = x.row(r);
                    let amax = row.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-8);
                    let scale = amax / qmax as f32;
                    for (c, v) in row.iter().enumerate() {
                        let q = (v / scale).round().clamp(-(qmax as f32) - 1.0, qmax as f32);
                        payload[r * x.cols + c] = q as i32;
                    }
                    row_scale.push(scale);
                }
                Codes::Uniform { bits: self.bits, rows: x.rows, cols: x.cols, axis: self.axis, payload, row_scale }
            }
            GroupAxis::PerChannel => {
                let mut col_scale = vec![0.0f32; x.cols];
                for c in 0..x.cols {
                    let mut amax = 0.0f32;
                    for r in 0..x.rows {
                        amax = amax.max(x.row(r)[c].abs());
                    }
                    col_scale[c] = amax.max(1e-8) / qmax as f32;
                }
                for r in 0..x.rows {
                    for (c, v) in x.row(r).iter().enumerate() {
                        let scale = col_scale[c];
                        let q = (v / scale).round().clamp(-(qmax as f32) - 1.0, qmax as f32);
                        payload[r * x.cols + c] = q as i32;
                    }
                }
                Codes::Uniform {
                    bits: self.bits,
                    rows: x.rows,
                    cols: x.cols,
                    axis: self.axis,
                    payload,
                    row_scale: col_scale,
                }
            }
        }
    }

    fn decode(&self, c: &Codes) -> Tensor2 {
        let Codes::Uniform { rows, cols, axis, payload, row_scale, .. } = c else {
            panic!("group_rtn::decode called with non-Uniform codes");
        };
        let mut out = Tensor2::zeros(*rows, *cols);
        for r in 0..*rows {
            for ci in 0..*cols {
                let scale = match axis {
                    GroupAxis::PerToken => row_scale[r],
                    GroupAxis::PerChannel => row_scale[ci],
                };
                out.row_mut(r)[ci] = payload[r * cols + ci] as f32 * scale;
            }
        }
        out
    }

    fn bits_per_element(&self) -> f32 {
        self.bits as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_token_round_trip_is_close() {
        let q = GroupRtn { bits: 4, axis: GroupAxis::PerToken };
        let x = Tensor2::from_rows(&[vec![1.0, -2.0, 3.0, -0.5], vec![0.1, 0.2, -0.3, 0.05]]);
        let codes = q.encode(&x);
        let recon = q.decode(&codes);
        assert!(x.mse(&recon) < 0.05);
    }

    #[test]
    fn per_channel_round_trip_is_close() {
        let q = GroupRtn { bits: 4, axis: GroupAxis::PerChannel };
        let x = Tensor2::from_rows(&[
            vec![1.0, -20.0, 3.0, -0.5],
            vec![0.9, -18.0, 2.8, -0.4],
            vec![1.1, -19.0, 3.2, -0.6],
        ]);
        let codes = q.encode(&x);
        let recon = q.decode(&codes);
        assert!(x.mse(&recon) < 1.0);
    }
}
