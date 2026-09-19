//! `per_token_scale`: normalize each token (row) by its max-abs value.
//!
//! Kept separate from any quantizer's own per-row scaling so hybrid configs
//! can ask the question in SPEC §8 M4 headline experiment H1: does adding
//! omni-token scaling on top of `sign_residual` help at matched bytes?

use crate::codes::SideInfo;
use crate::shape::ShapeCtx;
use crate::tensor::Tensor2;
use crate::traits::{Stage, Transform};

pub struct PerTokenScale {
    pub eps: f32,
}

impl Default for PerTokenScale {
    fn default() -> Self {
        PerTokenScale { eps: 1e-8 }
    }
}

impl Stage for PerTokenScale {
    fn name(&self) -> &'static str {
        "per_token_scale"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        0
    }

    fn per_token_meta_bytes(&self, _ctx: &ShapeCtx) -> f64 {
        4.0 // one f32 scale per token
    }
}

impl Transform for PerTokenScale {
    fn forward(&self, x: &mut Tensor2, side: &mut SideInfo) {
        let mut scales = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row = x.row_mut(r);
            let max_abs = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(self.eps);
            for v in row.iter_mut() {
                *v /= max_abs;
            }
            scales.push(max_abs);
        }
        side.row_scale = Some(scales);
    }

    fn inverse(&self, x: &mut Tensor2, side: &SideInfo) {
        let scales = side.row_scale.as_ref().expect("per_token_scale: missing side info");
        for r in 0..x.rows {
            let s = scales[r];
            for v in x.row_mut(r).iter_mut() {
                *v *= s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let stage = PerTokenScale::default();
        let orig = Tensor2::from_rows(&[vec![1.0, -4.0, 2.0], vec![0.0, 0.0, 0.0]]);
        let mut x = orig.clone();
        let mut side = SideInfo::default();
        stage.forward(&mut x, &mut side);
        assert!((x.row(0)[1] - (-1.0)).abs() < 1e-6); // max-abs channel maps to -1
        stage.inverse(&mut x, &side);
        for (a, b) in x.data.iter().zip(orig.data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
