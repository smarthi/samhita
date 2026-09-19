//! `NoResidual`: the M1 default. `qjl_1bit` (TurboQuant Prod) and other
//! score-side residual correctors are M2 scope (SPEC §4.3, §8).

use crate::shape::ShapeCtx;
use crate::tensor::Tensor2;
use crate::traits::{Residual, ResidualCodes, Stage};

pub struct NoResidual;

impl Stage for NoResidual {
    fn name(&self) -> &'static str {
        "none"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        0
    }

    fn per_token_meta_bytes(&self, _ctx: &ShapeCtx) -> f64 {
        0.0
    }
}

impl Residual for NoResidual {
    fn encode(&self, _x: &Tensor2, _recon: &Tensor2) -> ResidualCodes {
        ResidualCodes::default()
    }

    fn apply(&self, _recon: &mut Tensor2, _r: &ResidualCodes) {}

    fn score_correction(&self, q: &Tensor2, _r: &ResidualCodes) -> Tensor2 {
        Tensor2::zeros(q.rows, q.cols)
    }
}
