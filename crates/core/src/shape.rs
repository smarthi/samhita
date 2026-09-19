use crate::tensor::Tensor2;

/// Shape a stage needs to know to report its byte cost. `n_tokens` is the
/// number of tokens actually passing through the stage (i.e. excluding the
/// sink/recent high-precision window, which the layout stage carves off
/// before the rotation/normalization/quantizer chain ever sees them).
#[derive(Clone, Copy, Debug)]
pub struct ShapeCtx {
    pub n_tokens: usize,
    pub head_dim: usize,
}

/// Calibration data a stage's `fit` may consult (e.g. `eigenbasis` in later
/// milestones). M1 stages are all calibration-free, so this is currently
/// only a placeholder threaded through the trait for forward compatibility.
#[derive(Clone, Debug, Default)]
pub struct CalibData {
    pub samples: Vec<Tensor2>,
}
