use anyhow::Result;

use crate::codes::{Codes, SideInfo};
use crate::shape::{CalibData, ShapeCtx};
use crate::tensor::Tensor2;

/// Common surface every pipeline stage implements, per SPEC §4.2. A stage
/// declares its own byte cost; the accounting function in `bytes.rs` never
/// hardcodes per-stage knowledge, it only sums these declarations.
pub trait Stage {
    fn name(&self) -> &'static str;

    /// Optional calibration from captured activations. Must be
    /// deterministic given `seed`. M1 stages are all calibration-free and
    /// use the default no-op.
    fn fit(&mut self, _calib: &CalibData, _seed: u64) -> Result<()> {
        Ok(())
    }

    /// Bytes of persistent state this stage needs (rotation matrices,
    /// codebooks): counted once per (layer, head), not per token.
    fn state_bytes(&self, ctx: &ShapeCtx) -> usize;

    /// Bytes of per-token metadata this stage emits at encode time
    /// (e.g. a per-token scale). Fractional because some stages amortize
    /// (e.g. a shared codebook has no per-token cost at all).
    fn per_token_meta_bytes(&self, ctx: &ShapeCtx) -> f64;
}

/// Geometry-changing stages: rotation and normalization. Must be exactly
/// invertible, or must declare their reconstruction error via the
/// implementor's own docs/tests (SPEC §4.1).
pub trait Transform: Stage {
    fn forward(&self, x: &mut Tensor2, side: &mut SideInfo);
    fn inverse(&self, x: &mut Tensor2, side: &SideInfo);
}

/// The rounder. `score` and `decode` are kept as separate concerns at the
/// `Pipeline` level (not here) so K (score fidelity) and V (reconstruction
/// fidelity) can diverge, per SPEC §4.2 design note.
pub trait Quantizer: Stage {
    fn encode(&self, x: &Tensor2) -> Codes;
    fn decode(&self, c: &Codes) -> Tensor2;

    /// Payload-only bits per element; metadata is counted separately via
    /// `Stage::per_token_meta_bytes` / `Codes::packed_size_bytes`.
    fn bits_per_element(&self) -> f32;
}

/// Score/reconstruction correction on top of a quantizer's output.
/// M1 ships only `NoResidual`; `qjl_1bit` (TurboQuant Prod) lands in M2.
pub trait Residual: Stage {
    fn encode(&self, x: &Tensor2, recon: &Tensor2) -> ResidualCodes;
    fn apply(&self, recon: &mut Tensor2, r: &ResidualCodes);
    fn score_correction(&self, q: &Tensor2, r: &ResidualCodes) -> Tensor2;
}

#[derive(Clone, Debug, Default)]
pub struct ResidualCodes {
    pub payload: Vec<f32>,
}
