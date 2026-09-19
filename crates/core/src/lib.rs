//! samhita-core: stage traits, the M1 stage inventory, the pipeline that
//! composes them, and the single byte-accounting function. See
//! `/docs/design.md` at the repo root for the rationale behind the trait
//! shapes and any deviations from SPEC.md §4.2.

pub mod bytes;
pub mod codes;
pub mod normal;
pub mod pipeline;
pub mod preset;
pub mod rng;
pub mod shape;
pub mod stages;
pub mod tensor;
pub mod traits;

pub use bytes::{compute_bytes, AccountingInput, ByteReport};
pub use codes::{Codes, GroupAxis, SideInfo};
pub use pipeline::{Packed, Pipeline, PipelineConfig};
pub use preset::Preset;
pub use shape::{CalibData, ShapeCtx};
pub use tensor::Tensor2;
pub use traits::{Quantizer, Residual, ResidualCodes, Stage, Transform};
