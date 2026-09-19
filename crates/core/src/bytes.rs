//! The single byte-accounting function, per SPEC §5.3. Every memory figure
//! anywhere else in the project (harness, reports, docs) must be traced
//! back to this module — never `bits / 8` computed ad hoc elsewhere.
//!
//! The payload/metadata component is *not* re-derived from each stage's
//! declared `per_token_meta_bytes()` in isolation (that would let a wrong
//! declaration silently drift from what the serving path actually
//! allocates). Instead this function runs a real `Pipeline::encode` on the
//! given sample and reads the resulting `Codes::packed_size_bytes()` — the
//! literal bytes a serving-path buffer would need — so the accounting is
//! bound by construction to the same code path that would ship the data.
//! Persistent per-(layer,head) state (rotation matrices, codebooks) is
//! *not* observable from a single `encode` call, so that component still
//! comes from each stage's declared `state_bytes()`.

use crate::pipeline::Pipeline;
use crate::tensor::Tensor2;

#[derive(Debug, Clone, Copy)]
pub struct AccountingInput {
    pub seq_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub batch: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteReport {
    /// Packed quantizer payload + its per-group scale/zero-point, summed
    /// across all layers/heads/batch.
    pub payload_bytes: f64,
    /// Sink + recent tokens stored at their true high-precision dtype.
    pub window_bytes: f64,
    /// Rotation matrices / codebooks: counted once per (layer, head, batch).
    pub persistent_state_bytes: f64,
    /// `persistent_state_bytes / seq_len` — the amortized per-token view
    /// SPEC §5.3 asks be reported alongside the raw total.
    pub persistent_state_amortized_per_token_bytes: f64,
    /// Largest transient workspace buffer allocated during one encode call
    /// (not resident, reported separately, never added into totals that
    /// claim to be steady-state memory).
    pub peak_transient_bytes: f64,
    /// `payload_bytes + window_bytes + persistent_state_bytes`. Excludes
    /// transient workspace, which is not steady-state resident memory.
    pub total_resident_bytes: f64,
    pub effective_bits_per_element: f64,
}

/// Compute the byte report for `pipeline` at the shape described by
/// `input`, using `sample` (shape `[seq_len, head_dim]`) as the concrete
/// tensor to actually run through `Pipeline::encode`. The reported bytes
/// depend only on shapes/config, not on `sample`'s values, but running a
/// genuine encode keeps this function honest about what the serving path
/// would allocate (see module docs).
pub fn compute_bytes(pipeline: &Pipeline, sample: &Tensor2, input: &AccountingInput) -> ByteReport {
    assert_eq!(sample.rows, input.seq_len, "sample must have seq_len rows");
    assert_eq!(sample.cols, pipeline.head_dim, "sample must have head_dim cols");

    let packed = pipeline.encode(sample);
    let per_instance_payload = packed.codes.packed_size_bytes()
        + packed.residual_codes.packed_size_bytes()
        + packed.side.per_token_bytes() * packed.window_split.middle as f64;
    let per_instance_window = (packed.window_split.sink + packed.window_split.recent) as f64
        * pipeline.head_dim as f64
        * pipeline.window.high_prec_dtype_bytes as f64;
    let per_instance_state = pipeline.persistent_state_bytes(input.seq_len) as f64;
    let per_instance_transient = pipeline.peak_transient_bytes(input.seq_len);

    let instances = (input.num_layers * input.num_kv_heads * input.batch) as f64;

    let payload_bytes = per_instance_payload * instances;
    let window_bytes = per_instance_window * instances;
    let persistent_state_bytes = per_instance_state * instances;
    let peak_transient_bytes = per_instance_transient * instances;
    let total_resident_bytes = payload_bytes + window_bytes + persistent_state_bytes;

    let seq_len_f = input.seq_len.max(1) as f64;
    let persistent_state_amortized_per_token_bytes = persistent_state_bytes / seq_len_f;

    let total_elements = instances * input.seq_len as f64 * pipeline.head_dim as f64;
    let effective_bits_per_element =
        if total_elements > 0.0 { total_resident_bytes * 8.0 / total_elements } else { 0.0 };

    ByteReport {
        payload_bytes,
        window_bytes,
        persistent_state_bytes,
        persistent_state_amortized_per_token_bytes,
        peak_transient_bytes,
        total_resident_bytes,
        effective_bits_per_element,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::GroupAxis;
    use crate::pipeline::{NormalizationKind, PipelineConfig, QuantizerConfig, RotationKind, WindowConfig};
    use proptest::prelude::*;

    fn no_window_config(bits: u8, axis: GroupAxis) -> PipelineConfig {
        PipelineConfig {
            rotation: RotationKind::None,
            scale: NormalizationKind::None,
            residual: crate::pipeline::ResidualKind::None,
            quantizer: match axis {
                GroupAxis::PerToken => QuantizerConfig::GroupRtn { bits, axis: crate::pipeline::GroupAxisCfg::PerToken },
                GroupAxis::PerChannel => {
                    QuantizerConfig::GroupRtn { bits, axis: crate::pipeline::GroupAxisCfg::PerChannel }
                }
            },
            window: WindowConfig { sink: 0, recent: 0, dtype_bytes: 2 },
        }
    }

    fn deterministic_sample(rows: usize, cols: usize) -> Tensor2 {
        let mut t = Tensor2::zeros(rows, cols);
        for (i, v) in t.data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin()) * 3.0;
        }
        t
    }

    proptest! {
        /// Multiplying instance count must scale payload bytes exactly
        /// (accounting is per-instance-bytes * instances, no fixed cost
        /// smuggled in anywhere).
        #[test]
        fn scales_linearly_with_instance_count(
            rows in 4usize..40,
            cols in 4usize..32,
            bits in 2u8..=8,
            layers in 1usize..6,
            heads in 1usize..6,
            batch in 1usize..3,
        ) {
            let cfg = no_window_config(bits, GroupAxis::PerToken);
            let pipeline = Pipeline::build(&cfg, cols, 1);
            let sample = deterministic_sample(rows, cols);

            let one = compute_bytes(&pipeline, &sample, &AccountingInput {
                seq_len: rows, num_layers: 1, num_kv_heads: 1, batch: 1,
            });
            let many = compute_bytes(&pipeline, &sample, &AccountingInput {
                seq_len: rows, num_layers: layers, num_kv_heads: heads, batch,
            });

            let instances = (layers * heads * batch) as f64;
            prop_assert!((many.payload_bytes - one.payload_bytes * instances).abs() < 1e-6);
            prop_assert!((many.window_bytes - one.window_bytes * instances).abs() < 1e-6);
            prop_assert!((many.persistent_state_bytes - one.persistent_state_bytes * instances).abs() < 1e-6);
        }

        /// SPEC §5.3: "Property tests must show accounting matches actual
        /// allocated buffer sizes in the serving path." Compare the
        /// accounting function's payload prediction directly against the
        /// real `Codes::packed_size_bytes()` for the same encode call.
        #[test]
        fn payload_matches_real_packed_codes(
            rows in 1usize..40,
            cols in 1usize..32,
            bits in 2u8..=8,
        ) {
            let cfg = no_window_config(bits, GroupAxis::PerToken);
            let pipeline = Pipeline::build(&cfg, cols, 1);
            let sample = deterministic_sample(rows, cols);
            let packed = pipeline.encode(&sample);

            let report = compute_bytes(&pipeline, &sample, &AccountingInput {
                seq_len: rows, num_layers: 1, num_kv_heads: 1, batch: 1,
            });

            prop_assert!((report.payload_bytes - packed.codes.packed_size_bytes()).abs() < 1e-6);
        }

        /// Fewer bits must never cost more measured bytes (SPEC §3
        /// principle 2: measured bytes only, and they must behave sanely).
        #[test]
        fn fewer_bits_never_costs_more(
            rows in 4usize..40,
            cols in 4usize..32,
            bits_a in 2u8..=7,
        ) {
            let bits_b = bits_a + 1;
            let cfg_a = no_window_config(bits_a, GroupAxis::PerToken);
            let cfg_b = no_window_config(bits_b, GroupAxis::PerToken);
            let sample = deterministic_sample(rows, cols);

            let pa = Pipeline::build(&cfg_a, cols, 1);
            let pb = Pipeline::build(&cfg_b, cols, 1);
            let input = AccountingInput { seq_len: rows, num_layers: 2, num_kv_heads: 2, batch: 1 };

            let ra = compute_bytes(&pa, &sample, &input);
            let rb = compute_bytes(&pb, &sample, &input);
            prop_assert!(ra.payload_bytes <= rb.payload_bytes + 1e-6);
        }

        /// Window bytes must scale linearly in (sink+recent) * dtype_bytes,
        /// independent of the quantizer used on the middle tokens.
        #[test]
        fn window_bytes_scale_with_dtype_and_count(
            sink in 0usize..8,
            recent in 0usize..8,
            dtype_bytes in 1usize..=4,
            cols in 4usize..16,
        ) {
            let seq_len = sink + recent + 16;
            let cfg = PipelineConfig {
                rotation: RotationKind::None,
                scale: NormalizationKind::None,
                residual: crate::pipeline::ResidualKind::None,
                quantizer: QuantizerConfig::GroupRtn { bits: 4, axis: crate::pipeline::GroupAxisCfg::PerToken },
                window: WindowConfig { sink, recent, dtype_bytes },
            };
            let pipeline = Pipeline::build(&cfg, cols, 1);
            let sample = deterministic_sample(seq_len, cols);
            let report = compute_bytes(&pipeline, &sample, &AccountingInput {
                seq_len, num_layers: 1, num_kv_heads: 1, batch: 1,
            });
            let expected = (sink + recent) as f64 * cols as f64 * dtype_bytes as f64;
            prop_assert!((report.window_bytes - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn total_resident_is_sum_of_components() {
        let cfg = no_window_config(4, GroupAxis::PerToken);
        let pipeline = Pipeline::build(&cfg, 16, 1);
        let sample = deterministic_sample(20, 16);
        let report = compute_bytes(&pipeline, &sample, &AccountingInput {
            seq_len: 20,
            num_layers: 3,
            num_kv_heads: 2,
            batch: 1,
        });
        let sum = report.payload_bytes + report.window_bytes + report.persistent_state_bytes;
        assert!((sum - report.total_resident_bytes).abs() < 1e-6);
    }
}
