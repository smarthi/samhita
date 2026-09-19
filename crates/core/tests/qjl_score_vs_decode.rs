//! Integration test for `qjl_1bit` residual wired into `Pipeline`.
//!
//! This deliberately does NOT assert "`score()` with `qjl_1bit` beats plain
//! decode+dot" — an earlier version of this test tried exactly that on iid
//! random Gaussian Q/K data and found the *opposite*: qjl_1bit's score
//! error was consistently a bit higher, even averaged over 40 independent
//! trials of 1024 (q,k) pairs each. That's not a bug; it's the honest
//! result of a real property of the estimator, worth recording:
//!
//! `qjl_1bit`'s correction is unbiased (`E[correction] = q . r` exactly,
//! over the randomness of the projection matrix S — see
//! `stages::qjl_residual`'s own unit test for that claim, tested the right
//! way: fixed (q, r), many resampled S). Unbiased is not the same as
//! lower-error. For iid random Q/K, `q . r` (the true residual
//! contribution the correction is trying to recover) already averages to
//! ~0 by symmetry — there's no systematic bias in ignoring the residual
//! entirely, so adding the correction's own estimation variance can only
//! make things worse on this kind of unstructured data. Real attention
//! Q/K are *not* independent (that's the entire point of attention), so
//! this bias/variance tradeoff plausibly resolves the other way on real
//! activations — but that is an empirical question for the M2 harness on
//! captured activations (SPEC.md §6, and SPEC.md §2/§10's explicit ban on
//! synthetic-data-only validation of any codec) to answer, not something a
//! synthetic unit-test fixture can honestly claim either way. See
//! docs/codecs/turboquant_prod.md.
//!
//! What *is* safe to assert unconditionally, and is what this test checks:
//! `decode()` (reconstruction fidelity) is bit-identical with and without
//! `qjl_1bit`, on every trial — it's score-only by construction (a 1-bit
//! sign projection doesn't invert to a vector correction).

use samhita_core::pipeline::{
    NormalizationKind, PipelineConfig, QuantizerConfig, ResidualKind, RotationKind, WindowConfig,
};
use samhita_core::rng::SplitMix64;
use samhita_core::{Pipeline, Tensor2};

fn random_tensor(rows: usize, cols: usize, seed: u64) -> Tensor2 {
    let mut rng = SplitMix64::new(seed);
    let data = (0..rows * cols).map(|_| rng.next_gaussian() as f32).collect();
    Tensor2 { rows, cols, data }
}

#[test]
fn qjl_residual_leaves_decode_unchanged_and_score_well_defined() {
    let head_dim = 32;
    let no_window = WindowConfig { sink: 0, recent: 0, dtype_bytes: 2 };

    let baseline_cfg = PipelineConfig {
        rotation: RotationKind::Hadamard,
        scale: NormalizationKind::None,
        quantizer: QuantizerConfig::SignResidual { bits: 3 },
        residual: ResidualKind::None,
        window: no_window,
    };
    let qjl_cfg = PipelineConfig {
        rotation: RotationKind::Hadamard,
        scale: NormalizationKind::None,
        quantizer: QuantizerConfig::SignResidual { bits: 3 },
        residual: ResidualKind::Qjl1Bit,
        window: no_window,
    };

    for trial in 0..10u64 {
        let seed = 1000 + trial;
        let baseline = Pipeline::build(&baseline_cfg, head_dim, seed);
        let qjl = Pipeline::build(&qjl_cfg, head_dim, seed);

        let k = random_tensor(64, head_dim, 2_000_000 + trial);
        let q = random_tensor(16, head_dim, 3_000_000 + trial);

        let baseline_packed = baseline.encode(&k);
        let qjl_packed = qjl.encode(&k);

        // decode() must be identical: qjl_1bit is score-only by design.
        let baseline_recon = baseline.decode(&baseline_packed);
        let qjl_recon = qjl.decode(&qjl_packed);
        let recon_diff: f32 = baseline_recon
            .data
            .iter()
            .zip(qjl_recon.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(recon_diff < 1e-4, "trial {trial}: decode() should be identical with/without qjl_1bit, diff={recon_diff}");

        // score() must run and produce the right shape either way.
        let baseline_score = baseline.score(&q, &baseline_packed);
        let qjl_score = qjl.score(&q, &qjl_packed);
        assert_eq!(baseline_score.rows, q.rows);
        assert_eq!(baseline_score.cols, k.rows);
        assert_eq!(qjl_score.rows, q.rows);
        assert_eq!(qjl_score.cols, k.rows);

        // score() must actually change when qjl_1bit is added (i.e. the
        // residual really is being read and applied, not silently dropped).
        let score_diff: f32 = baseline_score
            .data
            .iter()
            .zip(qjl_score.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(score_diff > 1e-6, "trial {trial}: qjl_1bit should change score() vs. no residual");
    }
}

#[test]
fn score_shape_and_window_columns_are_correct_with_a_real_window() {
    // Regression test: an earlier version of `score()`'s residual branch
    // only accounted for the "middle" (non-window) tokens, silently
    // returning a truncated score matrix whenever sink/recent > 0 (i.e.
    // for every real preset — `turboquant_prod.toml` has sink=4,
    // recent=128). Caught by actually exercising a real window here
    // instead of the sink=0/recent=0 setup the other tests in this file
    // use to isolate the residual behavior.
    let head_dim = 16;
    let cfg = PipelineConfig {
        rotation: RotationKind::Hadamard,
        scale: NormalizationKind::None,
        quantizer: QuantizerConfig::SignResidual { bits: 3 },
        residual: ResidualKind::Qjl1Bit,
        window: WindowConfig { sink: 2, recent: 3, dtype_bytes: 2 },
    };
    let pipeline = Pipeline::build(&cfg, head_dim, 7);

    let seq_len = 20;
    let k = random_tensor(seq_len, head_dim, 11);
    let q = random_tensor(4, head_dim, 12);

    let packed = pipeline.encode(&k);
    let score = pipeline.score(&q, &packed);
    assert_eq!(score.rows, q.rows);
    assert_eq!(score.cols, seq_len, "score() must cover the full sequence, not just the middle");

    // Sink/recent columns are exact: score()'s columns there must match a
    // plain dot product against the original (unquantized) k rows.
    for (col, row) in (0..2).chain(seq_len - 3..seq_len).enumerate() {
        let _ = col;
        for qi in 0..q.rows {
            let expected: f32 = q.row(qi).iter().zip(k.row(row).iter()).map(|(a, b)| a * b).sum();
            let actual = score.row(qi)[row];
            assert!((expected - actual).abs() < 1e-3, "row={row} qi={qi} expected={expected} actual={actual}");
        }
    }
}
