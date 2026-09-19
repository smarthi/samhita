//! `Pipeline`: one K- or V-side composition of layout -> rotation ->
//! normalization -> quantizer -> residual (SPEC §4.1). Built from a
//! `PipelineConfig` (in turn loaded from a preset TOML file, SPEC §4.3).
//!
//! M1 scope was: `allocation` always uniform (no `mixed`/`water_filling`),
//! `residual` always absent. M2 adds `qjl_1bit` (TurboQuant Prod,
//! arXiv:2504.19874) as the first real `Residual` — see
//! `stages::qjl_residual` and the `score()` method below for why it needs
//! `Pipeline` to rotate `q` before scoring, which nothing in M1 needed to
//! do.

use serde::Deserialize;

use crate::codes::{Codes, GroupAxis, SideInfo};
use crate::shape::ShapeCtx;
use crate::stages::{
    GroupRtn, Hadamard, LloydMax, PerTokenScale, QjlResidual, RandomOrthogonal, SignResidual,
    WindowPolicy, WindowSplit,
};
use crate::tensor::Tensor2;
use crate::traits::{Quantizer, Residual, ResidualCodes, Transform};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RotationKind {
    None,
    Hadamard,
    RandomOrthogonal,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationKind {
    None,
    PerToken,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GroupAxisCfg {
    PerToken,
    PerChannel,
}

impl From<GroupAxisCfg> for GroupAxis {
    fn from(g: GroupAxisCfg) -> Self {
        match g {
            GroupAxisCfg::PerToken => GroupAxis::PerToken,
            GroupAxisCfg::PerChannel => GroupAxis::PerChannel,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuantizerConfig {
    GroupRtn { bits: u8, axis: GroupAxisCfg },
    LloydMax { bits: u8 },
    SignResidual { bits: u8 },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    None,
    #[serde(rename = "qjl_1bit")]
    Qjl1Bit,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct WindowConfig {
    pub sink: usize,
    pub recent: usize,
    pub dtype_bytes: usize,
}

fn default_rotation() -> RotationKind {
    RotationKind::None
}
fn default_scale() -> NormalizationKind {
    NormalizationKind::None
}
fn default_residual() -> ResidualKind {
    ResidualKind::None
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    #[serde(default = "default_rotation")]
    pub rotation: RotationKind,
    #[serde(default = "default_scale")]
    pub scale: NormalizationKind,
    pub quantizer: QuantizerConfig,
    #[serde(default = "default_residual")]
    pub residual: ResidualKind,
    pub window: WindowConfig,
}

pub struct Packed {
    pub window_split: WindowSplit,
    /// Sink tokens followed by recent tokens, kept at their true
    /// (high-precision) dtype conceptually — stored here as f32 since this
    /// is the CPU reference; `window_split` + the pipeline's configured
    /// `dtype_bytes` is what byte accounting uses, not this tensor's
    /// in-memory Rust representation.
    pub high_prec: Tensor2,
    pub side: SideInfo,
    pub codes: Codes,
    pub residual_codes: ResidualCodes,
}

pub struct Pipeline {
    pub head_dim: usize,
    /// Post-rotation channel count. Equal to `head_dim` unless the
    /// rotation is `hadamard` on a non-power-of-two `head_dim`, in which
    /// case it is `head_dim` padded up to the next power of two.
    working_dim: usize,
    pub window: WindowPolicy,
    rotation: Option<Box<dyn Transform>>,
    scale: Option<Box<dyn Transform>>,
    quantizer: Box<dyn Quantizer>,
    residual: Option<Box<dyn Residual>>,
}

fn concat_rows(a: &Tensor2, b: &Tensor2, cols: usize) -> Tensor2 {
    let mut out = Tensor2::zeros(a.rows + b.rows, cols);
    out.data[..a.data.len()].copy_from_slice(&a.data);
    out.data[a.data.len()..].copy_from_slice(&b.data);
    out
}

fn slice_rows(x: &Tensor2, start: usize, end: usize) -> Tensor2 {
    let mut out = Tensor2::zeros(end - start, x.cols);
    out.data.copy_from_slice(&x.data[start * x.cols..end * x.cols]);
    out
}

/// Horizontally stack three same-row-count matrices (`a | b | c`) along
/// columns — used to reassemble a score matrix's key axis as
/// [sink keys, middle keys, recent keys], matching `decode()`'s row order.
fn concat_cols(a: &Tensor2, b: &Tensor2, c: &Tensor2) -> Tensor2 {
    let rows = a.rows;
    assert_eq!(b.rows, rows);
    assert_eq!(c.rows, rows);
    let cols = a.cols + b.cols + c.cols;
    let mut out = Tensor2::zeros(rows, cols);
    for r in 0..rows {
        let (a_part, rest) = out.row_mut(r).split_at_mut(a.cols);
        let (b_part, c_part) = rest.split_at_mut(b.cols);
        a_part.copy_from_slice(a.row(r));
        b_part.copy_from_slice(b.row(r));
        c_part.copy_from_slice(c.row(r));
    }
    out
}

fn dot_all(q: &Tensor2, k: &Tensor2) -> Tensor2 {
    assert_eq!(q.cols, k.cols);
    let mut out = Tensor2::zeros(q.rows, k.rows);
    for i in 0..q.rows {
        for j in 0..k.rows {
            let dot: f32 = q.row(i).iter().zip(k.row(j).iter()).map(|(a, b)| a * b).sum();
            out.row_mut(i)[j] = dot;
        }
    }
    out
}

impl Pipeline {
    pub fn build(cfg: &PipelineConfig, head_dim: usize, seed: u64) -> Self {
        let (rotation, working_dim): (Option<Box<dyn Transform>>, usize) = match cfg.rotation {
            RotationKind::None => (None, head_dim),
            RotationKind::Hadamard => {
                let h = Hadamard::new(head_dim, seed);
                let wd = h.padded_dim;
                (Some(Box::new(h)), wd)
            }
            RotationKind::RandomOrthogonal => {
                (Some(Box::new(RandomOrthogonal::new(head_dim, seed))), head_dim)
            }
        };
        let scale: Option<Box<dyn Transform>> = match cfg.scale {
            NormalizationKind::None => None,
            NormalizationKind::PerToken => Some(Box::new(PerTokenScale::default())),
        };
        let quantizer: Box<dyn Quantizer> = match &cfg.quantizer {
            QuantizerConfig::GroupRtn { bits, axis } => {
                Box::new(GroupRtn { bits: *bits, axis: (*axis).into() })
            }
            QuantizerConfig::LloydMax { bits } => Box::new(LloydMax::new(*bits)),
            QuantizerConfig::SignResidual { bits } => Box::new(SignResidual { residual_bits: *bits }),
        };
        let residual: Option<Box<dyn Residual>> = match cfg.residual {
            ResidualKind::None => None,
            ResidualKind::Qjl1Bit => {
                assert!(
                    matches!(cfg.scale, NormalizationKind::None),
                    "qjl_1bit residual + a `scale` stage isn't supported yet: `scale` is \
                     per-token/data-dependent on K, so there's no shared transform `score()` \
                     could apply to `q` to keep it in the same space as the scaled K \
                     reconstruction. See docs/open-questions.md."
                );
                // Distinct seed offset from the rotation's own seed so the
                // QJL projection matrix isn't accidentally correlated with
                // the Hadamard sign vector.
                Some(Box::new(QjlResidual::new(working_dim, seed.wrapping_add(0x5115_5115))))
            }
        };
        let window = WindowPolicy {
            sink: cfg.window.sink,
            recent: cfg.window.recent,
            high_prec_dtype_bytes: cfg.window.dtype_bytes,
        };
        Pipeline { head_dim, working_dim, window, rotation, scale, quantizer, residual }
    }

    pub fn encode(&self, x: &Tensor2) -> Packed {
        assert_eq!(x.cols, self.head_dim);
        let split = self.window.split(x.rows);

        let sink_rows = slice_rows(x, 0, split.sink);
        let recent_rows = slice_rows(x, x.rows - split.recent, x.rows);
        let high_prec = concat_rows(&sink_rows, &recent_rows, self.head_dim);

        let mut middle = slice_rows(x, split.sink, split.sink + split.middle);
        let mut side = SideInfo::default();
        if let Some(rot) = &self.rotation {
            rot.forward(&mut middle, &mut side);
        }
        if let Some(sc) = &self.scale {
            sc.forward(&mut middle, &mut side);
        }
        let codes = self.quantizer.encode(&middle);
        let residual_codes = match &self.residual {
            Some(res) => {
                let stage1_recon = self.quantizer.decode(&codes);
                res.encode(&middle, &stage1_recon)
            }
            None => ResidualCodes::None,
        };

        Packed { window_split: split, high_prec, side, codes, residual_codes }
    }

    pub fn decode(&self, p: &Packed) -> Tensor2 {
        let mut middle = self.quantizer.decode(&p.codes);
        if let Some(res) = &self.residual {
            res.apply(&mut middle, &p.residual_codes);
        }
        if let Some(sc) = &self.scale {
            sc.inverse(&mut middle, &p.side);
        }
        if let Some(rot) = &self.rotation {
            rot.inverse(&mut middle, &p.side);
        }

        let total_rows = p.window_split.sink + p.window_split.middle + p.window_split.recent;
        let mut out = Tensor2::zeros(total_rows, self.head_dim);
        out.data[..p.window_split.sink * self.head_dim]
            .copy_from_slice(&p.high_prec.data[..p.window_split.sink * self.head_dim]);
        out.data[p.window_split.sink * self.head_dim
            ..(p.window_split.sink + p.window_split.middle) * self.head_dim]
            .copy_from_slice(&middle.data);
        out.data[(p.window_split.sink + p.window_split.middle) * self.head_dim..]
            .copy_from_slice(&p.high_prec.data[p.window_split.sink * self.head_dim..]);
        out
    }

    /// Estimate `q @ k^T` directly from packed keys.
    ///
    /// With no `residual` stage this is exactly the reference path
    /// (`decode` then dot) — SPEC §4.2's design note allows `score` to
    /// equal decode+dot in the reference path, and M1 had no codec that
    /// needed anything else.
    ///
    /// With `qjl_1bit`, this is genuinely different from `decode`+dot: `q`
    /// is rotated by the *same* rotation stage `k` went through (rotation
    /// preserves inner products, so `<q, k> = <rotate(q), rotate(k)>`),
    /// dotted against the stage-1 (still-rotated-space) reconstruction,
    /// then corrected by the residual's unbiased inner-product estimate —
    /// all without ever fully reconstructing `k`. This is the first stage
    /// where `score` and `decode` really diverge (`decode`'s reconstruction
    /// quality is unaffected by `qjl_1bit`; only `score`'s is).
    pub fn score(&self, q: &Tensor2, p: &Packed) -> Tensor2 {
        let Some(residual) = &self.residual else {
            let k = self.decode(p);
            return dot_all(q, &k);
        };

        // Sink/recent window tokens are exact (full-precision, never
        // rotated or quantized), so they're scored by a plain dot product
        // against the *original* q — same as what `decode()`+dot would
        // give for those columns. Only the "middle" tokens go through the
        // rotate-then-stage1-plus-correction path.
        let sink = slice_rows(&p.high_prec, 0, p.window_split.sink);
        let recent = slice_rows(
            &p.high_prec,
            p.window_split.sink,
            p.window_split.sink + p.window_split.recent,
        );
        let sink_scores = dot_all(q, &sink);
        let recent_scores = dot_all(q, &recent);

        let mut q_rot = q.clone();
        let mut dummy_side = SideInfo::default();
        if let Some(rot) = &self.rotation {
            rot.forward(&mut q_rot, &mut dummy_side);
        }
        let k_stage1 = self.quantizer.decode(&p.codes); // still in rotated space
        let mut middle_scores = dot_all(&q_rot, &k_stage1);
        let correction = residual.score_correction(&q_rot, &p.residual_codes);
        for (o, c) in middle_scores.data.iter_mut().zip(correction.data.iter()) {
            *o += c;
        }

        concat_cols(&sink_scores, &middle_scores, &recent_scores)
    }

    /// Total persistent state this pipeline instance holds (rotation
    /// matrix, codebook, ...), counted once per (layer, head).
    pub fn persistent_state_bytes(&self, seq_len: usize) -> usize {
        let split = self.window.split(seq_len);
        let ctx = ShapeCtx { n_tokens: split.middle, head_dim: self.head_dim };
        let mut total = 0;
        if let Some(r) = &self.rotation {
            total += r.state_bytes(&ctx);
        }
        if let Some(s) = &self.scale {
            total += s.state_bytes(&ctx);
        }
        total += self.quantizer.state_bytes(&ctx);
        if let Some(res) = &self.residual {
            total += res.state_bytes(&ctx);
        }
        total
    }

    pub fn window_split(&self, seq_len: usize) -> WindowSplit {
        self.window.split(seq_len)
    }

    /// Peak transient workspace estimate for one `encode`/`decode` call:
    /// the largest intermediate buffer the reference implementation
    /// allocates (the post-rotation f32 tensor).
    pub fn peak_transient_bytes(&self, seq_len: usize) -> f64 {
        let split = self.window.split(seq_len);
        split.middle as f64 * self.working_dim as f64 * 4.0
    }
}
