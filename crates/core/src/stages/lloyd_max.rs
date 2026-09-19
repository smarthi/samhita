//! `lloyd_max`: non-uniform scalar quantizer with a fixed Gaussian-marginal
//! codebook, per SPEC §4.3 stage inventory ("Beta/Gaussian marginal
//! codebook"). M1 ships the Gaussian variant.
//!
//! The codebook is built once (deterministically, no RNG) by running
//! Lloyd's algorithm on a deterministic quantile grid of the standard
//! normal — `x_i = Phi^{-1}((i+0.5)/N)` — rather than on drawn samples, so
//! the result doesn't depend on a seed at all. Each row is normalized by
//! its own std before indexing into the shared codebook, matching the
//! "marginal" framing (the codebook models a unit-variance source; the
//! row's scale is stored as ordinary per-token metadata, exactly like
//! `group_rtn`'s per-token scale).

use crate::codes::Codes;
use crate::normal::inv_norm_cdf;
use crate::shape::{CalibData, ShapeCtx};
use crate::tensor::Tensor2;
use crate::traits::{Quantizer, Stage};
use anyhow::Result;

pub struct LloydMax {
    pub bits: u8,
    /// Ascending codebook centroids for a unit-variance Gaussian source,
    /// length `2^bits`.
    pub codebook: Vec<f32>,
}

impl LloydMax {
    /// Build the codebook immediately (no separate `fit` call needed since
    /// it's calibration-free); `fit` is still implemented as a no-op to
    /// satisfy the `Stage` contract for pipelines that call it uniformly.
    pub fn new(bits: u8) -> Self {
        let levels = 1usize << bits;
        LloydMax { bits, codebook: build_gaussian_codebook(levels) }
    }

    fn nearest_index(&self, v: f32) -> usize {
        // Codebook is sorted ascending; boundaries are midpoints.
        match self
            .codebook
            .binary_search_by(|c| c.partial_cmp(&v).unwrap())
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) if i >= self.codebook.len() => self.codebook.len() - 1,
            Err(i) => {
                let lo = self.codebook[i - 1];
                let hi = self.codebook[i];
                if (v - lo).abs() <= (hi - v).abs() {
                    i - 1
                } else {
                    i
                }
            }
        }
    }
}

/// N deterministic quantile samples of N(0,1), used in place of Monte
/// Carlo sampling so codebook construction has no seed dependence.
fn gaussian_quantile_grid(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            inv_norm_cdf(p)
        })
        .collect()
}

fn build_gaussian_codebook(levels: usize) -> Vec<f32> {
    const N_SAMPLES: usize = 20_000;
    const ITERS: usize = 60;

    let samples = gaussian_quantile_grid(N_SAMPLES); // already sorted ascending
    // Initialize centroids at evenly spaced quantiles of the same grid.
    let mut centroids: Vec<f64> = (0..levels)
        .map(|i| {
            let p = (i as f64 + 0.5) / levels as f64;
            inv_norm_cdf(p)
        })
        .collect();

    for _ in 0..ITERS {
        // Boundaries = midpoints between sorted centroids.
        let mut boundaries = Vec::with_capacity(levels.saturating_sub(1));
        for w in centroids.windows(2) {
            boundaries.push((w[0] + w[1]) / 2.0);
        }

        let mut sums = vec![0.0f64; levels];
        let mut counts = vec![0u64; levels];
        let mut b_idx = 0usize;
        for &s in &samples {
            while b_idx < boundaries.len() && s > boundaries[b_idx] {
                b_idx += 1;
            }
            sums[b_idx] += s;
            counts[b_idx] += 1;
        }
        for i in 0..levels {
            if counts[i] > 0 {
                centroids[i] = sums[i] / counts[i] as f64;
            }
        }
    }

    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    centroids.into_iter().map(|v| v as f32).collect()
}

impl Stage for LloydMax {
    fn name(&self) -> &'static str {
        "lloyd_max"
    }

    fn fit(&mut self, _calib: &CalibData, _seed: u64) -> Result<()> {
        Ok(())
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        self.codebook.len() * 4
    }

    fn per_token_meta_bytes(&self, _ctx: &ShapeCtx) -> f64 {
        4.0 // per-token f32 scale mapping the unit-variance codebook onto the row
    }
}

impl Quantizer for LloydMax {
    fn encode(&self, x: &Tensor2) -> Codes {
        let mut payload = vec![0i32; x.rows * x.cols];
        let mut row_scale = Vec::with_capacity(x.rows);
        for r in 0..x.rows {
            let row = x.row(r);
            let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
            let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / row.len() as f64;
            let std = var.sqrt().max(1e-8) as f32;
            for (c, &v) in row.iter().enumerate() {
                let normalized = v / std;
                payload[r * x.cols + c] = self.nearest_index(normalized) as i32;
            }
            row_scale.push(std);
        }
        Codes::Codebook { bits: self.bits, rows: x.rows, cols: x.cols, payload, row_scale }
    }

    fn decode(&self, c: &Codes) -> Tensor2 {
        let Codes::Codebook { rows, cols, payload, row_scale, .. } = c else {
            panic!("lloyd_max::decode called with non-Codebook codes");
        };
        let mut out = Tensor2::zeros(*rows, *cols);
        for r in 0..*rows {
            let scale = row_scale[r];
            for ci in 0..*cols {
                let idx = payload[r * cols + ci] as usize;
                out.row_mut(r)[ci] = self.codebook[idx] * scale;
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
    fn codebook_is_sorted_and_symmetric_ish() {
        let lm = LloydMax::new(3);
        assert_eq!(lm.codebook.len(), 8);
        for w in lm.codebook.windows(2) {
            assert!(w[0] < w[1]);
        }
        // Gaussian source -> codebook should be roughly symmetric about 0.
        let sum: f32 = lm.codebook.iter().sum();
        assert!(sum.abs() < 0.05, "sum={sum}");
    }

    #[test]
    fn round_trip_reasonable_on_gaussian_row() {
        let lm = LloydMax::new(4);
        // A row that looks roughly Gaussian.
        let row = vec![
            -2.0, -1.5, -1.0, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, -0.5, 0.5, 0.05,
            -0.05,
        ];
        let x = Tensor2::from_rows(&[row]);
        let codes = lm.encode(&x);
        let recon = lm.decode(&codes);
        assert!(x.mse(&recon) < 0.2);
    }
}
