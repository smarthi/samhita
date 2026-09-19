//! `hadamard`: randomized Hadamard rotation, i.e. `diag(signs) @ H`.
//!
//! Ported from the reference in the author's `dhurandhar` repo
//! (`src/dhurandhar/turboquant.py`, itself implementing the rotation used
//! by TurboQuant, arXiv:2504.19874), replacing the O(d^2) matmul reference
//! with an O(d log d) fast Walsh-Hadamard butterfly and adding padding
//! bookkeeping so the transform round-trips for non-power-of-two head_dim.

use crate::codes::SideInfo;
use crate::rng::SplitMix64;
use crate::shape::ShapeCtx;
use crate::tensor::Tensor2;
use crate::traits::{Stage, Transform};

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1usize << (usize::BITS - (n - 1).leading_zeros())
    }
}

/// In-place fast Walsh-Hadamard butterfly (unnormalized: applying it twice
/// scales the input by `len`).
fn fwht_raw(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let u = a[j];
                let v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
            i += 2 * len;
        }
        len *= 2;
    }
}

/// Apply the orthonormal Hadamard matrix `H` (H@H = I) to a row in place.
fn apply_h(row: &mut [f32]) {
    fwht_raw(row);
    let norm = 1.0 / (row.len() as f32).sqrt();
    for v in row.iter_mut() {
        *v *= norm;
    }
}

pub struct Hadamard {
    pub orig_dim: usize,
    pub padded_dim: usize,
    /// Deterministic random signs, `diag(signs)`. Fixed at construction
    /// from `seed` — the whole point of "randomized" Hadamard is that the
    /// signs are reproducible, not resampled per call.
    pub signs: Vec<f32>,
}

impl Hadamard {
    pub fn new(head_dim: usize, seed: u64) -> Self {
        let padded_dim = next_pow2(head_dim);
        let mut rng = SplitMix64::new(seed);
        let signs = (0..padded_dim).map(|_| rng.next_sign()).collect();
        Hadamard { orig_dim: head_dim, padded_dim, signs }
    }

    fn pad(&self, x: &Tensor2) -> Tensor2 {
        if self.padded_dim == self.orig_dim {
            return x.clone();
        }
        let mut out = Tensor2::zeros(x.rows, self.padded_dim);
        for r in 0..x.rows {
            out.row_mut(r)[..self.orig_dim].copy_from_slice(x.row(r));
        }
        out
    }

    fn truncate(&self, x: &Tensor2) -> Tensor2 {
        if self.padded_dim == self.orig_dim {
            return x.clone();
        }
        let mut out = Tensor2::zeros(x.rows, self.orig_dim);
        for r in 0..x.rows {
            out.row_mut(r).copy_from_slice(&x.row(r)[..self.orig_dim]);
        }
        out
    }
}

impl Stage for Hadamard {
    fn name(&self) -> &'static str {
        "hadamard"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        // Signs are +/-1: a real implementation persists them bit-packed.
        (self.padded_dim as u64).div_ceil(8) as usize
    }

    fn per_token_meta_bytes(&self, _ctx: &ShapeCtx) -> f64 {
        0.0
    }
}

impl Transform for Hadamard {
    fn forward(&self, x: &mut Tensor2, _side: &mut SideInfo) {
        let mut padded = self.pad(x);
        for r in 0..padded.rows {
            let row = padded.row_mut(r);
            for (v, s) in row.iter_mut().zip(self.signs.iter()) {
                *v *= s;
            }
            apply_h(row);
        }
        *x = padded;
    }

    fn inverse(&self, x: &mut Tensor2, _side: &SideInfo) {
        let mut padded = x.clone();
        for r in 0..padded.rows {
            let row = padded.row_mut(r);
            apply_h(row);
            for (v, s) in row.iter_mut().zip(self.signs.iter()) {
                *v *= s;
            }
        }
        *x = self.truncate(&padded);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_power_of_two_dim() {
        let h = Hadamard::new(8, 42);
        let orig = Tensor2::from_rows(&[
            vec![1.0, -2.0, 3.0, 0.5, -0.25, 4.0, -1.5, 2.5],
            vec![0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        ]);
        let mut x = orig.clone();
        let mut side = SideInfo::default();
        h.forward(&mut x, &mut side);
        assert_eq!(x.cols, 8);
        h.inverse(&mut x, &side);
        for (a, b) in x.data.iter().zip(orig.data.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn round_trips_non_power_of_two_dim() {
        let h = Hadamard::new(5, 7);
        let orig = Tensor2::from_rows(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]]);
        let mut x = orig.clone();
        let mut side = SideInfo::default();
        h.forward(&mut x, &mut side);
        assert_eq!(x.cols, 8); // padded to next pow2
        h.inverse(&mut x, &side);
        assert_eq!(x.cols, 5);
        for (a, b) in x.data.iter().zip(orig.data.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let h = Hadamard::new(16, 1);
        let orig = Tensor2::from_rows(&[vec![
            1.0, -2.0, 3.0, 0.5, -0.25, 4.0, -1.5, 2.5, 1.0, -2.0, 3.0, 0.5, -0.25, 4.0, -1.5,
            2.5,
        ]]);
        let orig_norm: f32 = orig.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut x = orig.clone();
        let mut side = SideInfo::default();
        h.forward(&mut x, &mut side);
        let rot_norm: f32 = x.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((orig_norm - rot_norm).abs() < 1e-3, "{orig_norm} vs {rot_norm}");
    }
}
