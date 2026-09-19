//! `random_orthogonal`: a full `d x d` random orthogonal rotation, built via
//! Gram-Schmidt on a deterministically-seeded Gaussian matrix.
//!
//! Unlike `hadamard` (O(d) state, O(d log d) apply), this is O(d^2) state
//! and O(d^2) apply — the framework intentionally keeps both so ablations
//! can show the state-byte cost of "true" random rotation vs. the
//! structured Hadamard approximation (SPEC §4.3 `block_diagonal` note makes
//! the same O(d) vs O(d^2) point for a different rotation family).

use crate::codes::SideInfo;
use crate::rng::SplitMix64;
use crate::shape::ShapeCtx;
use crate::tensor::Tensor2;
use crate::traits::{Stage, Transform};

pub struct RandomOrthogonal {
    pub dim: usize,
    /// Row-major `dim x dim` orthonormal matrix Q, rows are orthonormal
    /// basis vectors: forward is `x @ Q^T`, inverse is `y @ Q`.
    pub q: Vec<f32>,
}

impl RandomOrthogonal {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut rows: Vec<Vec<f64>> = (0..dim)
            .map(|_| (0..dim).map(|_| rng.next_gaussian()).collect())
            .collect();

        // Modified Gram-Schmidt orthonormalization.
        for i in 0..dim {
            for j in 0..i {
                let dot: f64 = (0..dim).map(|k| rows[i][k] * rows[j][k]).sum();
                for k in 0..dim {
                    rows[i][k] -= dot * rows[j][k];
                }
            }
            let norm: f64 = rows[i].iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            for v in rows[i].iter_mut() {
                *v /= norm;
            }
        }

        let q = rows.into_iter().flatten().map(|v| v as f32).collect();
        RandomOrthogonal { dim, q }
    }

    #[inline]
    fn q_at(&self, r: usize, c: usize) -> f32 {
        self.q[r * self.dim + c]
    }

    /// `out = x @ Q^T`, i.e. `out[c] = sum_k x[k] * Q[c][k]`.
    fn matvec_qt(&self, x: &[f32], out: &mut [f32]) {
        for c in 0..self.dim {
            let mut acc = 0.0f32;
            for k in 0..self.dim {
                acc += x[k] * self.q_at(c, k);
            }
            out[c] = acc;
        }
    }

    /// `out = y @ Q`, i.e. `out[c] = sum_k y[k] * Q[k][c]`.
    fn matvec_q(&self, y: &[f32], out: &mut [f32]) {
        for c in 0..self.dim {
            let mut acc = 0.0f32;
            for k in 0..self.dim {
                acc += y[k] * self.q_at(k, c);
            }
            out[c] = acc;
        }
    }
}

impl Stage for RandomOrthogonal {
    fn name(&self) -> &'static str {
        "random_orthogonal"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        self.dim * self.dim * 4
    }

    fn per_token_meta_bytes(&self, _ctx: &ShapeCtx) -> f64 {
        0.0
    }
}

impl Transform for RandomOrthogonal {
    fn forward(&self, x: &mut Tensor2, _side: &mut SideInfo) {
        assert_eq!(x.cols, self.dim);
        let mut out = Tensor2::zeros(x.rows, self.dim);
        for r in 0..x.rows {
            self.matvec_qt(x.row(r), out.row_mut(r));
        }
        *x = out;
    }

    fn inverse(&self, x: &mut Tensor2, _side: &SideInfo) {
        assert_eq!(x.cols, self.dim);
        let mut out = Tensor2::zeros(x.rows, self.dim);
        for r in 0..x.rows {
            self.matvec_q(x.row(r), out.row_mut(r));
        }
        *x = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let ro = RandomOrthogonal::new(6, 123);
        let orig = Tensor2::from_rows(&[
            vec![1.0, -2.0, 3.0, 0.5, -0.25, 4.0],
            vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
        ]);
        let mut x = orig.clone();
        let mut side = SideInfo::default();
        ro.forward(&mut x, &mut side);
        ro.inverse(&mut x, &side);
        for (a, b) in x.data.iter().zip(orig.data.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn preserves_norm() {
        let ro = RandomOrthogonal::new(10, 5);
        let orig = Tensor2::from_rows(&[vec![1.0; 10]]);
        let orig_norm: f32 = orig.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut x = orig.clone();
        let mut side = SideInfo::default();
        ro.forward(&mut x, &mut side);
        let rot_norm: f32 = x.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((orig_norm - rot_norm).abs() < 1e-2, "{orig_norm} vs {rot_norm}");
    }
}
