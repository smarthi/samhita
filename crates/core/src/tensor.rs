//! Minimal row-major f32 matrix used throughout the pipeline.
//!
//! Rows are tokens, columns are channels (head_dim). Kept intentionally
//! small (no external linear-algebra crate) so the byte-accounting and
//! stage code stays auditable.

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor2 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor2 {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor2 { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_rows(rows: &[Vec<f32>]) -> Self {
        let r = rows.len();
        let c = rows.first().map(|row| row.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c, "ragged rows not allowed in Tensor2");
            data.extend_from_slice(row);
        }
        Tensor2 { rows: r, cols: c, data }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let cols = self.cols;
        &mut self.data[i * cols..(i + 1) * cols]
    }

    pub fn mse(&self, other: &Tensor2) -> f64 {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let n = self.data.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let sum: f64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                let d = (*a - *b) as f64;
                d * d
            })
            .sum();
        sum / n
    }
}
