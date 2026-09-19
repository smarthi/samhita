//! `sink_window` / `recent_window` layout policy: the first `sink` tokens
//! and the last `recent` tokens of a sequence are kept at high precision
//! (their true dtype, e.g. bf16) and never enter the
//! rotation/normalization/quantizer chain at all. This is the layout stage
//! at the front of the pipeline diagram in SPEC §4.1.
//!
//! Unlike the other stages this one doesn't implement `Transform` or
//! `Quantizer` — its job is routing tokens, not transforming values — so it
//! only implements `Stage`, and `Pipeline` calls its `split` directly.

use crate::shape::ShapeCtx;
use crate::traits::Stage;

#[derive(Clone, Copy, Debug)]
pub struct WindowPolicy {
    pub sink: usize,
    pub recent: usize,
    /// Bytes per element of the high-precision dtype the window is stored
    /// at (e.g. 2 for bf16/f16, 4 for f32).
    pub high_prec_dtype_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowSplit {
    pub sink: usize,
    pub recent: usize,
    pub middle: usize,
}

impl WindowPolicy {
    /// Given a total sequence length, returns the (sink, middle, recent)
    /// token counts. If the window is wider than the sequence, everything
    /// is high-precision and `middle == 0` (nothing reaches the pipeline).
    pub fn split(&self, seq_len: usize) -> WindowSplit {
        let window = (self.sink + self.recent).min(seq_len);
        let sink = self.sink.min(window);
        let recent = window - sink;
        WindowSplit { sink, recent, middle: seq_len - window }
    }
}

impl Stage for WindowPolicy {
    fn name(&self) -> &'static str {
        "sink_recent_window"
    }

    fn state_bytes(&self, _ctx: &ShapeCtx) -> usize {
        0
    }

    fn per_token_meta_bytes(&self, _ctx: &ShapeCtx) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_normally() {
        let w = WindowPolicy { sink: 4, recent: 8, high_prec_dtype_bytes: 2 };
        let s = w.split(100);
        assert_eq!(s, WindowSplit { sink: 4, recent: 8, middle: 88 });
    }

    #[test]
    fn window_wider_than_sequence() {
        let w = WindowPolicy { sink: 4, recent: 128, high_prec_dtype_bytes: 2 };
        let s = w.split(10);
        assert_eq!(s.middle, 0);
        assert_eq!(s.sink + s.recent, 10);
    }
}
