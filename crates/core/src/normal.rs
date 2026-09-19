//! Inverse standard-normal CDF, needed by `lloyd_max` to build a
//! deterministic Gaussian-source codebook without any RNG dependency.
//!
//! Acklam's rational approximation (public domain), |error| < 1.15e-9.
//! Reference: P.J. Acklam, "An algorithm for computing the inverse normal
//! cumulative distribution function" (2003), the standard implementation
//! reproduced in many numerical libraries.

const A: [f64; 6] = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
];
const B: [f64; 5] = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
];
const C: [f64; 6] = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
];
const D: [f64; 4] = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
];

/// Inverse of the standard normal CDF. `p` must be in `(0, 1)`.
pub fn inv_norm_cdf(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0, "inv_norm_cdf domain is (0,1), got {p}");
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_quantiles() {
        assert!((inv_norm_cdf(0.5)).abs() < 1e-9);
        assert!((inv_norm_cdf(0.975) - 1.959963985).abs() < 1e-6);
        assert!((inv_norm_cdf(0.025) - (-1.959963985)).abs() < 1e-6);
        assert!((inv_norm_cdf(0.8413447) - 1.0).abs() < 1e-5);
    }
}
