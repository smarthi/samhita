"""Deterministic PRNG mirroring `crates/core/src/rng.rs` and `normal.rs`
bit-for-bit, so the Python torch reference and the Rust core derive the
*same* rotation signs / codebook seeds given the same `seed`. This is what
lets `tests/test_bytes_parity_with_rust.py` compare the two
implementations at a tight tolerance instead of "roughly similar".
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1


class SplitMix64:
    """Public-domain SplitMix64, same constants as the Rust port."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_open01(self) -> float:
        bits = self.next_u64() >> 11  # 53 significant bits
        u = bits / float(1 << 53)
        return min(max(u, 1e-12), 1.0 - 1e-12)

    def next_sign(self) -> float:
        return 1.0 if (self.next_u64() & 1) == 0 else -1.0

    def next_gaussian(self) -> float:
        return inv_norm_cdf(self.next_open01())


# Acklam's rational approximation to the inverse standard-normal CDF,
# reproduced verbatim from crates/core/src/normal.rs so the two languages
# build identical (to within ~1e-9) Gaussian codebooks/rotation matrices.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _horner(coeffs: tuple[float, ...], x: float) -> float:
    acc = coeffs[0]
    for c in coeffs[1:]:
        acc = acc * x + c
    return acc


def inv_norm_cdf(p: float) -> float:
    import math

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return _horner(_C, q) / (_horner(_D + (1.0,), q))
    if p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        return (_horner(_A, r) * q) / _horner(_B + (1.0,), r)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_horner(_C, q) / (_horner(_D + (1.0,), q))
