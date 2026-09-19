//! Tiny deterministic PRNG so rotation stages don't need an external `rand`
//! dependency for something as simple as "reproducible signs/gaussians
//! given a seed". SplitMix64 (public domain, Vigna).

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in the open interval (0, 1), never exactly 0 or 1 so
    /// it is always safe to feed to `inv_norm_cdf`.
    pub fn next_open01(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) as f64; // 53 significant bits
        let u = bits / (1u64 << 53) as f64;
        u.clamp(1e-12, 1.0 - 1e-12)
    }

    pub fn next_sign(&mut self) -> f32 {
        if self.next_u64() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn next_gaussian(&mut self) -> f64 {
        crate::normal::inv_norm_cdf(self.next_open01())
    }
}
