//! Stateless, counter-based randomness: every draw is a pure function of
//! (seed, coordinates), so synthetic traces are reproducible and order-free.

const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finaliser: a bijective 64-bit mix.
pub fn mix(x: u64) -> u64 {
    let x = x.wrapping_add(GOLDEN);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub fn hash(seed: u64, a: u64, b: u64) -> u64 {
    mix(mix(seed ^ mix(a)) ^ b.wrapping_mul(GOLDEN))
}

/// Uniform in [0, 1).
pub fn uniform(seed: u64, a: u64, b: u64) -> f64 {
    (hash(seed, a, b) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal via Box–Muller.
pub fn normal(seed: u64, a: u64, b: u64) -> f64 {
    let u1 = uniform(seed, a, b.wrapping_mul(2)).max(f64::MIN_POSITIVE);
    let u2 = uniform(seed, a, b.wrapping_mul(2) + 1);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_in_range_with_sane_mean() {
        let xs: Vec<f64> = (0..10_000).map(|i| uniform(1, i, 0)).collect();
        assert!(xs.iter().all(|x| (0.0..1.0).contains(x)));
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        assert!((mean - 0.5).abs() < 0.02);
    }

    #[test]
    fn normal_has_unit_variance() {
        let xs: Vec<f64> = (0..20_000).map(|i| normal(7, i, 3)).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05);
    }
}
