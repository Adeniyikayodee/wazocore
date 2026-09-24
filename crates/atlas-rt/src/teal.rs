//! TEAL activation sparsity (PROJECT-PLAN-B §3.3): per-(layer, site)
//! magnitude thresholds from calibration histograms, and the masks they
//! imply. Same histogram, threshold rule and profile schema as
//! `atlas_trace.teal`, so profiles from either tool are interchangeable.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Sparsified inputs, in the order thresholds are stored.
pub const SITES: [&str; 4] = ["qkv", "o", "gate_up", "down"];
pub const QKV: usize = 0;
pub const O: usize = 1;
pub const GATE_UP: usize = 2;
pub const DOWN: usize = 3;

/// Histogram of log10|x| in 0.001-decade bins over [1e-6, 1e3].
pub const LOG_LO: f64 = -6.0;
pub const LOG_HI: f64 = 3.0;
pub const BINS: usize = 9000;

/// One histogram per (layer, site).
pub type Histograms = Vec<[Vec<u64>; 4]>;

pub fn empty_histograms(layers: usize) -> Histograms {
    (0..layers).map(|_| std::array::from_fn(|_| vec![0u64; BINS])).collect()
}

pub fn bin(x: f32) -> usize {
    let l = (x.abs() as f64).max(10f64.powf(LOG_LO)).log10();
    (((l - LOG_LO) / (LOG_HI - LOG_LO) * BINS as f64) as usize).min(BINS - 1)
}

/// Adds every value of every token to `h`.
pub fn accumulate(h: &mut [u64], xs: &[Vec<f32>]) {
    xs.iter().flatten().for_each(|&x| h[bin(x)] += 1);
}

/// Element-wise sum: the pooled profile weights each language by its
/// calibration tokens (equal budgets give equal weight).
pub fn pool(hs: &[Histograms]) -> Option<Histograms> {
    let first = hs.first()?;
    Some(
        (0..first.len())
            .map(|l| std::array::from_fn(|s| (0..BINS).map(|b| hs.iter().map(|h| h[l][s][b]).sum()).collect()))
            .collect(),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub enum TealError {
    Sparsity(f64),
    Empty,
}

impl fmt::Display for TealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sparsity(p) => write!(f, "sparsity {p} outside [0, 1)"),
            Self::Empty => write!(f, "empty histogram"),
        }
    }
}

impl std::error::Error for TealError {}

/// |x| below which a fraction `sparsity` of the histogram's mass lies
/// (linear interpolation inside the crossing bin), as `threshold_from_hist`
/// in atlas_trace.teal.
pub fn threshold(counts: &[u64], sparsity: f64) -> Result<f32, TealError> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(TealError::Sparsity(sparsity));
    }
    if sparsity == 0.0 {
        return Ok(0.0);
    }
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return Err(TealError::Empty);
    }
    let target = sparsity * total as f64;
    let (i, below) = counts
        .iter()
        .scan(0u64, |acc, &c| {
            let before = *acc;
            *acc += c;
            Some((before, *acc))
        })
        .enumerate()
        .find(|(_, (_, after))| *after as f64 >= target)
        .map(|(i, (before, _))| (i, before))
        .unwrap_or((BINS - 1, total));
    let frac = if counts[i] > 0 { (target - below as f64) / counts[i] as f64 } else { 0.0 };
    let width = (LOG_HI - LOG_LO) / BINS as f64;
    Ok(10f64.powf(LOG_LO + (i as f64 + frac) * width) as f32)
}

/// thresholds[layer][site] for one target sparsity.
pub fn thresholds(h: &Histograms, sparsity: f64) -> Result<Vec<[f32; 4]>, TealError> {
    h.iter()
        .map(|sites| {
            let t: Vec<f32> = sites.iter().map(|c| threshold(c, sparsity)).collect::<Result<_, _>>()?;
            Ok([t[0], t[1], t[2], t[3]])
        })
        .collect()
}

/// Zeroes |x| ≤ t; returns the masked values and the number kept.
pub fn mask(xs: &[Vec<f32>], t: f32) -> (Vec<Vec<f32>>, u64) {
    let masked: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().map(|&v| if v.abs() > t { v } else { 0.0 }).collect()).collect();
    let kept = masked.iter().flatten().filter(|v| **v != 0.0).count() as u64;
    (masked, kept)
}

/// Active-channel bitmap in trace layout: channel c is bit c % 64 of word c / 64.
pub fn bitmap(x: &[f32]) -> Vec<u64> {
    x.chunks(64).map(|w| w.iter().enumerate().fold(0u64, |acc, (i, v)| if *v != 0.0 { acc | 1 << i } else { acc })).collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub lang: String,
    pub split: String,
    pub tokens: usize,
    pub text_sha256: String,
}

/// A per-language profile (profiles/{lang}-{pct}.json).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub model: String,
    pub calibration: Calibration,
    pub sparsity: f64,
    pub sites: Vec<String>,
    pub thresholds: Vec<[f32; 4]>,
}

/// Mean with a 95% bootstrap interval (deterministic LCG resampling).
pub fn bootstrap(xs: &[f64], resamples: usize) -> (f64, f64, f64) {
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n.max(1) as f64;
    if n < 2 {
        return (mean, mean, mean);
    }
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    xs[(state >> 33) as usize % n]
                })
                .sum::<f64>()
                / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    (mean, means[resamples * 25 / 1000], means[resamples * 975 / 1000 - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalish(n: usize) -> Vec<f32> {
        // Deterministic heavy-ish tail: ratio of two uniforms.
        (0..n)
            .map(|i| {
                let a = ((i as u64).wrapping_mul(2_654_435_761) % 10_007) as f32 / 10_007.0 - 0.5;
                let b = ((i as u64).wrapping_mul(40_503) % 997) as f32 / 997.0 + 0.2;
                a / b
            })
            .collect()
    }

    #[test]
    fn threshold_hits_the_requested_quantile() {
        let x = normalish(200_000);
        let mut h = vec![0u64; BINS];
        accumulate(&mut h, std::slice::from_ref(&x));
        for p in [0.25, 0.4, 0.5] {
            let t = threshold(&h, p).unwrap();
            let below = x.iter().filter(|v| v.abs() <= t).count() as f64 / x.len() as f64;
            assert!((below - p).abs() < 3e-3, "p {p}: {below}");
        }
        assert_eq!(threshold(&h, 0.0), Ok(0.0));
        assert!(threshold(&h, 1.0).is_err());
        assert_eq!(threshold(&[0; BINS], 0.5), Err(TealError::Empty));
    }

    #[test]
    fn mask_counts_kept_and_bitmap_matches_trace_layout() {
        let (m, kept) = mask(&[vec![0.5, -0.1, 0.3, -2.0]], 0.2);
        assert_eq!((m[0].clone(), kept), (vec![0.5, 0.0, 0.3, -2.0], 3));
        let mut x = vec![0.0f32; 128];
        x[0] = 1.0;
        x[63] = 1.0;
        x[64] = 1.0;
        assert_eq!(bitmap(&x), vec![1 | 1 << 63, 1]);
    }

    #[test]
    fn pooling_sums_and_thresholds_are_per_site() {
        let one: Histograms = vec![[vec![1; BINS], vec![2; BINS], vec![3; BINS], vec![4; BINS]]];
        let p = pool(&[one.clone(), one]).unwrap();
        assert_eq!((p[0][0][0], p[0][3][5]), (2, 8));
        assert_eq!(thresholds(&p, 0.5).unwrap().len(), 1);
        assert!(pool(&[]).is_none());
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_mean() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (m, lo, hi) = bootstrap(&xs, 1000);
        assert_eq!(m, 3.0);
        assert!(lo <= m && m <= hi);
        assert_eq!(bootstrap(&xs, 1000), (m, lo, hi));
    }
}
