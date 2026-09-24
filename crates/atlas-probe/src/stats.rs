//! Order statistics for timing samples.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Summary {
    pub n: usize,
    pub min: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

/// Linearly interpolated percentile of an ascending slice; `q` in [0, 1].
pub fn percentile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64))
}

pub fn summarize(samples: &[f64]) -> Option<Summary> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(Summary {
        n: sorted.len(),
        min: *sorted.first()?,
        median: percentile(&sorted, 0.5)?,
        p95: percentile(&sorted, 0.95)?,
        max: *sorted.last()?,
    })
}

/// Bytes per second, or `None` for a non-positive or non-finite duration.
pub fn rate(bytes: u64, secs: f64) -> Option<f64> {
    (secs.is_finite() && secs > 0.0).then(|| bytes as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&xs, 0.0), Some(1.0));
        assert_eq!(percentile(&xs, 0.5), Some(2.5));
        assert_eq!(percentile(&xs, 1.0), Some(4.0));
        assert_eq!(percentile(&xs, 1.5), None);
        assert_eq!(percentile(&[], 0.5), None);
    }

    #[test]
    fn summarize_does_not_depend_on_input_order() {
        let s = summarize(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!((s.n, s.min, s.median, s.max), (3, 1.0, 2.0, 3.0));
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn rate_rejects_zero_duration() {
        assert_eq!(rate(10, 2.0), Some(5.0));
        assert_eq!(rate(10, 0.0), None);
        assert_eq!(rate(10, f64::NAN), None);
    }
}
