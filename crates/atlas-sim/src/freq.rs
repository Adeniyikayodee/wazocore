//! Activation frequencies from calibration traces, and the static hot sets
//! built from them (global vs per-language: claim C4).

use crate::trace::{Layout, Trace};

/// Fraction of tokens in which each item was active, indexed by item id.
pub fn activation_freqs(trace: &Trace) -> Vec<f64> {
    let lay = &trace.layout;
    let mut counts = vec![0u32; lay.item_count()];
    for t in 0..lay.tokens {
        for l in 0..lay.layers {
            for m in 0..lay.matrices() {
                for c in trace.active_channels(t, l, m) {
                    counts[lay.item(l, m, c) as usize] += 1;
                }
            }
        }
    }
    let n = lay.tokens.max(1) as f64;
    counts.into_iter().map(|c| c as f64 / n).collect()
}

/// Element-wise mean of per-language frequencies (a language-agnostic
/// profile weighting every language equally). `None` if shapes differ.
pub fn mean_freqs(per_lang: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = per_lang.first()?.len();
    if per_lang.iter().any(|f| f.len() != n) {
        return None;
    }
    let k = per_lang.len() as f64;
    Some((0..n).map(|i| per_lang.iter().map(|f| f[i]).sum::<f64>() / k).collect())
}

/// Greedy byte-knapsack: the hottest cacheable items that fit `capacity`.
/// An item's expected hit bytes per token are freq × size at cost size, so
/// ranking by freq alone is the optimal fractional ordering.
pub fn hot_set(freqs: &[f64], lay: &Layout, cacheable: &[bool], capacity: u64) -> Vec<bool> {
    let sizes = lay.item_sizes();
    let mut ranked: Vec<u32> = (0..lay.item_count() as u32)
        .filter(|&id| cacheable[lay.locate(id).1])
        .collect();
    ranked.sort_by(|&a, &b| freqs[b as usize].total_cmp(&freqs[a as usize]).then(a.cmp(&b)));
    let (hot, _) = ranked.iter().fold((vec![false; lay.item_count()], 0u64), |(mut hot, used), &id| {
        let size = sizes[id as usize] as u64;
        if used + size <= capacity {
            hot[id as usize] = true;
            (hot, used + size)
        } else {
            (hot, used)
        }
    });
    hot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::tests::{header, trace_from};

    #[test]
    fn frequencies_count_tokens() {
        let t = trace_from(header(4, 1, &[3]), |t, _, _, c| c == 0 || (c == 1 && t < 1));
        assert_eq!(activation_freqs(&t), vec![1.0, 0.25, 0.0]);
    }

    #[test]
    fn mean_requires_same_shape() {
        assert_eq!(mean_freqs(&[vec![1.0, 0.0], vec![0.0, 0.0]]), Some(vec![0.5, 0.0]));
        assert_eq!(mean_freqs(&[vec![1.0], vec![0.0, 0.0]]), None);
        assert_eq!(mean_freqs(&[]), None);
    }

    #[test]
    fn hot_set_takes_hottest_that_fit() {
        // Matrix 0: 3 channels of 100 B; matrix 1 (not cacheable): 2 of 200 B.
        let lay = Layout::new(&header(1, 1, &[3, 2]));
        let freqs = [0.2, 0.9, 0.5, 1.0, 1.0];
        let hot = hot_set(&freqs, &lay, &[true, false], 250);
        assert_eq!(hot, vec![false, true, true, false, false]);
    }
}
