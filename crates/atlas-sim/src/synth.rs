//! Synthetic activation traces, for testing the simulator and for asking how
//! skewed real traces must be before storage streaming can work (Gate G1).
//!
//! Channel i fires with stationary probability p_i = sigmoid(mu + z_i + d_i):
//! - z_i ~ N(0, skew²) is shared by all languages (intrinsic importance);
//! - d_i ~ N(0, lang_skew²) for a `lang_frac` subset, drawn per language;
//! - mu is solved so the mean firing rate is 1 − sparsity.
//!
//! Tokens are a Markov chain: with probability `persist` a channel repeats
//! its previous state, otherwise it redraws from p_i (stationary p_i holds).

use crate::rng;
use crate::trace::{FORMAT, Header, Layout, MatrixSpec, Trace, TraceError, VERSION};

#[derive(Debug, Clone, PartialEq)]
pub struct SynthParams {
    pub lang: String,
    pub tokens: u64,
    pub layers: u32,
    pub matrices: Vec<MatrixSpec>,
    pub sparsity: f64,
    pub skew: f64,
    pub lang_frac: f64,
    pub lang_skew: f64,
    pub persist: f64,
    /// Shared across languages: fixes z_i.
    pub model_seed: u64,
    /// Fixes token draws and the stream of this trace.
    pub seed: u64,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Stable per-language seed derived from its code.
pub fn lang_seed(lang: &str) -> u64 {
    lang.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// Pre-sigmoid logit of every item, before the mean shift.
fn logits(p: &SynthParams, n_items: usize) -> Vec<f64> {
    let ls = lang_seed(&p.lang);
    (0..n_items as u64)
        .map(|i| {
            let z = p.skew * rng::normal(p.model_seed, i, 0);
            let lang_specific = rng::uniform(p.model_seed, i, 1) < p.lang_frac;
            let d = if lang_specific { p.lang_skew * rng::normal(ls, i, 2) } else { 0.0 };
            z + d
        })
        .collect()
}

/// Finds mu with mean(sigmoid(mu + x)) = target by bisection.
pub fn solve_shift(xs: &[f64], target: f64) -> f64 {
    let mean_at = |mu: f64| xs.iter().map(|x| sigmoid(mu + x)).sum::<f64>() / xs.len() as f64;
    (0..80)
        .fold((-40.0f64, 40.0f64), |(lo, hi), _| {
            let mid = 0.5 * (lo + hi);
            if mean_at(mid) < target { (mid, hi) } else { (lo, mid) }
        })
        .0
}

/// Stationary firing probability of every item.
pub fn probabilities(p: &SynthParams, n_items: usize) -> Vec<f64> {
    let xs = logits(p, n_items);
    let mu = solve_shift(&xs, 1.0 - p.sparsity);
    xs.iter().map(|x| sigmoid(mu + x)).collect()
}

pub fn generate(p: &SynthParams) -> Result<Trace, TraceError> {
    let header = Header {
        format: FORMAT.into(),
        version: VERSION,
        model: "synthetic".into(),
        lang: p.lang.clone(),
        source: format!(
            "synth skew={} lang_frac={} lang_skew={} persist={} seed={}",
            p.skew, p.lang_frac, p.lang_skew, p.persist, p.seed
        ),
        profile: format!("sparsity={}", p.sparsity),
        layers: p.layers,
        tokens: p.tokens,
        matrices: p.matrices.clone(),
    };
    let lay = Layout::new(&header);
    let probs = probabilities(p, lay.item_count());
    let stream = p.seed ^ lang_seed(&p.lang);
    let fires = |t: usize, id: u32| rng::uniform(stream, t as u64, id as u64) < probs[id as usize];
    let keeps = |t: usize, id: u32| rng::uniform(stream, t as u64, (id as u64) | 1 << 40) < p.persist;

    // Token t depends on token t-1, so words are built token by token.
    let words = (0..lay.tokens).fold(Vec::with_capacity(lay.total_words()), |mut words: Vec<u64>, t| {
        let prev_start = words.len().checked_sub(lay.total_words() / lay.tokens.max(1));
        for l in 0..lay.layers {
            for m in 0..lay.matrices() {
                let base = lay.word_index(t, l, m) - lay.word_index(t, 0, 0);
                for w in 0..lay.words_per_matrix(m) {
                    let word = (0..64usize)
                        .map(|b| w * 64 + b)
                        .filter(|&c| c < lay.channels[m])
                        .filter(|&c| {
                            let id = lay.item(l, m, c);
                            match prev_start {
                                Some(ps) if t > 0 && keeps(t, id) => {
                                    words[ps + base + c / 64] >> (c % 64) & 1 == 1
                                }
                                _ => fires(t, id),
                            }
                        })
                        .fold(0u64, |acc, c| acc | 1 << (c % 64));
                    words.push(word);
                }
            }
        }
        words
    });
    Trace::from_parts(header, words)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(persist: f64) -> SynthParams {
        SynthParams {
            lang: "yo".into(),
            tokens: 400,
            layers: 2,
            matrices: vec![MatrixSpec { name: "m".into(), channels: 256, bytes_per_channel: 1 }],
            sparsity: 0.5,
            skew: 1.5,
            lang_frac: 0.2,
            lang_skew: 2.0,
            persist,
            model_seed: 1,
            seed: 2,
        }
    }

    fn active_fraction(t: &Trace) -> f64 {
        let lay = &t.layout;
        let on: usize = (0..lay.tokens)
            .flat_map(|tk| (0..lay.layers).map(move |l| (tk, l)))
            .map(|(tk, l)| t.active_channels(tk, l, 0).count())
            .sum();
        on as f64 / (lay.tokens * lay.layers * lay.channels[0]) as f64
    }

    #[test]
    fn hits_target_sparsity() {
        for persist in [0.0, 0.7] {
            let f = active_fraction(&generate(&params(persist)).unwrap());
            assert!((f - 0.5).abs() < 0.03, "persist {persist}: active {f}");
        }
    }

    #[test]
    fn deterministic_and_language_dependent() {
        let a = generate(&params(0.3)).unwrap();
        assert_eq!(a, generate(&params(0.3)).unwrap());
        let ha = generate(&SynthParams { lang: "ha".into(), ..params(0.3) }).unwrap();
        assert_ne!(a, ha);
    }

    #[test]
    fn language_only_moves_language_specific_channels() {
        let n = 2000;
        let yo = probabilities(&params(0.0), n);
        let ha = probabilities(&SynthParams { lang: "ha".into(), ..params(0.0) }, n);
        let moved = yo.iter().zip(&ha).filter(|(a, b)| (*a - *b).abs() > 0.05).count();
        assert!(moved > n / 20 && moved < n / 2, "moved {moved}");
    }

    #[test]
    fn persistence_raises_token_to_token_agreement() {
        let agree = |t: &Trace| {
            (1..t.layout.tokens)
                .map(|tk| (0..256).filter(|&c| t.is_active(tk, 0, 0, c) == t.is_active(tk - 1, 0, 0, c)).count())
                .sum::<usize>()
        };
        assert!(agree(&generate(&params(0.8)).unwrap()) > agree(&generate(&params(0.0)).unwrap()));
    }
}
