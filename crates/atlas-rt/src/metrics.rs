//! Fidelity metrics between two logit vectors (PROJECT-PLAN-B §4.2), in f64
//! via log-softmax for stability at a 128k vocabulary.

pub fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = logits.iter().map(|&l| (l as f64 - max).exp()).sum::<f64>().ln() + max;
    logits.iter().map(|&l| l as f64 - lse).collect()
}

/// KL(p ‖ q) in nats, where p and q are the softmaxes of the logits.
pub fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p_logits), log_softmax(q_logits));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum::<f64>().max(0.0)
}

/// Negative log-likelihood of `token` under the logits.
pub fn nll(logits: &[f32], token: usize) -> f64 {
    -log_softmax(logits)[token]
}

/// One perplexity chunk: the tokens to run and which predictions count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PplChunk {
    pub tokens: Vec<u32>,
    /// (logit position, target token) pairs within the chunk.
    pub scored: Vec<(usize, u32)>,
}

/// llama.cpp's perplexity protocol: split `tokens` into chunks of `n_ctx`,
/// replace each chunk's first token with `bos`, and score predictions for
/// the second half of the chunk.
pub fn ppl_chunks(tokens: &[u32], n_ctx: usize, max_chunks: usize, bos: u32) -> Vec<PplChunk> {
    tokens
        .chunks_exact(n_ctx)
        .take(max_chunks)
        .map(|c| {
            let tokens: Vec<u32> = std::iter::once(bos).chain(c[1..].iter().copied()).collect();
            let scored = (n_ctx / 2..n_ctx - 1).map(|i| (i, tokens[i + 1])).collect();
            PplChunk { tokens, scored }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_is_zero_for_identical_and_shift_invariant() {
        let a = [1.0, 2.0, 3.0];
        assert!(kl(&a, &a).abs() < 1e-12);
        assert!(kl(&a, &[11.0, 12.0, 13.0]).abs() < 1e-12);
    }

    #[test]
    fn kl_matches_closed_form() {
        // p = softmax(ln 3, 0) = (0.75, 0.25); q = (0.5, 0.5).
        let p = [3.0f32.ln(), 0.0];
        let want = 0.75 * (0.75f64 / 0.5).ln() + 0.25 * (0.25f64 / 0.5).ln();
        assert!((kl(&p, &[0.0, 0.0]) - want).abs() < 1e-6);
    }

    #[test]
    fn ppl_chunks_follow_llama_cpp() {
        let toks: Vec<u32> = (10..30).collect();
        let c = ppl_chunks(&toks, 8, 10, 1);
        assert_eq!(c.len(), 2); // 20 tokens -> 2 full chunks of 8
        assert_eq!(c[1].tokens, vec![1, 19, 20, 21, 22, 23, 24, 25]);
        assert_eq!(c[0].scored, vec![(4, 15), (5, 16), (6, 17)]);
        assert_eq!(ppl_chunks(&toks, 8, 1, 1).len(), 1);
    }

    #[test]
    fn nll_of_uniform_is_log_n() {
        assert!((nll(&[0.0; 4], 2) - 4f64.ln()).abs() < 1e-12);
    }
}
