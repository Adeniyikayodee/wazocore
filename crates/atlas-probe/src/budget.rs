//! Byte budget of one decoded token (PROJECT-PLAN-B §2). Decode is
//! memory-bandwidth-bound, so bytes read per token sets the speed ceiling.

/// Transformer dimensions of a Llama-3-style decoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub hidden: u64,
    pub ffn: u64,
    pub layers: u64,
    pub n_heads: u64,
    pub n_kv_heads: u64,
    pub head_dim: u64,
    pub vocab: u64,
}

/// N-ATLAS is a Llama-3-8B fine-tune.
pub const N_ATLAS: Shape = Shape {
    hidden: 4096,
    ffn: 14336,
    layers: 32,
    n_heads: 32,
    n_kv_heads: 8,
    head_dim: 128,
    vocab: 128_256,
};

/// Bits per weight for the sparsifiable projections and the dense lm_head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Precision {
    pub weight_bits: f64,
    pub head_bits: f64,
}

/// Group-wise 4-bit (≈ Q4_K_M) with a Q6_K lm_head.
pub const Q4_GROUP: Precision = Precision { weight_bits: 4.5, head_bits: 6.56 };

/// Q, K, V and O projections across all layers.
pub fn attn_params(s: &Shape) -> u64 {
    let q_or_o = s.hidden * s.n_heads * s.head_dim;
    let k_or_v = s.hidden * s.n_kv_heads * s.head_dim;
    s.layers * 2 * (q_or_o + k_or_v)
}

/// gate, up and down projections across all layers.
pub fn mlp_params(s: &Shape) -> u64 {
    s.layers * 3 * s.hidden * s.ffn
}

pub fn head_params(s: &Shape) -> u64 {
    s.vocab * s.hidden
}

/// Weight bytes streamed per decoded token when a fraction `sparsity` of
/// attention and MLP input channels is skipped. The lm_head stays dense; the
/// embedding lookup (one row) is negligible. `None` outside [0, 1).
pub fn bytes_per_token(s: &Shape, p: &Precision, sparsity: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&sparsity) {
        return None;
    }
    let sparsifiable = (attn_params(s) + mlp_params(s)) as f64 * p.weight_bits / 8.0;
    let head = head_params(s) as f64 * p.head_bits / 8.0;
    Some(sparsifiable * (1.0 - sparsity) + head)
}

/// Upper bound on decode tokens/s at a given read bandwidth (bytes/s).
pub fn decode_ceiling_tok_s(bytes_per_token: f64, bandwidth: f64) -> Option<f64> {
    (bytes_per_token > 0.0 && bandwidth.is_finite()).then(|| bandwidth / bytes_per_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_counts_match_llama3_8b() {
        assert_eq!(attn_params(&N_ATLAS), 1_342_177_280);
        assert_eq!(mlp_params(&N_ATLAS), 5_637_144_576);
        assert_eq!(head_params(&N_ATLAS), 525_336_576);
    }

    #[test]
    fn bytes_per_token_matches_plan_table() {
        let gb = |sp| bytes_per_token(&N_ATLAS, &Q4_GROUP, sp).unwrap() / 1e9;
        assert!((gb(0.0) - 4.36).abs() < 0.005);
        assert!((gb(0.25) - 3.38).abs() < 0.005);
        assert!((gb(0.40) - 2.79).abs() < 0.005);
        assert!((gb(0.50) - 2.39).abs() < 0.005);
        assert!(bytes_per_token(&N_ATLAS, &Q4_GROUP, 1.0).is_none());
    }

    #[test]
    fn dense_ceiling_at_8_gbps() {
        let b = bytes_per_token(&N_ATLAS, &Q4_GROUP, 0.0).unwrap();
        let t = decode_ceiling_tok_s(b, 8e9).unwrap();
        assert!((t - 1.84).abs() < 0.01);
    }
}
