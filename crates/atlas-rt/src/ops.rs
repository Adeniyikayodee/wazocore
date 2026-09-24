//! Llama-3 elementwise and attention ops, matching Hugging Face
//! `LlamaForCausalLM` in f32.

use std::fmt;

use serde_json::Value;

use atlas_satl::format::ModelConfig;

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| g * (v * inv)).collect()
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU activation: silu(gate) ⊙ up.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

pub fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best }).0
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeError(String);

impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rope_scaling: {}", self.0)
    }
}

impl std::error::Error for RopeError {}

/// Per-pair inverse frequencies, with Llama-3 frequency scaling applied
/// when `rope_scaling.rope_type == "llama3"` (as HF computes them, in f32).
pub fn rope_inv_freq(cfg: &ModelConfig) -> Result<Vec<f32>, RopeError> {
    let d = cfg.head_dim;
    let base = cfg.rope_theta as f32;
    let plain: Vec<f32> = (0..d / 2).map(|i| 1.0 / base.powf((2 * i) as f32 / d as f32)).collect();
    let Some(s) = cfg.rope_scaling.as_ref() else { return Ok(plain) };
    let kind = s.get("rope_type").or_else(|| s.get("type")).and_then(Value::as_str).unwrap_or("default");
    if kind == "default" {
        return Ok(plain);
    }
    if kind != "llama3" {
        return Err(RopeError(format!("unsupported rope_type {kind}")));
    }
    let num = |k: &str| s.get(k).and_then(Value::as_f64).map(|v| v as f32).ok_or_else(|| RopeError(format!("missing {k}")));
    let (factor, low, high, old) = (
        num("factor")?,
        num("low_freq_factor")?,
        num("high_freq_factor")?,
        num("original_max_position_embeddings")?,
    );
    let (low_wavelen, high_wavelen) = (old / low, old / high);
    Ok(plain
        .into_iter()
        .map(|f| {
            let wavelen = 2.0 * std::f32::consts::PI / f;
            if wavelen < high_wavelen {
                f
            } else if wavelen > low_wavelen {
                f / factor
            } else {
                let smooth = (old / wavelen - low) / (high - low);
                (1.0 - smooth) * f / factor + smooth * f
            }
        })
        .collect())
}

/// HF `rotate_half` RoPE on each head of `x` at position `pos`: pairs are
/// (x[i], x[i + d/2]).
pub fn rope(x: &[f32], head_dim: usize, inv_freq: &[f32], pos: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let (cos, sin): (Vec<f32>, Vec<f32>) = inv_freq.iter().map(|&f| {
        let a = pos as f32 * f;
        (a.cos(), a.sin())
    }).unzip();
    x.chunks_exact(head_dim)
        .flat_map(|h| {
            let first = (0..half).map(|i| h[i] * cos[i] - h[i + half] * sin[i]);
            let second = (0..half).map(|i| h[i + half] * cos[i] + h[i] * sin[i]);
            first.chain(second).collect::<Vec<_>>()
        })
        .collect()
}

/// Causal GQA attention for one query position over cached keys/values
/// (`keys`/`values`: `len` rows of `n_kv_heads * head_dim`).
pub fn attention(q: &[f32], keys: &[f32], values: &[f32], len: usize, cfg: &ModelConfig) -> Vec<f32> {
    let (hd, kv_dim) = (cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
    let group = cfg.n_heads / cfg.n_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    (0..cfg.n_heads)
        .flat_map(|h| {
            let qh = &q[h * hd..(h + 1) * hd];
            let kv_off = (h / group) * hd;
            let scores: Vec<f32> = (0..len)
                .map(|t| {
                    let k = &keys[t * kv_dim + kv_off..][..hd];
                    qh.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let p = softmax(&scores);
            // Positions outer, head dimension inner: contiguous value rows,
            // and each output element still sums over t in order 0..len.
            (0..len).fold(vec![0.0f32; hd], |mut out, t| {
                let v = &values[t * kv_dim + kv_off..][..hd];
                out.iter_mut().zip(v).for_each(|(o, &vi)| *o += p[t] * vi);
                out
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(scaling: Option<Value>) -> ModelConfig {
        ModelConfig {
            hidden: 64,
            ffn: 128,
            layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            vocab: 10,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: scaling,
            rope_layout: "hf_rotate_half".into(),
        }
    }

    #[test]
    fn rmsnorm_unit_rms() {
        let y = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_at_zero_is_identity_and_preserves_norm() {
        let f = rope_inv_freq(&cfg(None)).unwrap();
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.0).collect();
        assert_eq!(rope(&x, 16, &f, 0), x);
        let y = rope(&x, 16, &f, 37);
        let n = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        assert!((n(&x) - n(&y)).abs() < 1e-3);
    }

    #[test]
    fn llama3_scaling_hits_all_three_bands() {
        let s = serde_json::json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 64});
        let plain = rope_inv_freq(&cfg(None)).unwrap();
        let scaled = rope_inv_freq(&cfg(Some(s))).unwrap();
        assert_eq!(scaled[0], plain[0]); // wavelength 6.3 < 16: unchanged
        assert!(scaled[1] < plain[1] && scaled[1] > plain[1] / 8.0); // smoothed
        assert!((scaled[7] - plain[7] / 8.0).abs() < 1e-12); // long: divided
    }

    #[test]
    fn single_key_attention_returns_its_value() {
        let c = cfg(None);
        let q = vec![0.3; 64];
        let k = vec![0.1; 32];
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out = attention(&q, &k, &v, 1, &c);
        // Heads 0,1 share kv head 0 (values 0..16); heads 2,3 share kv head 1.
        assert_eq!(&out[..16], &v[..16]);
        assert_eq!(&out[48..], &v[16..]);
    }

    #[test]
    fn rejects_unknown_scaling() {
        assert!(rope_inv_freq(&cfg(Some(serde_json::json!({"rope_type": "yarn"})))).is_err());
    }
}
