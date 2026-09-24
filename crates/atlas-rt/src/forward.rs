//! One decoder step of Llama-3. The KV cache is the only state that
//! outlives a step, and it is passed explicitly.

use atlas_satl::format::ModelConfig;

use rayon::prelude::*;

use crate::kernels::{KernelError, columns_gemv_ref, row, rows_gemv_ref};
use crate::{fast, gemm};
use crate::ops::{add, attention, rmsnorm, rope, swiglu};
use crate::weights::{Columns, Rows, Weights};

/// Keys and values per layer, `n_kv_heads * head_dim` floats per position.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    kv_dim: usize,
    capacity: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn new(cfg: &ModelConfig, capacity: usize) -> Self {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        Self {
            kv_dim,
            capacity,
            keys: vec![Vec::with_capacity(capacity * kv_dim); cfg.layers],
            values: vec![Vec::with_capacity(capacity * kv_dim); cfg.layers],
            len: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }
}

/// Which kernel implementations a step uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernels {
    /// Scalar, single-threaded: the spec.
    Reference,
    /// Parallel and vectorised; checked against `Reference`.
    Fast,
}

/// Everything a step needs besides the weights: RoPE frequencies, kernel
/// choice and the active input channels of each matrix (all = dense).
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub inv_freq: Vec<f32>,
    pub kernels: Kernels,
    pub all_hidden: Vec<u32>,
    pub all_q: Vec<u32>,
    pub all_ffn: Vec<u32>,
}

impl Plan {
    pub fn dense(cfg: &ModelConfig, inv_freq: Vec<f32>, kernels: Kernels) -> Self {
        let range = |n: usize| (0..n as u32).collect();
        Self { inv_freq, kernels, all_hidden: range(cfg.hidden), all_q: range(cfg.n_heads * cfg.head_dim), all_ffn: range(cfg.ffn) }
    }

    fn columns(&self, w: &Columns, x: &[f32], active: &[u32]) -> Result<Vec<f32>, KernelError> {
        match self.kernels {
            Kernels::Reference => columns_gemv_ref(w, x, active),
            Kernels::Fast => fast::columns_gemv(w, x, active),
        }
    }

    fn rows(&self, w: &Rows, x: &[f32]) -> Result<Vec<f32>, KernelError> {
        match self.kernels {
            Kernels::Reference => rows_gemv_ref(w, x),
            Kernels::Fast => fast::rows_gemv(w, x),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    CacheFull(usize),
    BadToken { token: u32, vocab: usize },
    Kernel(KernelError),
}

impl From<KernelError> for StepError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheFull(n) => write!(f, "KV cache full at {n} positions"),
            Self::BadToken { token, vocab } => write!(f, "token {token} outside vocabulary of {vocab}"),
            Self::Kernel(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StepError {}

/// Runs token `token` at position `kv.len`, appends its K/V, returns logits.
pub fn step(w: &Weights, plan: &Plan, kv: &mut KvCache, token: u32) -> Result<Vec<f32>, StepError> {
    let c = &w.config;
    if kv.is_full() {
        return Err(StepError::CacheFull(kv.len));
    }
    if token as usize >= c.vocab {
        return Err(StepError::BadToken { token, vocab: c.vocab });
    }
    let eps = c.rms_norm_eps as f32;
    let (q_dim, kv_dim) = (c.n_heads * c.head_dim, kv.kv_dim);
    let pos = kv.len;
    let mut x = row(&w.embed, token as usize)?;
    for (l, layer) in w.layers.iter().enumerate() {
        let h = rmsnorm(&x, &layer.attn_norm, eps);
        let qkv = plan.columns(&layer.qkv, &h, &plan.all_hidden)?;
        let q = rope(&qkv[..q_dim], c.head_dim, &plan.inv_freq, pos);
        let k = rope(&qkv[q_dim..q_dim + kv_dim], c.head_dim, &plan.inv_freq, pos);
        kv.keys[l].extend_from_slice(&k);
        kv.values[l].extend_from_slice(&qkv[q_dim + kv_dim..]);
        let att = attention(&q, &kv.keys[l], &kv.values[l], pos + 1, c);
        x = add(&x, &plan.columns(&layer.o, &att, &plan.all_q)?);

        let h = rmsnorm(&x, &layer.mlp_norm, eps);
        let gu = plan.columns(&layer.gate_up, &h, &plan.all_hidden)?;
        let a = swiglu(&gu[..c.ffn], &gu[c.ffn..]);
        x = add(&x, &plan.columns(&layer.down, &a, &plan.all_ffn)?);
    }
    kv.len += 1;
    Ok(plan.rows(&w.lm_head, &rmsnorm(&x, &w.final_norm, eps))?)
}

/// Which logits `prefill` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// Only the last token's (generation).
    Last,
    /// Every token's (perplexity, fidelity).
    All,
}

/// Tokens per batched forward pass: bounds activation memory, not results.
pub const PREFILL_BATCH: usize = 64;

/// Runs `tokens` at positions `kv.len ..` and returns logits per `want`.
/// With `Kernels::Fast` tokens go through the model in batches, sharing
/// each unpacked weight tile; results equal calling `step` token by token,
/// bit for bit. `Kernels::Reference` simply calls `step`.
pub fn prefill(w: &Weights, plan: &Plan, kv: &mut KvCache, tokens: &[u32], want: Want) -> Result<Vec<Vec<f32>>, StepError> {
    if kv.len + tokens.len() > kv.capacity {
        return Err(StepError::CacheFull(kv.capacity));
    }
    if let Some(&token) = tokens.iter().find(|&&t| t as usize >= w.config.vocab) {
        return Err(StepError::BadToken { token, vocab: w.config.vocab });
    }
    let logits = match plan.kernels {
        Kernels::Reference => tokens.iter().map(|&t| step(w, plan, kv, t)).collect::<Result<Vec<_>, _>>()?,
        Kernels::Fast => {
            let batches: Vec<&[u32]> = tokens.chunks(PREFILL_BATCH).collect();
            let last = batches.len().saturating_sub(1);
            batches
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let need = match want {
                        Want::All => Logits::All,
                        Want::Last if i == last => Logits::Last,
                        Want::Last => Logits::None,
                    };
                    forward_batch(w, plan, kv, b, need)
                })
                .collect::<Result<Vec<_>, _>>()?
                .concat()
        }
    };
    Ok(match want {
        Want::All => logits,
        Want::Last => logits.into_iter().last().into_iter().collect(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Logits {
    None,
    Last,
    All,
}

/// One batch through every layer; mirrors `step` token by token.
fn forward_batch(w: &Weights, plan: &Plan, kv: &mut KvCache, tokens: &[u32], need: Logits) -> Result<Vec<Vec<f32>>, StepError> {
    let c = &w.config;
    let eps = c.rms_norm_eps as f32;
    let (q_dim, kv_dim, pos0) = (c.n_heads * c.head_dim, kv.kv_dim, kv.len);
    let mut xs = tokens.iter().map(|&t| row(&w.embed, t as usize)).collect::<Result<Vec<_>, _>>()?;
    for (l, layer) in w.layers.iter().enumerate() {
        let hs: Vec<Vec<f32>> = xs.iter().map(|x| rmsnorm(x, &layer.attn_norm, eps)).collect();
        let qkv = gemm::columns_gemm(&layer.qkv, &hs, &plan.all_hidden)?;
        let qs: Vec<Vec<f32>> = qkv.iter().enumerate().map(|(t, v)| rope(&v[..q_dim], c.head_dim, &plan.inv_freq, pos0 + t)).collect();
        for (t, v) in qkv.iter().enumerate() {
            kv.keys[l].extend(rope(&v[q_dim..q_dim + kv_dim], c.head_dim, &plan.inv_freq, pos0 + t));
            kv.values[l].extend_from_slice(&v[q_dim + kv_dim..]);
        }
        let (keys, values) = (&kv.keys[l], &kv.values[l]);
        let att: Vec<Vec<f32>> = qs.par_iter().enumerate().map(|(t, q)| attention(q, keys, values, pos0 + t + 1, c)).collect();
        let o = gemm::columns_gemm(&layer.o, &att, &plan.all_q)?;
        xs = xs.iter().zip(&o).map(|(x, o)| add(x, o)).collect();

        let hs: Vec<Vec<f32>> = xs.iter().map(|x| rmsnorm(x, &layer.mlp_norm, eps)).collect();
        let gu = gemm::columns_gemm(&layer.gate_up, &hs, &plan.all_hidden)?;
        let acts: Vec<Vec<f32>> = gu.iter().map(|v| swiglu(&v[..c.ffn], &v[c.ffn..])).collect();
        let d = gemm::columns_gemm(&layer.down, &acts, &plan.all_ffn)?;
        xs = xs.iter().zip(&d).map(|(x, d)| add(x, d)).collect();
    }
    kv.len += tokens.len();
    let normed = |x: &Vec<f32>| rmsnorm(x, &w.final_norm, eps);
    Ok(match need {
        Logits::None => Vec::new(),
        Logits::Last => vec![fast::rows_gemv(&w.lm_head, &normed(xs.last().expect("non-empty batch")))?],
        Logits::All => gemm::rows_gemm(&w.lm_head, &xs.iter().map(normed).collect::<Vec<_>>())?,
    })
}
