//! One decoder step of Llama-3. The KV cache is the only state that
//! outlives a step, and it is passed explicitly.

use atlas_satl::format::ModelConfig;

use crate::fast;
use crate::kernels::{KernelError, columns_gemv_ref, row, rows_gemv_ref};
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
