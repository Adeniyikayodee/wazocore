//! Maps a Hugging Face Llama checkpoint onto `.satl` tensors: validates
//! source shapes, sizes every segment and assigns page-aligned offsets.

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;

use crate::codec;
use crate::format::{Codec, Kind, ModelConfig, PAGE, Part, TensorEntry, align_up};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub q4_group: usize,
    pub q8_group: usize,
    pub q4_fit: codec::Fit,
    /// Store every matrix as lossless f32 (reference files for validation).
    pub f32: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { q4_group: 64, q8_group: 32, q4_fit: codec::Fit::MinMax, f32: false }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    MissingConfig(&'static str),
    MissingTensor(String),
    BadShape { tensor: String, expected: Vec<usize>, actual: Vec<usize> },
    BadGroup { tensor: String, rows: usize, group: usize },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingConfig(k) => write!(f, "config.json lacks {k}"),
            Self::MissingTensor(t) => write!(f, "checkpoint lacks tensor {t}"),
            Self::BadShape { tensor, expected, actual } => {
                write!(f, "{tensor}: expected shape {expected:?}, found {actual:?}")
            }
            Self::BadGroup { tensor, rows, group } => {
                write!(f, "{tensor}: {rows} values not divisible into groups of {group}")
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// RoPE base and scaling. transformers >= 5 nests both under
/// `rope_parameters`; earlier versions use top-level `rope_theta` and
/// `rope_scaling`. A missing base is an error, never a silent default:
/// a wrong base corrupts every position but the first.
fn rope_from_hf(v: &Value) -> Result<(f64, Option<Value>), PlanError> {
    if let Some(p) = v.get("rope_parameters").filter(|p| p.is_object()) {
        let theta = p.get("rope_theta").and_then(Value::as_f64).ok_or(PlanError::MissingConfig("rope_parameters.rope_theta"))?;
        let scaled = p.get("rope_type").and_then(Value::as_str).is_some_and(|t| t != "default");
        return Ok((theta, scaled.then(|| p.clone())));
    }
    let theta = v.get("rope_theta").and_then(Value::as_f64).ok_or(PlanError::MissingConfig("rope_theta"))?;
    Ok((theta, v.get("rope_scaling").filter(|s| !s.is_null()).cloned()))
}

/// Reads the fields `.satl` needs from a Hugging Face `config.json`.
pub fn config_from_hf(v: &Value) -> Result<ModelConfig, PlanError> {
    let (rope_theta, rope_scaling) = rope_from_hf(v)?;
    let int = |k: &'static str| v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or(PlanError::MissingConfig(k));
    let (hidden, n_heads) = (int("hidden_size")?, int("num_attention_heads")?);
    Ok(ModelConfig {
        hidden,
        ffn: int("intermediate_size")?,
        layers: int("num_hidden_layers")?,
        n_heads,
        n_kv_heads: int("num_key_value_heads").unwrap_or(n_heads),
        head_dim: int("head_dim").unwrap_or(hidden / n_heads),
        vocab: int("vocab_size")?,
        rms_norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).ok_or(PlanError::MissingConfig("rms_norm_eps"))?,
        rope_theta,
        rope_scaling,
        rope_layout: "hf_rotate_half".into(),
    })
}

/// A tensor to write, before offsets are assigned.
/// Encoded bytes of `len` values under `codec`.
pub fn codec_bytes(codec: &Codec, len: usize) -> usize {
    match codec {
        Codec::Q4 { group } => codec::q4_bytes(len, *group),
        Codec::Q8 { group } => codec::q8_bytes(len, *group),
        Codec::F32 => len * 4,
    }
}

fn columns(name: String, parts: Vec<(String, usize)>, count: usize, codec: Codec, page_stride: bool) -> (String, Kind) {
    let column_bytes: usize = parts.iter().map(|(_, rows)| codec_bytes(&codec, *rows)).sum();
    let stride = if page_stride { align_up(column_bytes as u64, PAGE) as usize } else { column_bytes };
    let parts = parts.into_iter().map(|(source, rows)| Part { source, rows }).collect();
    (name, Kind::Columns { count, parts, codec, column_bytes, stride })
}

fn rows(name: &str, source: &str, count: usize, cols: usize, codec: Codec) -> (String, Kind) {
    let row_bytes = codec_bytes(&codec, cols);
    let kind = Kind::Rows { count, cols, source: source.into(), codec, row_bytes, stride: row_bytes };
    (name.into(), kind)
}

fn vector(name: String, source: String, len: usize) -> (String, Kind) {
    (name, Kind::Vector { len, source })
}

/// Source tensor name and expected [rows, cols] shape each kind reads.
pub fn sources(kind: &Kind) -> Vec<(String, Vec<usize>)> {
    match kind {
        Kind::Columns { count, parts, .. } => parts.iter().map(|p| (p.source.clone(), vec![p.rows, *count])).collect(),
        Kind::Rows { count, cols, source, .. } => vec![(source.clone(), vec![*count, *cols])],
        Kind::Vector { len, source } => vec![(source.clone(), vec![*len])],
    }
}

fn byte_length(kind: &Kind) -> u64 {
    match kind {
        Kind::Columns { count, stride, .. } | Kind::Rows { count, stride, .. } => (*count * *stride) as u64,
        Kind::Vector { len, .. } => (*len * 4) as u64,
    }
}

fn layout(cfg: &ModelConfig, o: &Options) -> Vec<(String, Kind)> {
    let (h, f) = (cfg.hidden, cfg.ffn);
    let (g, r) = if o.f32 { (Codec::F32, Codec::F32) } else { (Codec::Q4 { group: o.q4_group }, Codec::Q8 { group: o.q8_group }) };
    let q_rows = cfg.n_heads * cfg.head_dim;
    let kv_rows = cfg.n_kv_heads * cfg.head_dim;
    let src = |l: usize, s: &str| format!("model.layers.{l}.{s}.weight");
    let per_layer = (0..cfg.layers).flat_map(|l| {
        [
            vector(format!("layers.{l}.attn_norm"), src(l, "input_layernorm"), h),
            columns(
                format!("layers.{l}.attn.qkv"),
                vec![(src(l, "self_attn.q_proj"), q_rows), (src(l, "self_attn.k_proj"), kv_rows), (src(l, "self_attn.v_proj"), kv_rows)],
                h,
                g,
                false,
            ),
            columns(format!("layers.{l}.attn.o"), vec![(src(l, "self_attn.o_proj"), h)], q_rows, g, false),
            vector(format!("layers.{l}.mlp_norm"), src(l, "post_attention_layernorm"), h),
            columns(
                format!("layers.{l}.mlp.gate_up"),
                vec![(src(l, "mlp.gate_proj"), f), (src(l, "mlp.up_proj"), f)],
                h,
                g,
                true,
            ),
            columns(format!("layers.{l}.mlp.down"), vec![(src(l, "mlp.down_proj"), h)], f, g, false),
        ]
    });
    std::iter::once(rows("embed", "model.embed_tokens.weight", cfg.vocab, h, r))
        .chain(per_layer)
        .chain([
            vector("final_norm".into(), "model.norm.weight".into(), h),
            rows("lm_head", "lm_head.weight", cfg.vocab, h, r),
        ])
        .collect()
}

fn check_groups(name: &str, kind: &Kind) -> Result<(), PlanError> {
    let bad = |rows: usize, group: usize| group == 0 || !group.is_multiple_of(codec::BLOCK) || !rows.is_multiple_of(group);
    match kind {
        Kind::Columns { parts, codec: Codec::Q4 { group }, .. } => match parts.iter().find(|p| bad(p.rows, *group)) {
            Some(p) => Err(PlanError::BadGroup { tensor: name.into(), rows: p.rows, group: *group }),
            None => Ok(()),
        },
        Kind::Rows { cols, codec: Codec::Q8 { group }, .. } if bad(*cols, *group) => {
            Err(PlanError::BadGroup { tensor: name.into(), rows: *cols, group: *group })
        }
        _ => Ok(()),
    }
}

/// Source tensors (name, shape) a checkpoint must contain for `cfg`.
pub fn required_shapes(cfg: &ModelConfig, o: &Options) -> Vec<(String, Vec<usize>)> {
    layout(cfg, o).iter().flat_map(|(_, k)| sources(k)).collect()
}

/// Every output tensor with its offset, after checking the checkpoint's
/// tensor shapes (`shapes`: source name -> shape).
pub fn plan(cfg: &ModelConfig, shapes: &HashMap<String, Vec<usize>>, o: &Options) -> Result<Vec<TensorEntry>, PlanError> {
    let tensors = layout(cfg, o);
    for (name, kind) in &tensors {
        check_groups(name, kind)?;
        for (source, expected) in sources(kind) {
            let actual = shapes.get(&source).ok_or_else(|| PlanError::MissingTensor(source.clone()))?;
            if *actual != expected {
                return Err(PlanError::BadShape { tensor: source, expected, actual: actual.clone() });
            }
        }
    }
    let (entries, _) = tensors.into_iter().fold((Vec::new(), 0u64), |(mut acc, offset), (name, kind)| {
        let length = byte_length(&kind);
        acc.push(TensorEntry { name, offset, length, kind });
        (acc, align_up(offset + length, PAGE))
    });
    Ok(entries)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden: 128,
            ffn: 256,
            layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 64,
            vocab: 96,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: None,
            rope_layout: "hf_rotate_half".into(),
        }
    }

    pub fn shapes_for(cfg: &ModelConfig) -> HashMap<String, Vec<usize>> {
        required_shapes(cfg, &Options::default()).into_iter().collect()
    }

    fn n_atlas() -> ModelConfig {
        ModelConfig { hidden: 4096, ffn: 14336, layers: 32, n_heads: 32, n_kv_heads: 8, head_dim: 128, vocab: 128_256, ..tiny_config() }
    }

    #[test]
    fn n_atlas_bundle_sizes_match_spec() {
        let cfg = n_atlas();
        let entries = plan(&cfg, &shapes_for(&cfg), &Options::default()).unwrap();
        let get = |n: &str| entries.iter().find(|e| e.name == n).unwrap();
        let sizes = |n: &str| match &get(n).kind {
            Kind::Columns { column_bytes, stride, .. } => (*column_bytes, *stride),
            _ => panic!(),
        };
        assert_eq!(sizes("layers.0.mlp.gate_up"), (16128, 16384));
        assert_eq!(sizes("layers.0.mlp.down"), (2304, 2304));
        assert_eq!(sizes("layers.0.attn.qkv"), (3456, 3456));
        assert_eq!(sizes("layers.31.attn.o"), (2304, 2304));
        assert!(entries.iter().all(|e| e.offset % PAGE == 0));
        assert!(entries.windows(2).all(|w| w[1].offset >= w[0].offset + w[0].length));
        let total = entries.last().map(|e| e.offset + e.length).unwrap() as f64 / 1e9;
        assert!((5.0..5.5).contains(&total), "total {total} GB");
    }

    #[test]
    fn rejects_missing_or_misshapen_tensors() {
        let cfg = tiny_config();
        let shapes = shapes_for(&cfg);
        let without = |k: &str| shapes.iter().filter(|(n, _)| *n != k).map(|(n, s)| (n.clone(), s.clone())).collect();
        assert!(matches!(plan(&cfg, &without("lm_head.weight"), &Options::default()), Err(PlanError::MissingTensor(_))));
        let mut bad = shapes.clone();
        bad.insert("model.norm.weight".into(), vec![7]);
        assert!(matches!(plan(&cfg, &bad, &Options::default()), Err(PlanError::BadShape { .. })));
        let odd = Options { q4_group: 48, ..Options::default() };
        assert!(matches!(plan(&cfg, &shapes, &odd), Err(PlanError::BadGroup { .. })));
    }

    #[test]
    fn reads_hf_config() {
        let v = serde_json::json!({
            "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
            "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0}
        });
        let c = config_from_hf(&v).unwrap();
        assert_eq!((c.head_dim, c.n_kv_heads, c.vocab), (128, 8, 128_256));
        assert!(c.rope_scaling.is_some());
        assert!(matches!(config_from_hf(&serde_json::json!({})), Err(PlanError::MissingConfig(_))));
    }

    #[test]
    fn reads_transformers_v5_rope_parameters() {
        let v = serde_json::json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 256, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "llama3", "factor": 8.0}
        });
        let c = config_from_hf(&v).unwrap();
        assert_eq!(c.rope_theta, 500000.0);
        assert_eq!(c.rope_scaling.unwrap()["factor"], 8.0);
    }

    #[test]
    fn missing_rope_theta_is_an_error_not_a_default() {
        let v = serde_json::json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 256, "rms_norm_eps": 1e-5
        });
        assert_eq!(config_from_hf(&v), Err(PlanError::MissingConfig("rope_theta")));
    }
}
