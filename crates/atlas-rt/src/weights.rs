//! Typed, borrowed views of the tensors in a `.satl` file. Building the
//! views is a pure function of the header and the data section bytes.

use std::fmt;

use atlas_satl::codec;
use atlas_satl::format::{Codec, Header, Kind, ModelConfig};

#[derive(Debug, Clone, Copy)]
pub struct Columns<'a> {
    pub count: usize,
    pub stride: usize,
    pub codec: Codec,
    /// Output rows of each part; the kernel output is their concatenation.
    pub rows: [usize; 3],
    pub parts: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct Rows<'a> {
    pub count: usize,
    pub cols: usize,
    pub stride: usize,
    pub codec: Codec,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub qkv: Columns<'a>,
    pub o: Columns<'a>,
    pub mlp_norm: Vec<f32>,
    pub gate_up: Columns<'a>,
    pub down: Columns<'a>,
}

#[derive(Debug, Clone)]
pub struct Weights<'a> {
    pub config: ModelConfig,
    pub embed: Rows<'a>,
    pub layers: Vec<Layer<'a>>,
    pub final_norm: Vec<f32>,
    pub lm_head: Rows<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightsError {
    Missing(String),
    WrongKind { tensor: String, expected: &'static str },
    WrongShape { tensor: String, detail: String },
    OutOfBounds(String),
}

impl fmt::Display for WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(t) => write!(f, "model lacks tensor {t}"),
            Self::WrongKind { tensor, expected } => write!(f, "{tensor} is not a {expected} tensor"),
            Self::WrongShape { tensor, detail } => write!(f, "{tensor}: {detail}"),
            Self::OutOfBounds(t) => write!(f, "{t} extends past the end of the file"),
        }
    }
}

impl std::error::Error for WeightsError {}

/// Resolves tensors by name against a data section.
struct Lookup<'h, 'a> {
    header: &'h Header,
    data: &'a [u8],
}

impl<'a> Lookup<'_, 'a> {
    fn entry(&self, name: &str) -> Result<(&Kind, &'a [u8]), WeightsError> {
        let t = self.header.tensors.iter().find(|t| t.name == name).ok_or_else(|| WeightsError::Missing(name.into()))?;
        let (start, end) = (t.offset as usize, (t.offset + t.length) as usize);
        let bytes = self.data.get(start..end).ok_or_else(|| WeightsError::OutOfBounds(name.into()))?;
        Ok((&t.kind, bytes))
    }

    fn columns(&self, name: &str, count: usize, rows: &[usize]) -> Result<Columns<'a>, WeightsError> {
        let (kind, data) = self.entry(name)?;
        let Kind::Columns { count: c, parts, codec, stride, .. } = kind else {
            return Err(WeightsError::WrongKind { tensor: name.into(), expected: "columns" });
        };
        if matches!(codec, Codec::Q8 { .. }) {
            return Err(WeightsError::WrongKind { tensor: name.into(), expected: "q4 or f32 columns" });
        }
        let got: Vec<usize> = parts.iter().map(|p| p.rows).collect();
        if *c != count || got != rows {
            let detail = format!("expected {count} columns of {rows:?}, found {c} of {got:?}");
            return Err(WeightsError::WrongShape { tensor: name.into(), detail });
        }
        let mut fixed = [0usize; 3];
        fixed[..rows.len()].copy_from_slice(rows);
        Ok(Columns { count, stride: *stride, codec: *codec, rows: fixed, parts: rows.len(), data })
    }

    fn rows(&self, name: &str, count: usize, cols: usize) -> Result<Rows<'a>, WeightsError> {
        let (kind, data) = self.entry(name)?;
        let Kind::Rows { count: n, cols: c, codec, stride, .. } = kind else {
            return Err(WeightsError::WrongKind { tensor: name.into(), expected: "rows" });
        };
        if matches!(codec, Codec::Q4 { .. }) {
            return Err(WeightsError::WrongKind { tensor: name.into(), expected: "q8 or f32 rows" });
        }
        if (*n, *c) != (count, cols) {
            let detail = format!("expected {count}x{cols}, found {n}x{c}");
            return Err(WeightsError::WrongShape { tensor: name.into(), detail });
        }
        Ok(Rows { count, cols, stride: *stride, codec: *codec, data })
    }

    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, WeightsError> {
        let (kind, data) = self.entry(name)?;
        match kind {
            Kind::Vector { len: n, .. } if *n == len => codec::decode_f32(data, len)
                .map_err(|e| WeightsError::WrongShape { tensor: name.into(), detail: e.to_string() }),
            Kind::Vector { len: n, .. } => {
                Err(WeightsError::WrongShape { tensor: name.into(), detail: format!("expected {len}, found {n}") })
            }
            _ => Err(WeightsError::WrongKind { tensor: name.into(), expected: "vector" }),
        }
    }
}

pub fn weights<'a>(header: &Header, data: &'a [u8]) -> Result<Weights<'a>, WeightsError> {
    let c = &header.config;
    let lk = Lookup { header, data };
    let (q, kv) = (c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
    let layers = (0..c.layers)
        .map(|l| {
            Ok(Layer {
                attn_norm: lk.vector(&format!("layers.{l}.attn_norm"), c.hidden)?,
                qkv: lk.columns(&format!("layers.{l}.attn.qkv"), c.hidden, &[q, kv, kv])?,
                o: lk.columns(&format!("layers.{l}.attn.o"), q, &[c.hidden])?,
                mlp_norm: lk.vector(&format!("layers.{l}.mlp_norm"), c.hidden)?,
                gate_up: lk.columns(&format!("layers.{l}.mlp.gate_up"), c.hidden, &[c.ffn, c.ffn])?,
                down: lk.columns(&format!("layers.{l}.mlp.down"), c.ffn, &[c.hidden])?,
            })
        })
        .collect::<Result<_, WeightsError>>()?;
    Ok(Weights {
        config: c.clone(),
        embed: lk.rows("embed", c.vocab, c.hidden)?,
        layers,
        final_norm: lk.vector("final_norm", c.hidden)?,
        lm_head: lk.rows("lm_head", c.vocab, c.hidden)?,
    })
}
