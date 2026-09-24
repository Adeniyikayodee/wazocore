//! Typed, borrowed views of the tensors in a `.satl` file. Building the
//! views is a pure function of the header and the data section bytes.

use std::fmt;

use atlas_satl::codec;
use atlas_satl::format::{Codec, Header, Kind, ModelConfig};

/// A column-major tensor. Only `Columns::new` builds one, so every value
/// satisfies the layout invariants the kernels rely on.
#[derive(Debug, Clone, Copy)]
pub struct Columns<'a> {
    pub(crate) count: usize,
    pub(crate) stride: usize,
    pub(crate) codec: Codec,
    /// Output rows of each part; the kernel output is their concatenation.
    pub(crate) rows: [usize; 3],
    pub(crate) parts: usize,
    pub(crate) data: &'a [u8],
}

/// A row-major tensor, built only through `Rows::new`.
#[derive(Debug, Clone, Copy)]
pub struct Rows<'a> {
    pub(crate) count: usize,
    pub(crate) cols: usize,
    pub(crate) stride: usize,
    pub(crate) codec: Codec,
    pub(crate) data: &'a [u8],
}

/// Why a byte range cannot be viewed as a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    Codec(&'static str),
    BadGroup { len: usize, group: usize },
    Parts(usize),
    StrideTooSmall { stride: usize, needed: usize },
    TooShort { needed: usize, have: usize },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(why) => write!(f, "{why}"),
            Self::BadGroup { len, group } => write!(f, "group {group} does not tile {len} values"),
            Self::Parts(n) => write!(f, "{n} parts (1 to 3 supported)"),
            Self::StrideTooSmall { stride, needed } => write!(f, "stride {stride} < {needed} encoded bytes"),
            Self::TooShort { needed, have } => write!(f, "needs {needed} bytes, segment has {have}"),
        }
    }
}

impl std::error::Error for LayoutError {}

fn check_group(len: usize, group: usize, max: usize) -> Result<(), LayoutError> {
    let ok = group > 0 && group.is_multiple_of(codec::BLOCK) && group <= max && len.is_multiple_of(group);
    if ok { Ok(()) } else { Err(LayoutError::BadGroup { len, group }) }
}

fn check_extent(count: usize, stride: usize, needed: usize, have: usize) -> Result<(), LayoutError> {
    if stride < needed {
        return Err(LayoutError::StrideTooSmall { stride, needed });
    }
    let total = count.checked_mul(stride).ok_or(LayoutError::TooShort { needed: usize::MAX, have })?;
    if total > have { Err(LayoutError::TooShort { needed: total, have }) } else { Ok(()) }
}

impl<'a> Columns<'a> {
    pub fn new(count: usize, stride: usize, codec: Codec, rows: &[usize], data: &'a [u8]) -> Result<Self, LayoutError> {
        if rows.is_empty() || rows.len() > 3 {
            return Err(LayoutError::Parts(rows.len()));
        }
        let needed: usize = match codec {
            Codec::Q4 { group } => rows
                .iter()
                .map(|&r| check_group(r, group, codec::MAX_Q4_GROUP).map(|()| codec::q4_bytes(r, group)))
                .sum::<Result<usize, _>>()?,
            Codec::F32 => rows.iter().map(|r| r * 4).sum(),
            Codec::Q8 { .. } => return Err(LayoutError::Codec("columns must be q4 or f32")),
        };
        check_extent(count, stride, needed, data.len())?;
        let mut fixed = [0usize; 3];
        fixed[..rows.len()].copy_from_slice(rows);
        Ok(Self { count, stride, codec, rows: fixed, parts: rows.len(), data })
    }

    /// Input channels (columns).
    pub fn count(&self) -> usize {
        self.count
    }

    /// Output length: all parts' rows.
    pub fn outputs(&self) -> usize {
        self.rows[..self.parts].iter().sum()
    }

    /// Bytes the tensor occupies (what a dense GEMV reads).
    pub fn bytes(&self) -> usize {
        self.count * self.stride
    }
}

impl<'a> Rows<'a> {
    pub fn new(count: usize, cols: usize, stride: usize, codec: Codec, data: &'a [u8]) -> Result<Self, LayoutError> {
        let needed = match codec {
            Codec::Q8 { group } => {
                check_group(cols, group, usize::MAX)?;
                codec::q8_bytes(cols, group)
            }
            Codec::F32 => cols * 4,
            Codec::Q4 { .. } => return Err(LayoutError::Codec("rows must be q8 or f32")),
        };
        check_extent(count, stride, needed, data.len())?;
        Ok(Self { count, cols, stride, codec, data })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn bytes(&self) -> usize {
        self.count * self.stride
    }
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
        let got: Vec<usize> = parts.iter().map(|p| p.rows).collect();
        if *c != count || got != rows {
            let detail = format!("expected {count} columns of {rows:?}, found {c} of {got:?}");
            return Err(WeightsError::WrongShape { tensor: name.into(), detail });
        }
        Columns::new(count, *stride, *codec, rows, data)
            .map_err(|e| WeightsError::WrongShape { tensor: name.into(), detail: e.to_string() })
    }

    fn rows(&self, name: &str, count: usize, cols: usize) -> Result<Rows<'a>, WeightsError> {
        let (kind, data) = self.entry(name)?;
        let Kind::Rows { count: n, cols: c, codec, stride, .. } = kind else {
            return Err(WeightsError::WrongKind { tensor: name.into(), expected: "rows" });
        };
        if (*n, *c) != (count, cols) {
            let detail = format!("expected {count}x{cols}, found {n}x{c}");
            return Err(WeightsError::WrongShape { tensor: name.into(), detail });
        }
        Rows::new(count, cols, *stride, *codec, data)
            .map_err(|e| WeightsError::WrongShape { tensor: name.into(), detail: e.to_string() })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_reject_layouts_the_kernels_cannot_read_safely() {
        let data = vec![0u8; 4 * 2304];
        let q4 = Codec::Q4 { group: 64 };
        assert!(Columns::new(4, 2304, q4, &[4096], &data).is_ok());
        assert_eq!(Columns::new(5, 2304, q4, &[4096], &data).unwrap_err(), LayoutError::TooShort { needed: 5 * 2304, have: 4 * 2304 });
        assert!(matches!(Columns::new(4, 2000, q4, &[4096], &data), Err(LayoutError::StrideTooSmall { .. })));
        assert!(matches!(Columns::new(1, 2304, Codec::Q4 { group: 256 }, &[4096], &data), Err(LayoutError::BadGroup { .. })));
        assert!(matches!(Columns::new(1, 2304, Codec::Q4 { group: 48 }, &[4096], &data), Err(LayoutError::BadGroup { .. })));
        assert!(matches!(Columns::new(1, 2304, Codec::Q8 { group: 32 }, &[4096], &data), Err(LayoutError::Codec(_))));
        assert!(matches!(Columns::new(1, 2304, q4, &[], &data), Err(LayoutError::Parts(0))));
        assert!(matches!(Columns::new(usize::MAX, 2304, q4, &[4096], &data), Err(LayoutError::TooShort { .. })));
    }

    #[test]
    fn rows_reject_bad_layouts() {
        let data = vec![0u8; 3 * 34];
        assert!(Rows::new(3, 32, 34, Codec::Q8 { group: 32 }, &data).is_ok());
        assert!(Rows::new(4, 32, 34, Codec::Q8 { group: 32 }, &data).is_err());
        assert!(Rows::new(3, 32, 34, Codec::Q4 { group: 32 }, &data).is_err());
        assert!(Rows::new(3, 40, 34, Codec::Q8 { group: 32 }, &data).is_err());
    }
}
