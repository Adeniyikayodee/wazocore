//! `.satl` container: header types and framing (docs/satl-format.md).

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAGIC: &[u8; 4] = b"SATL";
pub const FORMAT: &str = "satl";
pub const VERSION: u32 = 1;
pub const PAGE: u64 = 4096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    pub source: Source,
    pub config: ModelConfig,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub model: String,
    pub files: Vec<SourceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    pub name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<Value>,
    pub rope_layout: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    /// Relative to the data section; a multiple of `PAGE`.
    pub offset: u64,
    pub length: u64,
    pub kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Codec {
    Q4 { group: usize },
    Q8 { group: usize },
    /// Lossless little-endian f32; for reference files that validate the
    /// runtime's forward pass, not for deployment.
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Part {
    pub source: String,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Columns { count: usize, parts: Vec<Part>, codec: Codec, column_bytes: usize, stride: usize },
    Rows { count: usize, cols: usize, source: String, codec: Codec, row_bytes: usize, stride: usize },
    Vector { len: usize, source: String },
}

pub fn align_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to) * to
}

/// Magic, length and padded header JSON. The data section starts at
/// `data_start(prefix.len())`.
pub fn encode_prefix(h: &Header) -> Vec<u8> {
    let json = serde_json::to_vec(h).expect("header serialises");
    let header_len = json.len() as u32;
    MAGIC.iter().copied().chain(header_len.to_le_bytes()).chain(json).collect()
}

pub fn data_start(prefix_len: usize) -> u64 {
    align_up(prefix_len as u64, PAGE)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    BadMagic,
    Truncated,
    BadHeader(String),
    Unsupported { format: String, version: u32 },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a .satl file"),
            Self::Truncated => write!(f, ".satl file is truncated"),
            Self::BadHeader(e) => write!(f, "bad .satl header: {e}"),
            Self::Unsupported { format, version } => {
                write!(f, "unsupported {format} v{version} (want {FORMAT} v{VERSION})")
            }
        }
    }
}

impl std::error::Error for FormatError {}

/// Parses the header from the start of a file; returns it with the data
/// section offset.
pub fn decode_prefix(bytes: &[u8]) -> Result<(Header, u64), FormatError> {
    if bytes.get(0..4).ok_or(FormatError::Truncated)? != MAGIC {
        return Err(FormatError::BadMagic);
    }
    let len: [u8; 4] = bytes.get(4..8).ok_or(FormatError::Truncated)?.try_into().expect("4 bytes");
    let end = 8 + u32::from_le_bytes(len) as usize;
    let json = bytes.get(8..end).ok_or(FormatError::Truncated)?;
    let h: Header = serde_json::from_slice(json).map_err(|e| FormatError::BadHeader(e.to_string()))?;
    if h.format != FORMAT || h.version != VERSION {
        return Err(FormatError::Unsupported { format: h.format, version: h.version });
    }
    Ok((h, data_start(end)))
}
