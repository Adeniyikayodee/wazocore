//! Activation traces (`docs/trace-format.md`): which input channels of each
//! sparsified matrix were active, per token and layer.

use std::fmt;

use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 4] = b"ATRC";
pub const FORMAT: &str = "atlas-trace";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixSpec {
    pub name: String,
    pub channels: u32,
    pub bytes_per_channel: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    pub model: String,
    pub lang: String,
    pub source: String,
    pub profile: String,
    pub layers: u32,
    pub tokens: u64,
    pub matrices: Vec<MatrixSpec>,
}

/// N-ATLAS MLP bundles at 4.5 bits/weight (PROJECT-PLAN-B §3.2).
pub fn n_atlas_mlp() -> Vec<MatrixSpec> {
    vec![
        MatrixSpec { name: "gate_up".into(), channels: 4096, bytes_per_channel: 16128 },
        MatrixSpec { name: "down".into(), channels: 14336, bytes_per_channel: 2304 },
    ]
}

/// Index arithmetic derived from a header. Items are (layer, matrix, channel)
/// triples numbered densely, layer-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub tokens: usize,
    pub layers: usize,
    pub channels: Vec<usize>,
    pub bytes: Vec<u32>,
    word_offset: Vec<usize>,
    words_per_layer: usize,
    item_offset: Vec<usize>,
    items_per_layer: usize,
}

fn prefix_sums(xs: &[usize]) -> (Vec<usize>, usize) {
    let offsets = xs
        .iter()
        .scan(0, |acc, &x| {
            let start = *acc;
            *acc += x;
            Some(start)
        })
        .collect();
    (offsets, xs.iter().sum())
}

impl Layout {
    pub fn new(h: &Header) -> Self {
        let channels: Vec<usize> = h.matrices.iter().map(|m| m.channels as usize).collect();
        let words: Vec<usize> = channels.iter().map(|c| c.div_ceil(64)).collect();
        let (word_offset, words_per_layer) = prefix_sums(&words);
        let (item_offset, items_per_layer) = prefix_sums(&channels);
        Self {
            tokens: h.tokens as usize,
            layers: h.layers as usize,
            bytes: h.matrices.iter().map(|m| m.bytes_per_channel).collect(),
            channels,
            word_offset,
            words_per_layer,
            item_offset,
            items_per_layer,
        }
    }

    pub fn matrices(&self) -> usize {
        self.channels.len()
    }

    pub fn words_per_matrix(&self, m: usize) -> usize {
        self.channels[m].div_ceil(64)
    }

    pub fn total_words(&self) -> usize {
        self.tokens * self.layers * self.words_per_layer
    }

    pub fn word_index(&self, t: usize, l: usize, m: usize) -> usize {
        (t * self.layers + l) * self.words_per_layer + self.word_offset[m]
    }

    pub fn item_count(&self) -> usize {
        self.layers * self.items_per_layer
    }

    pub fn item(&self, l: usize, m: usize, c: usize) -> u32 {
        (l * self.items_per_layer + self.item_offset[m] + c) as u32
    }

    /// Inverse of `item`.
    pub fn locate(&self, id: u32) -> (usize, usize, usize) {
        let id = id as usize;
        let (l, rest) = (id / self.items_per_layer, id % self.items_per_layer);
        let m = self.item_offset.iter().rposition(|&o| o <= rest).unwrap_or(0);
        (l, m, rest - self.item_offset[m])
    }

    /// Byte size of every item, indexed by item id.
    pub fn item_sizes(&self) -> Vec<u32> {
        let per_layer: Vec<u32> = self
            .channels
            .iter()
            .zip(&self.bytes)
            .flat_map(|(&c, &b)| std::iter::repeat_n(b, c))
            .collect();
        per_layer.repeat(self.layers)
    }

    pub fn same_shape(&self, other: &Layout) -> bool {
        self.layers == other.layers && self.channels == other.channels && self.bytes == other.bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    pub header: Header,
    pub layout: Layout,
    words: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    BadMagic,
    Truncated,
    BadHeader(String),
    Unsupported { format: String, version: u32 },
    EmptyShape,
    LengthMismatch { expected_words: usize, actual_words: usize },
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not an atlas trace (bad magic)"),
            Self::Truncated => write!(f, "trace file is truncated"),
            Self::BadHeader(e) => write!(f, "bad trace header: {e}"),
            Self::Unsupported { format, version } => {
                write!(f, "unsupported trace {format} v{version} (want {FORMAT} v{VERSION})")
            }
            Self::EmptyShape => write!(f, "trace has no layers, matrices or channels"),
            Self::LengthMismatch { expected_words, actual_words } => {
                write!(f, "payload has {actual_words} words, header implies {expected_words}")
            }
        }
    }
}

impl std::error::Error for TraceError {}

/// Iterates the set bit positions of a bitmap, below `limit`.
pub fn set_bits(words: &[u64], limit: usize) -> impl Iterator<Item = usize> + '_ {
    words
        .iter()
        .enumerate()
        .flat_map(|(i, &w)| {
            std::iter::successors((w != 0).then_some(w), |&w| {
                let rest = w & (w - 1);
                (rest != 0).then_some(rest)
            })
            .map(move |w| i * 64 + w.trailing_zeros() as usize)
        })
        .take_while(move |&c| c < limit)
}

impl Trace {
    pub fn from_parts(header: Header, words: Vec<u64>) -> Result<Self, TraceError> {
        if header.format != FORMAT || header.version != VERSION {
            return Err(TraceError::Unsupported { format: header.format, version: header.version });
        }
        if header.layers == 0 || header.matrices.is_empty() || header.matrices.iter().any(|m| m.channels == 0)
        {
            return Err(TraceError::EmptyShape);
        }
        let layout = Layout::new(&header);
        if words.len() != layout.total_words() {
            return Err(TraceError::LengthMismatch {
                expected_words: layout.total_words(),
                actual_words: words.len(),
            });
        }
        Ok(Self { header, layout, words })
    }

    pub fn bitmap(&self, t: usize, l: usize, m: usize) -> &[u64] {
        let start = self.layout.word_index(t, l, m);
        &self.words[start..start + self.layout.words_per_matrix(m)]
    }

    pub fn is_active(&self, t: usize, l: usize, m: usize, c: usize) -> bool {
        self.words[self.layout.word_index(t, l, m) + c / 64] >> (c % 64) & 1 == 1
    }

    pub fn active_channels(&self, t: usize, l: usize, m: usize) -> impl Iterator<Item = usize> + '_ {
        set_bits(self.bitmap(t, l, m), self.layout.channels[m])
    }
}

pub fn encode(trace: &Trace) -> Vec<u8> {
    let json = serde_json::to_vec(&trace.header).expect("header serialises");
    let pad = (8 - (8 + json.len()) % 8) % 8;
    let header_len = (json.len() + pad) as u32;
    MAGIC
        .iter()
        .copied()
        .chain(header_len.to_le_bytes())
        .chain(json)
        .chain(std::iter::repeat_n(b' ', pad))
        .chain(trace.words.iter().flat_map(|w| w.to_le_bytes()))
        .collect()
}

pub fn decode(bytes: &[u8]) -> Result<Trace, TraceError> {
    let magic = bytes.get(0..4).ok_or(TraceError::Truncated)?;
    if magic != MAGIC {
        return Err(TraceError::BadMagic);
    }
    let len_bytes: [u8; 4] = bytes.get(4..8).ok_or(TraceError::Truncated)?.try_into().expect("4 bytes");
    let header_end = 8 + u32::from_le_bytes(len_bytes) as usize;
    let header_json = bytes.get(8..header_end).ok_or(TraceError::Truncated)?;
    let header: Header =
        serde_json::from_slice(header_json).map_err(|e| TraceError::BadHeader(e.to_string()))?;
    let payload = &bytes[header_end..];
    let (chunks, rest) = payload.as_chunks::<8>();
    if !rest.is_empty() {
        return Err(TraceError::Truncated);
    }
    let words = chunks.iter().map(|&c| u64::from_le_bytes(c)).collect();
    Trace::from_parts(header, words)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn header(tokens: u64, layers: u32, channels: &[u32]) -> Header {
        Header {
            format: FORMAT.into(),
            version: VERSION,
            model: "test".into(),
            lang: "yo".into(),
            source: "unit".into(),
            profile: "none".into(),
            layers,
            tokens,
            matrices: channels
                .iter()
                .enumerate()
                .map(|(i, &c)| MatrixSpec { name: format!("m{i}"), channels: c, bytes_per_channel: 100 * (i as u32 + 1) })
                .collect(),
        }
    }

    /// Builds a trace from a predicate over (token, layer, matrix, channel).
    pub fn trace_from(h: Header, active: impl Fn(usize, usize, usize, usize) -> bool) -> Trace {
        let lay = Layout::new(&h);
        let words = (0..lay.tokens)
            .flat_map(|t| (0..lay.layers).map(move |l| (t, l)))
            .flat_map(|(t, l)| (0..lay.matrices()).map(move |m| (t, l, m)))
            .flat_map(|(t, l, m)| {
                (0..lay.words_per_matrix(m)).map(move |w| (t, l, m, w))
            })
            .map(|(t, l, m, w)| {
                (0..64)
                    .filter(|b| {
                        let c = w * 64 + b;
                        c < lay.channels[m] && active(t, l, m, c)
                    })
                    .fold(0u64, |acc, b| acc | 1 << b)
            })
            .collect();
        Trace::from_parts(h, words).unwrap()
    }

    #[test]
    fn round_trips_through_bytes() {
        let t = trace_from(header(3, 2, &[70, 5]), |t, l, m, c| (t + l + m + c) % 3 == 0);
        let bytes = encode(&t);
        assert_eq!((8 + u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize) % 8, 0);
        assert_eq!(decode(&bytes).unwrap(), t);
    }

    #[test]
    fn reads_back_active_channels() {
        let t = trace_from(header(2, 1, &[130]), |t, _, _, c| c == 3 + t || c == 129);
        assert_eq!(t.active_channels(0, 0, 0).collect::<Vec<_>>(), vec![3, 129]);
        assert!(t.is_active(1, 0, 0, 4) && !t.is_active(1, 0, 0, 3));
    }

    #[test]
    fn items_round_trip() {
        let lay = Layout::new(&header(1, 3, &[70, 5]));
        assert_eq!(lay.item_count(), 225);
        assert_eq!(lay.locate(lay.item(2, 1, 4)), (2, 1, 4));
        assert_eq!(lay.locate(lay.item(1, 0, 69)), (1, 0, 69));
        let sizes = lay.item_sizes();
        assert_eq!((sizes.len(), sizes[lay.item(1, 1, 0) as usize]), (225, 200));
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(decode(b"NOPE\0\0\0\0"), Err(TraceError::BadMagic));
        assert_eq!(decode(b"ATR"), Err(TraceError::Truncated));
        let h = header(2, 1, &[64]);
        assert!(matches!(Trace::from_parts(h, vec![0]), Err(TraceError::LengthMismatch { .. })));
        assert_eq!(Trace::from_parts(header(1, 0, &[64]), vec![]), Err(TraceError::EmptyShape));
    }

    #[test]
    fn n_atlas_token_is_73_7_kb() {
        let h = Header { matrices: n_atlas_mlp(), layers: 32, tokens: 1, ..header(1, 1, &[1]) };
        assert_eq!(Layout::new(&h).total_words() * 8, 73_728);
    }
}
