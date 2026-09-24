//! Encodes and decodes `.satl` tensor segments from source matrices. Pure:
//! a segment is a function of the source bytes and its `Kind`, and ranges
//! of columns/rows can be encoded independently (the shell parallelises).

use std::fmt;
use std::ops::Range;

use half::{bf16, f16};

use crate::codec::{self, CodecError};
use crate::format::{Codec, Kind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// A row-major source tensor viewed as `rows × cols` (vectors: 1 × len).
#[derive(Debug, Clone, Copy)]
pub struct Matrix<'a> {
    pub rows: usize,
    pub cols: usize,
    pub dtype: Dtype,
    pub data: &'a [u8],
}

impl Matrix<'_> {
    pub fn get(&self, r: usize, c: usize) -> f32 {
        let i = (r * self.cols + c) * self.dtype.size();
        let b = &self.data[i..i + self.dtype.size()];
        match self.dtype {
            Dtype::Bf16 => bf16::from_le_bytes([b[0], b[1]]).to_f32(),
            Dtype::F16 => f16::from_le_bytes([b[0], b[1]]).to_f32(),
            Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }

    pub fn row(&self, r: usize) -> Vec<f32> {
        (0..self.cols).map(|c| self.get(r, c)).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EncodeError {
    Codec(CodecError),
    WrongSources { expected: usize, given: usize },
    CodecMismatch(&'static str),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(e) => write!(f, "{e}"),
            Self::WrongSources { expected, given } => write!(f, "tensor needs {expected} sources, got {given}"),
            Self::CodecMismatch(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<CodecError> for EncodeError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}

/// Columns gathered in blocks so each source row is read once per block
/// rather than once per column (the source is row-major).
const GATHER: usize = 32;

fn gather_columns(m: &Matrix, cols: Range<usize>) -> Vec<Vec<f32>> {
    let width = cols.len();
    let flat: Vec<f32> = (0..m.rows).flat_map(|r| cols.clone().map(move |c| m.get(r, c))).collect();
    (0..width).map(|k| (0..m.rows).map(|r| flat[r * width + k]).collect()).collect()
}

fn encode_values(codec: &Codec, values: &[f32]) -> Result<Vec<u8>, EncodeError> {
    Ok(match codec {
        Codec::Q4 { group } => codec::encode_q4(values, *group)?,
        Codec::Q8 { group } => codec::encode_q8(values, *group)?,
        Codec::F32 => codec::encode_f32(values),
    })
}

fn decode_values(codec: &Codec, bytes: &[u8], len: usize) -> Result<Vec<f32>, EncodeError> {
    Ok(match codec {
        Codec::Q4 { group } => codec::decode_q4(bytes, len, *group)?,
        Codec::Q8 { group } => codec::decode_q8(bytes, len, *group)?,
        Codec::F32 => codec::decode_f32(bytes, len)?,
    })
}

/// Bytes for columns `range` of a `Columns` tensor, each padded to `stride`.
pub fn encode_columns(kind: &Kind, parts: &[Matrix], range: Range<usize>) -> Result<Vec<u8>, EncodeError> {
    let Kind::Columns { parts: specs, codec, stride, column_bytes, .. } = kind else {
        return Err(EncodeError::CodecMismatch("encode_columns needs a columns tensor"));
    };
    if parts.len() != specs.len() {
        return Err(EncodeError::WrongSources { expected: specs.len(), given: parts.len() });
    }
    let blocks: Vec<Range<usize>> =
        range.clone().step_by(GATHER).map(|s| s..(s + GATHER).min(range.end)).collect();
    let mut out = Vec::with_capacity(range.len() * stride);
    for block in blocks {
        let per_part: Vec<Vec<Vec<f32>>> = parts.iter().map(|m| gather_columns(m, block.clone())).collect();
        for k in 0..block.len() {
            for col in &per_part {
                out.extend(encode_values(codec, &col[k])?);
            }
            out.resize(out.len() + stride - column_bytes, 0);
        }
    }
    Ok(out)
}

/// Bytes for rows `range` of a `Rows` tensor.
pub fn encode_rows(kind: &Kind, m: &Matrix, range: Range<usize>) -> Result<Vec<u8>, EncodeError> {
    let Kind::Rows { codec, .. } = kind else {
        return Err(EncodeError::CodecMismatch("encode_rows needs a rows tensor"));
    };
    range.map(|r| encode_values(codec, &m.row(r))).collect::<Result<Vec<_>, _>>().map(|rows| rows.concat())
}

pub fn encode_vector(m: &Matrix) -> Vec<u8> {
    (0..m.cols).flat_map(|c| m.get(0, c).to_le_bytes()).collect()
}

/// Decodes column `j` of a `Columns` segment: one vector per part.
pub fn decode_column(kind: &Kind, segment: &[u8], j: usize) -> Result<Vec<Vec<f32>>, EncodeError> {
    let Kind::Columns { parts, codec, stride, .. } = kind else {
        return Err(EncodeError::CodecMismatch("decode_column needs a columns tensor"));
    };
    let base = j * stride;
    let (_, decoded) = parts.iter().try_fold((base, Vec::new()), |(at, mut acc), p| {
        let len = crate::plan::codec_bytes(codec, p.rows);
        acc.push(decode_values(codec, &segment[at..at + len], p.rows)?);
        Ok::<_, EncodeError>((at + len, acc))
    })?;
    Ok(decoded)
}

pub fn decode_row(kind: &Kind, segment: &[u8], r: usize) -> Result<Vec<f32>, EncodeError> {
    let Kind::Rows { cols, codec, stride, row_bytes, .. } = kind else {
        return Err(EncodeError::CodecMismatch("decode_row needs a rows tensor"));
    };
    decode_values(codec, &segment[r * stride..r * stride + row_bytes], *cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Part;

    fn bf16_bytes(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect()
    }

    fn values(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
        (0..rows * cols).map(|i| (i as f32 * 0.13 + seed).sin() * 0.1).collect()
    }

    #[test]
    fn columns_round_trip_with_padding() {
        let (rows_a, rows_b, cols) = (64, 128, 40);
        let (a, b) = (bf16_bytes(&values(rows_a, cols, 0.0)), bf16_bytes(&values(rows_b, cols, 1.0)));
        let ma = Matrix { rows: rows_a, cols, dtype: Dtype::Bf16, data: &a };
        let mb = Matrix { rows: rows_b, cols, dtype: Dtype::Bf16, data: &b };
        let column_bytes = codec::q4_bytes(rows_a, 64) + codec::q4_bytes(rows_b, 64);
        let kind = Kind::Columns {
            count: cols,
            parts: vec![Part { source: "a".into(), rows: rows_a }, Part { source: "b".into(), rows: rows_b }],
            codec: Codec::Q4 { group: 64 },
            column_bytes,
            stride: 128,
        };
        let seg = encode_columns(&kind, &[ma, mb], 0..cols).unwrap();
        assert_eq!(seg.len(), cols * 128);
        // Encoding a sub-range yields exactly that slice of the full segment.
        assert_eq!(encode_columns(&kind, &[ma, mb], 33..35).unwrap(), seg[33 * 128..35 * 128]);
        for j in [0, 17, 39] {
            let dec = decode_column(&kind, &seg, j).unwrap();
            let src_a: Vec<f32> = (0..rows_a).map(|r| ma.get(r, j)).collect();
            let src_b: Vec<f32> = (0..rows_b).map(|r| mb.get(r, j)).collect();
            assert!(codec::error_stats(&src_a, &dec[0]).rel_rmse < 0.1);
            assert!(codec::error_stats(&src_b, &dec[1]).rel_rmse < 0.1);
        }
        assert!(seg[column_bytes..128].iter().all(|&b| b == 0));
    }

    #[test]
    fn rows_round_trip() {
        let xs = values(5, 64, 2.0);
        let data: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let m = Matrix { rows: 5, cols: 64, dtype: Dtype::F32, data: &data };
        let row_bytes = codec::q8_bytes(64, 32);
        let kind = Kind::Rows { count: 5, cols: 64, source: "e".into(), codec: Codec::Q8 { group: 32 }, row_bytes, stride: row_bytes };
        let seg = encode_rows(&kind, &m, 0..5).unwrap();
        let r3 = decode_row(&kind, &seg, 3).unwrap();
        assert!(codec::error_stats(&m.row(3), &r3).rel_rmse < 0.01);
    }

    #[test]
    fn reads_all_dtypes() {
        let f16b: Vec<u8> = f16::from_f32(1.5).to_le_bytes().to_vec();
        assert_eq!(Matrix { rows: 1, cols: 1, dtype: Dtype::F16, data: &f16b }.get(0, 0), 1.5);
        let v = bf16_bytes(&[0.5, -2.0]);
        assert_eq!(encode_vector(&Matrix { rows: 1, cols: 2, dtype: Dtype::Bf16, data: &v }), [0.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat());
    }
}
