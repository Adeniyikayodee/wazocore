//! Quantisation codecs of `.satl` (docs/satl-format.md). Pure functions over
//! slices; every encoder has a decoder so tests can bound the error.

use std::fmt;

use half::f16;

pub const BLOCK: usize = 32;

/// Largest q4 group (rows per scale) the runtime kernels accept.
pub const MAX_Q4_GROUP: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// `group` must be a positive multiple of 32 dividing the length.
    BadGroup { len: usize, group: usize },
    /// Encoded bytes do not match the expected size.
    BadLength { expected: usize, actual: usize },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadGroup { len, group } => {
                write!(f, "group {group} must be a multiple of {BLOCK} dividing length {len}")
            }
            Self::BadLength { expected, actual } => write!(f, "expected {expected} bytes, got {actual}"),
        }
    }
}

impl std::error::Error for CodecError {}

fn check_group(len: usize, group: usize) -> Result<(), CodecError> {
    if group == 0 || !group.is_multiple_of(BLOCK) || !len.is_multiple_of(group) {
        Err(CodecError::BadGroup { len, group })
    } else {
        Ok(())
    }
}

/// Encoded size of one q4 part of `rows` values.
pub fn q4_bytes(rows: usize, group: usize) -> usize {
    rows / 2 + rows / group * 4
}

/// How a q4 group's (scale, min) is chosen. The stored format is identical;
/// only the encoder's choice differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// scale = (max − min) / 15, min = min.
    MinMax,
    /// Search candidate scales and least-squares-fit (scale, min) to the
    /// resulting codes under weights |x| + rms(x), keeping the best weighted
    /// error (the approach of llama.cpp's k-quants).
    Search,
}

/// (scale, min) as stored, i.e. already rounded to f16, so the encoder
/// quantises against exactly what the decoder will use.
fn q4_params(group: &[f32], fit: Fit) -> (f16, f16) {
    let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    match fit {
        Fit::MinMax => {
            let min = f16::from_f32(lo);
            (f16::from_f32((hi - min.to_f32()).max(0.0) / 15.0), min)
        }
        Fit::Search => {
            let (scale, min) = search_params(group, lo, hi);
            (f16::from_f32(scale), f16::from_f32(min))
        }
    }
}

/// Weighted squared error of reconstructing `x` with codes from (scale, min).
fn weighted_error(x: &[f32], w: &[f32], scale: f32, min: f32) -> f32 {
    x.iter().zip(w).map(|(&v, &wi)| {
        let e = scale * q4_code(v, scale, min) as f32 + min - v;
        wi * e * e
    }).sum()
}

/// Candidate inverse scales around 15 / (max − min); for each, fixes the
/// codes and solves the weighted least-squares (scale, min) for them.
fn search_params(x: &[f32], lo: f32, hi: f32) -> (f32, f32) {
    const NMAX: f32 = 15.0;
    let range = hi - lo;
    if range <= 0.0 {
        return (0.0, lo);
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let w: Vec<f32> = x.iter().map(|v| rms + v.abs()).collect();
    let start = (range / NMAX, lo);
    let candidates = (0..=20).filter_map(|step| {
        let iscale = (NMAX - 1.0 + 0.1 * step as f32) / range;
        let codes: Vec<f32> = x.iter().map(|&v| ((v - lo) * iscale).round().clamp(0.0, NMAX)).collect();
        let (sw, sl, sl2, sx, sxl) = codes.iter().zip(x).zip(&w).fold((0.0f32, 0.0, 0.0, 0.0, 0.0), |a, ((&l, &v), &wi)| {
            (a.0 + wi, a.1 + wi * l, a.2 + wi * l * l, a.3 + wi * v, a.4 + wi * v * l)
        });
        let det = sw * sl2 - sl * sl;
        let scale = (sw * sxl - sx * sl) / det;
        let min = (sl2 * sx - sl * sxl) / det;
        (det > 0.0 && scale > 0.0 && scale.is_finite() && min.is_finite()).then_some((scale, min))
    });
    std::iter::once(start)
        .chain(candidates)
        .map(|(s, m)| (weighted_error(x, &w, s, m), (s, m)))
        .fold((f32::INFINITY, start), |best, (e, sm)| if e < best.0 { (e, sm) } else { best })
        .1
}

fn q4_code(w: f32, scale: f32, min: f32) -> u8 {
    if scale == 0.0 { 0 } else { ((w - min) / scale).round().clamp(0.0, 15.0) as u8 }
}

/// 4-bit asymmetric, `group` rows per (scale, min): nibbles, then params.
pub fn encode_q4(values: &[f32], group: usize) -> Result<Vec<u8>, CodecError> {
    encode_q4_with(values, group, Fit::MinMax)
}

/// `encode_q4` with an explicit parameter fit.
pub fn encode_q4_with(values: &[f32], group: usize, fit: Fit) -> Result<Vec<u8>, CodecError> {
    check_group(values.len(), group)?;
    let params: Vec<(f16, f16)> = values.chunks(group).map(|g| q4_params(g, fit)).collect();
    let codes: Vec<u8> = values
        .chunks(group)
        .zip(&params)
        .flat_map(|(g, &(s, m))| g.iter().map(move |&w| q4_code(w, s.to_f32(), m.to_f32())))
        .collect();
    let nibbles = codes
        .chunks(BLOCK)
        .flat_map(|b| (0..BLOCK / 2).map(move |k| b[k] | b[k + BLOCK / 2] << 4));
    let param_bytes = params.iter().flat_map(|(s, m)| [s.to_le_bytes(), m.to_le_bytes()]).flatten();
    Ok(nibbles.chain(param_bytes).collect())
}

pub fn decode_q4(bytes: &[u8], rows: usize, group: usize) -> Result<Vec<f32>, CodecError> {
    check_group(rows, group)?;
    let expected = q4_bytes(rows, group);
    if bytes.len() != expected {
        return Err(CodecError::BadLength { expected, actual: bytes.len() });
    }
    let (nibbles, params) = bytes.split_at(rows / 2);
    let codes: Vec<u8> = nibbles
        .chunks(BLOCK / 2)
        .flat_map(|b| b.iter().map(|x| x & 0x0f).chain(b.iter().map(|x| x >> 4)))
        .collect();
    let params: Vec<(f32, f32)> = params
        .chunks(4)
        .map(|p| (f16::from_le_bytes([p[0], p[1]]).to_f32(), f16::from_le_bytes([p[2], p[3]]).to_f32()))
        .collect();
    Ok(codes
        .chunks(group)
        .zip(params)
        .flat_map(|(g, (s, m))| g.iter().map(move |&q| q as f32 * s + m))
        .collect())
}

/// Encoded size of `len` values in q8 blocks of `group`.
pub fn q8_bytes(len: usize, group: usize) -> usize {
    len / group * (2 + group)
}

/// 8-bit symmetric: per block, f16 scale then `group` × i8.
pub fn encode_q8(values: &[f32], group: usize) -> Result<Vec<u8>, CodecError> {
    check_group(values.len(), group)?;
    Ok(values
        .chunks(group)
        .flat_map(|g| {
            let amax = g.iter().fold(0.0f32, |a, w| a.max(w.abs()));
            let scale = f16::from_f32(amax / 127.0);
            let s = scale.to_f32();
            let qs = g.iter().map(move |&w| if s == 0.0 { 0 } else { (w / s).round().clamp(-127.0, 127.0) as i8 as u8 });
            scale.to_le_bytes().into_iter().chain(qs)
        })
        .collect())
}

pub fn decode_q8(bytes: &[u8], len: usize, group: usize) -> Result<Vec<f32>, CodecError> {
    check_group(len, group)?;
    let expected = q8_bytes(len, group);
    if bytes.len() != expected {
        return Err(CodecError::BadLength { expected, actual: bytes.len() });
    }
    Ok(bytes
        .chunks(2 + group)
        .flat_map(|b| {
            let s = f16::from_le_bytes([b[0], b[1]]).to_f32();
            b[2..].iter().map(move |&q| q as i8 as f32 * s)
        })
        .collect())
}

pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn decode_f32(bytes: &[u8], len: usize) -> Result<Vec<f32>, CodecError> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if chunks.len() != len || !rest.is_empty() {
        return Err(CodecError::BadLength { expected: len * 4, actual: bytes.len() });
    }
    Ok(chunks.iter().map(|&c| f32::from_le_bytes(c)).collect())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorStats {
    /// ‖a − b‖ / ‖a‖.
    pub rel_rmse: f64,
    pub max_abs: f64,
}

pub fn error_stats(reference: &[f32], approx: &[f32]) -> ErrorStats {
    let (num, den, max_abs) = reference.iter().zip(approx).fold((0.0f64, 0.0f64, 0.0f64), |(n, d, m), (&a, &b)| {
        let e = (a - b) as f64;
        (n + e * e, d + (a as f64) * (a as f64), m.max(e.abs()))
    });
    ErrorStats { rel_rmse: if den > 0.0 { (num / den).sqrt() } else { num.sqrt() }, max_abs }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed as f32).sin() * 0.05) + 0.01 * (i % 7) as f32).collect()
    }

    #[test]
    fn q4_round_trip_error_within_half_step() {
        let xs = wave(512, 1);
        let enc = encode_q4(&xs, 64).unwrap();
        assert_eq!(enc.len(), q4_bytes(512, 64));
        let dec = decode_q4(&enc, 512, 64).unwrap();
        for (g, dg) in xs.chunks(64).zip(dec.chunks(64)) {
            let (s, _) = q4_params(g, Fit::MinMax);
            let half_step = s.to_f32() / 2.0 + 1e-3; // plus f16 rounding of min
            assert!(g.iter().zip(dg).all(|(a, b)| (a - b).abs() <= half_step));
        }
        assert!(error_stats(&xs, &dec).rel_rmse < 0.1);
    }

    /// Heavy-tailed weight-like values (a few outliers per group).
    fn weights_like(n: usize, seed: u64) -> Vec<f32> {
        (0..n as u64)
            .map(|i| {
                let h = (i.wrapping_mul(6364136223846793005) ^ seed.wrapping_mul(1442695040888963407)) >> 33;
                let u = (h % 10_000) as f32 / 10_000.0 - 0.5;
                if h.is_multiple_of(37) { u * 0.3 } else { u * 0.04 }
            })
            .collect()
    }

    #[test]
    fn search_fit_beats_min_max_and_keeps_the_format() {
        let xs = weights_like(64 * 400, 3);
        let mm = decode_q4(&encode_q4_with(&xs, 64, Fit::MinMax).unwrap(), xs.len(), 64).unwrap();
        let enc = encode_q4_with(&xs, 64, Fit::Search).unwrap();
        assert_eq!(enc.len(), q4_bytes(xs.len(), 64));
        let se = decode_q4(&enc, xs.len(), 64).unwrap();
        let (e_mm, e_se) = (error_stats(&xs, &mm).rel_rmse, error_stats(&xs, &se).rel_rmse);
        assert!(e_se < e_mm * 0.95, "search {e_se} vs min-max {e_mm}");
        assert_eq!(encode_q4_with(&[0.25; 64], 64, Fit::Search).map(|b| decode_q4(&b, 64, 64).unwrap()).unwrap(), vec![0.25; 64]);
    }

    #[test]
    fn q4_is_4_5_bits_per_weight_at_group_64() {
        assert_eq!(q4_bytes(14336, 64) * 8, 14336 * 9 / 2);
        assert_eq!(2 * q4_bytes(14336, 64), 16128);
        assert_eq!(q4_bytes(4096, 64), 2304);
        assert_eq!(q4_bytes(4096, 32) * 8, 4096 * 5);
    }

    #[test]
    fn q4_nibble_layout_splits_block_halves() {
        // Values 0..15 then 15..0 in one block, group 32: codes equal values.
        let xs: Vec<f32> = (0..16).chain((0..16).rev()).map(|v| v as f32).collect();
        let enc = encode_q4(&xs, 32).unwrap();
        assert_eq!(enc[0], 0xf0); // q[0] = 0 low, q[16] = 15 high
        assert_eq!(enc[15], 0x0f); // q[15] = 15 low, q[31] = 0 high
        assert_eq!(decode_q4(&enc, 32, 32).unwrap(), xs);
    }

    #[test]
    fn q4_constant_group_is_exact() {
        let xs = vec![0.25f32; 64];
        assert_eq!(decode_q4(&encode_q4(&xs, 64).unwrap(), 64, 64).unwrap(), xs);
    }

    #[test]
    fn q8_round_trip_is_tight() {
        let xs = wave(256, 3);
        let enc = encode_q8(&xs, 32).unwrap();
        assert_eq!(enc.len(), q8_bytes(256, 32));
        let dec = decode_q8(&enc, 256, 32).unwrap();
        assert!(error_stats(&xs, &dec).rel_rmse < 0.01);
    }

    #[test]
    fn f32_is_lossless() {
        let xs = wave(10, 4);
        assert_eq!(decode_f32(&encode_f32(&xs), 10).unwrap(), xs);
        assert!(decode_f32(&[0; 7], 2).is_err());
    }

    #[test]
    fn rejects_bad_groups_and_lengths() {
        assert!(encode_q4(&[0.0; 48], 32).is_err());
        assert!(encode_q4(&[0.0; 64], 48).is_err());
        assert!(encode_q8(&[0.0; 64], 0).is_err());
        assert!(matches!(decode_q4(&[0; 10], 64, 64), Err(CodecError::BadLength { .. })));
    }
}
