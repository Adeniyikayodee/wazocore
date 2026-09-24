//! Scalar reference kernels: the bit-exact spec that NEON/AVX2 paths are
//! tested against (PROJECT-PLAN-B §3.4). Fixed reduction order.

use half::f16;

use atlas_satl::codec::BLOCK;
use atlas_satl::format::Codec;

use crate::weights::{Columns, Rows};

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// y += xj · column, for one q4 part (`nibbles` then `params`).
fn axpy_q4(y: &mut [f32], xj: f32, part: &[u8], group: usize) {
    let rows = y.len();
    let (nibbles, params) = part.split_at(rows / 2);
    for (g, ys) in y.chunks_exact_mut(group).enumerate() {
        let xs = xj * f16_at(params, g * 4);
        let xm = xj * f16_at(params, g * 4 + 2);
        let first_block = g * group / BLOCK;
        for (b, yb) in ys.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
            let bytes = &nibbles[(first_block + b) * BLOCK / 2..][..BLOCK / 2];
            let (lo, hi) = yb.split_at_mut(BLOCK / 2);
            for k in 0..BLOCK / 2 {
                lo[k] += xs * (bytes[k] & 0x0f) as f32 + xm;
                hi[k] += xs * (bytes[k] >> 4) as f32 + xm;
            }
        }
    }
}

fn axpy_f32(y: &mut [f32], xj: f32, part: &[u8]) {
    let (w, _) = part.as_chunks::<4>();
    y.iter_mut().zip(w).for_each(|(yi, &b)| *yi += xj * f32::from_le_bytes(b));
}

fn part_bytes(codec: &Codec, rows: usize) -> usize {
    match codec {
        Codec::Q4 { group } => atlas_satl::codec::q4_bytes(rows, *group),
        Codec::Q8 { group } => atlas_satl::codec::q8_bytes(rows, *group),
        Codec::F32 => rows * 4,
    }
}

/// Input-sparse GEMV: y = Σ_{j ∈ active} x[j] · W[:, j]. Output is the
/// concatenation of all parts (e.g. [q; k; v]).
pub fn columns_gemv(w: &Columns, x: &[f32], active: &[u32]) -> Vec<f32> {
    let rows = &w.rows[..w.parts];
    let total: usize = rows.iter().sum();
    let mut y = vec![0.0f32; total];
    for &j in active {
        let j = j as usize;
        let xj = x[j];
        let mut at = j * w.stride;
        let mut out = 0;
        for &r in rows {
            let len = part_bytes(&w.codec, r);
            let part = &w.data[at..at + len];
            let ys = &mut y[out..out + r];
            match w.codec {
                Codec::Q4 { group } => axpy_q4(ys, xj, part, group),
                Codec::F32 => axpy_f32(ys, xj, part),
                Codec::Q8 { .. } => unreachable!("columns are never q8; rejected at load"),
            }
            at += len;
            out += r;
        }
    }
    y
}

fn dot_q8(row: &[u8], x: &[f32], group: usize) -> f32 {
    row.chunks_exact(2 + group)
        .zip(x.chunks_exact(group))
        .map(|(b, xs)| {
            let s = f16_at(b, 0);
            s * b[2..].iter().zip(xs).map(|(&q, &xi)| q as i8 as f32 * xi).sum::<f32>()
        })
        .sum()
}

fn dot_f32(row: &[u8], x: &[f32]) -> f32 {
    let (w, _) = row.as_chunks::<4>();
    w.iter().zip(x).map(|(&b, &xi)| f32::from_le_bytes(b) * xi).sum()
}

fn row_bytes<'a>(w: &Rows<'a>, r: usize) -> &'a [u8] {
    &w.data[r * w.stride..(r + 1) * w.stride]
}

/// Dense row-major GEMV (lm_head).
pub fn rows_gemv(w: &Rows, x: &[f32]) -> Vec<f32> {
    (0..w.count)
        .map(|r| match w.codec {
            Codec::Q8 { group } => dot_q8(row_bytes(w, r), x, group),
            Codec::F32 => dot_f32(row_bytes(w, r), x),
            Codec::Q4 { .. } => unreachable!("rows are never q4; rejected at load"),
        })
        .collect()
}

/// One decoded row (embedding lookup).
pub fn row(w: &Rows, r: usize) -> Vec<f32> {
    let b = row_bytes(w, r);
    match w.codec {
        Codec::Q8 { group } => b
            .chunks_exact(2 + group)
            .flat_map(|blk| {
                let s = f16_at(blk, 0);
                blk[2..].iter().map(move |&q| q as i8 as f32 * s)
            })
            .collect(),
        Codec::F32 => b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        Codec::Q4 { .. } => unreachable!("rows are never q4; rejected at load"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_satl::codec::{decode_q4, encode_q4, encode_q8};

    fn wave(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.29 + seed).sin() * 0.1).collect()
    }

    /// Two-part q4 columns tensor built from column-major values.
    fn q4_columns(cols: &[Vec<f32>], split: usize, stride_pad: usize) -> (Vec<u8>, usize) {
        let enc: Vec<Vec<u8>> = cols
            .iter()
            .map(|c| [encode_q4(&c[..split], 64).unwrap(), encode_q4(&c[split..], 64).unwrap()].concat())
            .collect();
        let stride = enc[0].len() + stride_pad;
        let data = enc.iter().flat_map(|e| e.iter().copied().chain(std::iter::repeat_n(0, stride_pad))).collect();
        (data, stride)
    }

    #[test]
    fn q4_gemv_matches_dequantised_reference_and_respects_active_set() {
        let (split, rows, n) = (64, 192, 10);
        let cols: Vec<Vec<f32>> = (0..n).map(|j| wave(rows, j as f32)).collect();
        let (data, stride) = q4_columns(&cols, split, 16);
        let w = Columns { count: n, stride, codec: Codec::Q4 { group: 64 }, rows: [split, rows - split, 0], parts: 2, data: &data };
        let x = wave(n, 5.0);
        let active: Vec<u32> = vec![0, 3, 4, 9];
        let y = columns_gemv(&w, &x, &active);
        let deq: Vec<Vec<f32>> = cols
            .iter()
            .map(|c| [decode_q4(&encode_q4(&c[..split], 64).unwrap(), split, 64).unwrap(), decode_q4(&encode_q4(&c[split..], 64).unwrap(), rows - split, 64).unwrap()].concat())
            .collect();
        for r in 0..rows {
            let expect: f32 = active.iter().map(|&j| x[j as usize] * deq[j as usize][r]).sum();
            assert!((y[r] - expect).abs() < 1e-6, "row {r}: {} vs {expect}", y[r]);
        }
    }

    #[test]
    fn full_active_set_equals_dense_matvec() {
        let (rows, n) = (4, 3);
        let cols: Vec<Vec<f32>> = (0..n).map(|j| wave(rows, j as f32)).collect();
        let data: Vec<u8> = cols.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let w = Columns { count: n, stride: rows * 4, codec: Codec::F32, rows: [rows, 0, 0], parts: 1, data: &data };
        let x = [1.0, -2.0, 0.5];
        let y = columns_gemv(&w, &x, &[0, 1, 2]);
        for r in 0..rows {
            let expect: f32 = (0..n).map(|j| x[j] * cols[j][r]).sum();
            assert!((y[r] - expect).abs() < 1e-7);
        }
        assert_eq!(columns_gemv(&w, &x, &[]), vec![0.0; rows]);
    }

    #[test]
    fn q8_rows_gemv_and_lookup() {
        let (count, cols) = (3, 64);
        let rows_f: Vec<Vec<f32>> = (0..count).map(|r| wave(cols, r as f32)).collect();
        let data: Vec<u8> = rows_f.iter().flat_map(|r| encode_q8(r, 32).unwrap()).collect();
        let w = Rows { count, cols, stride: data.len() / count, codec: Codec::Q8 { group: 32 }, data: &data };
        let x = wave(cols, 9.0);
        let y = rows_gemv(&w, &x);
        for r in 0..count {
            let dq = row(&w, r);
            let expect: f32 = dq.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((y[r] - expect).abs() < 1e-5);
            assert!(dq.iter().zip(&rows_f[r]).all(|(a, b)| (a - b).abs() < 1e-3));
        }
    }
}
