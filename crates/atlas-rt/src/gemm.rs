//! Batched kernels for prefill: several tokens through one weight matrix.
//!
//! Per token, the arithmetic is exactly `fast::columns_gemv` / `rows_gemv`
//! (same tiling, folding, integer sums and reduction order), so batched
//! prefill equals token-by-token decode bit for bit. The saving is that each
//! (column block, group) is unpacked into an int8 tile once and reused by
//! every token in the batch.

use rayon::prelude::*;

use atlas_satl::format::Codec;

use crate::fast::{self, Band, Part, parts, split, tiling};
use crate::int8::{self, COLS, DecodeTile, FoldParams, TileDot};
use crate::kernels::{self, KernelError, check_columns, check_rows, f16_at, row_bytes};
use crate::weights::{Columns, Rows};

/// Inner kernels of the batched q4 GEMM: NEON by default, the spec in tests.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Inner {
    pub fold: FoldParams,
    pub decode: DecodeTile,
    pub dot: TileDot,
}

pub(crate) const DISPATCH: Inner = Inner { fold: int8::fold_params, decode: int8::decode_tile, dot: int8::tile_dot };
#[cfg(test)]
pub(crate) const SPEC: Inner = Inner { fold: int8::fold_params_scalar, decode: int8::decode_tile_scalar, dot: int8::tile_dot_scalar };

/// `columns_gemv` for every token in `xs`: returns one output per token.
pub fn columns_gemm(w: &Columns, xs: &[Vec<f32>], active: &[u32]) -> Result<Vec<Vec<f32>>, KernelError> {
    columns_gemm_with(w, xs, active, DISPATCH)
}

pub(crate) fn columns_gemm_with(w: &Columns, xs: &[Vec<f32>], active: &[u32], inner: Inner) -> Result<Vec<Vec<f32>>, KernelError> {
    xs.iter().try_for_each(|x| check_columns(w, x, active))?;
    let Codec::Q4 { group } = w.codec else {
        return xs.iter().map(|x| kernels::columns_gemv_ref(w, x, active)).collect();
    };
    debug_assert!(group <= fast::MAX_GROUP, "Columns::new bounds the group");
    let parts = parts(w);
    let (bands, slices) = tiling(&parts, active.len());
    let cols = split(active.len(), slices);
    let partials: Vec<Vec<f32>> = (0..bands.len() * slices)
        .into_par_iter()
        .map(|t| {
            let b = bands[t / slices];
            band_batch(w, parts[b.part], b, group, xs, &active[cols[t % slices].clone()], inner)
        })
        .collect();

    // Each token is reduced independently (in parallel), band by band and
    // slice by slice in the same order as `fast::columns_gemv`.
    let total: usize = parts.iter().map(|p| p.rows).sum();
    Ok((0..xs.len())
        .into_par_iter()
        .map(|t| {
            let mut y = vec![0.0f32; total];
            for (b, tile) in bands.iter().zip(partials.chunks(slices)) {
                let dst = &mut y[parts[b.part].out + b.row0..][..b.rows];
                for partial in tile {
                    dst.iter_mut().zip(&partial[t * b.rows..(t + 1) * b.rows]).for_each(|(d, p)| *d += p);
                }
            }
            y
        })
        .collect())
}

/// One tile for all tokens: `acc[t · rows + r]`, token-major.
fn band_batch(w: &Columns, part: Part, band: Band, group: usize, xs: &[Vec<f32>], cols: &[u32], inner: Inner) -> Vec<f32> {
    let (tokens, groups) = (xs.len(), band.rows / group);
    let nibbles = part.byte + band.row0 / 2;
    let params = part.byte + part.rows / 2 + band.row0 / group * 4;
    let mut acc = vec![0.0f32; tokens * band.rows];
    let mut bias = vec![0.0f32; tokens * groups];
    let mut tile = vec![0u8; group * COLS];
    let mut isum = vec![0i32; group];
    for block in cols.chunks(COLS) {
        // Pad a short block with zero-weighted copies of its first column,
        // exactly as the decode kernel does.
        let js: [Option<usize>; COLS] = std::array::from_fn(|k| block.get(k).map(|&j| j as usize));
        let bases: [usize; COLS] = std::array::from_fn(|k| js[k].unwrap_or(block[0] as usize) * w.stride);
        let block_x: Vec<[f32; COLS]> = xs.iter().map(|x| std::array::from_fn(|k| js[k].map_or(0.0, |j| x[j]))).collect();
        for g in 0..groups {
            (inner.decode)(w.data, &bases, nibbles + g * group / 2, &mut tile);
            let p = int8::params(w.data, &bases, params + g * 4);
            for (t, xs_t) in block_x.iter().enumerate() {
                let f = (inner.fold)(&p, xs_t);
                bias[t * groups + g] += f.bias;
                if f.t == 0.0 {
                    continue;
                }
                (inner.dot)(&tile, &f.xq, &mut isum);
                acc[t * band.rows + g * group..][..group].iter_mut().zip(&isum).for_each(|(a, &s)| *a += f.t * s as f32);
            }
        }
    }
    for t in 0..tokens {
        acc[t * band.rows..(t + 1) * band.rows]
            .iter_mut()
            .enumerate()
            .for_each(|(r, a)| *a += bias[t * groups + r / group]);
    }
    acc
}

/// `rows_gemv` for every token: each row's bytes are read once and used by
/// all tokens. Returns one output per token.
pub fn rows_gemm(w: &Rows, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, KernelError> {
    xs.iter().try_for_each(|x| check_rows(w, x))?;
    let Codec::Q8 { group } = w.codec else {
        return xs.iter().map(|x| kernels::rows_gemv_ref(w, x)).collect();
    };
    let quantised: Vec<(Vec<i8>, Vec<f32>)> = xs
        .iter()
        .map(|x| {
            let (q, s): (Vec<Vec<i8>>, Vec<f32>) = x.chunks(group).map(int8::quantize).unzip();
            (q.concat(), s)
        })
        .collect();
    let by_row: Vec<Vec<f32>> = (0..w.count)
        .into_par_iter()
        .with_min_len(256)
        .map(|r| {
            let row = row_bytes(w, r);
            quantised
                .iter()
                .map(|(xq, sx)| {
                    row.chunks_exact(2 + group)
                        .enumerate()
                        .map(|(b, blk)| f16_at(blk, 0) * sx[b] * int8::dot_i8(&blk[2..], &xq[b * group..(b + 1) * group]) as f32)
                        .sum()
                })
                .collect()
        })
        .collect();
    Ok((0..xs.len()).map(|t| by_row.iter().map(|r| r[t]).collect()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_satl::codec::{encode_q4, encode_q8};

    fn wave(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed).sin() * 0.1) + 0.02 * ((i * 7) % 5) as f32).collect()
    }

    fn tensor(rows: &[usize], n: usize, group: usize) -> (Vec<u8>, usize) {
        let enc: Vec<Vec<u8>> = (0..n)
            .map(|j| rows.iter().enumerate().flat_map(|(p, &r)| encode_q4(&wave(r, (j * 3 + p) as f32), group).unwrap()).collect())
            .collect();
        let stride = enc[0].len() + 48;
        let data = enc.iter().flat_map(|e| e.iter().copied().chain(std::iter::repeat_n(0, 48))).collect();
        (data, stride)
    }

    #[test]
    fn batched_columns_equal_per_token_decode_bit_for_bit() {
        let rows = [fast::BAND + 64, 128];
        let n = 70;
        for group in [32, 64] {
            let (data, stride) = tensor(&rows, n, group);
            let w = Columns::new(n, stride, Codec::Q4 { group }, &rows, &data).unwrap();
            // Token 2 is all zeros (exercises the t == 0 skip).
            let xs: Vec<Vec<f32>> = vec![wave(n, 1.5), wave(n, -0.4), vec![0.0; n]];
            for mask in [(0..n as u32).collect::<Vec<_>>(), (0..n as u32).filter(|j| j % 3 != 1).collect(), vec![]] {
                let batched = columns_gemm(&w, &xs, &mask).unwrap();
                let single: Vec<Vec<f32>> = xs.iter().map(|x| fast::columns_gemv(&w, x, &mask).unwrap()).collect();
                assert_eq!(batched, single, "group {group}, mask {}", mask.len());
                assert_eq!(batched, columns_gemm_with(&w, &xs, &mask, SPEC).unwrap());
            }
            assert!(columns_gemm(&w, &[wave(n - 1, 0.0)], &[0]).is_err());
        }
    }

    #[test]
    fn batched_rows_equal_per_token_decode_bit_for_bit() {
        let (count, cols) = (700, 64);
        let data: Vec<u8> = (0..count).flat_map(|r| encode_q8(&wave(cols, r as f32), 32).unwrap()).collect();
        let w = Rows::new(count, cols, data.len() / count, Codec::Q8 { group: 32 }, &data).unwrap();
        let xs = vec![wave(cols, 4.0), wave(cols, -2.0)];
        let batched = rows_gemm(&w, &xs).unwrap();
        let single: Vec<Vec<f32>> = xs.iter().map(|x| fast::rows_gemv(&w, x).unwrap()).collect();
        assert_eq!(batched, single);
    }
}
