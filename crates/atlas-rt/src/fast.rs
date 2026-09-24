//! Parallel int8-activation kernels (the `Fast` path).
//!
//! Correctness is layered (PROJECT-PLAN-B §3.4):
//! - vs the f32 reference (`kernels`): within int8-activation tolerance;
//! - vs the scalar int8 spec (`int8::*_scalar`): bit-identical, so NEON
//!   (and any future AVX2 path) gives the same logits on every device;
//! - vs itself at any thread count: bit-identical, because tiling depends
//!   only on the matrix shape and partial sums are reduced in fixed order.
//!
//! q4 columns are tiled in 2D: at most `BAND` output rows (16 KiB of f32,
//! L1-resident) by a contiguous slice of the active columns. Within a tile,
//! per 32-column block and 64-row group, folded activations
//! `x_j · s_{j,g}` are quantised to int8 and reduced across columns by
//! `int8::dot_cols`; the `x_j · min` terms are summed once per group.

use std::ops::Range;

use rayon::prelude::*;

use atlas_satl::format::Codec;

use crate::int8::{self, COLS, DotCols, Fold};
use crate::kernels::{self, KernelError, check_columns, check_rows, f16_at, part_bytes, row_bytes};
use crate::weights::{Columns, Rows};

/// Output rows per tile.
pub const BAND: usize = 4096;
/// Target tiles per matrix: at least 4 per thread up to 16 threads, so
/// work-stealing can even out uneven (big.LITTLE) cores. Deliberately not
/// derived from the thread count, which would change the reduction order.
const TILES: usize = 64;
/// Largest q4 group (rows per scale), enforced by `Columns::new`.
const MAX_GROUP: usize = atlas_satl::codec::MAX_Q4_GROUP;

#[derive(Debug, Clone, Copy)]
struct Part {
    /// Offset of this part's rows in the output vector.
    out: usize,
    /// Offset of this part's bytes within a column.
    byte: usize,
    rows: usize,
}

#[derive(Debug, Clone, Copy)]
struct Band {
    part: usize,
    row0: usize,
    rows: usize,
}

fn parts(w: &Columns) -> Vec<Part> {
    w.rows[..w.parts]
        .iter()
        .scan((0usize, 0usize), |(out, byte), &rows| {
            let p = Part { out: *out, byte: *byte, rows };
            *out += rows;
            *byte += part_bytes(&w.codec, rows);
            Some(p)
        })
        .collect()
}

/// `k` contiguous ranges covering `0..n` (some may be empty).
fn split(n: usize, k: usize) -> Vec<Range<usize>> {
    (0..k).map(|i| i * n / k..(i + 1) * n / k).collect()
}

/// Inner kernels of the q4 column GEMV: NEON by default, the scalar spec
/// in tests.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Inner {
    pub fold: Fold,
    pub dot: DotCols,
}

pub(crate) const DISPATCH: Inner = Inner { fold: int8::fold, dot: int8::dot_cols };
#[cfg(test)]
pub(crate) const SPEC: Inner = Inner { fold: int8::fold_scalar, dot: int8::dot_cols_scalar };

/// y = Σ_{j ∈ active} x[j] · W[:, j] with int8 activations.
pub fn columns_gemv(w: &Columns, x: &[f32], active: &[u32]) -> Result<Vec<f32>, KernelError> {
    columns_gemv_with(w, x, active, DISPATCH)
}

/// `columns_gemv` with explicit inner kernels (tests pass `SPEC`).
pub(crate) fn columns_gemv_with(w: &Columns, x: &[f32], active: &[u32], inner: Inner) -> Result<Vec<f32>, KernelError> {
    let Codec::Q4 { group } = w.codec else {
        return kernels::columns_gemv_ref(w, x, active);
    };
    check_columns(w, x, active)?;
    debug_assert!(group <= MAX_GROUP, "Columns::new bounds the group");
    let parts = parts(w);
    let bands: Vec<Band> = parts
        .iter()
        .enumerate()
        .flat_map(|(part, p)| (0..p.rows).step_by(BAND).map(move |row0| Band { part, row0, rows: BAND.min(p.rows - row0) }))
        .collect();
    let slices = TILES.div_ceil(bands.len()).clamp(1, active.len().div_ceil(COLS).max(1));
    let cols = split(active.len(), slices);
    let partials: Vec<Vec<f32>> = (0..bands.len() * slices)
        .into_par_iter()
        .map(|t| {
            let b = bands[t / slices];
            band_q4(w, parts[b.part], b, group, x, &active[cols[t % slices].clone()], inner)
        })
        .collect();

    let total: usize = parts.iter().map(|p| p.rows).sum();
    let mut y = vec![0.0f32; total];
    for (b, tile) in bands.iter().zip(partials.chunks(slices)) {
        let dst = &mut y[parts[b.part].out + b.row0..][..b.rows];
        for partial in tile {
            dst.iter_mut().zip(partial).for_each(|(d, p)| *d += p);
        }
    }
    Ok(y)
}

/// One tile: rows `band` of one part, over `cols` (a slice of active columns).
fn band_q4(w: &Columns, part: Part, band: Band, group: usize, x: &[f32], cols: &[u32], inner: Inner) -> Vec<f32> {
    let groups = band.rows / group;
    let nibbles = part.byte + band.row0 / 2;
    let params = part.byte + part.rows / 2 + band.row0 / group * 4;
    let mut acc = vec![0.0f32; band.rows];
    let mut bias = vec![0.0f32; groups];
    let mut isum = [0i32; MAX_GROUP];
    for block in cols.chunks(COLS) {
        // Pad a short block with zero-weighted copies of its first column.
        let cx: [(usize, f32); COLS] = std::array::from_fn(|k| {
            block.get(k).map_or((block[0] as usize, 0.0), |&j| (j as usize, x[j as usize]))
        });
        let bases: [usize; COLS] = cx.map(|(j, _)| j * w.stride);
        let xs: [f32; COLS] = cx.map(|(_, v)| v);
        for g in 0..groups {
            let f = (inner.fold)(w.data, &bases, params + g * 4, &xs);
            bias[g] += f.bias;
            if f.t == 0.0 {
                continue;
            }
            let out = &mut isum[..group];
            (inner.dot)(w.data, &bases, nibbles + g * group / 2, &f.xq, out);
            acc[g * group..(g + 1) * group].iter_mut().zip(out.iter()).for_each(|(a, &s)| *a += f.t * s as f32);
        }
    }
    acc.iter_mut().enumerate().for_each(|(r, a)| *a += bias[r / group]);
    acc
}

/// Signature of the int8 row dot product (spec or NEON).
pub(crate) type DotI8 = fn(&[u8], &[i8]) -> i32;

/// Dense row-major GEMV with int8 activations, parallel over rows.
pub fn rows_gemv(w: &Rows, x: &[f32]) -> Result<Vec<f32>, KernelError> {
    rows_gemv_with(w, x, int8::dot_i8)
}

/// `rows_gemv` with an explicit inner kernel (tests pass the spec).
pub(crate) fn rows_gemv_with(w: &Rows, x: &[f32], dot: DotI8) -> Result<Vec<f32>, KernelError> {
    let Codec::Q8 { group } = w.codec else {
        return kernels::rows_gemv_ref(w, x);
    };
    check_rows(w, x)?;
    let (xq, sx): (Vec<Vec<i8>>, Vec<f32>) = x.chunks(group).map(int8::quantize).unzip();
    let xq: Vec<i8> = xq.concat();
    Ok((0..w.count)
        .into_par_iter()
        .with_min_len(512)
        .map(|r| {
            row_bytes(w, r)
                .chunks_exact(2 + group)
                .enumerate()
                .map(|(b, blk)| f16_at(blk, 0) * sx[b] * dot(&blk[2..], &xq[b * group..(b + 1) * group]) as f32)
                .sum()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{columns_gemv_ref, rows_gemv_ref};
    use atlas_satl::codec::{encode_q4, encode_q8};

    fn wave(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed).sin() * 0.1) + 0.02 * ((i * 7) % 5) as f32).collect()
    }

    /// q4 columns tensor with parts of `rows` each, `n` columns, padded stride.
    fn tensor(rows: &[usize], n: usize, group: usize) -> (Vec<u8>, usize) {
        let enc: Vec<Vec<u8>> = (0..n)
            .map(|j| rows.iter().enumerate().flat_map(|(p, &r)| encode_q4(&wave(r, (j * 3 + p) as f32), group).unwrap()).collect())
            .collect();
        let stride = enc[0].len() + 48;
        let data = enc.iter().flat_map(|e| e.iter().copied().chain(std::iter::repeat_n(0, 48))).collect();
        (data, stride)
    }

    fn view<'a>(data: &'a [u8], stride: usize, rows: &[usize], n: usize, group: usize) -> Columns<'a> {
        Columns::new(n, stride, Codec::Q4 { group }, rows, data).unwrap()
    }

    /// Int8 rounding moves each folded activation by at most t/2 = amax/254,
    /// so a GEMV's relative error is bounded by about 1/127 when few columns
    /// are active (errors cannot average out) and shrinks as more are.
    const INT8_TOL: f32 = 1.0 / 127.0;

    /// ‖a − b‖ / ‖b‖.
    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum();
        (num / den.max(1e-30)).sqrt()
    }

    fn masks(n: usize) -> [Vec<u32>; 4] {
        [
            (0..n as u32).collect(),
            (0..n as u32).filter(|j| j % 3 != 1).collect(),
            vec![5, 6, 7],
            vec![],
        ]
    }

    #[test]
    fn columns_equal_the_int8_spec_bit_for_bit() {
        let rows = [BAND + 64, 128];
        let n = 70;
        for group in [32, 64] {
            let (data, stride) = tensor(&rows, n, group);
            let w = view(&data, stride, &rows, n, group);
            let x = wave(n, 1.5);
            for m in &masks(n) {
                assert_eq!(columns_gemv(&w, &x, m).unwrap(), columns_gemv_with(&w, &x, m, SPEC).unwrap());
            }
        }
    }

    #[test]
    fn columns_stay_close_to_the_f32_reference() {
        let rows = [BAND + 64, 128];
        let n = 70;
        let (data, stride) = tensor(&rows, n, 64);
        let w = view(&data, stride, &rows, n, 64);
        let x = wave(n, 1.5);
        for m in masks(n).iter().filter(|m| !m.is_empty()) {
            let e = rel_err(&columns_gemv(&w, &x, m).unwrap(), &columns_gemv_ref(&w, &x, m).unwrap());
            assert!(e < INT8_TOL, "relative error {e} for mask of {}", m.len());
        }
        assert_eq!(columns_gemv(&w, &x, &[]).unwrap(), vec![0.0; BAND + 64 + 128]);
        assert!(columns_gemv(&w, &x, &[70]).is_err());
    }

    #[test]
    fn identical_for_any_thread_count() {
        let rows = [256, 256];
        let (data, stride) = tensor(&rows, 300, 64);
        let w = view(&data, stride, &rows, 300, 64);
        let x = wave(300, 0.7);
        let all: Vec<u32> = (0..300).collect();
        let run = |t| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap().install(|| columns_gemv(&w, &x, &all).unwrap());
        let one = run(1);
        assert_eq!(one, run(3));
        assert_eq!(one, run(8));
    }

    #[test]
    fn rows_equal_spec_and_stay_close_to_reference() {
        let (count, cols) = (1000, 64);
        let data: Vec<u8> = (0..count).flat_map(|r| encode_q8(&wave(cols, r as f32), 32).unwrap()).collect();
        let w = Rows::new(count, cols, data.len() / count, Codec::Q8 { group: 32 }, &data).unwrap();
        let x = wave(cols, 4.0);
        let fast = rows_gemv(&w, &x).unwrap();
        assert_eq!(fast, rows_gemv_with(&w, &x, int8::dot_i8_scalar).unwrap());
        assert!(rel_err(&fast, &rows_gemv_ref(&w, &x).unwrap()) < INT8_TOL);
    }
}
