//! Int8-activation inner kernels. The scalar functions are the exact spec:
//! the NEON versions must return bit-identical results (integer sums are
//! exact; float epilogues are done by shared scalar code in both paths).
//!
//! q4 columns: for one 64-row group and a block of 32 active columns, the
//! folded activations `x'_j = x_j · s_{j,g}` are quantised to int8 with one
//! scale `t`, so y[r] += t · Σ_j xq_j · q_{r,j} is an integer dot product
//! across columns. NEON transposes 4 columns into row order (ZIP1/ZIP2)
//! and reduces them with SDOT.

use atlas_satl::codec::BLOCK;

/// Active columns quantised together.
pub const COLS: usize = 32;

/// Int8 scale for `v`: v ≈ t · q with |q| ≤ 127 (0 for an all-zero `v`).
fn scale(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs())) / 127.0
}

fn code(x: f32, inv: f32) -> i8 {
    (x * inv).round() as i8
}

/// Int8 quantisation with one scale: v ≈ t · q, |q| ≤ 127.
pub fn quantize(v: &[f32]) -> (Vec<i8>, f32) {
    let t = scale(v);
    let inv = if t == 0.0 { 0.0 } else { 1.0 / t };
    (v.iter().map(|&x| code(x, inv)).collect(), t)
}

/// `quantize` for one column block, without allocating.
pub fn quantize_cols(v: &[f32; COLS]) -> ([i8; COLS], f32) {
    let t = scale(v);
    let inv = if t == 0.0 { 0.0 } else { 1.0 / t };
    (v.map(|x| code(x, inv)), t)
}

/// Folded, quantised activations of one column block at one group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Folded {
    pub xq: [i8; COLS],
    /// Int8 scale: x_j · s_j ≈ t · xq_j.
    pub t: f32,
    /// Σ_j x_j · min_j, the group's constant term.
    pub bias: f32,
}

/// Signature shared by the spec and NEON folding.
pub type Fold = fn(&[u8], &[usize; COLS], usize, &[f32; COLS]) -> Folded;

/// Spec: for each column c, reads (scale, min) as f16 at `bases[c] + off`,
/// folds `x[c] · scale`, quantises the block to int8 and sums x·min.
/// The bias uses 4 fused lanes (c mod 4) combined as (l0+l1)+(l2+l3), the
/// order the NEON version uses, so both are bit-identical.
pub fn fold_scalar(data: &[u8], bases: &[usize; COLS], off: usize, x: &[f32; COLS]) -> Folded {
    let param = |c: usize, o: usize| {
        let at = bases[c] + off + o;
        half::f16::from_le_bytes([data[at], data[at + 1]]).to_f32()
    };
    let folded: [f32; COLS] = std::array::from_fn(|c| x[c] * param(c, 0));
    let lanes = (0..COLS).fold([0.0f32; 4], |mut l, c| {
        l[c % 4] = x[c].mul_add(param(c, 2), l[c % 4]);
        l
    });
    let (xq, t) = quantize_cols(&folded);
    Folded { xq, t, bias: (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) }
}

/// One column block's (scale, min) at one group, decoded to f32. Shared by
/// every token of a batch, so prefill converts parameters once, not per token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub s: [f32; COLS],
    pub m: [f32; COLS],
}

pub fn params(data: &[u8], bases: &[usize; COLS], off: usize) -> Params {
    let param = |c: usize, o: usize| {
        let at = bases[c] + off + o;
        half::f16::from_le_bytes([data[at], data[at + 1]]).to_f32()
    };
    Params { s: std::array::from_fn(|c| param(c, 0)), m: std::array::from_fn(|c| param(c, 2)) }
}

/// Signature shared by the spec and NEON parameter folding.
pub type FoldParams = fn(&Params, &[f32; COLS]) -> Folded;

/// Spec: `fold_scalar` with pre-decoded parameters (identical arithmetic).
pub fn fold_params_scalar(p: &Params, x: &[f32; COLS]) -> Folded {
    let folded: [f32; COLS] = std::array::from_fn(|c| x[c] * p.s[c]);
    let lanes = (0..COLS).fold([0.0f32; 4], |mut l, c| {
        l[c % 4] = x[c].mul_add(p.m[c], l[c % 4]);
        l
    });
    let (xq, t) = quantize_cols(&folded);
    Folded { xq, t, bias: (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) }
}

/// Where one column's 16-byte nibble block for a 32-row block starts.
#[inline(always)]
fn nib(base: usize, nib_off: usize, blk: usize) -> usize {
    base + nib_off + blk * BLOCK / 2
}

/// Signature shared by the spec and the NEON kernel, so callers can be
/// run with either and compared bit for bit.
pub type DotCols = fn(&[u8], &[usize; COLS], usize, &[i8; COLS], &mut [i32]);

/// Spec: out[r] = Σ_c xq[c] · q_c[r] for `out.len()` rows (a multiple of
/// 32) of the columns whose nibbles start at `bases[c] + nib_off`.
pub fn dot_cols_scalar(data: &[u8], bases: &[usize; COLS], nib_off: usize, xq: &[i8; COLS], out: &mut [i32]) {
    for (r, o) in out.iter_mut().enumerate() {
        let (blk, k) = (r / BLOCK, r % BLOCK);
        *o = (0..COLS)
            .map(|c| {
                let byte = data[nib(bases[c], nib_off, blk) + k % 16];
                let q = if k < 16 { byte & 0x0f } else { byte >> 4 };
                q as i32 * xq[c] as i32
            })
            .sum();
    }
}

/// Spec: Σ_i a[i] · b[i], with `a` holding int8 values as raw bytes.
pub fn dot_i8_scalar(a: &[u8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i8 as i32 * y as i32).sum()
}

/// Tile layout (prefill): the columns of one block, for `rows` rows, as
/// int8 values 0..15 grouped exactly as the decode kernel's SDOT operands.
/// Vector v = (blk · 8 + quad) · 8 + i holds rows `blk·32 + 4i .. +4`, each
/// row the 4 columns `4·quad .. +4`: byte v·16 + r·4 + k.
pub fn tile_index(row: usize, col: usize) -> usize {
    let (blk, i, r) = (row / BLOCK, (row % BLOCK) / 4, row % 4);
    (((blk * (COLS / 4) + col / 4) * 8 + i) * 16) + r * 4 + col % 4
}

/// Signatures shared by the spec and NEON tile kernels.
pub type DecodeTile = fn(&[u8], &[usize; COLS], usize, &mut [u8]);
pub type TileDot = fn(&[u8], &[i8; COLS], &mut [i32]);

/// Spec: unpacks `tile.len() / COLS` rows of the block's columns into the
/// tile layout.
pub fn decode_tile_scalar(data: &[u8], bases: &[usize; COLS], nib_off: usize, tile: &mut [u8]) {
    let rows = tile.len() / COLS;
    for row in 0..rows {
        let (blk, k) = (row / BLOCK, row % BLOCK);
        for (col, &base) in bases.iter().enumerate() {
            let byte = data[nib(base, nib_off, blk) + k % 16];
            tile[tile_index(row, col)] = if k < 16 { byte & 0x0f } else { byte >> 4 };
        }
    }
}

/// Spec: out[r] = Σ_c tile[r, c] · xq[c] for `out.len()` rows.
pub fn tile_dot_scalar(tile: &[u8], xq: &[i8; COLS], out: &mut [i32]) {
    for (r, o) in out.iter_mut().enumerate() {
        *o = (0..COLS).map(|c| tile[tile_index(r, c)] as i32 * xq[c] as i32).sum();
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    use super::{BLOCK, COLS, nib};

    pub fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }

    /// Exact f16 → f32 for finite values, on u32 lanes holding f16 bits:
    /// shift the magnitude into f32 position and rescale by 2^112 (exact
    /// for normals and subnormals alike), then restore the sign.
    #[inline]
    #[target_feature(enable = "neon")]
    fn f16_to_f32(h: uint32x4_t) -> float32x4_t {
        let sign = vshlq_n_u32::<16>(vandq_u32(h, vdupq_n_u32(0x8000)));
        let mag = vreinterpretq_f32_u32(vshlq_n_u32::<13>(vandq_u32(h, vdupq_n_u32(0x7fff))));
        let scaled = vmulq_f32(mag, vdupq_n_f32(f32::from_bits(0x7780_0000)));
        vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(scaled), sign))
    }

    /// NEON twin of `fold_scalar`.
    ///
    /// # Safety
    /// Requires `dotprod`; every `bases[c] + off + 4` must be ≤ `data.len()`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn fold(data: &[u8], bases: &[usize; COLS], off: usize, x: &[f32; COLS]) -> super::Folded {
        let p = data.as_ptr();
        let mut folded = [vdupq_n_f32(0.0); COLS / 4];
        let mut lanes = vdupq_n_f32(0.0);
        let mut amax = vdupq_n_f32(0.0);
        for q in 0..COLS / 4 {
            let words: [u32; 4] = std::array::from_fn(|k| unsafe {
                p.add(bases[q * 4 + k] + off).cast::<u32>().read_unaligned().to_le()
            });
            let pairs = unsafe { vld1q_u32(words.as_ptr()) };
            let s = f16_to_f32(vandq_u32(pairs, vdupq_n_u32(0xffff)));
            let m = f16_to_f32(vshrq_n_u32::<16>(pairs));
            let xv = unsafe { vld1q_f32(x[q * 4..].as_ptr()) };
            folded[q] = vmulq_f32(xv, s);
            lanes = vfmaq_f32(lanes, xv, m);
            amax = vmaxq_f32(amax, vabsq_f32(folded[q]));
        }
        let t = vmaxvq_f32(amax) / 127.0;
        let inv = if t == 0.0 { 0.0 } else { 1.0 / t };
        let mut xq = [0i8; COLS];
        for (q, f) in folded.iter().enumerate().step_by(2) {
            let a = vcvtaq_s32_f32(vmulq_n_f32(*f, inv));
            let b = vcvtaq_s32_f32(vmulq_n_f32(folded[q + 1], inv));
            let n16 = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
            let n8 = vqmovn_s16(vcombine_s16(vget_low_s16(n16), vget_high_s16(n16)));
            unsafe { vst1_s8(xq[q * 4..].as_mut_ptr(), n8) };
        }
        let l = [vgetq_lane_f32::<0>(lanes), vgetq_lane_f32::<1>(lanes), vgetq_lane_f32::<2>(lanes), vgetq_lane_f32::<3>(lanes)];
        super::Folded { xq, t, bias: (l[0] + l[1]) + (l[2] + l[3]) }
    }

    /// NEON twin of `fold_params_scalar`.
    ///
    /// # Safety
    /// Requires `dotprod`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn fold_params(p: &super::Params, x: &[f32; COLS]) -> super::Folded {
        let mut folded = [vdupq_n_f32(0.0); COLS / 4];
        let mut lanes = vdupq_n_f32(0.0);
        let mut amax = vdupq_n_f32(0.0);
        for q in 0..COLS / 4 {
            let (s, m, xv) = unsafe { (vld1q_f32(p.s[q * 4..].as_ptr()), vld1q_f32(p.m[q * 4..].as_ptr()), vld1q_f32(x[q * 4..].as_ptr())) };
            folded[q] = vmulq_f32(xv, s);
            lanes = vfmaq_f32(lanes, xv, m);
            amax = vmaxq_f32(amax, vabsq_f32(folded[q]));
        }
        let t = vmaxvq_f32(amax) / 127.0;
        let inv = if t == 0.0 { 0.0 } else { 1.0 / t };
        let mut xq = [0i8; COLS];
        for (q, f) in folded.iter().enumerate().step_by(2) {
            let a = vcvtaq_s32_f32(vmulq_n_f32(*f, inv));
            let b = vcvtaq_s32_f32(vmulq_n_f32(folded[q + 1], inv));
            let n16 = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
            let n8 = vqmovn_s16(vcombine_s16(vget_low_s16(n16), vget_high_s16(n16)));
            unsafe { vst1_s8(xq[q * 4..].as_mut_ptr(), n8) };
        }
        let l = [vgetq_lane_f32::<0>(lanes), vgetq_lane_f32::<1>(lanes), vgetq_lane_f32::<2>(lanes), vgetq_lane_f32::<3>(lanes)];
        super::Folded { xq, t, bias: (l[0] + l[1]) + (l[2] + l[3]) }
    }

    /// NEON twin of `dot_cols_scalar`.
    ///
    /// # Safety
    /// Requires the `dotprod` feature, and every column's `out.len() / 2`
    /// nibble bytes from `bases[c] + nib_off` must lie inside `data`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_cols(data: &[u8], bases: &[usize; COLS], nib_off: usize, xq: &[i8; COLS], out: &mut [i32]) {
        for blk in 0..out.len() / BLOCK {
            // acc[i]: rows 4i..4i+4 of this 32-row block.
            let mut acc = [vdupq_n_s32(0); 8];
            for quad in 0..COLS / 4 {
                let c = quad * 4;
                let packed = i32::from_le_bytes([xq[c] as u8, xq[c + 1] as u8, xq[c + 2] as u8, xq[c + 3] as u8]);
                let xv = vreinterpretq_s8_s32(vdupq_n_s32(packed));
                let load = |k: usize| unsafe { vld1q_u8(data.as_ptr().add(nib(bases[c + k], nib_off, blk))) };
                let (v0, v1, v2, v3) = (load(0), load(1), load(2), load(3));
                let mask = vdupq_n_u8(0x0f);
                let lo = [vandq_u8(v0, mask), vandq_u8(v1, mask), vandq_u8(v2, mask), vandq_u8(v3, mask)];
                let hi = [vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1), vshrq_n_u8::<4>(v2), vshrq_n_u8::<4>(v3)];
                for (half, q) in [lo, hi].iter().enumerate() {
                    let z01l = vreinterpretq_u16_u8(vzip1q_u8(q[0], q[1]));
                    let z01h = vreinterpretq_u16_u8(vzip2q_u8(q[0], q[1]));
                    let z23l = vreinterpretq_u16_u8(vzip1q_u8(q[2], q[3]));
                    let z23h = vreinterpretq_u16_u8(vzip2q_u8(q[2], q[3]));
                    let rows4 = [
                        vzip1q_u16(z01l, z23l),
                        vzip2q_u16(z01l, z23l),
                        vzip1q_u16(z01h, z23h),
                        vzip2q_u16(z01h, z23h),
                    ];
                    for (i, r) in rows4.iter().enumerate() {
                        let a = &mut acc[half * 4 + i];
                        *a = vdotq_s32(*a, vreinterpretq_s8_u16(*r), xv);
                    }
                }
            }
            for (i, a) in acc.iter().enumerate() {
                let dst: &mut [i32; 4] = (&mut out[blk * BLOCK + i * 4..][..4]).try_into().expect("4 lanes");
                unsafe { vst1q_s32(dst.as_mut_ptr(), *a) };
            }
        }
    }

    /// NEON twin of `decode_tile_scalar`.
    ///
    /// # Safety
    /// Requires `dotprod`; every column's `tile.len() / COLS / 2` nibble
    /// bytes from `bases[c] + nib_off` must lie inside `data`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn decode_tile(data: &[u8], bases: &[usize; COLS], nib_off: usize, tile: &mut [u8]) {
        let blocks = tile.len() / COLS / BLOCK;
        for blk in 0..blocks {
            for quad in 0..COLS / 4 {
                let c = quad * 4;
                let load = |k: usize| unsafe { vld1q_u8(data.as_ptr().add(nib(bases[c + k], nib_off, blk))) };
                let (v0, v1, v2, v3) = (load(0), load(1), load(2), load(3));
                let mask = vdupq_n_u8(0x0f);
                let lo = [vandq_u8(v0, mask), vandq_u8(v1, mask), vandq_u8(v2, mask), vandq_u8(v3, mask)];
                let hi = [vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1), vshrq_n_u8::<4>(v2), vshrq_n_u8::<4>(v3)];
                for (half, q) in [lo, hi].iter().enumerate() {
                    let z01l = vreinterpretq_u16_u8(vzip1q_u8(q[0], q[1]));
                    let z01h = vreinterpretq_u16_u8(vzip2q_u8(q[0], q[1]));
                    let z23l = vreinterpretq_u16_u8(vzip1q_u8(q[2], q[3]));
                    let z23h = vreinterpretq_u16_u8(vzip2q_u8(q[2], q[3]));
                    let rows4 = [vzip1q_u16(z01l, z23l), vzip2q_u16(z01l, z23l), vzip1q_u16(z01h, z23h), vzip2q_u16(z01h, z23h)];
                    for (j, r) in rows4.iter().enumerate() {
                        let v = (blk * (COLS / 4) + quad) * 8 + half * 4 + j;
                        let dst: &mut [u8; 16] = (&mut tile[v * 16..v * 16 + 16]).try_into().expect("16 bytes");
                        unsafe { vst1q_u8(dst.as_mut_ptr(), vreinterpretq_u8_u16(*r)) };
                    }
                }
            }
        }
    }

    /// NEON twin of `tile_dot_scalar`.
    ///
    /// # Safety
    /// Requires `dotprod`; `tile` holds at least `out.len() · COLS` bytes.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn tile_dot(tile: &[u8], xq: &[i8; COLS], out: &mut [i32]) {
        let xv: [int8x16_t; COLS / 4] = std::array::from_fn(|q| {
            let packed = i32::from_le_bytes([xq[q * 4] as u8, xq[q * 4 + 1] as u8, xq[q * 4 + 2] as u8, xq[q * 4 + 3] as u8]);
            vreinterpretq_s8_s32(vdupq_n_s32(packed))
        });
        for blk in 0..out.len() / BLOCK {
            let mut acc = [vdupq_n_s32(0); 8];
            for (quad, x) in xv.iter().enumerate() {
                for (i, a) in acc.iter_mut().enumerate() {
                    let v = (blk * (COLS / 4) + quad) * 8 + i;
                    *a = vdotq_s32(*a, unsafe { vld1q_s8(tile.as_ptr().add(v * 16).cast::<i8>()) }, *x);
                }
            }
            for (i, a) in acc.iter().enumerate() {
                let dst: &mut [i32; 4] = (&mut out[blk * BLOCK + i * 4..][..4]).try_into().expect("4 lanes");
                unsafe { vst1q_s32(dst.as_mut_ptr(), *a) };
            }
        }
    }

    /// NEON twin of `dot_i8_scalar` for lengths that are multiples of 16.
    ///
    /// # Safety
    /// Requires the `dotprod` feature; `a` and `b` have equal length, a
    /// multiple of 16.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_i8(a: &[u8], b: &[i8]) -> i32 {
        let (xa, _) = a.as_chunks::<16>();
        let (ya, _) = b.as_chunks::<16>();
        let acc = xa.iter().zip(ya).fold(vdupq_n_s32(0), |acc, (x, y)| unsafe {
            vdotq_s32(acc, vreinterpretq_s8_u8(vld1q_u8(x.as_ptr())), vld1q_s8(y.as_ptr()))
        });
        vaddvq_s32(acc)
    }
}

#[cfg(target_arch = "aarch64")]
fn max_base(bases: &[usize; COLS]) -> usize {
    bases.iter().copied().max().unwrap_or(0)
}

/// Dispatches to NEON when available, else the scalar spec.
pub fn dot_cols(data: &[u8], bases: &[usize; COLS], nib_off: usize, xq: &[i8; COLS], out: &mut [i32]) {
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        assert!(out.len().is_multiple_of(BLOCK) && max_base(bases) + nib_off + out.len() / 2 <= data.len(), "column block out of bounds");
        // SAFETY: dotprod checked; every column's loads are within the bound asserted above.
        return unsafe { neon::dot_cols(data, bases, nib_off, xq, out) };
    }
    dot_cols_scalar(data, bases, nib_off, xq, out)
}

/// Dispatches to NEON when available, else the scalar spec.
pub fn fold(data: &[u8], bases: &[usize; COLS], off: usize, x: &[f32; COLS]) -> Folded {
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        assert!(max_base(bases) + off + 4 <= data.len(), "column params out of bounds");
        // SAFETY: dotprod checked; every 4-byte param read is within the bound asserted above.
        return unsafe { neon::fold(data, bases, off, x) };
    }
    fold_scalar(data, bases, off, x)
}

/// Dispatches to NEON when available, else the scalar spec.
pub fn fold_params(p: &Params, x: &[f32; COLS]) -> Folded {
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        // SAFETY: dotprod checked; only fixed-size arrays are read.
        return unsafe { neon::fold_params(p, x) };
    }
    fold_params_scalar(p, x)
}

/// Dispatches to NEON when available, else the scalar spec.
pub fn decode_tile(data: &[u8], bases: &[usize; COLS], nib_off: usize, tile: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        let rows = tile.len() / COLS;
        assert!(rows.is_multiple_of(BLOCK) && max_base(bases) + nib_off + rows / 2 <= data.len(), "tile out of bounds");
        // SAFETY: dotprod checked; every column's loads are within the bound asserted above.
        return unsafe { neon::decode_tile(data, bases, nib_off, tile) };
    }
    decode_tile_scalar(data, bases, nib_off, tile)
}

/// Dispatches to NEON when available, else the scalar spec.
pub fn tile_dot(tile: &[u8], xq: &[i8; COLS], out: &mut [i32]) {
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        assert!(out.len().is_multiple_of(BLOCK) && tile.len() >= out.len() * COLS, "tile too small");
        // SAFETY: dotprod checked; the tile covers every row read (asserted above).
        return unsafe { neon::tile_dot(tile, xq, out) };
    }
    tile_dot_scalar(tile, xq, out)
}

pub fn dot_i8(a: &[u8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if neon::available() && a.len() == b.len() && a.len().is_multiple_of(16) {
        // SAFETY: dotprod checked; equal lengths, multiple of 16.
        return unsafe { neon::dot_i8(a, b) };
    }
    dot_i8_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        (0..n as u64).map(|i| ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7) as u8).collect()
    }

    fn xq(seed: i32) -> [i8; COLS] {
        std::array::from_fn(|i| (((i as i32 * 37 + seed * 11) % 255) - 127) as i8)
    }

    #[test]
    fn dispatch_equals_scalar_spec_exactly() {
        let stride = 1000;
        let data = bytes(COLS * stride + 64, 3);
        let bases: [usize; COLS] = std::array::from_fn(|c| ((c * 7) % COLS) * stride);
        for rows in [32, 64] {
            for off in [0, 16, 200] {
                let x = xq(off as i32);
                let (mut fast, mut spec) = (vec![0; rows], vec![0; rows]);
                dot_cols(&data, &bases, off, &x, &mut fast);
                dot_cols_scalar(&data, &bases, off, &x, &mut spec);
                assert_eq!(fast, spec);
            }
        }
    }

    #[test]
    fn scalar_spec_decodes_the_nibble_layout() {
        // One column whose block holds q[k] = k for rows 0..15 and 15-k for 16..31.
        let block: Vec<u8> = (0..16u8).map(|k| k | (15 - k) << 4).collect();
        let data = [block, vec![0; 16]].concat();
        let bases = [0usize; COLS];
        let mut x = [0i8; COLS];
        x[0] = 2;
        let mut got = vec![0; 32];
        dot_cols_scalar(&data, &bases, 0, &x, &mut got);
        let want: Vec<i32> = (0..32).map(|r| if r < 16 { 2 * r } else { 2 * (31 - r) }).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn i8_dot_dispatch_matches_spec() {
        let a = bytes(64, 1);
        let b: Vec<i8> = bytes(64, 2).into_iter().map(|b| b as i8).collect();
        assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b));
    }

    #[test]
    fn fold_dispatch_equals_scalar_spec_exactly() {
        // Params include zero, subnormal, negative and large f16 values.
        let specials = [0.0f32, 6.0e-8, -3.1e-5, 0.0123, -0.5, 1.0, 1234.0, -60000.0];
        let data: Vec<u8> = (0..COLS * 8)
            .flat_map(|i| {
                let (s, m) = (specials[i % 8], specials[(i * 3 + 1) % 8]);
                [half::f16::from_f32(s).to_le_bytes(), half::f16::from_f32(m).to_le_bytes()].concat()
            })
            .collect();
        let bases: [usize; COLS] = std::array::from_fn(|c| c * 16);
        for off in [0, 4, 8] {
            let x: [f32; COLS] = std::array::from_fn(|c| ((c as f32) * 0.7 + off as f32).sin() * 3.0);
            assert_eq!(fold(&data, &bases, off, &x), fold_scalar(&data, &bases, off, &x));
        }
        let zero = [0.0f32; COLS];
        assert_eq!(fold(&data, &bases, 0, &zero), fold_scalar(&data, &bases, 0, &zero));
    }

    #[test]
    fn fold_with_decoded_params_equals_fold_exactly() {
        let specials = [0.0f32, 6.0e-8, -3.1e-5, 0.0123, -0.5, 1.0, 1234.0, -60000.0];
        let data: Vec<u8> = (0..COLS * 8)
            .flat_map(|i| [half::f16::from_f32(specials[i % 8]).to_le_bytes(), half::f16::from_f32(specials[(i * 5 + 2) % 8]).to_le_bytes()].concat())
            .collect();
        let bases: [usize; COLS] = std::array::from_fn(|c| c * 16);
        for off in [0, 4] {
            let x: [f32; COLS] = std::array::from_fn(|c| ((c as f32) * 0.9 + off as f32).cos() * 2.0);
            let p = params(&data, &bases, off);
            let want = fold_scalar(&data, &bases, off, &x);
            assert_eq!(fold_params(&p, &x), want);
            assert_eq!(fold_params_scalar(&p, &x), want);
        }
    }

    #[test]
    fn tile_kernels_equal_spec_and_compose_to_dot_cols() {
        let stride = 700;
        let data = bytes(COLS * stride + 64, 5);
        let bases: [usize; COLS] = std::array::from_fn(|c| ((c * 11) % COLS) * stride);
        for rows in [32, 64] {
            let (mut fast, mut spec) = (vec![0u8; rows * COLS], vec![0u8; rows * COLS]);
            decode_tile(&data, &bases, 24, &mut fast);
            decode_tile_scalar(&data, &bases, 24, &mut spec);
            assert_eq!(fast, spec);
            let x = xq(rows as i32);
            let (mut a, mut b, mut c) = (vec![0; rows], vec![0; rows], vec![0; rows]);
            tile_dot(&spec, &x, &mut a);
            tile_dot_scalar(&spec, &x, &mut b);
            dot_cols_scalar(&data, &bases, 24, &x, &mut c);
            assert_eq!(a, b);
            assert_eq!(b, c, "tile path must equal the decode path");
        }
    }

    #[test]
    fn quantize_bounds_error() {
        let v: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.3).collect();
        let (q, t) = quantize(&v);
        assert!(v.iter().zip(&q).all(|(a, &b)| (a - t * b as f32).abs() <= t / 2.0 + 1e-6));
        assert_eq!(quantize(&[0.0; 4]), (vec![0; 4], 0.0));
    }
}
