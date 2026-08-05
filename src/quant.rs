//! Block-quantised tensor views and the fused dequantise + matmul kernels.
//!
//! Every weight in the engine is exposed as a [`QT`]: a borrowed, row-major 2-D
//! view over bytes that live in a memory map. Kernels dequantise one row into a
//! small f32 scratch buffer and then dot it against every activation column in
//! the batch, so the dequantisation cost is amortised over the batch and the hot
//! loop is a single SIMD dot product regardless of the storage format.

use rayon::prelude::*;

/// Elements per quantisation block. Chosen so a block is one AVX2 register pair.
pub const BLK: usize = 32;

/// Storage format of a weight tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dt {
    F32,
    F16,
    BF16,
    /// 8-bit symmetric, f16 scale per 32 values (8.5 bits/weight).
    Q8,
    /// 4-bit symmetric, f16 scale per 32 values (4.5 bits/weight).
    Q4,
    /// 6-bit symmetric, f16 scale per 32 values (6.5 bits/weight).
    Q6,
    /// 5-bit symmetric, f16 scale per 32 values (5.5 bits/weight).
    Q5,
}

impl Dt {
    pub fn parse(s: &str) -> Option<Dt> {
        Some(match s {
            "F32" | "f32" => Dt::F32,
            "F16" | "f16" => Dt::F16,
            "BF16" | "bf16" => Dt::BF16,
            "Q8" | "q8" => Dt::Q8,
            "Q4" | "q4" => Dt::Q4,
            "Q5" | "q5" => Dt::Q5,
            "Q6" | "q6" => Dt::Q6,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Dt::F32 => "F32",
            Dt::F16 => "F16",
            Dt::BF16 => "BF16",
            Dt::Q8 => "Q8",
            Dt::Q4 => "Q4",
            Dt::Q5 => "Q5",
            Dt::Q6 => "Q6",
        }
    }

    /// Bytes needed for `cols` contiguous elements.
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            Dt::F32 => cols * 4,
            Dt::F16 | Dt::BF16 => cols * 2,
            Dt::Q8 => cols / BLK * (2 + BLK),
            Dt::Q4 => cols / BLK * (2 + BLK / 2),
            // A nibble each, plus one spare bit per value packed 8 to a byte.
            Dt::Q5 => cols / BLK * (2 + BLK / 2 + BLK / 8),
            // A nibble each, plus two spare bits per value packed 4 to a byte.
            Dt::Q6 => cols / BLK * (2 + BLK / 2 + BLK / 4),
        }
    }

    /// Whether a row of `cols` elements can be stored in this format.
    pub fn fits(self, cols: usize) -> bool {
        match self {
            Dt::Q8 | Dt::Q4 | Dt::Q5 | Dt::Q6 => cols % BLK == 0,
            _ => true,
        }
    }
}

/// A row-major 2-D weight view. `data` points into a memory map that outlives
/// the process, which is what lets the whole engine stay copy-free.
#[derive(Clone, Copy)]
pub struct QT {
    pub dt: Dt,
    pub rows: usize,
    pub cols: usize,
    pub data: &'static [u8],
}

impl QT {
    pub fn new(dt: Dt, rows: usize, cols: usize, data: &'static [u8]) -> QT {
        debug_assert_eq!(data.len(), rows * dt.row_bytes(cols));
        QT { dt, rows, cols, data }
    }

    pub fn row(&self, r: usize) -> &'static [u8] {
        let s = self.dt.row_bytes(self.cols);
        &self.data[r * s..(r + 1) * s]
    }

    /// Dequantise row `r` into `out` (length `cols`).
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        dequant(self.dt, self.row(r), out)
    }

    /// Read the whole tensor as f32. Used for norms and other tiny tensors.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            self.dequant_row(r, &mut v[r * self.cols..(r + 1) * self.cols]);
        }
        v
    }
}

#[inline]
fn f16_to_f32(b: u16) -> f32 {
    let sign = ((b >> 15) as u32) << 31;
    let exp = ((b >> 10) & 0x1f) as u32;
    let man = (b & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: value is man * 2^-24, renormalised into a f32 exponent.
            let lz = man.leading_zeros();
            sign | ((134 - lz) << 23) | ((man << (lz - 8)) & 0x7f_ffff)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[inline]
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let man = x & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let man = (man | 0x80_0000) >> (1 - exp) as u32;
        return sign | ((man + 0x1000) >> 13) as u16;
    }
    // Round to nearest even, letting mantissa overflow bump the exponent.
    let rounded = man + 0x0fff + ((man >> 13) & 1);
    if rounded & 0x80_0000 != 0 {
        exp += 1;
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | ((exp as u16) << 10) | ((rounded & 0x7f_ffff) >> 13) as u16
}

/// Expand one packed row into f32.
pub fn dequant(dt: Dt, src: &[u8], out: &mut [f32]) {
    match dt {
        Dt::F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Dt::F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Dt::BF16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        Dt::Q8 => {
            for (blk, o) in src.chunks_exact(2 + BLK).zip(out.chunks_mut(BLK)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..BLK {
                    o[i] = d * (blk[2 + i] as i8) as f32;
                }
            }
        }
        Dt::Q4 => {
            for (blk, o) in src.chunks_exact(2 + BLK / 2).zip(out.chunks_mut(BLK)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..BLK / 2 {
                    let b = blk[2 + i];
                    o[i] = d * ((b & 0x0f) as i32 - 8) as f32;
                    o[i + BLK / 2] = d * ((b >> 4) as i32 - 8) as f32;
                }
            }
        }
        // Q5 and Q6 keep Q4's nibble pairing — value `i` in the low half of byte
        // `i`, value `i + 16` in the high half — and add the remaining bits in a
        // trailing plane. Sharing the layout means the same indexing reasoning
        // holds for all three.
        Dt::Q5 => {
            for (blk, o) in src.chunks_exact(2 + BLK / 2 + BLK / 8).zip(out.chunks_mut(BLK)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let hi = &blk[2 + BLK / 2..];
                let bit = |j: usize| ((hi[j / 8] >> (j % 8)) & 1) as i32;
                for i in 0..BLK / 2 {
                    let b = blk[2 + i];
                    o[i] = d * (((b & 0x0f) as i32 | (bit(i) << 4)) - 16) as f32;
                    o[i + BLK / 2] = d * (((b >> 4) as i32 | (bit(i + BLK / 2) << 4)) - 16) as f32;
                }
            }
        }
        Dt::Q6 => {
            for (blk, o) in src.chunks_exact(2 + BLK / 2 + BLK / 4).zip(out.chunks_mut(BLK)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let hi = &blk[2 + BLK / 2..];
                let pair = |j: usize| ((hi[j / 4] >> ((j % 4) * 2)) & 0x3) as i32;
                for i in 0..BLK / 2 {
                    let b = blk[2 + i];
                    o[i] = d * (((b & 0x0f) as i32 | (pair(i) << 4)) - 32) as f32;
                    o[i + BLK / 2] = d * (((b >> 4) as i32 | (pair(i + BLK / 2) << 4)) - 32) as f32;
                }
            }
        }
    }
}

/// Pack one f32 row into `dt`. `dst` must be `dt.row_bytes(src.len())` long.
pub fn quantize(dt: Dt, src: &[f32], dst: &mut [u8]) {
    match dt {
        Dt::F32 => {
            for (v, c) in src.iter().zip(dst.chunks_exact_mut(4)) {
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        Dt::F16 => {
            for (v, c) in src.iter().zip(dst.chunks_exact_mut(2)) {
                c.copy_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
        }
        Dt::BF16 => {
            for (v, c) in src.iter().zip(dst.chunks_exact_mut(2)) {
                // Round to nearest even on the truncated mantissa.
                let b = v.to_bits();
                let r = ((b >> 16) & 1) + 0x7fff;
                c.copy_from_slice(&(((b + r) >> 16) as u16).to_le_bytes());
            }
        }
        Dt::Q8 => {
            for (s, blk) in src.chunks_exact(BLK).zip(dst.chunks_exact_mut(2 + BLK)) {
                let amax = s.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                blk[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                let inv = if d > 0.0 { 1.0 / f16_to_f32(f32_to_f16(d)) } else { 0.0 };
                for i in 0..BLK {
                    blk[2 + i] = (s[i] * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        }
        Dt::Q4 => {
            for (s, blk) in src.chunks_exact(BLK).zip(dst.chunks_exact_mut(2 + BLK / 2)) {
                let amax = s.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 7.0;
                blk[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                let inv = if d > 0.0 { 1.0 / f16_to_f32(f32_to_f16(d)) } else { 0.0 };
                let q = |v: f32| ((v * inv).round() as i32 + 8).clamp(0, 15) as u8;
                for i in 0..BLK / 2 {
                    blk[2 + i] = q(s[i]) | (q(s[i + BLK / 2]) << 4);
                }
            }
        }
        Dt::Q5 => {
            for (s, blk) in src.chunks_exact(BLK).zip(dst.chunks_exact_mut(2 + BLK / 2 + BLK / 8)) {
                let amax = s.iter().fold(0f32, |m, v| m.max(v.abs()));
                // 15 levels either side of zero, so the offset is 16.
                let d = amax / 15.0;
                blk[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                let inv = if d > 0.0 { 1.0 / f16_to_f32(f32_to_f16(d)) } else { 0.0 };
                let q = |v: f32| ((v * inv).round() as i32 + 16).clamp(0, 31) as u8;
                blk[2..].iter_mut().for_each(|b| *b = 0);
                for i in 0..BLK / 2 {
                    let (lo, hi) = (q(s[i]), q(s[i + BLK / 2]));
                    blk[2 + i] = (lo & 0x0f) | ((hi & 0x0f) << 4);
                    for (j, v) in [(i, lo), (i + BLK / 2, hi)] {
                        blk[2 + BLK / 2 + j / 8] |= ((v >> 4) & 1) << (j % 8);
                    }
                }
            }
        }
        Dt::Q6 => {
            for (s, blk) in src.chunks_exact(BLK).zip(dst.chunks_exact_mut(2 + BLK / 2 + BLK / 4)) {
                let amax = s.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 31.0;
                blk[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                let inv = if d > 0.0 { 1.0 / f16_to_f32(f32_to_f16(d)) } else { 0.0 };
                let q = |v: f32| ((v * inv).round() as i32 + 32).clamp(0, 63) as u8;
                blk[2..].iter_mut().for_each(|b| *b = 0);
                for i in 0..BLK / 2 {
                    let (lo, hi) = (q(s[i]), q(s[i + BLK / 2]));
                    blk[2 + i] = (lo & 0x0f) | ((hi & 0x0f) << 4);
                    for (j, v) in [(i, lo), (i + BLK / 2, hi)] {
                        blk[2 + BLK / 2 + j / 4] |= ((v >> 4) & 0x3) << ((j % 4) * 2);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 16 <= n {
        let x0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let y0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);
        let x1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let y1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(x1, y1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let x = _mm256_loadu_ps(a.as_ptr().add(i));
        let y = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(x, y, acc0);
        i += 8;
    }
    let s = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(s, 1);
    let lo = _mm256_castps256_ps128(s);
    let mut q = _mm_add_ps(hi, lo);
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    let mut out = _mm_cvtss_f32(q);
    while i < n {
        out += a[i] * b[i];
        i += 1;
    }
    out
}

/// NEON is part of the aarch64 baseline, so this needs no runtime check.
#[cfg(target_arch = "aarch64")]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    unsafe {
        let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0;
        while i + 8 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
            i += 8;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            i += 4;
        }
        let mut out = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            out += a[i] * b[i];
            i += 1;
        }
        out
    }
}

/// Dot product of two equal-length f32 slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return dot_neon(a, b);
    }
    #[allow(unreachable_code)]
    // Four independent accumulators so the scalar path still pipelines and
    // auto-vectorises (this is the hot loop on aarch64).
    let mut acc = [0.0f32; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..4 {
            acc[i] += x[i] * y[i];
        }
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

/// `out[t][r] = w[r] . x[t]` for a batch of `t` activation vectors.
///
/// Rows are split across the thread pool; each row is dequantised once and
/// reused for the whole batch, which is what makes prefill cheap. Results are
/// accumulated row-major and transposed at the end — an O(rows*t) shuffle that
/// disappears next to the O(rows*cols*t) arithmetic.
pub fn matmul(w: &QT, x: &[f32], out: &mut [f32]) {
    let (rows, cols) = (w.rows, w.cols);
    let t = x.len() / cols;
    debug_assert_eq!(out.len(), t * rows);
    let mut acc = vec![0.0f32; rows * t];
    let band = (rows / (rayon::current_num_threads() * 4)).clamp(1, 64);
    acc.par_chunks_mut(band * t).enumerate().for_each(|(bi, chunk)| {
        let mut buf = vec![0.0f32; cols];
        for (i, cell) in chunk.chunks_mut(t).enumerate() {
            w.dequant_row(bi * band + i, &mut buf);
            for (ti, o) in cell.iter_mut().enumerate() {
                *o = dot(&buf, &x[ti * cols..(ti + 1) * cols]);
            }
        }
    });
    if t == 1 {
        out.copy_from_slice(&acc);
    } else {
        for r in 0..rows {
            for ti in 0..t {
                out[ti * rows + r] = acc[r * t + ti];
            }
        }
    }
}

/// Single-vector convenience wrapper around [`matmul`].
pub fn matvec(w: &QT, x: &[f32], out: &mut [f32]) {
    matmul(w, x, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leak(v: Vec<u8>) -> &'static [u8] {
        Box::leak(v.into_boxed_slice())
    }

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, 1e-5, -3.5, 1e-8] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
    }

    #[test]
    fn quant_roundtrip_within_tolerance() {
        let row: Vec<f32> = (0..128).map(|i| ((i * 37 % 71) as f32 - 35.0) / 10.0).collect();
        for (dt, tol) in [
            (Dt::F32, 1e-6),
            (Dt::F16, 1e-2),
            (Dt::BF16, 5e-2),
            (Dt::Q8, 3e-2),
            (Dt::Q6, 6e-2),
            (Dt::Q5, 0.12),
            (Dt::Q4, 0.4),
        ] {
            let mut packed = vec![0u8; dt.row_bytes(row.len())];
            quantize(dt, &row, &mut packed);
            let mut back = vec![0.0; row.len()];
            dequant(dt, &packed, &mut back);
            for (a, b) in row.iter().zip(&back) {
                assert!((a - b).abs() < tol, "{:?}: {a} vs {b}", dt);
            }
        }
    }

    /// The block formats must sit in the right order on both axes: more bits
    /// means more bytes and less error. A layout bug in the spare-bit planes
    /// would show up here as Q5 or Q6 being no better than Q4.
    #[test]
    fn more_bits_costs_more_space_and_loses_less() {
        // A smooth ramp with an outlier, so the per-block scale is exercised.
        let row: Vec<f32> = (0..256).map(|i| (i as f32 / 255.0 - 0.5) * if i == 3 { 9.0 } else { 2.0 }).collect();
        let mut prev: Option<(Dt, usize, f32)> = None;
        for dt in [Dt::Q4, Dt::Q5, Dt::Q6, Dt::Q8] {
            let bytes = dt.row_bytes(row.len());
            let mut packed = vec![0u8; bytes];
            quantize(dt, &row, &mut packed);
            let mut back = vec![0.0; row.len()];
            dequant(dt, &packed, &mut back);
            let err = row.iter().zip(&back).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            if let Some((pdt, pbytes, perr)) = prev {
                assert!(bytes > pbytes, "{dt:?} is not larger than {pdt:?}: {bytes} vs {pbytes}");
                assert!(err < perr, "{dt:?} is not more accurate than {pdt:?}: {err} vs {perr}");
            }
            prev = Some((dt, bytes, err));
        }
        // The advertised widths, to the byte: 4.5, 5.5, 6.5 and 8.5 bits.
        for (dt, bits) in [(Dt::Q4, 4.5), (Dt::Q5, 5.5), (Dt::Q6, 6.5), (Dt::Q8, 8.5)] {
            let got = dt.row_bytes(BLK) as f32 * 8.0 / BLK as f32;
            assert!((got - bits).abs() < 1e-6, "{dt:?} costs {got} bits per weight, not {bits}");
        }
    }

    /// Every representable level must survive a round trip, or the high-bit
    /// plane is being packed or read at the wrong offset.
    #[test]
    fn every_level_round_trips() {
        for (dt, levels) in [(Dt::Q4, 15i32), (Dt::Q5, 31), (Dt::Q6, 63), (Dt::Q8, 255)] {
            let half = levels / 2;
            // Exactly the grid the format quantises to, scaled by 1.
            let row: Vec<f32> = (0..BLK).map(|i| (i as i32 % (half + 1) - half / 2) as f32).collect();
            let mut packed = vec![0u8; dt.row_bytes(row.len())];
            quantize(dt, &row, &mut packed);
            let mut back = vec![0.0; row.len()];
            dequant(dt, &packed, &mut back);
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let step = amax / (half as f32).max(1.0);
            for (i, (a, b)) in row.iter().zip(&back).enumerate() {
                assert!((a - b).abs() <= step * 0.75, "{dt:?} position {i}: {a} came back as {b} (step {step})");
            }
        }
    }

    #[test]
    fn names_round_trip_through_parse() {
        for dt in [Dt::F32, Dt::F16, Dt::BF16, Dt::Q8, Dt::Q6, Dt::Q5, Dt::Q4] {
            assert_eq!(Dt::parse(dt.name()), Some(dt));
            assert_eq!(Dt::parse(&dt.name().to_lowercase()), Some(dt));
        }
        assert_eq!(Dt::parse("q3"), None);
    }

    #[test]
    fn matmul_matches_reference() {
        let (rows, cols, t) = (35usize, 64usize, 3usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();
        let x: Vec<f32> = (0..t * cols).map(|i| ((i % 9) as f32 - 4.0) / 5.0).collect();
        let mut packed = vec![0u8; rows * Dt::F32.row_bytes(cols)];
        for r in 0..rows {
            let off = r * Dt::F32.row_bytes(cols);
            quantize(Dt::F32, &w[r * cols..(r + 1) * cols], &mut packed[off..off + cols * 4]);
        }
        let qt = QT::new(Dt::F32, rows, cols, leak(packed));
        let mut out = vec![0.0; t * rows];
        matmul(&qt, &x, &mut out);
        for ti in 0..t {
            for r in 0..rows {
                let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[ti * cols + c]).sum();
                assert!((out[ti * rows + r] - want).abs() < 1e-3);
            }
        }
    }
}
