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
}

impl Dt {
    pub fn parse(s: &str) -> Option<Dt> {
        Some(match s {
            "F32" | "f32" => Dt::F32,
            "F16" | "f16" => Dt::F16,
            "BF16" | "bf16" => Dt::BF16,
            "Q8" | "q8" => Dt::Q8,
            "Q4" | "q4" => Dt::Q4,
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
        }
    }

    /// Bytes needed for `cols` contiguous elements.
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            Dt::F32 => cols * 4,
            Dt::F16 | Dt::BF16 => cols * 2,
            Dt::Q8 => cols / BLK * (2 + BLK),
            Dt::Q4 => cols / BLK * (2 + BLK / 2),
        }
    }

    /// Whether a row of `cols` elements can be stored in this format.
    pub fn fits(self, cols: usize) -> bool {
        match self {
            Dt::Q8 | Dt::Q4 => cols % BLK == 0,
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
        for (dt, tol) in [(Dt::F32, 1e-6), (Dt::F16, 1e-2), (Dt::BF16, 5e-2), (Dt::Q8, 3e-2), (Dt::Q4, 0.4)] {
            let mut packed = vec![0u8; dt.row_bytes(row.len())];
            quantize(dt, &row, &mut packed);
            let mut back = vec![0.0; row.len()];
            dequant(dt, &packed, &mut back);
            for (a, b) in row.iter().zip(&back) {
                assert!((a - b).abs() < tol, "{:?}: {a} vs {b}", dt);
            }
        }
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
