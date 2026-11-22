use crate::dtype::{f16_to_f32, BYTES_Q4_0, BYTES_Q8_0};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_ps_avx2(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128::<8>(sum128));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128::<4>(sum64));
    _mm_cvtsi128_si32(sum32)
}

/// Computes `a · b` using 256-bit AVX2+FMA vector accumulators.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported at runtime
/// and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` guarantees all four 8-float unaligned loads are in bounds.
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            acc3,
        );
        i += 32;
    }

    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` guarantees 8-float load is in bounds.
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc);
        i += 8;
    }

    let mut sum = hsum_ps_avx2(acc);
    while i < n {
        sum = (*pa.add(i)).mul_add(*pb.add(i), sum);
        i += 1;
    }
    sum
}

/// Register-blocked `6x16` AVX2+FMA GEMM row-tile kernel with L1 `KC` blocking.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported at runtime and
/// `a_tile.len() >= tile_m * k`, `b.len() >= k * n`, `out_tile.len() >= tile_m * n`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_f32_row_tile_avx2(
    a_tile: &[f32],
    b: &[f32],
    tile_m: usize,
    k: usize,
    n: usize,
    out_tile: &mut [f32],
) {
    debug_assert!(a_tile.len() >= tile_m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(out_tile.len() >= tile_m * n);

    out_tile[..tile_m * n].fill(0.0);
    let pa = a_tile.as_ptr();
    let pb = b.as_ptr();
    let pc = out_tile.as_mut_ptr();

    const KC: usize = 128;
    let mut p0 = 0usize;
    while p0 < k {
        let p_end = (p0 + KC).min(k);

        let mut r = 0usize;
        while r + 6 <= tile_m {
            let mut c = 0usize;
            while c + 16 <= n {
                // SAFETY: `r + 6 <= tile_m` and `c + 16 <= n` keep all 6x16 loads/stores in bounds.
                let mut c00 = _mm256_loadu_ps(pc.add((r) * n + c));
                let mut c01 = _mm256_loadu_ps(pc.add((r) * n + c + 8));
                let mut c10 = _mm256_loadu_ps(pc.add((r + 1) * n + c));
                let mut c11 = _mm256_loadu_ps(pc.add((r + 1) * n + c + 8));
                let mut c20 = _mm256_loadu_ps(pc.add((r + 2) * n + c));
                let mut c21 = _mm256_loadu_ps(pc.add((r + 2) * n + c + 8));
                let mut c30 = _mm256_loadu_ps(pc.add((r + 3) * n + c));
                let mut c31 = _mm256_loadu_ps(pc.add((r + 3) * n + c + 8));
                let mut c40 = _mm256_loadu_ps(pc.add((r + 4) * n + c));
                let mut c41 = _mm256_loadu_ps(pc.add((r + 4) * n + c + 8));
                let mut c50 = _mm256_loadu_ps(pc.add((r + 5) * n + c));
                let mut c51 = _mm256_loadu_ps(pc.add((r + 5) * n + c + 8));

                for p in p0..p_end {
                    let b0 = _mm256_loadu_ps(pb.add(p * n + c));
                    let b1 = _mm256_loadu_ps(pb.add(p * n + c + 8));

                    let a0 = _mm256_set1_ps(*pa.add((r) * k + p));
                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c01 = _mm256_fmadd_ps(a0, b1, c01);

                    let a1 = _mm256_set1_ps(*pa.add((r + 1) * k + p));
                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                    c11 = _mm256_fmadd_ps(a1, b1, c11);

                    let a2 = _mm256_set1_ps(*pa.add((r + 2) * k + p));
                    c20 = _mm256_fmadd_ps(a2, b0, c20);
                    c21 = _mm256_fmadd_ps(a2, b1, c21);

                    let a3 = _mm256_set1_ps(*pa.add((r + 3) * k + p));
                    c30 = _mm256_fmadd_ps(a3, b0, c30);
                    c31 = _mm256_fmadd_ps(a3, b1, c31);

                    let a4 = _mm256_set1_ps(*pa.add((r + 4) * k + p));
                    c40 = _mm256_fmadd_ps(a4, b0, c40);
                    c41 = _mm256_fmadd_ps(a4, b1, c41);

                    let a5 = _mm256_set1_ps(*pa.add((r + 5) * k + p));
                    c50 = _mm256_fmadd_ps(a5, b0, c50);
                    c51 = _mm256_fmadd_ps(a5, b1, c51);
                }

                _mm256_storeu_ps(pc.add((r) * n + c), c00);
                _mm256_storeu_ps(pc.add((r) * n + c + 8), c01);
                _mm256_storeu_ps(pc.add((r + 1) * n + c), c10);
                _mm256_storeu_ps(pc.add((r + 1) * n + c + 8), c11);
                _mm256_storeu_ps(pc.add((r + 2) * n + c), c20);
                _mm256_storeu_ps(pc.add((r + 2) * n + c + 8), c21);
                _mm256_storeu_ps(pc.add((r + 3) * n + c), c30);
                _mm256_storeu_ps(pc.add((r + 3) * n + c + 8), c31);
                _mm256_storeu_ps(pc.add((r + 4) * n + c), c40);
                _mm256_storeu_ps(pc.add((r + 4) * n + c + 8), c41);
                _mm256_storeu_ps(pc.add((r + 5) * n + c), c50);
                _mm256_storeu_ps(pc.add((r + 5) * n + c + 8), c51);

                c += 16;
            }

            // Column tail for the 6-row group
            if c < n {
                for sub_r in r..r + 6 {
                    for p in p0..p_end {
                        let a_val = *pa.add(sub_r * k + p);
                        let av = _mm256_set1_ps(a_val);
                        let mut cj = c;
                        while cj + 8 <= n {
                            let cv = _mm256_loadu_ps(pc.add(sub_r * n + cj));
                            let bv = _mm256_loadu_ps(pb.add(p * n + cj));
                            _mm256_storeu_ps(pc.add(sub_r * n + cj), _mm256_fmadd_ps(av, bv, cv));
                            cj += 8;
                        }
                        while cj < n {
                            let dst = pc.add(sub_r * n + cj);
                            *dst = a_val.mul_add(*pb.add(p * n + cj), *dst);
                            cj += 1;
                        }
                    }
                }
            }

            r += 6;
        }

        // Remaining `< 6` rows
        while r < tile_m {
            for p in p0..p_end {
                let a_val = *pa.add(r * k + p);
                let av = _mm256_set1_ps(a_val);
                let mut cj = 0usize;
                while cj + 16 <= n {
                    let c0 = _mm256_loadu_ps(pc.add(r * n + cj));
                    let c1 = _mm256_loadu_ps(pc.add(r * n + cj + 8));
                    let b0 = _mm256_loadu_ps(pb.add(p * n + cj));
                    let b1 = _mm256_loadu_ps(pb.add(p * n + cj + 8));
                    _mm256_storeu_ps(pc.add(r * n + cj), _mm256_fmadd_ps(av, b0, c0));
                    _mm256_storeu_ps(pc.add(r * n + cj + 8), _mm256_fmadd_ps(av, b1, c1));
                    cj += 16;
                }
                while cj + 8 <= n {
                    let cv = _mm256_loadu_ps(pc.add(r * n + cj));
                    let bv = _mm256_loadu_ps(pb.add(p * n + cj));
                    _mm256_storeu_ps(pc.add(r * n + cj), _mm256_fmadd_ps(av, bv, cv));
                    cj += 8;
                }
                while cj < n {
                    let dst = pc.add(r * n + cj);
                    *dst = a_val.mul_add(*pb.add(p * n + cj), *dst);
                    cj += 1;
                }
            }
            r += 1;
        }

        p0 = p_end;
    }
}

/// Exact integer AVX2 `Q8_0 × Q8_0` dot-product kernel using `_mm256_sign_epi8` +
/// `_mm256_maddubs_epi16` + `_mm256_madd_epi16`.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported at runtime and
/// `w_blocks.len() == num_blocks * 34` and `x_blocks.len() == num_blocks * 34`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_q8_0_q8_0_avx2(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    debug_assert_eq!(w_blocks.len(), num_blocks * BYTES_Q8_0);
    debug_assert_eq!(x_blocks.len(), num_blocks * BYTES_Q8_0);

    let ones16 = _mm256_set1_epi16(1);
    let pw = w_blocks.as_ptr();
    let px = x_blocks.as_ptr();

    let mut acc = 0.0f32;
    for b in 0..num_blocks {
        let w_off = b * BYTES_Q8_0;
        let x_off = b * BYTES_Q8_0;
        // SAFETY: `b < num_blocks` and slice lengths equal `num_blocks * 34`, so
        // `[off .. off + 2]` scale bytes and `[off + 2 .. off + 34]` 32-byte loads are in bounds.
        let dw = f16_to_f32(u16::from_le_bytes([*pw.add(w_off), *pw.add(w_off + 1)]));
        let dx = f16_to_f32(u16::from_le_bytes([*px.add(x_off), *px.add(x_off + 1)]));

        let vw = _mm256_loadu_si256(pw.add(w_off + 2).cast::<__m256i>());
        let vx = _mm256_loadu_si256(px.add(x_off + 2).cast::<__m256i>());

        let ax = _mm256_sign_epi8(vw, vw);
        let sy = _mm256_sign_epi8(vx, vw);
        let dot16 = _mm256_maddubs_epi16(ax, sy);
        let dot32 = _mm256_madd_epi16(dot16, ones16);
        let isum = hsum_epi32_avx2(dot32);

        acc += (dw * dx) * (isum as f32);
    }
    acc
}

/// Exact integer AVX2 `Q4_0 × Q8_0` dot-product kernel unpacking 4-bit nibbles (`0..16` low,
/// `16..32` high), subtracting bias `8`, and reducing via `_mm256_maddubs_epi16`.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported at runtime and
/// `w_blocks.len() == num_blocks * 18` and `x_blocks.len() == num_blocks * 34`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_q4_0_q8_0_avx2(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    debug_assert_eq!(w_blocks.len(), num_blocks * BYTES_Q4_0);
    debug_assert_eq!(x_blocks.len(), num_blocks * BYTES_Q8_0);

    let mask_low = _mm_set1_epi8(0x0F);
    let bias8 = _mm256_set1_epi8(8);
    let ones16 = _mm256_set1_epi16(1);

    let pw = w_blocks.as_ptr();
    let px = x_blocks.as_ptr();

    let mut acc = 0.0f32;
    for b in 0..num_blocks {
        let w_off = b * BYTES_Q4_0;
        let x_off = b * BYTES_Q8_0;
        // SAFETY: `w_off + 18 <= w_blocks.len()` and `x_off + 34 <= x_blocks.len()`.
        let dw = f16_to_f32(u16::from_le_bytes([*pw.add(w_off), *pw.add(w_off + 1)]));
        let dx = f16_to_f32(u16::from_le_bytes([*px.add(x_off), *px.add(x_off + 1)]));

        let raw16 = _mm_loadu_si128(pw.add(w_off + 2).cast::<__m128i>());
        let lo = _mm_and_si128(raw16, mask_low);
        let hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask_low);
        let unpacked = _mm256_set_m128i(hi, lo);
        let vw = _mm256_sub_epi8(unpacked, bias8);

        let vx = _mm256_loadu_si256(px.add(x_off + 2).cast::<__m256i>());
        let ax = _mm256_sign_epi8(vw, vw);
        let sy = _mm256_sign_epi8(vx, vw);
        let dot16 = _mm256_maddubs_epi16(ax, sy);
        let dot32 = _mm256_madd_epi16(dot16, ones16);
        let isum = hsum_epi32_avx2(dot32);

        acc += (dw * dx) * (isum as f32);
    }
    acc
}

/// Fused AVX2+FMA RMSNorm over a single row.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported at runtime and
/// `x.len() == weight.len() == out.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rmsnorm_row_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let px = x.as_ptr();
    let pw = weight.as_ptr();
    let po = out.as_mut_ptr();

    let mut vsum = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= dim {
        // SAFETY: `i + 8 <= dim` keeps 8-float load in bounds.
        let vx = _mm256_loadu_ps(px.add(i));
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
        i += 8;
    }
    let mut sum_sq = hsum_ps_avx2(vsum);
    while i < dim {
        let v = *px.add(i);
        sum_sq = v.mul_add(v, sum_sq);
        i += 1;
    }

    let inv_rms = 1.0 / (sum_sq / (dim as f32) + eps).sqrt();
    let vscale = _mm256_set1_ps(inv_rms);

    let mut j = 0usize;
    while j + 8 <= dim {
        // SAFETY: `j + 8 <= dim` keeps loads/stores in bounds.
        let vx = _mm256_loadu_ps(px.add(j));
        let vw = _mm256_loadu_ps(pw.add(j));
        let vy = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(po.add(j), vy);
        j += 8;
    }
    while j < dim {
        *po.add(j) = (*px.add(j) * inv_rms) * (*pw.add(j));
        j += 1;
    }
}
