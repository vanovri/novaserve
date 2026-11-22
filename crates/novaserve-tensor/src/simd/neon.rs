use crate::dtype::{f16_to_f32, BYTES_Q4_0, BYTES_Q8_0};
use std::arch::aarch64::*;

/// NEON `f32` dot product.
///
/// # Safety
/// Caller must ensure `neon` is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` keeps two 4-float loads in bounds.
        acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
        i += 8;
    }
    let mut acc = vaddq_f32(acc0, acc1);
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n` keeps 4-float load in bounds.
        acc = vfmaq_f32(acc, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum = (*pa.add(i)).mul_add(*pb.add(i), sum);
        i += 1;
    }
    sum
}

/// NEON `f32` row-tile GEMM kernel.
///
/// # Safety
/// Caller must ensure `neon` is available and tile bounds hold.
#[target_feature(enable = "neon")]
pub unsafe fn matmul_f32_row_tile_neon(
    a_tile: &[f32],
    b: &[f32],
    tile_m: usize,
    k: usize,
    n: usize,
    out_tile: &mut [f32],
) {
    out_tile[..tile_m * n].fill(0.0);
    let pa = a_tile.as_ptr();
    let pb = b.as_ptr();
    let pc = out_tile.as_mut_ptr();

    for r in 0..tile_m {
        for p in 0..k {
            let a_val = *pa.add(r * k + p);
            let av = vdupq_n_f32(a_val);
            let mut c = 0usize;
            while c + 8 <= n {
                // SAFETY: `c + 8 <= n` keeps 8-float loads/stores in bounds.
                let c0 = vld1q_f32(pc.add(r * n + c));
                let c1 = vld1q_f32(pc.add(r * n + c + 4));
                let b0 = vld1q_f32(pb.add(p * n + c));
                let b1 = vld1q_f32(pb.add(p * n + c + 4));
                vst1q_f32(pc.add(r * n + c), vfmaq_f32(c0, av, b0));
                vst1q_f32(pc.add(r * n + c + 4), vfmaq_f32(c1, av, b1));
                c += 8;
            }
            while c < n {
                let dst = pc.add(r * n + c);
                *dst = a_val.mul_add(*pb.add(p * n + c), *dst);
                c += 1;
            }
        }
    }
}

/// NEON `Q8_0 × Q8_0` dot product.
///
/// # Safety
/// Caller must ensure `neon` is available and block slice bounds hold.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q8_0_q8_0_neon(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    let pw = w_blocks.as_ptr();
    let px = x_blocks.as_ptr();
    let mut acc = 0.0f32;
    for b in 0..num_blocks {
        let w_off = b * BYTES_Q8_0;
        let x_off = b * BYTES_Q8_0;
        // SAFETY: `b < num_blocks` guarantees 34-byte block bounds.
        let dw = f16_to_f32(u16::from_le_bytes([*pw.add(w_off), *pw.add(w_off + 1)]));
        let dx = f16_to_f32(u16::from_le_bytes([*px.add(x_off), *px.add(x_off + 1)]));

        let w0 = vld1q_s8(pw.add(w_off + 2).cast::<i8>());
        let w1 = vld1q_s8(pw.add(w_off + 18).cast::<i8>());
        let x0 = vld1q_s8(px.add(x_off + 2).cast::<i8>());
        let x1 = vld1q_s8(px.add(x_off + 18).cast::<i8>());

        let p0 = vaddq_s16(
            vmull_s8(vget_low_s8(w0), vget_low_s8(x0)),
            vmull_high_s8(w0, x0),
        );
        let p1 = vaddq_s16(
            vmull_s8(vget_low_s8(w1), vget_low_s8(x1)),
            vmull_high_s8(w1, x1),
        );
        let sum32 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
        let isum = vaddvq_s32(sum32);
        acc += (dw * dx) * (isum as f32);
    }
    acc
}

/// NEON `Q4_0 × Q8_0` dot product.
///
/// # Safety
/// Caller must ensure `neon` is available and block slice bounds hold.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q4_0_q8_0_neon(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    let pw = w_blocks.as_ptr();
    let px = x_blocks.as_ptr();
    let mask_low = vdupq_n_u8(0x0F);
    let bias8 = vdupq_n_s8(8);

    let mut acc = 0.0f32;
    for b in 0..num_blocks {
        let w_off = b * BYTES_Q4_0;
        let x_off = b * BYTES_Q8_0;
        // SAFETY: `b < num_blocks` guarantees 18-byte Q4_0 and 34-byte Q8_0 bounds.
        let dw = f16_to_f32(u16::from_le_bytes([*pw.add(w_off), *pw.add(w_off + 1)]));
        let dx = f16_to_f32(u16::from_le_bytes([*px.add(x_off), *px.add(x_off + 1)]));

        let raw = vld1q_u8(pw.add(w_off + 2));
        let w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_low)), bias8);
        let w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw)), bias8);

        let x0 = vld1q_s8(px.add(x_off + 2).cast::<i8>());
        let x1 = vld1q_s8(px.add(x_off + 18).cast::<i8>());

        let p0 = vaddq_s16(
            vmull_s8(vget_low_s8(w0), vget_low_s8(x0)),
            vmull_high_s8(w0, x0),
        );
        let p1 = vaddq_s16(
            vmull_s8(vget_low_s8(w1), vget_low_s8(x1)),
            vmull_high_s8(w1, x1),
        );
        let sum32 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
        let isum = vaddvq_s32(sum32);
        acc += (dw * dx) * (isum as f32);
    }
    acc
}

/// NEON `f32` RMSNorm row kernel.
///
/// # Safety
/// Caller must ensure `neon` is available and `x.len() == weight.len() == out.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn rmsnorm_row_neon(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let px = x.as_ptr();
    let pw = weight.as_ptr();
    let po = out.as_mut_ptr();

    let mut vsum = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= dim {
        // SAFETY: `i + 4 <= dim` keeps 4-float load in bounds.
        let vx = vld1q_f32(px.add(i));
        vsum = vfmaq_f32(vsum, vx, vx);
        i += 4;
    }
    let mut sum_sq = vaddvq_f32(vsum);
    while i < dim {
        let v = *px.add(i);
        sum_sq = v.mul_add(v, sum_sq);
        i += 1;
    }

    let inv_rms = 1.0 / (sum_sq / (dim as f32) + eps).sqrt();
    let vscale = vdupq_n_f32(inv_rms);
    let mut j = 0usize;
    while j + 4 <= dim {
        // SAFETY: `j + 4 <= dim` keeps loads/stores in bounds.
        let vx = vld1q_f32(px.add(j));
        let vw = vld1q_f32(pw.add(j));
        vst1q_f32(po.add(j), vmulq_f32(vmulq_f32(vx, vscale), vw));
        j += 4;
    }
    while j < dim {
        *po.add(j) = (*px.add(j) * inv_rms) * (*pw.add(j));
        j += 1;
    }
}
