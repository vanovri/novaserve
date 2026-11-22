//! GGML-compatible `Q8_0` (`34 B / 32 elems`) and `Q4_0` (`18 B / 32 elems`) block
//! quantization and dequantization routines (100% safe Rust).

use crate::alloc::AlignedBuf;
use crate::dtype::{f16_to_f32, f32_to_f16, BYTES_Q4_0, BYTES_Q8_0, QK4_0, QK8_0};
use crate::error::TensorError;

/// Quantizes a single 32-float block into 34-byte GGML `Q8_0` format:
/// `[0..2]: d (f16 LE)`, `[2..34]: qs ([i8; 32])`.
#[inline]
pub fn quantize_block_q8_0(src: &[f32; QK8_0], dst: &mut [u8; BYTES_Q8_0]) {
    let mut amax = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let d = amax / 127.0;
    let d_f16 = f32_to_f16(d);
    let d_eff = f16_to_f32(d_f16);
    let d_bytes = d_f16.to_le_bytes();
    dst[0] = d_bytes[0];
    dst[1] = d_bytes[1];

    if d_eff == 0.0 || !d_eff.is_finite() {
        dst[2..BYTES_Q8_0].fill(0);
        return;
    }

    let inv_d = 1.0 / d_eff;
    for i in 0..QK8_0 {
        let q = (src[i] * inv_d).round().clamp(-127.0, 127.0) as i8;
        dst[2 + i] = q as u8;
    }
}

/// Dequantizes a single 34-byte GGML `Q8_0` block into 32 `f32` values.
#[inline]
pub fn dequantize_block_q8_0(src: &[u8; BYTES_Q8_0], dst: &mut [f32; QK8_0]) {
    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    for i in 0..QK8_0 {
        let q = src[2 + i] as i8;
        dst[i] = d * (q as f32);
    }
}

/// Quantizes a row of arbitrary length `cols` into `cols.div_ceil(32)` `Q8_0` blocks,
/// zero-padding any partial tail block `[cols .. num_blocks * 32)`.
pub fn quantize_row_q8_0(src: &[f32], dst_bytes: &mut [u8]) -> Result<(), TensorError> {
    let cols = src.len();
    let num_blocks = if cols == 0 { 0 } else { cols.div_ceil(QK8_0) };
    let expected_bytes = num_blocks * BYTES_Q8_0;
    if dst_bytes.len() != expected_bytes {
        return Err(TensorError::ShapeMismatch {
            expected: vec![expected_bytes],
            actual: vec![dst_bytes.len()],
            context: "quantize_row_q8_0",
        });
    }

    let full_blocks = cols / QK8_0;
    for b in 0..full_blocks {
        let blk_src: &[f32; QK8_0] = src[b * QK8_0..(b + 1) * QK8_0]
            .try_into()
            .expect("exact 32 slice");
        let blk_dst: &mut [u8; BYTES_Q8_0] = (&mut dst_bytes[b * BYTES_Q8_0..(b + 1) * BYTES_Q8_0])
            .try_into()
            .expect("exact 34 slice");
        quantize_block_q8_0(blk_src, blk_dst);
    }

    let rem = cols % QK8_0;
    if rem > 0 {
        let mut tail = [0.0f32; QK8_0];
        tail[..rem].copy_from_slice(&src[full_blocks * QK8_0..]);
        let blk_dst: &mut [u8; BYTES_Q8_0] = (&mut dst_bytes
            [full_blocks * BYTES_Q8_0..(full_blocks + 1) * BYTES_Q8_0])
            .try_into()
            .expect("exact 34 tail slice");
        quantize_block_q8_0(&tail, blk_dst);
    }
    Ok(())
}

/// Dequantizes a row of `Q8_0` blocks back into `dst` (`cols` elements).
pub fn dequantize_row_q8_0(src_bytes: &[u8], dst: &mut [f32]) -> Result<(), TensorError> {
    let cols = dst.len();
    let num_blocks = if cols == 0 { 0 } else { cols.div_ceil(QK8_0) };
    let expected_bytes = num_blocks * BYTES_Q8_0;
    if src_bytes.len() != expected_bytes {
        return Err(TensorError::ShapeMismatch {
            expected: vec![expected_bytes],
            actual: vec![src_bytes.len()],
            context: "dequantize_row_q8_0",
        });
    }

    let full_blocks = cols / QK8_0;
    for b in 0..full_blocks {
        let blk_src: &[u8; BYTES_Q8_0] = src_bytes[b * BYTES_Q8_0..(b + 1) * BYTES_Q8_0]
            .try_into()
            .expect("exact 34 slice");
        let blk_dst: &mut [f32; QK8_0] = (&mut dst[b * QK8_0..(b + 1) * QK8_0])
            .try_into()
            .expect("exact 32 slice");
        dequantize_block_q8_0(blk_src, blk_dst);
    }

    let rem = cols % QK8_0;
    if rem > 0 {
        let blk_src: &[u8; BYTES_Q8_0] = src_bytes
            [full_blocks * BYTES_Q8_0..(full_blocks + 1) * BYTES_Q8_0]
            .try_into()
            .expect("exact 34 tail slice");
        let mut tail = [0.0f32; QK8_0];
        dequantize_block_q8_0(blk_src, &mut tail);
        dst[full_blocks * QK8_0..].copy_from_slice(&tail[..rem]);
    }
    Ok(())
}

/// Quantizes a `[rows, cols]` row-major `f32` matrix into an aligned `Q8_0` buffer.
pub fn quantize_matrix_q8_0(
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<AlignedBuf, TensorError> {
    if src.len() != rows.saturating_mul(cols) {
        return Err(TensorError::ShapeMismatch {
            expected: vec![rows, cols],
            actual: vec![src.len()],
            context: "quantize_matrix_q8_0",
        });
    }
    let row_bytes = cols.div_ceil(QK8_0) * BYTES_Q8_0;
    let mut out = AlignedBuf::new_zeroed(rows * row_bytes)?;
    let out_bytes = out.as_bytes_mut();
    for r in 0..rows {
        quantize_row_q8_0(
            &src[r * cols..(r + 1) * cols],
            &mut out_bytes[r * row_bytes..(r + 1) * row_bytes],
        )?;
    }
    Ok(out)
}

/// Quantizes a single 32-float block into 18-byte GGML `Q4_0` format:
/// `[0..2]: d (f16 LE)`, `[2..18]: qs ([u8; 16])` where low nibbles store elements `0..16`
/// biased by `+8` and high nibbles store elements `16..32` biased by `+8`.
#[inline]
pub fn quantize_block_q4_0(src: &[f32; QK4_0], dst: &mut [u8; BYTES_Q4_0]) {
    let mut amax = 0.0f32;
    let mut signed_max = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > amax {
            amax = a;
            signed_max = v;
        }
    }
    let d = signed_max / -8.0;
    let d_f16 = f32_to_f16(d);
    let d_eff = f16_to_f32(d_f16);
    let d_bytes = d_f16.to_le_bytes();
    dst[0] = d_bytes[0];
    dst[1] = d_bytes[1];

    if d_eff == 0.0 || !d_eff.is_finite() {
        dst[2..BYTES_Q4_0].fill(0x88);
        return;
    }

    let inv_d = 1.0 / d_eff;
    for j in 0..16 {
        let x0 = (src[j] * inv_d + 8.5).floor().clamp(0.0, 15.0) as u8;
        let x1 = (src[j + 16] * inv_d + 8.5).floor().clamp(0.0, 15.0) as u8;
        dst[2 + j] = x0 | (x1 << 4);
    }
}

/// Dequantizes a single 18-byte GGML `Q4_0` block into 32 `f32` values.
#[inline]
pub fn dequantize_block_q4_0(src: &[u8; BYTES_Q4_0], dst: &mut [f32; QK4_0]) {
    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    for j in 0..16 {
        let byte = src[2 + j];
        let q0 = ((byte & 0x0F) as i32) - 8;
        let q1 = ((byte >> 4) as i32) - 8;
        dst[j] = d * (q0 as f32);
        dst[j + 16] = d * (q1 as f32);
    }
}

/// Quantizes a row of arbitrary length `cols` into `cols.div_ceil(32)` `Q4_0` blocks.
pub fn quantize_row_q4_0(src: &[f32], dst_bytes: &mut [u8]) -> Result<(), TensorError> {
    let cols = src.len();
    let num_blocks = if cols == 0 { 0 } else { cols.div_ceil(QK4_0) };
    let expected_bytes = num_blocks * BYTES_Q4_0;
    if dst_bytes.len() != expected_bytes {
        return Err(TensorError::ShapeMismatch {
            expected: vec![expected_bytes],
            actual: vec![dst_bytes.len()],
            context: "quantize_row_q4_0",
        });
    }

    let full_blocks = cols / QK4_0;
    for b in 0..full_blocks {
        let blk_src: &[f32; QK4_0] = src[b * QK4_0..(b + 1) * QK4_0]
            .try_into()
            .expect("exact 32 slice");
        let blk_dst: &mut [u8; BYTES_Q4_0] = (&mut dst_bytes[b * BYTES_Q4_0..(b + 1) * BYTES_Q4_0])
            .try_into()
            .expect("exact 18 slice");
        quantize_block_q4_0(blk_src, blk_dst);
    }

    let rem = cols % QK4_0;
    if rem > 0 {
        let mut tail = [0.0f32; QK4_0];
        tail[..rem].copy_from_slice(&src[full_blocks * QK4_0..]);
        let blk_dst: &mut [u8; BYTES_Q4_0] = (&mut dst_bytes
            [full_blocks * BYTES_Q4_0..(full_blocks + 1) * BYTES_Q4_0])
            .try_into()
            .expect("exact 18 tail slice");
        quantize_block_q4_0(&tail, blk_dst);
    }
    Ok(())
}

/// Dequantizes a row of `Q4_0` blocks back into `dst` (`cols` elements).
pub fn dequantize_row_q4_0(src_bytes: &[u8], dst: &mut [f32]) -> Result<(), TensorError> {
    let cols = dst.len();
    let num_blocks = if cols == 0 { 0 } else { cols.div_ceil(QK4_0) };
    let expected_bytes = num_blocks * BYTES_Q4_0;
    if src_bytes.len() != expected_bytes {
        return Err(TensorError::ShapeMismatch {
            expected: vec![expected_bytes],
            actual: vec![src_bytes.len()],
            context: "dequantize_row_q4_0",
        });
    }

    let full_blocks = cols / QK4_0;
    for b in 0..full_blocks {
        let blk_src: &[u8; BYTES_Q4_0] = src_bytes[b * BYTES_Q4_0..(b + 1) * BYTES_Q4_0]
            .try_into()
            .expect("exact 18 slice");
        let blk_dst: &mut [f32; QK4_0] = (&mut dst[b * QK4_0..(b + 1) * QK4_0])
            .try_into()
            .expect("exact 32 slice");
        dequantize_block_q4_0(blk_src, blk_dst);
    }

    let rem = cols % QK4_0;
    if rem > 0 {
        let blk_src: &[u8; BYTES_Q4_0] = src_bytes
            [full_blocks * BYTES_Q4_0..(full_blocks + 1) * BYTES_Q4_0]
            .try_into()
            .expect("exact 18 tail slice");
        let mut tail = [0.0f32; QK4_0];
        dequantize_block_q4_0(blk_src, &mut tail);
        dst[full_blocks * QK4_0..].copy_from_slice(&tail[..rem]);
    }
    Ok(())
}

/// Quantizes a `[rows, cols]` row-major `f32` matrix into an aligned `Q4_0` buffer.
pub fn quantize_matrix_q4_0(
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<AlignedBuf, TensorError> {
    if src.len() != rows.saturating_mul(cols) {
        return Err(TensorError::ShapeMismatch {
            expected: vec![rows, cols],
            actual: vec![src.len()],
            context: "quantize_matrix_q4_0",
        });
    }
    let row_bytes = cols.div_ceil(QK4_0) * BYTES_Q4_0;
    let mut out = AlignedBuf::new_zeroed(rows * row_bytes)?;
    let out_bytes = out.as_bytes_mut();
    for r in 0..rows {
        quantize_row_q4_0(
            &src[r * cols..(r + 1) * cols],
            &mut out_bytes[r * row_bytes..(r + 1) * row_bytes],
        )?;
    }
    Ok(out)
}
