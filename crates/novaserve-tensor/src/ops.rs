use crate::alloc::AlignedScratchArena;
use crate::dtype::{bf16_to_f32, f16_to_f32, DType, BYTES_Q4_0, BYTES_Q8_0, QK4_0, QK8_0};
use crate::error::TensorError;
use crate::quant::quantize_row_q8_0;
use crate::simd;
use crate::tensor::TensorView;

#[cfg(all(not(target_arch = "wasm32"), not(miri)))]
use rayon::prelude::*;

/// Cache-blocked + SIMD + Rayon matrix multiplication: `out[m, n] = a[m, k] * b[k, n]`.
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<(), TensorError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TensorError::InvalidDimension(format!(
            "matmul_f32 requires non-zero dims, got ({m}, {k}, {n})"
        )));
    }
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m * k, k * n, m * n],
            actual: vec![a.len(), b.len(), out.len()],
            context: "matmul_f32",
        });
    }

    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    {
        const MC: usize = 24;
        if m >= MC && m.saturating_mul(k).saturating_mul(n) >= 65_536 {
            out.par_chunks_mut(MC * n)
                .enumerate()
                .for_each(|(tile_idx, out_tile)| {
                    let row_start = tile_idx * MC;
                    let tile_m = out_tile.len() / n;
                    let a_tile = &a[row_start * k..(row_start + tile_m) * k];
                    simd::matmul_f32_row_tile(a_tile, b, tile_m, k, n, out_tile);
                });
            return Ok(());
        }
    }

    simd::matmul_f32_row_tile(a, b, m, k, n, out);
    Ok(())
}

/// Row-major weight linear projection: `out[m, n] = x[m, k] * w_row_major^T[n, k]`.
pub fn linear_f32(
    x: &[f32],
    w_row_major: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<(), TensorError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TensorError::InvalidDimension(format!(
            "linear_f32 requires non-zero dims, got ({m}, {k}, {n})"
        )));
    }
    if x.len() != m * k || w_row_major.len() != n * k || out.len() != m * n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m * k, n * k, m * n],
            actual: vec![x.len(), w_row_major.len(), out.len()],
            context: "linear_f32",
        });
    }

    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    {
        if m.saturating_mul(k).saturating_mul(n) >= 65_536 {
            if m == 1 {
                let x_row = &x[..k];
                out.par_chunks_mut(32)
                    .enumerate()
                    .for_each(|(blk_idx, out_chunk)| {
                        let col_base = blk_idx * 32;
                        for (offset, dst) in out_chunk.iter_mut().enumerate() {
                            let j = col_base + offset;
                            let w_row = &w_row_major[j * k..(j + 1) * k];
                            *dst = simd::vec_dot_f32(x_row, w_row);
                        }
                    });
            } else {
                out.par_chunks_mut(n).enumerate().for_each(|(r, out_row)| {
                    let x_row = &x[r * k..(r + 1) * k];
                    for (j, dst) in out_row.iter_mut().enumerate() {
                        let w_row = &w_row_major[j * k..(j + 1) * k];
                        *dst = simd::vec_dot_f32(x_row, w_row);
                    }
                });
            }
            return Ok(());
        }
    }

    for r in 0..m {
        let x_row = &x[r * k..(r + 1) * k];
        let out_row = &mut out[r * n..(r + 1) * n];
        for (j, dst) in out_row.iter_mut().enumerate() {
            let w_row = &w_row_major[j * k..(j + 1) * k];
            *dst = simd::vec_dot_f32(x_row, w_row);
        }
    }
    Ok(())
}

/// On-the-fly activation-quantized `Q8_0` projection:
/// quantizes `x[m, k]` into `Q8_0` inside `arena` once, then computes `x_q8 * w_q8^T` (`[n, k]`).
pub fn matmul_q8_0(
    x: &[f32],
    w_q8: &[u8],
    m: usize,
    k: usize,
    n: usize,
    arena: &AlignedScratchArena,
    out: &mut [f32],
) -> Result<(), TensorError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TensorError::InvalidDimension(format!(
            "matmul_q8_0 requires non-zero dims, got ({m}, {k}, {n})"
        )));
    }
    let num_blocks = k.div_ceil(QK8_0);
    let x_row_bytes = num_blocks * BYTES_Q8_0;
    let w_row_bytes = num_blocks * BYTES_Q8_0;

    if x.len() != m * k || w_q8.len() != n * w_row_bytes || out.len() != m * n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m * k, n * w_row_bytes, m * n],
            actual: vec![x.len(), w_q8.len(), out.len()],
            context: "matmul_q8_0",
        });
    }

    let x_q8_buf = arena.alloc_bytes(m * x_row_bytes)?;
    for r in 0..m {
        quantize_row_q8_0(
            &x[r * k..(r + 1) * k],
            &mut x_q8_buf[r * x_row_bytes..(r + 1) * x_row_bytes],
        )?;
    }
    let x_q8_ref: &[u8] = &*x_q8_buf;

    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    {
        if m.saturating_mul(num_blocks).saturating_mul(n) >= 2_048 {
            if m == 1 {
                let x_blk = &x_q8_ref[..x_row_bytes];
                out.par_chunks_mut(32)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let base_j = chunk_idx * 32;
                        for (off, dst) in out_chunk.iter_mut().enumerate() {
                            let j = base_j + off;
                            let w_blk = &w_q8[j * w_row_bytes..(j + 1) * w_row_bytes];
                            *dst = simd::vec_dot_q8_0_q8_0(w_blk, x_blk, num_blocks);
                        }
                    });
            } else {
                out.par_chunks_mut(n).enumerate().for_each(|(r, out_row)| {
                    let x_blk = &x_q8_ref[r * x_row_bytes..(r + 1) * x_row_bytes];
                    for (j, dst) in out_row.iter_mut().enumerate() {
                        let w_blk = &w_q8[j * w_row_bytes..(j + 1) * w_row_bytes];
                        *dst = simd::vec_dot_q8_0_q8_0(w_blk, x_blk, num_blocks);
                    }
                });
            }
            return Ok(());
        }
    }

    for r in 0..m {
        let x_blk = &x_q8_ref[r * x_row_bytes..(r + 1) * x_row_bytes];
        let out_row = &mut out[r * n..(r + 1) * n];
        for (j, dst) in out_row.iter_mut().enumerate() {
            let w_blk = &w_q8[j * w_row_bytes..(j + 1) * w_row_bytes];
            *dst = simd::vec_dot_q8_0_q8_0(w_blk, x_blk, num_blocks);
        }
    }
    Ok(())
}

/// On-the-fly activation-quantized `Q4_0` projection:
/// quantizes `x[m, k]` into `Q8_0` inside `arena` once, then computes `x_q8 * w_q4^T` (`[n, k]`).
pub fn matmul_q4_0(
    x: &[f32],
    w_q4: &[u8],
    m: usize,
    k: usize,
    n: usize,
    arena: &AlignedScratchArena,
    out: &mut [f32],
) -> Result<(), TensorError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TensorError::InvalidDimension(format!(
            "matmul_q4_0 requires non-zero dims, got ({m}, {k}, {n})"
        )));
    }
    let num_blocks = k.div_ceil(QK4_0);
    let x_row_bytes = num_blocks * BYTES_Q8_0;
    let w_row_bytes = num_blocks * BYTES_Q4_0;

    if x.len() != m * k || w_q4.len() != n * w_row_bytes || out.len() != m * n {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m * k, n * w_row_bytes, m * n],
            actual: vec![x.len(), w_q4.len(), out.len()],
            context: "matmul_q4_0",
        });
    }

    let x_q8_buf = arena.alloc_bytes(m * x_row_bytes)?;
    for r in 0..m {
        quantize_row_q8_0(
            &x[r * k..(r + 1) * k],
            &mut x_q8_buf[r * x_row_bytes..(r + 1) * x_row_bytes],
        )?;
    }
    let x_q8_ref: &[u8] = &*x_q8_buf;

    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    {
        if m.saturating_mul(num_blocks).saturating_mul(n) >= 2_048 {
            if m == 1 {
                let x_blk = &x_q8_ref[..x_row_bytes];
                out.par_chunks_mut(32)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let base_j = chunk_idx * 32;
                        for (off, dst) in out_chunk.iter_mut().enumerate() {
                            let j = base_j + off;
                            let w_blk = &w_q4[j * w_row_bytes..(j + 1) * w_row_bytes];
                            *dst = simd::vec_dot_q4_0_q8_0(w_blk, x_blk, num_blocks);
                        }
                    });
            } else {
                out.par_chunks_mut(n).enumerate().for_each(|(r, out_row)| {
                    let x_blk = &x_q8_ref[r * x_row_bytes..(r + 1) * x_row_bytes];
                    for (j, dst) in out_row.iter_mut().enumerate() {
                        let w_blk = &w_q4[j * w_row_bytes..(j + 1) * w_row_bytes];
                        *dst = simd::vec_dot_q4_0_q8_0(w_blk, x_blk, num_blocks);
                    }
                });
            }
            return Ok(());
        }
    }

    for r in 0..m {
        let x_blk = &x_q8_ref[r * x_row_bytes..(r + 1) * x_row_bytes];
        let out_row = &mut out[r * n..(r + 1) * n];
        for (j, dst) in out_row.iter_mut().enumerate() {
            let w_blk = &w_q4[j * w_row_bytes..(j + 1) * w_row_bytes];
            *dst = simd::vec_dot_q4_0_q8_0(w_blk, x_blk, num_blocks);
        }
    }
    Ok(())
}

/// Polymorphic zero-allocation linear projection `out[m, n] = x[m, k] * weight^T[n, k]`
/// supporting `F32`, `Q8_0`, `Q4_0`, `F16`, and `BF16` weights seamlessly.
pub fn linear_projection(
    x: &[f32],
    weight: &TensorView<'_>,
    m: usize,
    arena: &AlignedScratchArena,
    out: &mut [f32],
) -> Result<(), TensorError> {
    if weight.rank() != 2 {
        return Err(TensorError::InvalidDimension(format!(
            "linear_projection expects 2D weight [out_features, in_features], got {:?}",
            weight.dims()
        )));
    }
    let n = weight.dims()[0];
    let k = weight.dims()[1];
    match weight.dtype() {
        DType::F32 => {
            if let Ok(w_slice) = weight.as_f32_slice() {
                linear_f32(x, w_slice, m, k, n, out)
            } else {
                // Unaligned mmap fallback into scratch arena
                let w_tmp = arena.alloc_f32(n * k)?;
                let raw = weight.raw_bytes();
                for (i, slot) in w_tmp.iter_mut().enumerate() {
                    let off = i * 4;
                    *slot =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                }
                linear_f32(x, w_tmp, m, k, n, out)
            }
        }
        DType::Q8_0 => matmul_q8_0(x, weight.raw_bytes(), m, k, n, arena, out),
        DType::Q4_0 => matmul_q4_0(x, weight.raw_bytes(), m, k, n, arena, out),
        DType::F16 => {
            let raw = weight.raw_bytes();
            let row_tmp = arena.alloc_f32(k)?;
            for j in 0..n {
                for (i, slot) in row_tmp.iter_mut().enumerate().take(k) {
                    let off = (j * k + i) * 2;
                    *slot = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                }
                for r in 0..m {
                    out[r * n + j] = simd::vec_dot_f32(&x[r * k..(r + 1) * k], row_tmp);
                }
            }
            Ok(())
        }
        DType::BF16 => {
            let raw = weight.raw_bytes();
            let row_tmp = arena.alloc_f32(k)?;
            for j in 0..n {
                for (i, slot) in row_tmp.iter_mut().enumerate().take(k) {
                    let off = (j * k + i) * 2;
                    *slot = bf16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                }
                for r in 0..m {
                    out[r * n + j] = simd::vec_dot_f32(&x[r * k..(r + 1) * k], row_tmp);
                }
            }
            Ok(())
        }
        other => Err(TensorError::UnsupportedDType(format!(
            "linear_projection unsupported dtype {other:?}"
        ))),
    }
}

/// Fused RMSNorm (`y_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i`) supporting `[num_tokens, dim]`.
pub fn rmsnorm_f32(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<(), TensorError> {
    let dim = weight.len();
    if dim == 0 || x.is_empty() || x.len() != out.len() || !x.len().is_multiple_of(dim) {
        return Err(TensorError::ShapeMismatch {
            expected: vec![dim],
            actual: vec![x.len(), out.len()],
            context: "rmsnorm_f32",
        });
    }
    if !(eps > 0.0 && eps.is_finite()) {
        return Err(TensorError::InvalidDimension(format!(
            "rmsnorm eps must be positive finite, got {eps}"
        )));
    }

    let rows = x.len() / dim;
    for r in 0..rows {
        simd::rmsnorm_row_f32(
            &x[r * dim..(r + 1) * dim],
            weight,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
    }
    Ok(())
}

/// Fused SwiGLU activation: `out[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]`.
pub fn swiglu_f32(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), TensorError> {
    if gate.len() != up.len() || gate.len() != out.len() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![gate.len()],
            actual: vec![up.len(), out.len()],
            context: "swiglu_f32",
        });
    }

    for i in 0..gate.len() {
        let g = gate[i];
        let silu = if g < -88.0 {
            0.0
        } else if g > 88.0 {
            g
        } else {
            g / (1.0 + (-g).exp())
        };
        out[i] = silu * up[i];
    }
    Ok(())
}

/// In-place Rotary Positional Embedding (`RoPE`) over `[num_tokens, n_heads, head_dim]`
/// with arbitrary per-token `positions: &[u32]` (interleaved adjacent pairs `(2i, 2i+1)`).
pub fn rope_inplace_f32(
    qk: &mut [f32],
    positions: &[u32],
    n_heads: usize,
    head_dim: usize,
    freq_base: f32,
    scaling: f32,
) -> Result<(), TensorError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || n_heads == 0 {
        return Err(TensorError::InvalidDimension(format!(
            "rope_inplace_f32 requires even head_dim > 0 and n_heads > 0, got n_heads={n_heads}, head_dim={head_dim}"
        )));
    }
    let expected = positions
        .len()
        .checked_mul(n_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| TensorError::InvalidDimension("rope shape overflow".into()))?;
    if qk.len() != expected {
        return Err(TensorError::ShapeMismatch {
            expected: vec![expected],
            actual: vec![qk.len()],
            context: "rope_inplace_f32",
        });
    }
    if !(scaling > 0.0 && scaling.is_finite() && freq_base > 0.0 && freq_base.is_finite()) {
        return Err(TensorError::InvalidDimension(
            "rope freq_base and scaling must be positive finite".into(),
        ));
    }

    let half = head_dim / 2;
    let stride_token = n_heads * head_dim;
    let base_f64 = freq_base as f64;
    let scale_f64 = scaling as f64;
    for (t, &pos) in positions.iter().enumerate() {
        let p = (pos as f64) / scale_f64;
        let tok_slice = &mut qk[t * stride_token..(t + 1) * stride_token];
        for i in 0..half {
            // Compute phase angle in f64 so long-context positions (e.g., 128k) suffer zero
            // powf + trig range-reduction phase drift before casting sin/cos to f32.
            let exponent = -((2 * i) as f64) / (head_dim as f64);
            let theta = p * base_f64.powf(exponent);
            let (sin_d, cos_d) = theta.sin_cos();
            let sin_t = sin_d as f32;
            let cos_t = cos_d as f32;
            for h in 0..n_heads {
                let base = h * head_dim + 2 * i;
                let x0 = tok_slice[base];
                let x1 = tok_slice[base + 1];
                tok_slice[base] = x0 * cos_t - x1 * sin_t;
                tok_slice[base + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    Ok(())
}

/// Numerically stable `max`-shifted Softmax: `out[i] = exp(x[i] - max_j x[j]) / sum_k exp(x[k] - max_j x[j])`.
pub fn softmax_f32(x: &[f32], out: &mut [f32]) -> Result<(), TensorError> {
    if x.is_empty() || x.len() != out.len() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![out.len()],
            context: "softmax_f32",
        });
    }

    let mut max_val = f32::NEG_INFINITY;
    for &v in x {
        if v.is_nan() {
            return Err(TensorError::InvalidDimension(
                "NaN encountered in softmax_f32 input".into(),
            ));
        }
        if v > max_val {
            max_val = v;
        }
    }

    if max_val == f32::NEG_INFINITY {
        out.fill(0.0);
        return Ok(());
    }
    if max_val == f32::INFINITY {
        return Err(TensorError::InvalidDimension(
            "+Inf encountered in softmax_f32 input".into(),
        ));
    }

    let mut sum = 0.0f64;
    for (dst, &v) in out.iter_mut().zip(x.iter()) {
        let e = (v - max_val).exp();
        *dst = e;
        sum += e as f64;
    }

    if sum <= 0.0 {
        out.fill(0.0);
        return Ok(());
    }

    let inv_sum = (1.0 / sum) as f32;
    for dst in out.iter_mut() {
        *dst *= inv_sum;
    }
    Ok(())
}

/// In-place numerically stable `max`-shifted Softmax.
pub fn softmax_inplace_f32(x: &mut [f32]) -> Result<(), TensorError> {
    if x.is_empty() {
        return Err(TensorError::InvalidDimension(
            "softmax_inplace_f32 requires non-empty slice".into(),
        ));
    }
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v.is_nan() {
            return Err(TensorError::InvalidDimension(
                "NaN encountered in softmax_inplace_f32".into(),
            ));
        }
        if v > max_val {
            max_val = v;
        }
    }
    if max_val == f32::NEG_INFINITY {
        x.fill(0.0);
        return Ok(());
    }
    if max_val == f32::INFINITY {
        return Err(TensorError::InvalidDimension(
            "+Inf encountered in softmax_inplace_f32".into(),
        ));
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        let e = (*v - max_val).exp();
        *v = e;
        sum += e as f64;
    }
    let inv_sum = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
    Ok(())
}
