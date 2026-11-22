//! Runtime-dispatched SIMD compute backend (`novaserve-tensor::simd`).
//!
//! Under `cfg(miri)` or `wasm32`, dispatches cleanly to the portable `scalar` kernels.
//! On `x86_64`, detects `AVX2 + FMA` at runtime; on `aarch64`, uses `NEON`.

pub mod scalar;

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub mod avx2;

#[cfg(all(target_arch = "aarch64", not(miri)))]
pub mod neon;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    Avx2Fma,
    Neon,
}

#[inline]
pub fn active_backend() -> SimdBackend {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return SimdBackend::Avx2Fma;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return SimdBackend::Neon;
        }
    }
    SimdBackend::Scalar
}

#[inline]
pub fn vec_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime check verified AVX2 and FMA support; slice lengths match.
            return unsafe { avx2::vec_dot_f32_avx2(a, b) };
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Runtime check verified NEON support; slice lengths match.
            return unsafe { neon::vec_dot_f32_neon(a, b) };
        }
    }
    scalar::vec_dot_f32_scalar(a, b)
}

#[inline]
pub fn matmul_f32_row_tile(
    a_tile: &[f32],
    b: &[f32],
    tile_m: usize,
    k: usize,
    n: usize,
    out_tile: &mut [f32],
) {
    assert!(a_tile.len() >= tile_m * k);
    assert!(b.len() >= k * n);
    assert!(out_tile.len() >= tile_m * n);

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime check verified AVX2 and FMA support; buffer bounds asserted above.
            unsafe { avx2::matmul_f32_row_tile_avx2(a_tile, b, tile_m, k, n, out_tile) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Runtime check verified NEON support; buffer bounds asserted above.
            unsafe { neon::matmul_f32_row_tile_neon(a_tile, b, tile_m, k, n, out_tile) };
            return;
        }
    }
    scalar::matmul_f32_row_tile_scalar(a_tile, b, tile_m, k, n, out_tile);
}

#[inline]
pub fn vec_dot_q8_0_q8_0(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    assert_eq!(w_blocks.len(), num_blocks * 34);
    assert_eq!(x_blocks.len(), num_blocks * 34);

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime check verified AVX2+FMA; exact `num_blocks * 34` bounds verified.
            return unsafe { avx2::vec_dot_q8_0_q8_0_avx2(w_blocks, x_blocks, num_blocks) };
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Runtime check verified NEON; exact `num_blocks * 34` bounds verified.
            return unsafe { neon::vec_dot_q8_0_q8_0_neon(w_blocks, x_blocks, num_blocks) };
        }
    }
    scalar::vec_dot_q8_0_q8_0_scalar(w_blocks, x_blocks, num_blocks)
}

#[inline]
pub fn vec_dot_q4_0_q8_0(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    assert_eq!(w_blocks.len(), num_blocks * 18);
    assert_eq!(x_blocks.len(), num_blocks * 34);

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime check verified AVX2+FMA; exact block slice bounds verified.
            return unsafe { avx2::vec_dot_q4_0_q8_0_avx2(w_blocks, x_blocks, num_blocks) };
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Runtime check verified NEON; exact block slice bounds verified.
            return unsafe { neon::vec_dot_q4_0_q8_0_neon(w_blocks, x_blocks, num_blocks) };
        }
    }
    scalar::vec_dot_q4_0_q8_0_scalar(w_blocks, x_blocks, num_blocks)
}

#[inline]
pub fn rmsnorm_row_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime check verified AVX2+FMA; equal lengths asserted above.
            unsafe { avx2::rmsnorm_row_avx2(x, weight, eps, out) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Runtime check verified NEON; equal lengths asserted above.
            unsafe { neon::rmsnorm_row_neon(x, weight, eps, out) };
            return;
        }
    }
    scalar::rmsnorm_row_scalar(x, weight, eps, out);
}
