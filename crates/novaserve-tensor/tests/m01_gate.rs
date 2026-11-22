use novaserve_tensor::{
    bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, linear_f32, linear_projection, matmul_f32,
    matmul_q4_0, matmul_q8_0, rmsnorm_f32, rope_inplace_f32, serialize_gguf_v3,
    serialize_safetensors, simd, softmax_f32, swiglu_f32, AlignedScratchArena, DType, GgufArray,
    GgufFile, GgufValue, SafetensorsFile, SplitMix64, TensorBuf, CACHE_LINE_ALIGN,
};
use std::collections::BTreeMap;

fn f64_matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p] as f64;
            for j in 0..n {
                out[i * n + j] += a_ip * (b[p * n + j] as f64);
            }
        }
    }
    out
}

fn f64_linear_ref(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (x[i * k + p] as f64) * (w[j * k + p] as f64);
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let fa = va as f64;
        dot += fa * vb;
        na += fa * fa;
        nb += vb * vb;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn arena_alignment_disjoint_slices_and_zero_alloc_reset() {
    let mut arena = AlignedScratchArena::new(64 * 1024).expect("arena init");
    assert_eq!(arena.base_addr() % CACHE_LINE_ALIGN, 0);

    for iter in 0..16 {
        let a = arena.alloc_f32(17).expect("alloc a");
        let b = arena.alloc_f32(65).expect("alloc b");
        let c = arena.alloc_bytes(37).expect("alloc c");

        assert_eq!((a.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert_eq!((b.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert_eq!((c.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);

        for (i, slot) in a.iter_mut().enumerate() {
            *slot = (iter * 100 + i) as f32;
        }
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = -((iter * 50 + i) as f32);
        }
        c.fill(0xAB);

        assert_eq!(a[16], (iter * 100 + 16) as f32);
        assert_eq!(b[64], -((iter * 50 + 64) as f32));
        assert!(c.iter().all(|&byte| byte == 0xAB));
        assert_eq!(arena.alloc_count(), 3);

        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.alloc_count(), 0);
    }
}

#[test]
fn ieee754_f16_and_bf16_roundtrip_and_subnormals() {
    // Exhaustively check all 65,536 f16 bit patterns round-trip cleanly!
    #[cfg(not(miri))]
    let step = 1u32;
    #[cfg(miri)]
    let step = 257u32;

    let mut bits_u32 = 0u32;
    while bits_u32 <= 0xFFFF {
        let h = bits_u32 as u16;
        let f = f16_to_f32(h);
        let h2 = f32_to_f16(f);
        if f.is_nan() {
            assert!((h2 & 0x7C00) == 0x7C00 && (h2 & 0x03FF) != 0);
        } else {
            assert_eq!(h, h2, "f16 bit pattern 0x{h:04x} -> {f} -> 0x{h2:04x}");
        }
        bits_u32 += step;
    }

    // Specific bf16 values including subnormal/tie/NaN
    for &val in &[
        0.0f32,
        -0.0,
        1.0,
        -2.5,
        123.75,
        -65504.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let b = f32_to_bf16(val);
        let back = bf16_to_f32(b);
        if val.is_infinite() {
            assert_eq!(val, back);
        } else {
            let rel = if val == 0.0 {
                0.0
            } else {
                ((val - back) / val).abs()
            };
            assert!(rel < 0.01, "bf16 drift on {val}: got {back}");
        }
    }
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}

#[test]
fn tensor_views_slice_reshape_transpose_zero_copy() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = TensorBuf::from_f32_slice(&[2, 3, 4], &data).expect("tensor");
    let v = t.view();
    v.check_aligned(CACHE_LINE_ALIGN).expect("64B aligned");

    // Slice axis 1 [1..3] -> shape [2, 2, 4]
    let sliced = v.slice(1, 1, 3).expect("slice");
    assert_eq!(sliced.dims(), &[2, 2, 4]);
    assert_eq!(sliced.get_f32(&[0, 0, 2]).unwrap(), 6.0);
    assert_eq!(sliced.get_f32(&[1, 1, 3]).unwrap(), 23.0);

    // Transpose (0, 2) on original -> shape [4, 3, 2]
    let tr = v.transpose(0, 2).expect("transpose");
    assert_eq!(tr.dims(), &[4, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(
                    v.get_f32(&[i, j, k]).unwrap(),
                    tr.get_f32(&[k, j, i]).unwrap()
                );
            }
        }
    }

    // Reshape contiguous [2, 3, 4] -> [6, 4]
    let rs = v.reshape(&[6, 4]).expect("reshape");
    assert_eq!(rs.dims(), &[6, 4]);
    assert_eq!(rs.get_f32(&[5, 3]).unwrap(), 23.0);
}

#[test]
fn differential_1000_gemm_and_quantized_matmul_suite() {
    #[cfg(not(miri))]
    let (iterations, max_m, max_k_extra, max_n) = (1_000usize, 31usize, 128usize, 45usize);
    #[cfg(miri)]
    let (iterations, max_m, max_k_extra, max_n) = (12usize, 5usize, 11usize, 8usize);

    let mut rng = SplitMix64::new(0xDEC0_DED1_2026_1122);
    #[cfg(not(miri))]
    let arena_bytes = 4 * 1024 * 1024;
    #[cfg(miri)]
    let arena_bytes = 64 * 1024;
    let mut arena = AlignedScratchArena::new(arena_bytes).expect("arena");

    let mut worst_fp32_err = 0.0f64;
    let mut min_q8_cos = 1.0f64;
    let mut min_q4_cos = 1.0f64;
    let mut worst_simd_vs_scalar_q8 = 0.0f32;
    let mut worst_simd_vs_scalar_q4 = 0.0f32;

    for case_idx in 0..iterations {
        arena.reset();

        // Include 1xK GEMV, odd non-multiples of 32/64, and multi-tile GEMM dimensions
        let m = if case_idx % 4 == 0 {
            1
        } else {
            1 + (rng.next_u32() as usize % max_m)
        };
        let k = if case_idx % 3 == 0 {
            // Odd non-multiple of 32/64 tail!
            33 + (rng.next_u32() as usize % max_k_extra)
        } else {
            32 + (rng.next_u32() as usize % max_k_extra)
        };
        let n = 4 + (rng.next_u32() as usize % max_n);

        let a: Vec<f32> = (0..m * k).map(|_| rng.range_f32(-0.5, 0.5)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.range_f32(-0.5, 0.5)).collect();

        // 1. FP32 GEMM vs f64 triple-loop reference
        let mut c_simd = vec![0.0f32; m * n];
        matmul_f32(&a, &b, m, k, n, &mut c_simd).expect("matmul_f32");
        let c_ref = f64_matmul_ref(&a, &b, m, k, n);

        for idx in 0..(m * n) {
            let err = (c_simd[idx] as f64 - c_ref[idx]).abs();
            if err > worst_fp32_err {
                worst_fp32_err = err;
            }
            assert!(
                err < 1e-5,
                "FP32 GEMM drift at case {case_idx} ({m}x{k}x{n}): err={err:.3e} >= 1e-5"
            );
        }

        // 2. Linear FP32 (`x * W^T`) vs f64 reference
        let mut lin_out = vec![0.0f32; m * n];
        linear_f32(&a, &b, m, k, n, &mut lin_out).expect("linear_f32");
        let lin_ref = f64_linear_ref(&a, &b, m, k, n);
        for idx in 0..(m * n) {
            let err = (lin_out[idx] as f64 - lin_ref[idx]).abs();
            assert!(
                err < 1e-5,
                "FP32 linear_f32 drift at case {case_idx}: err={err:.3e}"
            );
        }

        // 3. Q8_0 and Q4_0 Quantized Matmul vs Dequantized f64 Reference + SIMD == Scalar check
        // Ensure non-degenerate vector length (`k >= 32`) for meaningful cosine similarity
        if k >= 32 && n >= 4 {
            let w_f32: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian_f32() * 0.5).collect();
            let x_f32: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian_f32() * 0.5).collect();

            let w_q8_buf = TensorBuf::from_f32_as_dtype(&[n, k], &w_f32, DType::Q8_0).unwrap();
            let w_q8_deq = w_q8_buf.view().to_f32_vec().unwrap();
            let ref_q8 = f64_linear_ref(&x_f32, &w_q8_deq, m, k, n);

            let mut out_q8 = vec![0.0f32; m * n];
            matmul_q8_0(&x_f32, w_q8_buf.as_bytes(), m, k, n, &arena, &mut out_q8)
                .expect("matmul_q8_0");

            let cos_q8 = cosine_similarity(&out_q8, &ref_q8);
            min_q8_cos = min_q8_cos.min(cos_q8);
            assert!(
                cos_q8 > 0.999,
                "Q8_0 cosine_similarity={cos_q8:.6} <= 0.999 at case {case_idx} ({m}x{k}x{n})"
            );

            let w_q4_buf = TensorBuf::from_f32_as_dtype(&[n, k], &w_f32, DType::Q4_0).unwrap();
            let w_q4_deq = w_q4_buf.view().to_f32_vec().unwrap();
            let ref_q4 = f64_linear_ref(&x_f32, &w_q4_deq, m, k, n);

            let mut out_q4 = vec![0.0f32; m * n];
            matmul_q4_0(&x_f32, w_q4_buf.as_bytes(), m, k, n, &arena, &mut out_q4)
                .expect("matmul_q4_0");

            let cos_q4 = cosine_similarity(&out_q4, &ref_q4);
            min_q4_cos = min_q4_cos.min(cos_q4);
            assert!(
                cos_q4 > 0.990,
                "Q4_0 cosine_similarity={cos_q4:.6} <= 0.990 at case {case_idx} ({m}x{k}x{n})"
            );

            // Direct SIMD vs Scalar quantized dot-product equivalence (`<= 1e-6`)
            let num_blocks = k.div_ceil(32);
            let mut x0_q8 = vec![0u8; num_blocks * 34];
            novaserve_tensor::quantize_row_q8_0(&x_f32[..k], &mut x0_q8).unwrap();
            let w0_q8 = &w_q8_buf.as_bytes()[..num_blocks * 34];
            let d_simd_q8 = simd::vec_dot_q8_0_q8_0(w0_q8, &x0_q8, num_blocks);
            let d_scal_q8 = simd::scalar::vec_dot_q8_0_q8_0_scalar(w0_q8, &x0_q8, num_blocks);
            let diff_q8 = (d_simd_q8 - d_scal_q8).abs();
            worst_simd_vs_scalar_q8 = worst_simd_vs_scalar_q8.max(diff_q8);
            assert!(
                diff_q8 <= 1e-6,
                "Q8_0 SIMD vs Scalar mismatch: {d_simd_q8} vs {d_scal_q8} (diff={diff_q8})"
            );

            let w0_q4 = &w_q4_buf.as_bytes()[..num_blocks * 18];
            let d_simd_q4 = simd::vec_dot_q4_0_q8_0(w0_q4, &x0_q8, num_blocks);
            let d_scal_q4 = simd::scalar::vec_dot_q4_0_q8_0_scalar(w0_q4, &x0_q8, num_blocks);
            let diff_q4 = (d_simd_q4 - d_scal_q4).abs();
            worst_simd_vs_scalar_q4 = worst_simd_vs_scalar_q4.max(diff_q4);
            assert!(
                diff_q4 <= 1e-6,
                "Q4_0 SIMD vs Scalar mismatch: {d_simd_q4} vs {d_scal_q4} (diff={diff_q4})"
            );
        }
    }

    // Polymorphic linear_projection smoke check across F16 & BF16
    arena.reset();
    let w_f16 = TensorBuf::from_f32_as_dtype(&[8, 32], &vec![0.25f32; 256], DType::F16).unwrap();
    let w_bf16 = TensorBuf::from_f32_as_dtype(&[8, 32], &vec![0.25f32; 256], DType::BF16).unwrap();
    let x_ones = vec![1.0f32; 64];
    let mut out_poly = vec![0.0f32; 16];
    linear_projection(&x_ones, &w_f16.view(), 2, &arena, &mut out_poly).unwrap();
    assert!((out_poly[0] - 8.0).abs() < 1e-4);
    linear_projection(&x_ones, &w_bf16.view(), 2, &arena, &mut out_poly).unwrap();
    assert!((out_poly[0] - 8.0).abs() < 1e-4);

    let _ = (
        worst_fp32_err,
        min_q8_cos,
        min_q4_cos,
        worst_simd_vs_scalar_q8,
        worst_simd_vs_scalar_q4,
    );
}

#[test]
fn transformer_primitives_rmsnorm_swiglu_rope_softmax_vs_f64() {
    let mut rng = SplitMix64::new(0xCAFE_BABE_1234_5678);
    let tokens = 7usize;
    let n_heads = 4usize;
    let head_dim = 32usize;
    let dim = n_heads * head_dim; // 128

    // 1. RMSNorm vs f64
    let x: Vec<f32> = (0..tokens * dim)
        .map(|_| rng.range_f32(-3.0, 3.0))
        .collect();
    let w: Vec<f32> = (0..dim).map(|_| rng.range_f32(0.5, 1.5)).collect();
    let eps = 1e-5f32;
    let mut out_rms = vec![0.0f32; tokens * dim];
    rmsnorm_f32(&x, &w, eps, &mut out_rms).unwrap();

    for t in 0..tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let mean_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (dim as f64);
        let inv = 1.0 / (mean_sq + eps as f64).sqrt();
        for i in 0..dim {
            let expected = (row[i] as f64) * inv * (w[i] as f64);
            let err = (out_rms[t * dim + i] as f64 - expected).abs();
            assert!(err < 1e-5, "rmsnorm err={err}");
        }
    }

    // 2. SwiGLU vs f64 (including extreme tails -100, +100)
    let mut gate: Vec<f32> = (0..257).map(|_| rng.range_f32(-6.0, 6.0)).collect();
    gate[0] = -120.0;
    gate[1] = 120.0;
    let up: Vec<f32> = (0..257).map(|_| rng.range_f32(-2.0, 2.0)).collect();
    let mut out_swi = vec![0.0f32; 257];
    swiglu_f32(&gate, &up, &mut out_swi).unwrap();
    for i in 0..257 {
        let g = gate[i] as f64;
        let silu = if g < -88.0 {
            0.0
        } else if g > 88.0 {
            g
        } else {
            g / (1.0 + (-g).exp())
        };
        let expected = silu * (up[i] as f64);
        let err = (out_swi[i] as f64 - expected).abs();
        assert!(err < 1e-5 && out_swi[i].is_finite(), "swiglu err={err}");
    }

    // 3. RoPE with arbitrary non-contiguous positions (including long-context 131,072) vs f64
    let positions = [0u32, 17, 3, 511, 42, 1024, 131_072];
    let mut qk = x.clone();
    rope_inplace_f32(&mut qk, &positions, n_heads, head_dim, 10_000.0, 1.0).unwrap();
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..n_heads {
            for i in 0..(head_dim / 2) {
                let theta = (pos as f64) * 10_000.0f64.powf(-((2 * i) as f64) / (head_dim as f64));
                let (s, c) = theta.sin_cos();
                let base = t * dim + h * head_dim + 2 * i;
                let x0 = x[base] as f64;
                let x1 = x[base + 1] as f64;
                let r0 = x0 * c - x1 * s;
                let r1 = x0 * s + x1 * c;
                assert!((qk[base] as f64 - r0).abs() < 1e-5);
                assert!((qk[base + 1] as f64 - r1).abs() < 1e-5);
            }
        }
    }

    // 4. Numerically stable Softmax under extreme shifts (+10000, -10000, all -inf)
    let extreme = [10_000.0f32, 9_999.0, 9_995.0, -10_000.0];
    let mut probs = [0.0f32; 4];
    softmax_f32(&extreme, &mut probs).unwrap();
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));

    let all_neg_inf = [f32::NEG_INFINITY; 8];
    let mut zero_probs = [1.0f32; 8];
    softmax_f32(&all_neg_inf, &mut zero_probs).unwrap();
    assert!(zero_probs.iter().all(|&p| p == 0.0));
}

#[test]
fn safetensors_and_gguf_v3_lossless_mmap_roundtrip_and_10000_hostile_fuzz() {
    let mut rng = SplitMix64::new(0x9988_7766_5544_3322);
    let f32_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.125 - 4.0).collect();
    let t_f32 = TensorBuf::from_f32_slice(&[2, 32], &f32_data).unwrap();
    let t_f16 = TensorBuf::from_f32_as_dtype(&[2, 32], &f32_data, DType::F16).unwrap();
    let t_bf16 = TensorBuf::from_f32_as_dtype(&[2, 32], &f32_data, DType::BF16).unwrap();
    let t_q8 = TensorBuf::from_f32_as_dtype(&[2, 32], &f32_data, DType::Q8_0).unwrap();
    let t_q4 = TensorBuf::from_f32_as_dtype(&[2, 32], &f32_data, DType::Q4_0).unwrap();

    // --- 1. Safetensors lossless round-trip (in-memory + disk mmap) ---
    let mut st_meta = BTreeMap::new();
    st_meta.insert("format".to_string(), "pt".to_string());
    st_meta.insert("producer".to_string(), "novaserve".to_string());

    let st_bytes = serialize_safetensors(
        &[
            ("blk.0.attn_norm.weight", t_f32.view()),
            ("blk.0.attn_q.weight", t_f16.view()),
            ("blk.0.attn_k.weight", t_bf16.view()),
        ],
        Some(&st_meta),
    )
    .expect("serialize_safetensors");

    let st_parsed = SafetensorsFile::from_bytes(&st_bytes).expect("parse safetensors");
    assert_eq!(
        st_parsed.metadata().get("producer").map(String::as_str),
        Some("novaserve")
    );
    assert_eq!(
        st_parsed
            .tensor("blk.0.attn_norm.weight")
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        f32_data
    );

    #[cfg(not(miri))]
    {
        let tmp_dir = std::env::temp_dir();
        let st_path = tmp_dir.join(format!("novaserve_m01_{}.safetensors", std::process::id()));
        novaserve_tensor::write_safetensors(&st_path, &[("w", t_f32.view())], Some(&st_meta))
            .expect("write_safetensors");
        let st_mmap = SafetensorsFile::open(&st_path).expect("mmap safetensors");
        assert_eq!(st_mmap.tensor("w").unwrap().to_f32_vec().unwrap(), f32_data);
        let _ = std::fs::remove_file(&st_path);
    }

    // --- 2. GGUF v3 lossless round-trip across ALL 13 metadata types + quantized tensors ---
    let mut gguf_meta = BTreeMap::new();
    gguf_meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    gguf_meta.insert("general.alignment".into(), GgufValue::U32(32));
    gguf_meta.insert("test.u8".into(), GgufValue::U8(250));
    gguf_meta.insert("test.i8".into(), GgufValue::I8(-42));
    gguf_meta.insert("test.u16".into(), GgufValue::U16(65_000));
    gguf_meta.insert("test.i16".into(), GgufValue::I16(-12_345));
    gguf_meta.insert("test.u32".into(), GgufValue::U32(1_000_000));
    gguf_meta.insert("test.i32".into(), GgufValue::I32(-999_999));
    gguf_meta.insert("test.f32".into(), GgufValue::F32(1.5));
    gguf_meta.insert("test.bool".into(), GgufValue::Bool(true));
    gguf_meta.insert("test.u64".into(), GgufValue::U64(9_876_543_210));
    gguf_meta.insert("test.i64".into(), GgufValue::I64(-9_876_543_210));
    gguf_meta.insert("test.f64".into(), GgufValue::F64(std::f64::consts::PI));
    gguf_meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(GgufArray {
            elem_type: 8,
            elements: vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
            ],
        }),
    );
    // Nested array (`Array` of `Array` of `U32`)
    gguf_meta.insert(
        "test.nested_array".into(),
        GgufValue::Array(GgufArray {
            elem_type: 9,
            elements: vec![GgufValue::Array(GgufArray {
                elem_type: 4,
                elements: vec![GgufValue::U32(7), GgufValue::U32(11)],
            })],
        }),
    );

    let gguf_bytes = serialize_gguf_v3(
        &gguf_meta,
        &[
            ("blk.0.attn_norm.weight", t_f32.view()),
            ("blk.0.attn_q.weight", t_q8.view()),
            ("blk.0.attn_k.weight", t_q4.view()),
            ("blk.0.attn_v.weight", t_f16.view()),
            ("blk.0.ffn_down.weight", t_bf16.view()),
        ],
    )
    .expect("serialize_gguf_v3");

    let gguf_parsed = GgufFile::from_bytes(&gguf_bytes).expect("parse gguf");
    assert_eq!(gguf_parsed.metadata(), &gguf_meta);
    assert_eq!(
        gguf_parsed
            .tensor("blk.0.attn_norm.weight")
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        f32_data
    );
    assert_eq!(
        gguf_parsed
            .tensor("blk.0.attn_q.weight")
            .unwrap()
            .raw_bytes(),
        t_q8.as_bytes()
    );
    assert_eq!(
        gguf_parsed
            .tensor("blk.0.attn_k.weight")
            .unwrap()
            .raw_bytes(),
        t_q4.as_bytes()
    );

    #[cfg(not(miri))]
    {
        let tmp_dir = std::env::temp_dir();
        let gguf_path = tmp_dir.join(format!("novaserve_m01_{}.gguf", std::process::id()));
        novaserve_tensor::write_gguf_v3(
            &gguf_path,
            &gguf_meta,
            &[("q4", t_q4.view()), ("q8", t_q8.view())],
        )
        .expect("write_gguf_v3");
        let gguf_mmap = GgufFile::open(&gguf_path).expect("mmap gguf");
        assert_eq!(gguf_mmap.tensor("q4").unwrap().raw_bytes(), t_q4.as_bytes());
        assert_eq!(gguf_mmap.tensor("q8").unwrap().raw_bytes(), t_q8.as_bytes());
        let _ = std::fs::remove_file(&gguf_path);
    }

    // --- 3. 10,000 Hostile Corrupted / Truncated / Overflowing Fuzz Payloads ---
    #[cfg(not(miri))]
    let fuzz_iters = 10_000usize;
    #[cfg(miri)]
    let fuzz_iters = 40usize;

    let mut clean_rejections = 0usize;
    for i in 0..fuzz_iters {
        if i % 2 == 0 {
            // Mutate valid GGUF bytes (truncations, u64::MAX overflows, corrupt types/offsets)
            let mut mutated = gguf_bytes.clone();
            match i % 6 {
                0 => {
                    // Truncate at arbitrary byte boundary
                    let cut = (rng.next_u32() as usize) % mutated.len();
                    mutated.truncate(cut);
                }
                1 => {
                    // Inject huge u64::MAX in tensor_count / kv_count / string len
                    let pos = 8 + ((rng.next_u32() as usize) % (mutated.len() - 16));
                    mutated[pos..pos + 8].copy_from_slice(&u64::MAX.to_le_bytes());
                }
                2 => {
                    // Corrupt random header bytes
                    for _ in 0..6 {
                        let idx = (rng.next_u32() as usize) % (mutated.len() / 2);
                        mutated[idx] ^= (rng.next_u32() as u8).max(1);
                    }
                }
                3 => {
                    // Pure random hostile blob
                    let len = (rng.next_u32() as usize) % 256;
                    mutated = (0..len).map(|_| rng.next_u32() as u8).collect();
                }
                4 => {
                    // Valid magic + v3 + deep recursive array attack (depth 32)
                    mutated.clear();
                    mutated.extend_from_slice(&novaserve_tensor::GGUF_MAGIC.to_le_bytes());
                    mutated.extend_from_slice(&3u32.to_le_bytes());
                    mutated.extend_from_slice(&0u64.to_le_bytes());
                    mutated.extend_from_slice(&1u64.to_le_bytes());
                    mutated.extend_from_slice(&1u64.to_le_bytes());
                    mutated.push(b'a');
                    mutated.extend_from_slice(&9u32.to_le_bytes());
                    for _ in 0..16 {
                        mutated.extend_from_slice(&9u32.to_le_bytes());
                        mutated.extend_from_slice(&1u64.to_le_bytes());
                    }
                }
                _ => {
                    // Append unaligned / corrupt tensor offset
                    let tail_header = 24 + ((rng.next_u32() as usize) % 120);
                    if tail_header < mutated.len() {
                        mutated[tail_header] = mutated[tail_header].wrapping_add(3);
                    }
                }
            }
            if GgufFile::from_bytes(&mutated).is_err() {
                clean_rejections += 1;
            }
        } else {
            // Mutate valid Safetensors bytes (OOB offsets, overlapping tensors, huge header_len)
            let mut mutated = st_bytes.clone();
            match i % 5 {
                0 => {
                    let cut = (rng.next_u32() as usize) % mutated.len();
                    mutated.truncate(cut);
                }
                1 => {
                    // Overflowing 8-byte header_len
                    mutated[..8].copy_from_slice(&(u64::MAX - 4).to_le_bytes());
                }
                2 => {
                    // Malicious overlapping offsets in JSON
                    let evil_json = br#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}}"#;
                    mutated.clear();
                    mutated.extend_from_slice(&(evil_json.len() as u64).to_le_bytes());
                    mutated.extend_from_slice(evil_json);
                    mutated.extend_from_slice(&[0u8; 24]);
                }
                3 => {
                    // Shape overflow attack `[usize::MAX, 4]`
                    let evil_json = format!(
                        r#"{{"a":{{"dtype":"F32","shape":[{},4],"data_offsets":[0,16]}}}}"#,
                        u64::MAX
                    );
                    mutated.clear();
                    mutated.extend_from_slice(&(evil_json.len() as u64).to_le_bytes());
                    mutated.extend_from_slice(evil_json.as_bytes());
                    mutated.extend_from_slice(&[0u8; 16]);
                }
                _ => {
                    // Trailing unreferenced bytes or random header corruption
                    mutated.push(0xFF);
                }
            }
            if SafetensorsFile::from_bytes(&mutated).is_err() {
                clean_rejections += 1;
            }
        }
    }
    assert!(clean_rejections > fuzz_iters / 2);
}

#[cfg(not(miri))]
#[test]
fn m01_kernel_telemetry_benchmark() {
    let mut rng = SplitMix64::new(0x1122_3344_5566_7788);
    let mut arena = AlignedScratchArena::new(8 * 1024 * 1024).unwrap();
    let m = 64usize;
    let k = 512usize;
    let n = 512usize;
    let a: Vec<f32> = (0..m * k).map(|_| rng.range_f32(-0.5, 0.5)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.range_f32(-0.5, 0.5)).collect();
    let mut out = vec![0.0f32; m * n];

    // Warmup
    matmul_f32(&a, &b, m, k, n, &mut out).unwrap();

    let reps = 20usize;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        matmul_f32(&a, &b, m, k, n, &mut out).unwrap();
    }
    let dt_gemm = t0.elapsed().as_secs_f64();
    let gflops_gemm =
        (2.0 * (m as f64) * (k as f64) * (n as f64) * (reps as f64)) / (dt_gemm * 1e9);

    // Decode GEMV: 1 x 1024 -> 2048 (`FP32` vs `Q8_0` vs `Q4_0`)
    let dk = 1024usize;
    let dn = 2048usize;
    let x_dec: Vec<f32> = (0..dk).map(|_| rng.next_gaussian_f32()).collect();
    let w_dec: Vec<f32> = (0..dn * dk)
        .map(|_| rng.next_gaussian_f32() * 0.1)
        .collect();
    let w_q8 = TensorBuf::from_f32_as_dtype(&[dn, dk], &w_dec, DType::Q8_0).unwrap();
    let w_q4 = TensorBuf::from_f32_as_dtype(&[dn, dk], &w_dec, DType::Q4_0).unwrap();
    let mut y_dec = vec![0.0f32; dn];

    let dec_reps = 40usize;
    let tf = std::time::Instant::now();
    for _ in 0..dec_reps {
        linear_f32(&x_dec, &w_dec, 1, dk, dn, &mut y_dec).unwrap();
    }
    let us_fp32 = tf.elapsed().as_secs_f64() * 1e6 / (dec_reps as f64);

    let t8 = std::time::Instant::now();
    for _ in 0..dec_reps {
        arena.reset();
        matmul_q8_0(&x_dec, w_q8.as_bytes(), 1, dk, dn, &arena, &mut y_dec).unwrap();
    }
    let us_q8 = t8.elapsed().as_secs_f64() * 1e6 / (dec_reps as f64);

    let t4 = std::time::Instant::now();
    for _ in 0..dec_reps {
        arena.reset();
        matmul_q4_0(&x_dec, w_q4.as_bytes(), 1, dk, dn, &arena, &mut y_dec).unwrap();
    }
    let us_q4 = t4.elapsed().as_secs_f64() * 1e6 / (dec_reps as f64);

    println!(
        "[M01 Telemetry] Backend={:?} | FP32 GEMM={gflops_gemm:.2} GFLOPS | 1x1024->2048 GEMV: FP32={us_fp32:.1}us, Q8_0={us_q8:.1}us, Q4_0={us_q4:.1}us",
        simd::active_backend()
    );
}
