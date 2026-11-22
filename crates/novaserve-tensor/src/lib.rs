#![deny(unsafe_code)]

//! `novaserve-tensor`: Zero-framework 64-byte aligned tensor arena, GGML `Q8_0`/`Q4_0`
//! quantization, SIMD (`AVX2+FMA` / `NEON` / Scalar) GEMM & GEMV kernels, Transformer
//! primitive operators, and zero-copy `Safetensors` & `GGUF v3` readers/writers.

#[allow(unsafe_code)]
pub mod alloc;
pub mod dtype;
pub mod error;
pub mod gguf_io;
pub mod ops;
pub mod quant;
pub mod rng;
pub mod safetensors_io;
#[allow(unsafe_code)]
pub mod simd;
pub mod tensor;

pub use alloc as arena;
pub use alloc::{AlignedBuf, AlignedScratchArena, ByteSource, CACHE_LINE_ALIGN};
pub use dtype::{
    bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, DType, BYTES_Q4_0, BYTES_Q8_0, QK4_0, QK8_0,
};
pub use error::TensorError;
#[cfg(not(target_arch = "wasm32"))]
pub use gguf_io::write_gguf_v3;
pub use gguf_io::{
    serialize_gguf_v3, GgufArray, GgufFile, GgufTensorInfo, GgufValue, GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC, GGUF_VERSION_V3,
};
pub use ops::{
    linear_f32, linear_projection, matmul_f32, matmul_q4_0, matmul_q8_0, rmsnorm_f32,
    rope_inplace_f32, softmax_f32, softmax_inplace_f32, swiglu_f32,
};
pub use quant::{
    dequantize_block_q4_0, dequantize_block_q8_0, dequantize_row_q4_0, dequantize_row_q8_0,
    quantize_block_q4_0, quantize_block_q8_0, quantize_matrix_q4_0, quantize_matrix_q8_0,
    quantize_row_q4_0, quantize_row_q8_0,
};
pub use rng::{ChaCha8Rng, SplitMix64};
#[cfg(not(target_arch = "wasm32"))]
pub use safetensors_io::write_safetensors;
pub use safetensors_io::{serialize_safetensors, SafetensorEntry, SafetensorsFile};
pub use tensor::{TensorBuf, TensorView, MAX_RANK};
