# Module M01 Design Specification — `crates/novaserve-tensor`

## 1. Architectural Scope
`novaserve-tensor` is the zero-framework numerical and binary serialization substrate of **NovaServe**. It provides:
1. **64-Byte Cache-Aligned Bump Arena (`AlignedScratchArena`) & Owned Buffers (`AlignedBlob`)** for zero-allocation forward passes.
2. **Multi-Dimensional Tensor Views (`TensorView<'a>`) & Owned Tensors (`TensorBuf`)** supporting up to 4 dimensions, arbitrary element/byte strides, zero-copy `slice`, `reshape`, and `transpose`.
3. **Pure-Rust IEEE-754 Half-Precision (`F16`, `BF16`) & GGML Block Quantization (`Q8_0`, `Q4_0`)** with exact binary compatibility with `llama.cpp` / `GGML`.
4. **Runtime-Dispatched SIMD Compute Kernels (`AVX2+FMA`, `NEON`, and Miri/WASM-Safe Scalar Fallback)** for tiled `FP32` GEMM/GEMV, quantized integer dot products (`Q8_0 × Q8_0`, `Q4_0 × Q8_0`), `RMSNorm`, `SwiGLU`, arbitrary-position `RoPE`, and `max`-shifted `Softmax`.
5. **Zero-Copy `Safetensors` & `GGUF v3` Readers and Writers** backed by `memmap2` on native hosts and byte buffers on `wasm32`, hardened against hostile/corrupted headers.

---

## 2. Memory Layouts & Data Types

### 2.1 Alignment & `AlignedScratchArena`
- Base allocation uses `std::alloc::Layout::from_size_align(capacity_bytes, 64)` stored behind a single `NonNull<u8>` root allocation to preserve pointer provenance under Stacked/Tree Borrows (`cargo +nightly miri test`).
- Every sub-allocation `alloc_bytes(len)` rounds cursor offsets up to a 64-byte boundary (`(cursor + 63) & !63`), guaranteeing cache-line alignment and safe 256-bit AVX2 / 128-bit NEON vector loads.
- `reset()` restores `cursor = 0` in $O(1)$ without touching the system allocator.

### 2.2 Quantization Block Layouts (`block_size = 32`)

| Format | Elements / Block | Bytes / Block | Binary Layout (Little-Endian) | Dequantization Formula |
| --- | --- | --- | --- | --- |
| `F32` | 1 | 4 B | `f32` IEEE-754 binary32 | Identity |
| `F16` | 1 | 2 B | `u16` IEEE-754 binary16 (`1s + 5e + 10m`) | Exact subnormal/normal/Inf/NaN bit expansion |
| `BF16` | 1 | 2 B | `u16` bfloat16 (`1s + 8e + 7m`) | `f32::from_bits((bits as u32) << 16)` |
| `Q8_0` | 32 | **34 B** | `[0..2]`: `d: f16`, `[2..34]`: `qs: [i8; 32]` | $\hat{x}_i = \text{f16}(d) \cdot q_i,\quad q_i \in [-127, 127]$ |
| `Q4_0` | 32 | **18 B** | `[0..2]`: `d: f16`, `[2..18]`: `qs: [u8; 16]` | $\hat{x}_j = \text{f16}(d) \cdot ((qs_j \& \text{0x0F}) - 8)$ for $j \in [0,15]$;<br>$\hat{x}_{j+16} = \text{f16}(d) \cdot ((qs_j \gg 4) - 8)$ for $j \in [0,15]$ |

- **Non-Multiple-of-32 Tail Handling:** For arbitrary matrix inner dimensions $K$, row quantization allocates $B = \lceil K / 32 \rceil$ blocks, zero-padding virtual positions $[K, 32B)$. Because zero-padded activation lanes quantize to $q_i = 0$, quantized dot products over $B$ blocks equal exact $K$-element dot products with zero tail branching inside SIMD micro-kernels.

---

## 3. SIMD Kernel Architecture & Numerical Stability

### 3.1 Tiled `FP32` GEMM (`matmul_f32`)
- Computes $C[M, N] = A[M, K] \cdot B[K, N]$ and row-major weight projection $Y[M, N] = X[M, K] \cdot W^T[N, K]$.
- Uses L1/L2 cache blocking (`MC = 64, KC = 256, NC = 128`) with a register-blocked `6x16` (`AVX2+FMA`, 12 `__m256` accumulators) / `4x8` (`NEON`) micro-kernel and scalar tail cleanup.
- Parallelized across output row tiles via `rayon` on native non-Miri targets when $M \cdot N \cdot K \ge 32,768$.

### 3.2 Exact Integer SIMD Quantized Dot Products (`vec_dot_q8_0_q8_0`, `vec_dot_q4_0_q8_0`)
- Incoming activation rows $x \in \mathbb{R}^K$ are quantized once into aligned `Q8_0` blocks ($y_i \in [-127, 127]$) inside `AlignedScratchArena`.
- **AVX2 Sign-Magnitude Trick (Overflow-Free):**
  - Given signed weight bytes $w_i \in [-128, 127]$ and activation bytes $y_i \in [-127, 127]$:
  - `abs_w = _mm256_sign_epi8(w, w)` produces exact unsigned $|w_i| \in [0, 128]$ (including $|-128| = \text{0x80}_{u8} = 128$).
  - `signed_y = _mm256_sign_epi8(y, w)` negates $y_i$ when $w_i < 0$ and zeroes $y_i$ when $w_i = 0$ without overflow since $-y_i \in [-127, 127]$.
  - `prod16 = _mm256_maddubs_epi16(abs_w, signed_y)` computes pairwise sums bounded by $2 \times 128 \times 127 = 32,512 \le 32,767$ (`i16::MAX` — zero saturation!).
  - `prod32 = _mm256_madd_epi16(prod16, ones_i16)` widens into exact `i32` block sums, multiplied by `d_w * d_y` in `f32`.

### 3.3 Transformer Primitive Ops & Numerical Guards
1. **`rmsnorm_f32`:** Accumulates $\frac{1}{D}\sum_{i=0}^{D-1} x_i^2$ with FMA reduction; computes $\text{inv\_rms} = (\text{mean\_sq} + \varepsilon)^{-1/2}$ ($\varepsilon > 0$) and writes $y_i = (x_i \cdot \text{inv\_rms}) \cdot w_i$.
2. **`swiglu_f32`:** Computes $\text{SiLU}(g_i) \cdot u_i = \frac{g_i}{1 + \exp(-g_i)} \cdot u_i$ with branchless overflow guard ($g_i < -88 \implies 0$, $g_i > 88 \implies g_i$).
3. **`rope_inplace_f32`:** Applies rotary embeddings over `[num_tokens, n_heads, head_dim]` with per-token `positions: &[u32]`, `freq_base` (e.g., `10000.0` or `500000.0`), and frequency `scaling` factor, supporting both interleaved/adjacent and half-rotated (`Llama`/`Qwen` NeoX-style) layouts.
4. **`softmax_f32`:** Two-pass max-shifted $\exp(x_i - m)$ where $m = \max_j x_j$:
   - All $-\infty$ inputs safely return uniform `0.0` or typed error without producing `NaN` (`-\infty - (-\infty)` guarded).

---

## 4. Binary Weight Parsers & Hostile Input Defenses

### 4.1 `Safetensors`
- Validates `header_len <= 64 MiB` and `8 + header_len <= buf.len()` via checked arithmetic before parsing JSON.
- Validates every tensor's `shape` product using `usize::checked_mul`, verifies `begin <= end`, `end - begin == expected_bytes`, `end <= data_buf.len()`, and enforces strict non-overlapping intervals sorted by `begin`.

### 4.2 `GGUF v3`
- Checks magic `0x46554747` (`"GGUF"`), version `3`, reasonable `tensor_count` and `metadata_kv_count` bounds against remaining file bytes before allocating `Vec` capacities.
- Implements all 13 metadata types (`U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `F32`, `Bool` strictly `0|1`, `String` validated UTF-8, nested `Array` bounded to depth 8, `U64`, `I64`, `F64`).
- Validates power-of-two `general.alignment` (default `32`), computes padded `tensor_data_offset`, and checks every tensor's byte span for overflow, alignment, OOB, and overlap.
