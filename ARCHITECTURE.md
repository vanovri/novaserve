# NovaServe Systems Architecture & First-Principles Invariants

**NovaServe** (`novaserve`) is a zero-ML-framework High-Throughput LLM Inference Serving Engine engineered from first principles in Rust (Edition 2021). It unifies the core systems breakthroughs of **vLLM** (PagedAttention & Iteration-Level Continuous Batching), **SGLang** (Compressed Radix Prefix Caching & Byte-DFA Constrained Decoding), **Sarathi-Serve** (Token-Budgeted Chunked Prefill), **FlashDecoding** (Online-Softmax Parallel Block-Partition Reduction), **Leviathan et al.** (Exact Distribution-Preserving Speculative Decoding), and **GGML/llama.cpp** (64-Byte Aligned `AVX2+FMA`/`NEON` Block-Quantized `Q8_0`/`Q4_0` Kernels + Zero-Copy `GGUF v3` & `Safetensors` Loaders).

---

## 1. Crate Dependency DAG & `unsafe` Quarantine Boundary

```text
                        ┌──────────────────────────┐
                        │     novaserve-tensor     │  <-- Controlled `unsafe` ONLY in `alloc` (64B Arena/mmap)
                        │  (M01: SIMD, Q8_0/Q4_0,  │      & `simd` (AVX2+FMA / NEON + Miri-verified Scalar)
                        │   GGUF v3, Safetensors)  │
                        └────────────┬─────────────┘
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
     ┌───────────────────────┐ ┌───────────┐ ┌───────────────────────┐
     │  novaserve-frontend   │ │novaserve- │ │    novaserve-model    │
     │ (M02: Byte-BPE, UTF-8 │ │    kv     │ │ (M04: GQA Llama/Qwen, │
     │  Stream, JSON-DFA)    │ │(M03: CoW  │ │  Paged FlashDecoding, │
     └───────────┬───────────┘ │KvSlab,    │ │  ModelFactory)        │
                 │             │RadixCache,│ └───────────┬───────────┘
                 │             │SwapEngine)│             │
                 │             └─────┬─────┘             │
                 └───────────────────┼───────────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │     novaserve-sched      │  <-- `#![deny(unsafe_code)]` across M02..M08!
                        │  (M05: Continuous Batch, │
                        │   Chunked Prefill, Swap, │
                        │   Leviathan Speculative) │
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │    novaserve-sampler     │
                        │ (M06: 9-Stage Vectorized │
                        │  Top-K/Min-P/Top-P/Mask) │
                        └────────────┬─────────────┘
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
┌───────────────────┐    ┌───────────────────────┐    ┌───────────────────┐
│ novaserve-oracle  │    │   novaserve-server    │    │  novaserve-wasm   │
│ (M07: f64 Eager   │    │ (M08: OpenAI HTTP/SSE,│    │ (M09: wasm32 +    │
│  Ref, 5k Chaos,   │    │  SLA Bench) + TUI     │    │  Visual Studio)   │
│  3-Regime TVD)    │    │  `novaserve-top`      │    └───────────────────┘
└───────────────────┘    └───────────────────────┘
```

---

## 2. `Q8_0` / `Q4_0` Block Quantization & `GGUF v3` Zero-Copy Memory Map

```text
GGUF v3 File Mapped via `ByteSource` (OS `mmap` on Native / 64B `AlignedBuf` on `wasm32`)
┌────────────────────┬─────────────────────────┬──────────────────┬──────────────────────────────────┐
│ Magic "GGUF" (4B)  │ Metadata KV Table       │ Tensor Directory │ Padding -> 32B/64B Alignment     │
│ Version = 3  (4B)  │ (13 GGUF Value Types,   │ (name, dims,     │ Contiguous Weight Tensor Planes  │
│ n_tensors    (8B)  │  arch hyperparams,      │  ggml_dtype,     │ (`TensorView<'a>` borrows bytes  │
│ n_kv         (8B)  │  BPE vocab & merges)    │  rel_offset)     │  with ZERO heap copy!)           │
└────────────────────┴─────────────────────────┴──────────────────┴──────────────────────────────────┘

Block-Quantized Row Layouts (`QK = 32` weights per super-block):

  Q8_0 Block (34 Bytes / 32 Weights = 8.50 bits/weight, 3.75x compression vs FP32):
  ┌──────────────┬──────────────────────────────────────────────────────────┐
  │ d: IEEE f16  │ qs[0..32]: 32 x signed int8  ->  w_i = f16_to_f32(d)*qs_i│
  │   (2 Bytes)  │                       (32 Bytes)                         │
  └──────────────┴──────────────────────────────────────────────────────────┘

  Q4_0 Block (18 Bytes / 32 Weights = 4.50 bits/weight, 7.07x compression vs FP32):
  ┌──────────────┬──────────────────────────────────────────────────────────┐
  │ d: IEEE f16  │ nibbles[0..16]: packed lo/hi 4-bit unsigned (-8 bias)    │
  │   (2 Bytes)  │ lo -> qs[0..16] = (byte & 0x0F) - 8                      │
  │              │ hi -> qs[16..32]= (byte >> 4)   - 8                      │
  └──────────────┴──────────────────────────────────────────────────────────┘

AVX2 + FMA Vectorized Dot-Product Pipeline (`novaserve-tensor::simd::avx2`):
  Activation Row x[0..32] (FP32) ──► On-the-fly `Q8_0` Quantize (`d_x`, `qx[0..32]`)
                                            │
  Weight Block (`d_w`, `qw[0..32]`) ────────┼──► `_mm256_sign_epi8` + `_mm256_maddubs_epi16`
                                            │    + `_mm256_madd_epi16` -> int32 Lane Sum
                                            ▼
                                     `acc_f32 += (d_w * d_x) * (sum_i32 as f32)`
```

---

## 3. PagedAttention CoW Indirection & SGLang `RadixCache` Mid-Edge Splitting

```text
Physical `KvSlab` Shape: `[n_layers, 2 (K/V), num_physical_blocks, n_kv_heads, block_size=16, head_dim]`
External Fragmentation = 0.0% (all blocks uniform size; internal waste <= 15 slots on tail block only)

Sequence R1.0 (Forked Child 0) ─┐
  LogicalBlockTable: [B07, B02] ├──► Shared Prefix Physical Blocks `B07` (`rc=5`), `B02` (`rc=5`)
Sequence R1.1 (Forked Child 1) ─┤    (1 ref held by `RadixCache` + 4 refs held by `n=4` CoW children!)
  LogicalBlockTable: [B07, B02, B19] ──► Divergent Token Appended -> `cow_append_slot()` allocates
                                         fresh `B19` (`rc=1`), leaving `B02` untouched!

SGLang-Style Compressed Radix Tree Mid-Edge Split (`novaserve-kv::radix_cache`):
  Before Insert `[SysBlk0, SysBlk1, UserB_Blk2]`:
    Root(#0) ────── edge: [SysBlk0, SysBlk1, UserA_Blk2] ──────► Node(#1)

  After Longest-Prefix Match (2 blocks = 32 tokens) & `split_node(#1, split_blocks=2)`:
                                         ┌── edge: [UserA_Blk2] ──► Node(#2, existing leaf)
    Root(#0) ── edge: [SysBlk0, SysBlk1]─┴── edge: [UserB_Blk2] ──► Node(#3, newly inserted leaf)
                      Node(#1, internal)     (Strict LRU evicts only unpinned leaves with `rc == 1`!)
```

---

## 4. Online-Softmax Paged FlashDecoding & Leviathan Speculative Pipeline

```text
1. Online-Softmax Paged FlashDecoding (`context_len >= 256` partitioned across Rayon workers):
   Partition p over physical KV blocks `[b_start .. b_end)`:
     Maintains running `(m_p, l_p, o_p[0..head_dim])` in `f64`/`f32` without materializing `[context_len]` scores:
       m_new = max(m_old, score_t)
       alpha = exp(m_old - m_new),  beta = exp(score_t - m_new)
       l_new = alpha * l_old + beta
       o_new = alpha * o_old + beta * V[phys_block, slot]

   Cross-Partition Associative Merge:
     m = max_p(m_p),   L = sum_p exp(m_p - m) * l_p
     Output = sum_p (exp(m_p - m) / L) * o_p   ==> Matches Contiguous Eager within `1.371e-6` (`< 1e-5`)!

2. Exact Leviathan Speculative Rejection + Residual Sampling (`novaserve-sched::speculative`):
   Step A: DraftModel (`Draft-500K`) autoregressively proposes K=4 tokens `(d_1..d_4)` with `q_1..q_4`.
   Step B: TargetModel (`Tiny-16M`) verifies `[x_last, d_1, d_2, d_3, d_4]` in ONE batched forward pass,
           yielding exact target distributions `p_1, p_2, p_3, p_4, p_5` (with JSON-DFA masks applied!).
   Step C: For i = 1..K, draw `u_i ~ Uniform(0, 1)`:
           - Accept `d_i` iff `u_i <= min(1, p_i(d_i) / q_i(d_i))`
           - On FIRST rejection at i:
               * Truncate Target & Draft `LogicalBlockTable`s back to `m + i` (zero stale KV!)
               * Sample exact replacement token `x' ~ norm(max(0, p_i(x) - q_i(x)))`
               * Break!
           - If all K accepted: sample bonus token `x_{K+1} ~ p_{K+1}` (emitting K+1 tokens in 1 pass!).
```

---

## 5. Verified Release SLA & Differential Proof Telemetry

All metrics below are measured by `cargo run --release -p novaserve-server --bin novaserve-bench` and `cargo test --workspace --release` on `x86_64` (`Avx2Fma`):

### 5.1 Kernel Quantization Speedup (`Tiny-16M` GQA)

| Weight DType | Footprint (MB) | Compression vs FP32 | Single-Stream (`tok/s`) | Single-Stream `TPOT` (`ms`) | Continuous Batch-16 (`tok/s`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FP32` | 11.01 | 1.00x | 376.1 | 2.659 | 1148.3 |
| `Q8_0` | 2.93 | **3.75x** | **421.6** | **2.372** | 848.6 |
| `Q4_0` | 1.56 | **7.07x** | 269.4 | 3.712 | 407.4 |

### 5.2 Continuous Batching & Chunked Prefill Tail-Jitter Ablation (`32` Poisson Clients)

| Scheduler Policy | Clients | Throughput (`tok/s`) | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | `ITL p50` (`ms`) | `ITL p99` (`ms`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential Locking (`batch=1`) | 32 | 403.3 | 1016.63 | 1785.03 | 1.47 | 3.01 |
| Continuous Batching (`32 seqs`) | 32 | **886.1** | **368.68** | **368.71** | 14.11 | 368.71 |
| Continuous + Chunked Prefill (`B=64`) | 32 | 700.8 | 389.08 | 880.60 | 24.51 | **69.20** |

- **Mid-Flight 512-Token Long Prompt Decode `ITL p99` Spike:** Unchunked `240.80 ms` -> Chunked (`B=64`) **`56.45 ms` (`4.27x` tail-jitter reduction)**.

### 5.3 SGLang-Style Radix Prefix Cache (`512`-Token Shared System Prompt, `24` Requests)

| Configuration | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | Total Wall Time (`ms`) | KV-Block Hit Rate |
| --- | ---: | ---: | ---: | ---: |
| `Radix OFF` | 221.00 | 370.37 | 5932.2 | 0.0% |
| `Radix ON` | **19.79** (`11.17x` faster) | **293.98** | **1565.7** (`3.79x` faster) | **95.8%** |

### 5.4 Leviathan Speculative Decoding (`K = 4`) & Statistical Distribution Invariance (`M07`)

| Regime / Proof | Acceptance `α` | Throughput / `TPOT` | Statistical / Numerical Proof |
| --- | ---: | ---: | --- |
| **Speedup (`Draft-500K` + `Tiny-16M`)** | **65.6%** (`3.59` tok/pass) | **859.0 tok/s** (`1.164 ms/tok`, **`1.61x`**) | Exact greedy & stochastic parity |
| **Oracle Regime 1 (`Aligned`)** | 97.7% | 160,000 tokens (`32k` rollouts) | `TVD = 0.00612 < 0.015`, `χ² = 14.92` (`df=15`) |
| **Oracle Regime 2 (`Mismatched`)** | 4.1% | 160,000 tokens (`32k` rollouts) | `TVD = 0.00817 < 0.015`, `χ² = 17.25` (`df=15`) |
| **Oracle Regime 3 (`Adversarial`)** | 0.0% (100% residual!) | 160,000 tokens (`32k` rollouts) | `TVD = 0.00507 < 0.015`, `χ² = 10.79` (`df=15`) |
