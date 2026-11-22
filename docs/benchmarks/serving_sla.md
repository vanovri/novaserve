# NovaServe Production Serving SLA & Systems Ablation Report

- **Active SIMD Kernel Backend:** `Avx2Fma`
- **Benchmark Model:** `Tiny-16M GQA (dim=256, layers=4, n_heads=8, n_kv_heads=2, vocab=512)`

## 1. Kernel Quantization Speedup (`FP32` vs `Q8_0` vs `Q4_0`)

| Weight DType | Footprint (MB) | Compression vs FP32 | Single-Stream (`tok/s`) | Single-Stream `TPOT` (`ms`) | Continuous Batch-16 (`tok/s`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FP32` | 11.01 | 1.00x | 187.9 | 5.323 | 771.1 |
| `Q8_0` | 2.93 | 3.75x | 588.5 | 1.699 | 1232.3 |
| `Q4_0` | 1.56 | 7.07x | 557.6 | 1.794 | 1189.9 |

## 2. Continuous Batching vs Sequential Locking (`32` Poisson Clients) & Chunked Prefill

| Scheduler Policy | Clients | Throughput (`tok/s`) | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | `ITL p50` (`ms`) | `ITL p99` (`ms`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential Locking (batch=1) | 32 | 410.5 | 919.85 | 1756.35 | 1.43 | 3.44 |
| Continuous Batching (32 seqs) | 32 | 928.6 | 375.98 | 376.00 | 13.91 | 376.00 |
| Continuous + Chunked Prefill (B=64) | 32 | 617.2 | 495.46 | 992.03 | 28.20 | 45.68 |

- **Mid-Flight 512-Token Prompt Decode Tail-Jitter (`ITL p99`):** Unchunked `274.55 ms` -> Chunked (`B=64`) `33.58 ms` (**`8.18x` tail-spike reduction**)

## 3. SGLang-Style Radix Prefix Cache Ablation (`512`-Token Shared System Prompt)

| Configuration | Requests | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | Total Wall Time (`ms`) | KV-Block Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Radix OFF` | 24 | 221.83 | 292.94 | 5803.0 | 0.0% |
| `Radix ON` | 24 | 11.13 | 249.87 | 897.8 | **95.8%** |

- **Radix `TTFT p50` Speedup:** **`19.94x` faster** (`6.46x` end-to-end wall speedup)

## 4. Exact Leviathan Speculative Decoding Speedup (`K = 4`)

| Decoding Engine | Lookahead `K` | Acceptance `α` | Mean Tokens / Target Pass | Throughput (`tok/s`) | `TPOT` (`ms/tok`) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure Target (`Tiny-16M`) | 0 | — | 1.00 | 781.9 | 1.279 | 1.00x |
| Speculative (`Draft-500K` + `Tiny-16M`) | 4 | **65.6%** | **3.59** | **1113.8** | **0.898** | **1.42x** |
