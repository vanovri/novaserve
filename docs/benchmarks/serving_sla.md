# NovaServe Production Serving SLA & Systems Ablation Report

- **Active SIMD Kernel Backend:** `Avx2Fma`
- **Benchmark Model:** `Tiny-16M GQA (dim=256, layers=4, n_heads=8, n_kv_heads=2, vocab=512)`

## 1. Kernel Quantization Speedup (`FP32` vs `Q8_0` vs `Q4_0`)

| Weight DType | Footprint (MB) | Compression vs FP32 | Single-Stream (`tok/s`) | Single-Stream `TPOT` (`ms`) | Continuous Batch-16 (`tok/s`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FP32` | 11.01 | 1.00x | 376.1 | 2.659 | 1148.3 |
| `Q8_0` | 2.93 | 3.75x | 421.6 | 2.372 | 848.6 |
| `Q4_0` | 1.56 | 7.07x | 269.4 | 3.712 | 407.4 |

## 2. Continuous Batching vs Sequential Locking (`32` Poisson Clients) & Chunked Prefill

| Scheduler Policy | Clients | Throughput (`tok/s`) | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | `ITL p50` (`ms`) | `ITL p99` (`ms`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential Locking (batch=1) | 32 | 403.3 | 1016.63 | 1785.03 | 1.47 | 3.01 |
| Continuous Batching (32 seqs) | 32 | 886.1 | 368.68 | 368.71 | 14.11 | 368.71 |
| Continuous + Chunked Prefill (B=64) | 32 | 700.8 | 389.08 | 880.60 | 24.51 | 69.20 |

- **Mid-Flight 512-Token Prompt Decode Tail-Jitter (`ITL p99`):** Unchunked `240.80 ms` -> Chunked (`B=64`) `56.45 ms` (**`4.27x` tail-spike reduction**)

## 3. SGLang-Style Radix Prefix Cache Ablation (`512`-Token Shared System Prompt)

| Configuration | Requests | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | Total Wall Time (`ms`) | KV-Block Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Radix OFF` | 24 | 221.00 | 370.37 | 5932.2 | 0.0% |
| `Radix ON` | 24 | 19.79 | 293.98 | 1565.7 | **95.8%** |

- **Radix `TTFT p50` Speedup:** **`11.17x` faster** (`3.79x` end-to-end wall speedup)

## 4. Exact Leviathan Speculative Decoding Speedup (`K = 4`)

| Decoding Engine | Lookahead `K` | Acceptance `α` | Mean Tokens / Target Pass | Throughput (`tok/s`) | `TPOT` (`ms/tok`) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure Target (`Tiny-16M`) | 0 | — | 1.00 | 534.7 | 1.870 | 1.00x |
| Speculative (`Draft-500K` + `Tiny-16M`) | 4 | **65.6%** | **3.59** | **859.0** | **1.164** | **1.61x** |
