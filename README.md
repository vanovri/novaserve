# NovaServe (`novaserve`)

**High-Throughput LLM Inference Serving Engine Engineered from First Principles in Rust** — uniting **vLLM PagedAttention & Continuous Batching**, **SGLang RadixAttention Prefix Caching & Compressed Byte-DFA Structured Outputs**, **Sarathi-Serve Chunked Prefill**, **Exact Leviathan Speculative Decoding**, and **SIMD Block-Quantized (`Q8_0`/`Q4_0`) CPU + WebAssembly Execution** with **zero external ML framework dependencies**.

---

## Key Capabilities & Verified Guarantees

- **Zero External ML Frameworks:** Tensor arenas, `IEEE-754` `f16`/`bf16`, `Q8_0`/`Q4_0` block quantization, `AVX2+FMA` & `NEON` tiled `GEMM`/`GEMV`, `RMSNorm`, `SwiGLU`, `RoPE`, Online-Softmax Paged FlashDecoding, `GGUF v3` & `Safetensors` zero-copy parsers, Byte-Level BPE, and Draft-2020-12 JSON-Schema -> Minimized Byte-DFA compilation are all implemented 100% from scratch.
- **Miri-Verified Memory Safety (`0` UB) & Strict `#![deny(unsafe_code)]`:** `unsafe` is quarantined exclusively to `novaserve-tensor::{alloc, simd}` and verified under `cargo +nightly miri test -p novaserve-tensor`. Every higher-level crate (`frontend`, `kv`, `model`, `sched`, `sampler`, `oracle`, `server`, `tui`) enforces `#![deny(unsafe_code)]`.
- **Exact PagedAttention Invariance (`1.371e-6 < 1e-5`):** Scrambled non-contiguous physical KV blocks + Radix CoW shared prefixes + irregular Chunked Prefill + 8-sequence Continuous Batching + `Swap-Out -> Swap-In` preemption match single-sequence `f64` eager execution across **5,000 randomized chaos scenarios** (`20,960` sequences, `6,377` preemptions, `0` leaked blocks).
- **Exact Distribution-Preserving Speculative Decoding (`TVD < 0.015`):** Proven empirically across `480,000` sampled tokens (`Aligned`, `Mismatched`, and `0%`-acceptance `Adversarial` draft regimes), delivering **`1.61x` wall-clock `TPOT` speedup** (`65.6%` acceptance at `K=4`).
- **100% Valid Constrained JSON Decoding (`5,000 / 5,000`):** Minimized Byte-DFA vocabulary bitmasks (`0.494 µs/step` at `V=4096`) + `forced_tokens()` deterministic jump-ahead guarantee 100% schema-valid JSON even under adversarial uniform random logits.
- **OpenAI-Compatible Async HTTP/SSE Gateway, `novaserve-top` TUI & In-Browser WASM Studio:** Supports `/v1/chat/completions`, `/v1/completions`, `stream: true` SSE with instantaneous TCP-disconnect KV reclamation, live 4-quadrant `ratatui` cockpit, and zero-backend `wasm32` Visual Studio (`studio/`).

---

## Workspace Map (`M01` – `M09`)

| Module | Path | Description |
| --- | --- | --- |
| **M01** | `crates/novaserve-tensor` | 64B Aligned Arena, `F16`/`BF16`/`Q8_0`/`Q4_0`, `AVX2+FMA`/`NEON`/Scalar `GEMM`/`GEMV` (`70.46 GFLOPS`), `GGUF v3` & `Safetensors` |
| **M02** | `crates/novaserve-frontend` | Byte-Level BPE Tokenizer, `TokenStreamDecoder`, Regex & Draft-2020-12 JSON-Schema -> Hopcroft-Minimized Byte-DFA |
| **M03** | `crates/novaserve-kv` | PagedAttention `KvSlab` (`FP32`/`Q8_0`), CoW `BlockAllocator`, SGLang `RadixCache` (mid-edge split + LRU), `SwapEngine` |
| **M04** | `crates/novaserve-model` | Pre-Norm GQA Transformer (`Llama`/`Qwen2`/`Mistral`), Online-Softmax Paged FlashDecoding, `ModelFactory` |
| **M05** | `crates/novaserve-sched` | Iteration-Level Continuous Batching, Chunked Prefill, `Swap`/`Recompute` Preemption, Leviathan Speculative Decoding |
| **M06** | `crates/novaserve-sampler` | Allocation-Free 9-Stage Vectorized Logits Pipeline (`Top-K`, `Min-P`, `Top-P`, Penalties, Grammar Bitmask, `Logprobs`) |
| **M07** | `crates/novaserve-oracle` | Pure-`f64` `EagerReferenceEngine`, 5,000-Scenario Batch/Preemption Chaos Fuzzer, 3-Regime Speculative `TVD`/`χ²` Verifier |
| **M08** | `crates/novaserve-server` & `tui` | OpenAI HTTP/SSE Server (`novaserve-server`), Release SLA Suite (`novaserve-bench`), 4-Quadrant `ratatui` `novaserve-top` |
| **M09** | `crates/novaserve-wasm` & `studio/` | `wasm32-unknown-unknown` Engine Bridge + Interactive TypeScript/Vite PagedAttention, Radix & Speculative Visual Studio |

---

## Quickstart

### 1. Run Full End-to-End Verification (`make verify-all`)
```bash
# Runs rustfmt, workspace clippy (-D warnings), all unit/integration tests,
# 5,000-scenario chaos oracle + 3-regime speculative TVD proof, 5,000 JSON-schema fuzz,
# release SLA ablation benchmarks, wasm32 engine compilation, Vite studio build & Vitest E2E!
make verify-all

# Run Nightly Miri undefined-behavior verification on novaserve-tensor:
make miri
```

### 2. Launch the OpenAI-Compatible HTTP & SSE Server
```bash
# Boot with built-in deterministic Tiny-16M Q8_0 + Draft-500K Speculative (K=4) weights:
cargo run --release -p novaserve-server --bin novaserve-server -- --port 8080 --dtype q8_0 --lookahead-k 4

# Or load an off-the-shelf GGUF v3 / Safetensors model:
cargo run --release -p novaserve-server --bin novaserve-server -- --model path/to/model.gguf --port 8080
```

Query with standard OpenAI `curl` (including `stream: true` SSE and `json_schema` structured outputs):
```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "novaserve-tiny-16m-q8_0",
    "messages": [{"role": "user", "content": "Invoke documentation search tool"}],
    "stream": true,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "tool_call",
        "schema": {
          "type": "object",
          "properties": {
            "tool": {"type": "string", "enum": ["search_docs", "query_kv"]},
            "limit": {"type": "integer"}
          },
          "required": ["tool", "limit"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### 3. Launch `novaserve-top` Terminal Cockpit
```bash
# Self-contained embedded demo mode with live multi-turn, CoW, JSON-DFA & Speculative traffic:
cargo run --release -p novaserve-tui --bin novaserve-top -- --demo

# Or attach to a running novaserve-server instance:
cargo run --release -p novaserve-tui --bin novaserve-top -- --url http://127.0.0.1:8080
```

### 4. Launch Interactive WebAssembly Visual Studio (`studio/`)
```bash
make wasm-studio
cd studio && npm run dev
```

---

## Verified Benchmark Highlights (`Tiny-16M` on `Avx2Fma`)

| Subsystem Ablation | Baseline | NovaServe Optimized | Verified Gain |
| --- | --- | --- | --- |
| **`Q8_0` SIMD Quantization** | `FP32` (`11.01 MB`, `376.1 tok/s`) | `Q8_0` (`2.93 MB`, `421.6 tok/s`) | **`3.75x` smaller, `1.12x` faster single-stream** |
| **`Q4_0` SIMD Quantization** | `FP32` (`11.01 MB`) | `Q4_0` (`1.56 MB`) | **`7.07x` memory reduction** |
| **Continuous Batching (`32` clients)** | Sequential Locking (`403.3 tok/s`, `TTFT p50 = 1017 ms`) | Continuous Batching (`886.1 tok/s`, `TTFT p50 = 369 ms`) | **`2.20x` throughput, `2.76x` lower `TTFT p50`** |
| **Sarathi Chunked Prefill (`B=64`)** | Unchunked 512-tok arrival (`ITL p99 = 240.80 ms`) | Chunked Prefill (`ITL p99 = 56.45 ms`) | **`4.27x` decode tail-jitter reduction** |
| **SGLang Radix Prefix Cache** | `Radix OFF` (`TTFT p50 = 221.00 ms`, `5932 ms` wall) | `Radix ON` (`TTFT p50 = 19.79 ms`, **`95.8%` hit rate**) | **`11.17x` faster `TTFT p50`, `3.79x` wall speedup** |
| **Leviathan Speculative (`K=4`)** | Pure Target (`534.7 tok/s`, `1.870 ms/tok` `TPOT`) | `Draft-500K` + `Tiny-16M` (`α = 65.6%`, `3.59` tok/pass) | **`859.0 tok/s` (`1.61x` `TPOT` speedup)** |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`docs/benchmarks/serving_sla.md`](docs/benchmarks/serving_sla.md) for full memory-layout diagrams and SLA tables.
