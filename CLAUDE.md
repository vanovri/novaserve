# CLAUDE.md — Constitutional Engineering Laws for NovaServe

## 1. Mission & Architectural Identity
You are building **NovaServe**, a production-grade LLM Inference Serving Engine written from first principles in Rust (Edition 2021/2024), combining the core systems breakthroughs of **vLLM** (PagedAttention & Continuous Batching), **SGLang** (RadixAttention Prefix Caching & Compressed FSM Structured Outputs), **Sarathi-Serve** (Chunked Prefill), and **GGML/llama.cpp** (SIMD Block-Quantized `Q4_0`/`Q8_0` CPU Inference + Zero-Copy `GGUF v3` / `Safetensors` weights).

Every subsystem must obey rigorous ML-systems invariants:
- **Zero External ML Frameworks:** Workspace crates MUST NOT depend on `candle`, `tch`, `ort`, `burn`, `dfdx`, `tract`, `ndarray`, `nalgebra`, `tokenizers`, `safetensors`, `gguf`, `outlines`, or `llguidance`. You MUST implement tensor layouts, SIMD matrix multiplication, `RoPE`, `RMSNorm`, Online-Softmax Paged Attention, `GGUF v3` / `Safetensors` binary parsers, Byte-Level BPE, JSON-Schema DFA compilation, and Paged KV-Cache management from scratch.
- **Exact Paged invariance Law:** Computing forward logits for any sequence $S$ via **PagedAttention** (with arbitrary physical block permutation, Copy-on-Write shared prefix blocks, `Chunked Prefill` split across arbitrary chunk sizes $C \in [1, |S|]$, and `Swap-Out` -> `Swap-In` preemption cycles) MUST produce logits matching a contiguous single-sequence eager forward pass within `max_abs_diff <= 1e-5` (`FP32`) / `1e-3` (`Q8_0`).
- **Exact Speculative Decoding Law (Leviathan et al., ICML 2023):** Speculative decoding with draft distribution $q(x)$ and target distribution $p(x)$ MUST accept draft token $x_i$ with probability $\min\left(1, \frac{p(x_i)}{q(x_i)}\right)$, and upon the first rejection at position $i$, sample a replacement token from the residual distribution $p'(x) = \text{norm}(\max(0, p(x) - q(x)))$ (or sample bonus token $x_{K+1} \sim p_{K+1}$ if all $K$ draft tokens are accepted). It is **STRICTLY FORBIDDEN** to use greedy-only approximation when `temperature > 0`—your speculative engine must preserve the exact mathematical output distribution of the target model even if the draft model emits adversarial uniform noise!

## 2. Permitted Dependencies & `unsafe` Quarantine
1. **Permitted Low-Level Dependencies Only:**
   - Core Engine (`novaserve-tensor`, `novaserve-frontend`, `novaserve-kv`, `novaserve-model`, `novaserve-sched`, `novaserve-sampler`, `novaserve-oracle`): `thiserror`, `bitflags`, `half` (optional for IEEE `f16`/`bf16` bit conversion, or implement pure-Rust IEEE-754 bit-shifts!), `memmap2`, `rayon` (thread-pool for parallel head/tile compute), `parking_lot`, `regex-syntax` (AST parser ONLY if desired, or hand-written JSON-schema/regex Thompson NFA -> DFA compiler), `serde`, `serde_json`.
   - Server & TUI (`novaserve-server`, `novaserve-tui`): `tokio`, `axum` / `hyper`, `clap`, `ratatui`, `crossterm`.
   - WASM (`novaserve-wasm`): `wasm-bindgen`, `js-sys`, `serde-wasm-bindgen`.
2. **ZERO Incomplete Code:** `todo!()`, `unimplemented!()`, `#[ignore]`, dummy random logits, and hardcoded mock responses are **STRICTLY FORBIDDEN**. Every commit must pass `cargo clippy --workspace --all-targets -- -D warnings`.
3. **`unsafe` Quarantine Policy:**
   - `#![deny(unsafe_code)]` MUST be active in `novaserve-frontend`, `novaserve-kv` (logical block/radix managers), `novaserve-model`, `novaserve-sched`, `novaserve-sampler`, `novaserve-oracle`, `novaserve-server`, and `novaserve-tui`.
   - Controlled `unsafe` is permitted ONLY inside `novaserve-tensor::alloc` (64-byte aligned arena / `mmap`), `novaserve-tensor::simd` (`x86_64` `AVX2`+`FMA` and `aarch64` `NEON` intrinsics guarded by runtime feature detection + 100% safe scalar fallback), and `novaserve-kv::arena` (pre-allocated physical KV slab slice indexing verified by bounds checks). Every `unsafe` block MUST carry a `// SAFETY:` proof comment and pass `cargo +nightly miri test -p novaserve-tensor` (using scalar fallback under `cfg(miri)` for arch intrinsics).

## 3. Self-Contained Model Weights Law (Zero External Download Dependency)
Tests, benchmarks, TUI demos, and WASM Studio MUST NEVER require downloading multi-gigabyte HuggingFace weights from the internet to verify correctness!
- Build a deterministic **Synthetic & Micro-Trained Weight Factory (`ModelFactory`)** capable of generating coherent structured test models (`LlamaConfig`: e.g., `Tiny-16M` `dim=256, layers=4, n_heads=8, n_kv_heads=2` GQA, and `Micro-2M` `dim=64, layers=2, n_heads=4, n_kv_heads=2`, plus matched `Draft-500K` models trained/distilled in-memory on synthetic structured JSON/arithmetic/story corpora or initialized with deterministic orthogonal/Xavier weights) AND exporting/loading them cleanly in real **`.gguf` (v3)** and **`.safetensors`** binary formats!
- Simultaneously ensure 100% compatibility with real off-the-shelf `Llama-3.2-1B` / `SmolLM2-135M` / `Qwen2.5-0.5B` `.gguf` and `.safetensors` files when a user passes `--model <path>`!

## 4. Autonomous Progression Protocol (`PROGRESS.md`)
Maintain `PROGRESS.md` tracking `M01` through `M09`. Never advance from `M(k)` to `M(k+1)` until:
1. All unit, Miri, numerical-equivalence, statistical-distribution, and memory-leak gates for `M01..M(k)` pass with `0` errors and `0` warnings.
2. Telemetry tables in `PROGRESS.md` and `docs/benchmarks/` reflect real `--release` benchmark executions.
3. Changes are committed atomically (`feat(mXX): ...`).
