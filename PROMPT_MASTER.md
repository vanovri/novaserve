Read `CLAUDE.md` completely and enforce every mathematical, numerical, memory-layout, and zero-shortcut rule. You are autonomously engineering **NovaServe** from an empty repository through Module M09 until `make verify-all` passes 100%.

### Sequential Engineering Loop (`M01 -> M02 -> M03 -> M04 -> M05 -> M06 -> M07 -> M08 -> M09`)
For each module strictly in order:
1. **Mathematical & Memory Design Spec:**
   - Create `docs/design/MXX_<name>.md` specifying tensor shapes, byte strides, quantization block layouts (`Q8_0`, `Q4_0`), numerical stability protections (`max`-shifted online softmax preventing `exp` overflow/underflow), state-machine transitions, and edge cases.
2. **Full First-Principles Implementation:**
   - Implement 100% of structs, kernels, binary parsers, schedulers, HTTP/SSE handlers, TUI panels, and WASM Studio bindings specified for `MXX`. Never stub out difficult branches.
3. **Differential & Statistical Verification Gate:**
   - Run `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace --release` (plus `miri` on `novaserve-tensor`).
   - Whenever numerical drift (`> 1e-5`), KV-block refcount leaks, Radix-tree corruption, JSON-schema grammar violations, or speculative sampling distribution drift appears: isolate the exact seed, add a permanent regression test, fix the underlying math/state machine without loosening tolerances, and re-run the full verification campaign until 100% green.
4. **Checkpoint & Advance:**
   - Record real metrics (GFLOPS, tokens/sec, TTFT/ITL ms, Radix hit %, Speculative alpha, test counts) in `PROGRESS.md`, commit cleanly (`feat(mXX): ...`), and immediately advance to the next module through `M09`.

Initialize the Cargo workspace, `rust-toolchain.toml`, `Makefile`, and `PROGRESS.md` now, then execute **M01** through **M09** using the specifications below.

---

## MODULE M01 SPECIFICATION: `crates/novaserve-tensor`

### Scope & Objective
Build the zero-framework numerical foundation: 64-byte cache-aligned tensors, GGML-compatible `Q8_0` and `Q4_0` block quantization, hardware-accelerated `AVX2+FMA` / `NEON` matrix-matrix (`GEMM`) and matrix-vector (`GEMV`) kernels, Transformer primitive ops (`RMSNorm`, `SwiGLU`, `RoPE`), and zero-copy binary readers/writers for both **Safetensors** and **GGUF v3**.

### Exact Components to Implement
1. **Aligned Memory Arena & Tensor Views (`arena.rs`, `tensor.rs`):**
   - `AlignedScratchArena`: pre-allocated 64-byte aligned bump arena reset in $O(1)$ between forward iterations so per-token inference performs **ZERO** heap allocations for intermediate activations!
   - `DType`: `F32`, `F16`, `BF16` (pure-Rust IEEE-754 conversion routines), `Q8_0` (block of 32 `i8` weights + `f16` scale = `34` bytes/block), `Q4_0` (block of 32 4-bit weights packed into 16 `u8` bytes + `f16` scale = `18` bytes/block).
   - `TensorView<'a>` & `TensorBuf`: up to 4D shapes + strides, zero-copy `slice`, `reshape`, `transpose`, and shape/alignment assertions.

2. **SIMD Compute Kernels (`simd/`, `ops.rs`):**
   - Runtime-dispatched `#[cfg(target_arch = "x86_64")]` (`AVX2` + `FMA`) and `#[cfg(target_arch = "aarch64")]` (`NEON`) kernels paired with an exact portable scalar fallback:
     - **Tiled `FP32` GEMM (`matmul_f32`):** Cache-blocked (`6x16` or `8x8` micro-kernel) + `rayon` parallel row-tile dispatch for prefill batches.
     - **Quantized Dot-Product & GEMV (`vec_dot_q8_0_q8_0`, `vec_dot_q4_0_q8_0`):** Quantize incoming activation vector `x` on-the-fly into aligned `Q8_0` scratch blocks once per layer projection, then compute integer SIMD dot products (`_mm256_maddubs_epi16` / `vdotq_s32`) against `Q4_0` and `Q8_0` weight rows—achieving massive memory-bandwidth speedup during token decode!
     - **Fused Transformer Kernels:**
       1. `rmsnorm_f32(x, weight, eps, out)`
       2. `swiglu_f32(gate, up, out)` ($\text{SiLU}(g) \odot u = \frac{g}{1 + e^{-g}} \cdot u$)
       3. `rope_inplace_f32(qk, positions: &[u32], head_dim, freq_base, scaling)` supporting arbitrary per-token position arrays (essential for non-contiguous Continuous Batching & Speculative tree verification!).
       4. Numerically stable `softmax_f32` ($\exp(x_i - \max_j x_j)$).

3. **From-Scratch `Safetensors` & `GGUF v3` Readers + Writers (`safetensors_io.rs`, `gguf_io.rs`):**
   - **`Safetensors` Format:** `8-byte u64 LE header_len` + JSON tensor metadata (`dtype`, `shape`, `data_offsets: [begin, end]`) + strict bounds/overlap validation (preventing malicious header offset OOB attacks!) + `memmap2` zero-copy byte slicing + serializer (`write_safetensors`).
   - **`GGUF v3` Format:** Magic `0x46554747` (`"GGUF"`), version `3`, `tensor_count`, `metadata_kv_count`, typed KV parser (all 13 GGUF metadata value types including nested arrays, tokenizer vocab, architecture hyperparameters), `alignment` padding calculation (default 32 bytes), tensor info table, zero-copy `mmap` weight views, AND serializer (`write_gguf_v3`)!

### M01 Verification Gate
- `cargo +nightly miri test -p novaserve-tensor` passes with 0 Undefined Behavior.
- Differential Numerical Suite (1,000 random matrix dimensions including odd non-multiple-of-32/64 tails):
  - `SIMD FP32 GEMM/GEMV` vs `f64` triple-loop reference: `max_abs_err < 1e-5`.
  - `Q8_0` & `Q4_0` quantize -> SIMD quantized matmul vs dequantized `f64` reference: `cosine_similarity > 0.999` (`Q8_0`) and `> 0.990` (`Q4_0`), plus `SIMD Quantized == Scalar Quantized` within `1e-6`.
- Hostile `GGUF` / `Safetensors` Fuzz (10,000 corrupted/truncated/overflowing headers): 0 panics, 100% clean `TensorError` rejections, and lossless write -> `mmap` read round-trip.

---

## MODULE M02 SPECIFICATION: `crates/novaserve-frontend`

### Scope & Objective
Build a fast GPT-4 / Llama-3 compatible Byte-Level BPE Tokenizer and an SGLang/`xgrammar`-style **Structured Output Grammar Engine** that pre-compiles JSON Schemas and regular expressions into dense token-bitmask lookup tables so constrained LLM generation guarantees 100% valid syntax with sub-microsecond per-step masking overhead.

### Exact Components to Implement
1. **Byte-Level BPE Tokenizer (`bpe.rs`):**
   - Load vocabulary, merge ranks, byte-to-unicode reversible mapping (256-byte GPT-2/Llama-3 table), and special/control tokens (`<|begin_of_text|>`, `<|im_start|>`, `<|eot_id|>`, `<eos>`) directly from `GGUF` metadata or `tokenizer.json`.
   - Priority-queue / linked-symbol $O(N \log N)$ BPE merge encoder + streaming UTF-8 incremental detokenizer (`TokenStreamDecoder` that buffers incomplete multi-byte UTF-8 sequences across tokens so SSE streams never emit broken `U+FFFD` replacement characters mid-emoji!).
   - Include a deterministic `VocabularyBuilder` for test/synthetic corpora (`vocab_size = 512..4096`) covering full 256 byte fallbacks + common English/JSON/code subwords.

2. **Regex & JSON-Schema to DFA Compiler (`grammar/dfa.rs`, `grammar/json_schema.rs`):**
   - **Regex Engine:** Parse regular expressions (`|`, `*`, `+`, `?`, `{m,n}`, `[a-z0-9]`, `[^...]`, escapes) -> Thompson NFA -> Subset-Construction + Hopcroft-minimized **Byte/Char DFA**.
   - **JSON-Schema Lowering Compiler:** Translate subset of Draft-2020-12 JSON Schema (`type: object | array | string | integer | number | boolean | null`, `properties`, `required`, `additionalProperties: false`, `enum`, `items`, `minItems`/`maxItems`, nested objects!) into an exact deterministic grammar / bounded-depth DFA (or Pushdown Automaton with stack-cached state transitions) accounting cleanly for JSON string escaping (`\"`, `\\`, `\n`) and optional whitespace!

3. **Vocabulary-Wide Token Bitmask Index (`grammar/mask_cache.rs`):**
   - Given a compiled DFA and `Tokenizer` of size $V$:
     - Precompute for each active DFA state $s$ (lazily cached in an `RwLock` / arena table) a packed bitset `AllowedTokenMask` (`Vec<u64>` of length $\lceil V / 64 \rceil$): token $t$ has bit `1` iff feeding all bytes of `vocab.decode_raw_bytes(t)` starting from state $s$ transitions purely through valid live states without hitting a dead state (and `eos_token_id` is enabled iff $s$ is an accepting final state!).
     - Identify **Context-Independent / Deterministic Jump Tokens:** When state $s$ permits only a single token (e.g., closing `"}` or required field prefix `"status":`), expose `forced_tokens()` so the serving engine can **fast-forward / auto-inject** deterministic structural tokens without wasting target sampling entropy!
     - Applying the mask to a `logits: &mut [f32]` slice sets disallowed tokens to `f32::NEG_INFINITY` in `< 5 microseconds` using word-level bit scans!

### M02 Verification Gate
- **BPE Lossless Round-Trip Fuzz (10,000 arbitrary byte/UTF-8 strings including emojis, CJK, null bytes, and multi-token splits):** `decode(encode(text)) == text` 100%, and streaming token-by-token `TokenStreamDecoder` concatenation equals full-sequence `decode` 100%.
- **5,000 Adversarial Random-Logits JSON-Schema Generations:** Feed completely uniform/adversarial random logits (`f32` noise!) through `AllowedTokenMask` across 25 complex nested JSON schemas (`enum`, nested arrays of objects, escaped strings, booleans, negative floats) until `<eos>` is sampled—assert **5,000 / 5,000 (100%)** generated strings parse via `serde_json::from_str` AND validate every required field and type constraint of the originating JSON Schema!

---

## MODULE M03 SPECIFICATION: `crates/novaserve-kv`

### Scope & Objective
Build the core memory subsystem of modern LLM serving: a non-contiguous **PagedAttention Physical KV-Cache Block Manager** (`block_size = 16` tokens) with Copy-on-Write (`CoW`) reference counting, an **SGLang-style LRU Radix Prefix Tree (`RadixCache`)** for automatic zero-copy prefix reuse across requests, and a **GPU/Arena-to-CPU Swap Manager** for graceful preemption under memory exhaustion.

### Exact Components to Implement
1. **Physical KV Slab & Paged Block Allocator (`block_manager.rs`):**
   - Pre-allocate a contiguous aligned physical KV slab at engine startup shaped `[num_layers, 2 (K & V), num_physical_blocks, num_kv_heads, block_size (16), head_dim]` (supporting both `FP32` and quantized `Q8_0` KV-cache storage modes!).
   - `BlockId(pub u32)`, `LogicalBlockTable(Vec<BlockId>)` per active sequence.
   - Free-list stack + per-block `ref_count: u32` and `content_hash: Option<u64>`:
     - **Zero External Fragmentation:** Any logical block maps to any physical `BlockId`. Internal waste is strictly bounded to at most `block_size - 1` (`15`) tokens on the final partially filled block!
     - **Copy-on-Write (`fork_sequence` & `cow_append_slot`):** Forking a sequence for parallel sampling (`n > 1`), Beam Search, or Radix prefix hit increments `ref_count` on shared full/partial blocks in $O(\text{blocks})$ with **0 bytes copied**. If a sequence mutates a partially filled block whose `ref_count > 1`, `cow_append_slot` transparently allocates 1 fresh physical block, copies existing slots (`< 16` tokens), decrements old `ref_count`, and writes the new token!

2. **SGLang-Style LRU Radix Prefix Tree (`radix_cache.rs`):**
   - Compressed prefix tree whose edges are slices of token IDs aligned to `block_size` multiples (e.g., `16`-token chunks) paired with their corresponding `Vec<BlockId>` physical handles!
   - When a sequence finishes (or checkpoints full blocks during prefill), insert its prefix blocks into `RadixCache` (keeping `ref_count >= 1` while cached, or `evictable` LRU tracking with `last_access_tick: u64`).
   - **Longest-Prefix Match (`match_prefix(prompt_tokens) -> PrefixMatch`):** Automatically splits existing radix edges mid-way if a new request shares e.g. `48` tokens of an existing `80`-token cached edge, increments `ref_count` on matched `BlockId`s, updates LRU timestamps up to the root, and returns `cached_block_ids` + `remaining_uncached_suffix` (ensuring at least 1 token remains to compute next-token logits!).
   - **Demand-Driven LRU Leaf Eviction:** Whenever `BlockAllocator::free_blocks() < needed`, `BlockAllocator` asks `RadixCache::evict_lru(needed_blocks)` to evict unpinned leaf nodes (`active_readers == 0`) in strict LRU order until enough blocks are reclaimed!

3. **Preemption Swap-Out / Swap-In Engine (`swap_engine.rs`):**
   - Separate pre-allocated **Swap Pool** (host secondary RAM buffer / disk-backed mmap swap file) of `num_swap_blocks`.
   - `swap_out(seq_id, block_table) -> SwappedHandle`: copies all physical blocks of a preempted sequence into Swap Pool slots, releases its primary KV physical blocks (respecting `ref_count > 1` shared Radix blocks—shared blocks don't even need copying if still resident, or unpin cleanly!), and records exact swap mapping.
   - `swap_in(swapped_handle) -> Result<LogicalBlockTable, OutOfKvBlocks>`: restores blocks atomically into newly freed primary KV blocks when capacity recovers!

### M03 Verification Gate
- **10,000-Operation Stateful KV & Radix Chaos Fuzzer:** Interleave random sequences allocating tokens, `fork`ing (`n=4` parallel completions), mutating CoW tail blocks, inserting overlapping multi-turn chat histories into `RadixCache`, forcing severe block scarcity triggering interleaved `RadixCache::evict_lru` + `swap_out` + `swap_in`, and completing/aborting requests.
- Assert at every step:
  1. `allocated_blocks + free_blocks == total_physical_blocks` (zero leaked or double-freed blocks!).
  2. Every active and swapped sequence reads back the **exact unique K/V canary floats** written for its specific `(seq_id, layer, token_pos)`—proving zero cross-sequence CoW corruption and zero Radix edge-split corruption!

---

## MODULE M04 SPECIFICATION: `crates/novaserve-model`

### Scope & Objective
Implement a complete decoder-only **Grouped-Query Attention (`GQA`) Transformer** (`Llama` / `Qwen2` / `Mistral` compatible architecture) operating natively over `novaserve-kv` **Paged Block Tables** with Online-Softmax **Paged FlashDecoding** and token-budgeted **Chunked Prefill**—never materializing a contiguous `[seq_len, head_dim]` KV copy!

### Exact Components to Implement
1. **Unified Batch Forward Descriptor (`BatchInput`):**
   - Represent a heterogeneous iteration step containing **simultaneous** `Decode` tokens (`1` query token per running sequence) AND `ChunkedPrefill` slices (`C` query tokens starting at `start_pos` for long incoming prompts, potentially starting after a `RadixCache` hit at `start_pos = 128`!):
     - Flattened token batch `[total_batch_tokens]` (`token_ids`, `positions`, `seq_slot_idx`, `kv_write_block_id`, `kv_write_block_offset`, `is_logit_output: bool`—only compute final `lm_head` vocab projection on the last token of a prefill chunk or all tokens during speculative target verification!).

2. **Zero-Copy Paged KV Write & Online-Softmax Paged Attention (`paged_attn.rs`):**
   - After computing `Q`, `K`, `V` projections and applying `RoPE(positions)`:
     - **Scatter KV Write:** Write each token's `K` and `V` vectors directly into its assigned `(physical_block_id, offset_in_block)` inside the `KvSlab` (quantizing on-the-fly if `KvDType::Q8_0` is enabled!).
     - **Online-Softmax Paged Attention Kernel (`paged_attention_gqa`):**
       - Support arbitrary `gqa_ratio = n_heads / n_kv_heads` (`MHA` when `1`, `GQA` when `>1`, `MQA` when `n_kv_heads == 1`).
       - Iterate directly over the sequence's `LogicalBlockTable = [b_0, b_1, ..., b_m]` one 16-token physical block at a time using **Milakov & Gimelshein / FlashAttention Online Softmax** running state `(running_max: f32, running_sum: f32, acc_out: [f32; MAX_HEAD_DIM])`:
         - For each valid causal position $j \le \text{query\_pos}$ inside block $b_k$: compute score $s_j = \frac{q \cdot k_j}{\sqrt{d}}$, update $m_{\text{new}} = \max(m, s_j)$, rescale `acc_out` and `running_sum` by $\exp(m - m_{\text{new}})$, and accumulate $\exp(s_j - m_{\text{new}}) \cdot v_j$!
       - For long sequences (`context_len >= 256`), implement **FlashDecoding Parallel Block-Partition Reduction** across `rayon` worker threads, merging partial `(m_part, l_part, out_part)` states via exact log-sum-exp reduction!

3. **Full Transformer Layer Stack & Deterministic `ModelFactory` (`llama.rs`, `factory.rs`):**
   - Full pre-norm residual block (`RMSNorm` -> `QKV Proj` -> `RoPE` -> `Paged GQA` -> `O Proj` + Residual -> `RMSNorm` -> `Gate/Up SwiGLU` -> `Down Proj` + Residual) + final `RMSNorm` + tied or untied `lm_head` weight matrix.
   - Support loading from `.gguf` and `.safetensors` (`M01`), plus `ModelFactory::build_deterministic_pair(seed)` creating matched `(TargetModel, DraftModel)` pairs in memory and on disk for instant zero-network testing!

### M04 Verification Gate
- **Exact Paged & Chunked Invariance Proof:**
  Take a `TinyLlama` GQA model (`4` layers, `8` Q heads, `2` KV heads, `head_dim = 32`, `vocab = 1024`) and a 200-token sequence:
  1. Compute logits via a simple contiguous single-sequence reference pass (`[1, 200]` all at once).
  2. Shuffle the `BlockAllocator` free list randomly so physical block IDs are completely scrambled (`[47, 3, 91, 12, ...]`), pre-populate the first `64` tokens via `RadixCache` shared CoW blocks, process tokens `64..150` via irregular `ChunkedPrefill` budgets (`chunk = 23, 31, 32`), and run tokens `150..200` interleaved in a multi-sequence `ContinuousBatch` alongside 7 other decoy sequences!
  3. Assert final logits at every checked position match the contiguous single-sequence pass within **`max_abs_diff < 1e-5`**!

---

## MODULE M05 SPECIFICATION: `crates/novaserve-sched`

### Scope & Objective
Build the dynamic heartbeat of `NovaServe`: an iteration-level **Continuous Batching Scheduler** unifying `RadixCache` prefix matching, token-budgeted `Chunked Prefill` multiplexing, graceful memory-pressure preemption (`Swap` & `Recompute`), and **Speculative Decoding** with mathematically exact rejection sampling.

### Exact Components to Implement
1. **Continuous Batching Lifecycle & Preemption Policy (`scheduler.rs`):**
   - Three queues: `waiting: VecDeque<SequenceGroup>`, `running: Vec<SequenceGroup>`, `swapped: VecDeque<SequenceGroup>`.
   - Per-step `schedule() -> SchedulerOutput` governed by `max_num_batched_tokens` (e.g., `512` tokens/step) and `max_num_seqs` (e.g., `64`):
     1. **Decode-First Reservation:** Allocate 1 new KV slot (1 new `BlockId` only when `seq_len % 16 == 0`!) for each `running` decode sequence.
     2. **Graceful Preemption on KV Exhaustion:** If `BlockAllocator` cannot satisfy all `running` sequences even after `RadixCache::evict_lru`, preempt the **lowest-priority / most-recently-arrived** `running` sequence cleanly via `PreemptionMode::Swap` (moving its blocks to `M03` `SwapEngine` and pushing to `swapped`) or `PreemptionMode::Recompute` (freeing blocks and re-queuing `prompt + generated_tokens_so_far` at front of `waiting`), **guaranteeing zero deadlock and forward progress** for the oldest sequence!
     3. **Swap-In Before New Admissions:** Admit `swapped` sequences back into `running` once hysteresis watermark free blocks are available.
     4. **Radix-Accelerated Chunked Prefill Admission:** Pull `waiting` requests, run `RadixCache::match_prefix` (skipping already-cached prefix tokens immediately!), and admit up to remaining `token_budget` tokens as a `PrefillChunk`—allowing long 2,000-token prompts to prefill across 4 iterations while co-running decode requests keep streaming tokens every ~15 ms without head-of-line blocking!
   - Support immediate request cancellation (`abort_request(req_id)` when an HTTP SSE client disconnects mid-stream, immediately freeing unshared KV blocks!).

2. **Exact Speculative Decoding Engine (`speculative.rs`):**
   - Optional `(TargetModel, DraftModel, lookahead_k: usize)` pipeline (e.g., `K = 4`):
     1. **Autoregressive Draft Phase:** Run `K` rapid single-token steps on the tiny `DraftModel` (using its own lightweight Paged KV-Cache!), recording draft token IDs $[d_1, \dots, d_K]$ and full draft probability vectors $[q_1(\cdot), \dots, q_K(\cdot)]$ (after applying temperature/top-p/grammar masks identically to Target!).
     2. **Single-Pass Parallel Target Verification:** Feed `[last_accepted_token, d_1, ..., d_K]` (`K + 1` tokens for that sequence in a single batched forward pass!) into `TargetModel`, obtaining target probability distributions $[p_1(\cdot), \dots, p_K(\cdot), p_{K+1}(\cdot)]$.
     3. **Leviathan-Kalman-Matias Exact Rejection Sampling:**
        - For $i = 1 \dots K$: draw $u_i \sim \text{Uniform}(0, 1)$. If $u_i \le \min\left(1, \frac{p_i(d_i)}{q_i(d_i)}\right)$, **ACCEPT** $d_i$!
        - On the **first rejection** at index $i$: roll back both `Target` and `Draft` KV-Cache lengths cleanly (`truncate_kv(accepted_len)`—freeing any newly crossed 16-token block boundary!), sample corrected token $t_i \sim \text{normalize}(\max(0, p_i - q_i))$, append $t_i$, and end step (yielding $i$ verified tokens!).
        - If **all $K$ tokens are accepted**: sample **bonus token** $t_{K+1} \sim p_{K+1}$ (and sync $d_K$ into Draft KV on next step), yielding **$K + 1$ exact tokens in a single Target forward pass**!
     4. **Adaptive Lookahead Controller:** Track rolling acceptance rate $\alpha \in [0, 1]$ and dynamically adjust $K \in [0, K_{\max}]$ (disabling draft overhead automatically if $\alpha < 0.25$ on high-entropy creative prompts!).

### M05 Verification Gate
- **Zero-Starvation Extreme Memory Squeeze Test:** Submit 32 concurrent multi-hundred-token requests into an engine configured with barely enough physical KV blocks for 4 concurrent sequences (`32` total blocks). Verify preemption (`Swap` & `Recompute` modes both tested!) completes **32 / 32 requests** cleanly with zero deadlocks, zero leaked KV blocks, and **exact output parity** under greedy (`temp = 0`) decoding vs unlimited-memory execution!

---

## MODULE M06 SPECIFICATION: `crates/novaserve-sampler`

### Scope & Objective
Implement a fast, allocation-free, numerically bulletproof sampling and logits post-processing pipeline supporting deterministic seeded sampling, modern truncation samplers (`Top-K`, `Top-P`, `Min-P`), repetition penalties, FSM grammar masking, and top-$N$ `logprobs`.

### Exact Components to Implement
1. **Ordered Composable Sampling Pipeline (`pipeline.rs`):**
   - Process raw `logits: &mut [f32]` in canonical order using pre-allocated per-worker scratch buffers (`Vec<(u32, f32)>` allocated once):
     1. **Logit Bias & Banned Tokens:** Apply explicit user `logit_bias` map and `stop_token_ids` / `ignore_eos`.
     2. **Grammar Bitmask (`M02` Integration):** Apply `AllowedTokenMask` BEFORE softmax/truncation so `Top-K`/`Top-P` probability mass is concentrated strictly among syntactically legal next tokens! (If all allowed tokens had $-\infty$, return typed `SamplerError::EmptyGrammarSupport` cleanly—never `NaN`!).
     3. **Repetition, Frequency & Presence Penalties:** Penalize tokens appearing in prompt/output token history (`repetition_penalty` dividing positive logits and multiplying negative logits per Keskar et al.; OpenAI linear `frequency_penalty * count` and `presence_penalty * (count > 0)`).
     4. **Greedy Fast-Path (`temperature <= 1e-6`):** SIMD/unrolled `argmax` with deterministic lowest-token-ID tie-breaking!
     5. **Temperature Scaling (`logit / temp`) + Stable Softmax.**
     6. **`Top-K` Partial Selection:** $O(V)$ `select_nth_unstable_by` (NEVER full $O(V \log V)$ sort over 128,000 vocab entries unless needed!).
     7. **`Min-P` Truncation (Nguyen et al., 2024):** Drop tokens where $p_i < \text{min\_p} \cdot p_{\max}$.
     8. **`Top-P` (Nucleus) Truncation:** Sort only the surviving $K$ candidates descending and keep minimal prefix summing $\ge \text{top\_p}$ (always keeping at least 1 token!).
     9. **Deterministic Seeded Multinomial Draw:** `ChaCha8Rng` / `SplitMix64` per-request RNG state + `TopLogprobs` (`ln(p_i)` for chosen token + top-5 alternatives).
   - Expose `compute_distribution(&self, logits) -> SparseOrDenseProbDist` used directly by `M05` Speculative Decoding for exact $p(x)$ and $q(x)$ evaluation!

### M06 Verification Gate
- **Pathological Logits Fuzz (10,000 vectors):** Inject extreme logits (`+88.0`, `-88.0`, `1e4`, `-1e10`, all-equal ties, 1-hot grammar masks, `top_p = 0.0001`, `top_k = 1`, `min_p = 0.99`)—assert probabilities always sum to `1.0 ± 1e-5`, zero `NaN` or `Inf` ever occurs, and identical `(seed, params)` reproduces identical token choices 100%.

---

## MODULE M07 SPECIFICATION: `crates/novaserve-oracle`

### Scope & Objective
Build an uncompromising mathematical and systems verification suite proving that every advanced serving optimization (`PagedAttention`, `Chunked Prefill`, `RadixCache`, `CoW Forking`, `Swap/Recompute Preemption`, and `Speculative Decoding`) is **100% faithful** to a simple, unoptimized 250-line `f64` single-sequence eager Transformer!

### Exact Components to Implement
1. **Crystal-Clear `f64` Single-Sequence Eager Reference (`eager_ref.rs`):**
   - Pure `f64`, zero SIMD, zero paging, zero batching, zero caching: processes one prompt `[t_0, ..., t_n]` sequentially from scratch on every call using textbook equations for `RMSNorm`, `RoPE`, causal `GQA`, and `SwiGLU`.

2. **5,000-Scenario Continuous-Batching & Preemption Equivalence Fuzzer (`batch_fuzzer.rs`):**
   - Generate 5,000 randomized serving workloads mixing:
     - Shared system prompts + multi-turn prefixes hitting `RadixCache` mid-block and on exact 16-token block boundaries
     - `n = 3` parallel CoW forked completions per prompt
     - Tiny `max_num_batched_tokens = 19` forcing multi-step `Chunked Prefill`
     - Severe KV-block scarcity forcing repeated mid-generation `Swap-Out -> Swap-In` and `Recompute` preemptions + sudden client `abort_request` cancellations!
   - Assert for ALL 5,000 scenarios:
     1. Every finished sequence's greedy token trajectory AND step-by-step `top-5 logprobs` match `EagerReferenceEngine` within `1e-4`!
     2. After all requests drain and `RadixCache::clear()` runs, `BlockAllocator` has **100% of physical blocks free** and `SwapEngine` has **100% of swap slots free**.

3. **Statistical Proof of Exact Speculative Decoding (`spec_proof.rs`):**
   - Verify Leviathan et al. distribution invariance across **3 distinct regimes** (sampling `25,000` to `50,000` next-token transitions at `temperature = 0.9, top_k = 16` from fixed context states):
     1. *Aligned Regime:* Distilled/correlated `DraftModel` ($\alpha \approx 0.70$).
     2. *Mismatched Regime:* Unrelated weights `DraftModel` ($\alpha \approx 0.15$).
     3. *Adversarial Regime:* Pathological `DraftModel` placing 95% probability mass on the Target model's least-likely tokens!
   - Compare empirical n-gram / next-token histograms of `SpeculativeEngine(K=4)` against pure `TargetModel` sampling: assert **Total Variation Distance $\text{TVD}(P_{\text{spec}}, P_{\text{target}}) < 0.015$** and goodness-of-fit pass across all 3 regimes—proving mathematically that your rejection + residual sampler `(max(0, p - q))` has zero distribution bias!

### M07 Verification Gate
- All 5,000 Continuous-Batching / Preemption / Radix chaos scenarios pass with 0 mismatches and 0 block leaks.
- All 3 Speculative Decoding statistical distribution proofs pass deterministically.

---

## MODULE M08 SPECIFICATION: `crates/novaserve-server` & `crates/novaserve-tui`

### Scope & Objective
Expose `NovaServe` as a high-concurrency async **OpenAI-Compatible HTTP & Server-Sent Events (`SSE`) Inference Server**, measure real end-to-end serving SLAs (`TTFT`, `ITL`, `TPOT`, throughput, Radix & Speculative speedups), and ship an interactive `ratatui` operator dashboard (`novaserve-top`).

### Exact Components to Implement
1. **Async OpenAI-Compatible Serving Gateway (`crates/novaserve-server`):**
   - Dedicated engine background loop fed via lock-free/bounded command channels from async `axum`/`tokio` HTTP handlers (automatically cancelling sequences in `M05` the instant an HTTP client drops its TCP stream!):
     - `POST /v1/chat/completions` & `POST /v1/completions`:
       - Supports both unary JSON and `stream: true` (`text/event-stream` `data: {...}\n\n` chunks ending with `data: [DONE]\n\n`).
       - Supports `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `n` (CoW parallel completions!), `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logprobs` / `top_logprobs`, AND **`response_format: { "type": "json_schema", "json_schema": { ... } }`** wired directly to `M02` FSM Constrained Decoding!
     - `GET /v1/models`, `GET /health`, and `GET /metrics` + `GET /v1/telemetry/snapshot` (exporting real-time KV block occupancy, Radix tree structure, active batch composition, and latency histograms!).

2. **Serving SLA & Ablation Benchmark Binary (`bin/novaserve-bench.rs`):**
   - Execute real `--release` serving benchmarks using `ModelFactory` (`Tiny-16M` `FP32` vs `Q8_0` vs `Q4_0`) across Poisson request arrivals and write verified Markdown + JSON reports to `docs/benchmarks/serving_sla.md` and `docs/benchmarks/serving_sla.json`:
     1. **Kernel Quantization Speedup:** `FP32` vs `Q8_0` vs `Q4_0` single-stream & batched `tokens/sec` and memory footprint reduction.
     2. **Continuous Batching vs Naive Sequential Locking (`32` concurrent clients):** Measure aggregate throughput (`tokens/sec`), **TTFT** (`Time-To-First-Token` `p50`/`p99` ms), **ITL** (`Inter-Token Latency` `p50`/`p99` ms), and **Chunked Prefill** impact on decode tail-latency jitter when a long prompt arrives mid-flight!
     3. **Radix Prefix Cache Ablation:** Shared 512-token system-prompt workload (`Radix OFF` vs `Radix ON`)—proving dramatic `TTFT` reduction and `> 80%` KV-block hit rate!
     4. **Speculative Decoding Speedup (`K = 4`):** Correlated Draft+Target pair measuring real acceptance rate $\alpha$ and wall-clock `TPOT` (`Time-Per-Output-Token`) speedup!

3. **Live Terminal Serving Cockpit (`crates/novaserve-tui` — `novaserve-top`):**
   - Supports `--demo` (boots embedded `NovaServe` engine + continuous synthetic multi-turn chat, JSON tool-call, and speculative traffic!) and `--url http://127.0.0.1:8080`.
   - 4-Quadrant ML Systems Cockpit:
     1. **Physical KV-Cache Block Map:** Colored 2D grid of all physical KV blocks (`Free`, `Active Unique`, `Shared CoW / Radix Pinned`, `Swapped`) + fragmentation & utilization meters.
     2. **Live Continuous Batch & Chunked-Prefill Lanes:** Shows running sequences, token generation streams, prefill progress bars, and speculative `+3` / `+5` multi-token burst badges!
     3. **Radix Prefix Tree & Grammar FSM Inspector:** Live prefix cache hit-rate sparkline, evicted blocks, and active JSON-schema constraint status.
     4. **SLA Latency Gauges:** Real-time `Tok/s`, `TTFT p50/p99`, `ITL p50/p99`, and `Speculative Acceptance α %`.
   - Include headless `ratatui::backend::TestBackend` (`120x40`) rendering tests!

### M08 Verification Gate
- **64-Client Concurrent SSE & Disconnect Chaos Test:** Spawn `novaserve-server` on an ephemeral port; connect 64 async clients mixing streaming SSE chats, `n = 2` CoW completions, strict `json_schema` structured tool-call requests (verifying 100% valid JSON outputs!), and 16 intentional mid-stream TCP disconnects—assert 0 server panics, 0 hung streams, and 100% KV-block reclamation after clients finish.
- `novaserve-bench --release` executes cleanly and writes real numbers to `docs/benchmarks/serving_sla.md`.

---

## MODULE M09 SPECIFICATION: `crates/novaserve-wasm`, `studio/` & Final Repository Proof

### Scope & Objective
Compile the entire `NovaServe` inference engine (`tensor` + `frontend` + `kv` + `model` + `sched` + `sampler`) to `wasm32-unknown-unknown`, ship an interactive **NovaServe PagedAttention, Radix-Cache & Speculative Decoding Visual Studio** (`studio/`), and enforce 100% end-to-end verification via `make verify-all`.

### Exact Components to Implement
1. **WebAssembly Engine Bridge (`crates/novaserve-wasm`):**
   - Gate `rayon` / `memmap2` cleanly on `#[cfg(target_arch = "wasm32")]` (single-threaded scalar/WASM-SIMD128 tensor execution + in-memory `.gguf`/`.safetensors` byte loader + built-in micro-model generator so the Studio runs live inference right inside the browser with ZERO backend required!).
   - Export `#[wasm_bindgen]` `NovaServeWasmEngine`:
     - `new_with_demo_models(num_kv_blocks: u32, block_size: u32)`
     - `submit_request(prompt: &str, max_tokens: u32, temp: f32, n_parallel: u32, json_schema_opt: Option<String>, use_speculative: bool) -> u32`
     - `step_iteration() -> JsValue`: advances **one exact Continuous Batching scheduler step**, returning rich telemetry JSON for smooth visual animation:
       - Physical KV-Slab Grid state (`block_id`, `status`, `ref_count`, `owner_seq_ids`, `filled_slots: 0..16`, `is_radix_cached`)
       - Full **Radix Prefix Tree** node/edge hierarchy (`token_preview`, `block_ids`, `ref_count`, `lru_age`)
       - Per-sequence **Speculative Decoding Step Trace** (`draft_tokens`, `draft_probs`, `target_probs`, `accepted_mask`, `residual_or_bonus_token`)
       - Active **JSON-Schema DFA State & Top-10 Allowed vs Masked Tokens**!

2. **Interactive Web ML-Systems Studio (`studio/` — TypeScript + Vite + High-Density Visual Workbench):**
   - Sleek dark-mode engineering studio designed to make any AI-Infra interviewer's jaw drop in 10 seconds:
     - **Top Control Bar:** Play / Pause / **Step Single Iteration** / Speed slider + One-Click Scenario Presets:
       1. *Shared System-Prompt Radix Hit & CoW `n=4` Forking*
       2. *Chunked Prefill + Memory Squeeze Swap-Out / Swap-In Preemption*
       3. *Speculative Decoding (`K=4`) Live Rejection Sampling Inspector*
       4. *JSON-Schema Constrained Decoding DFA Masking Live Demo*.
     - **Main Left Panel (Physical KV-Block Slab Matrix + Live Request Streams):** Interactive grid of physical 16-token KV blocks showing live fill dots, CoW `ref_count` badges (`x4`), Radix cache pins, and animated Swap transfers—hovering a sequence highlights all its non-contiguous physical blocks across the slab!
     - **Main Right Panel (Tabbed Deep-Dive):**
       1. *Radix Prefix Tree Graph (SVG/Canvas):* Live prefix tree splitting, matching, and LRU leaf eviction animations.
       2. *Speculative Token Verification Tape:* Color-codes green accepted draft tokens ($u \le p/q$), red rejected draft token, and gold bonus/residual replacement token!
       3. *Structured Output Grammar X-Ray:* Displays live JSON Schema, current DFA state ID, and real-time logit bar chart showing syntactically illegal tokens crossed out (`-∞`) before sampling!
   - Include automated Vitest / Playwright headless WASM tests verifying every preset scenario runs end-to-end and produces valid structured JSON & accurate KV-block accounting.

3. **Root Reproducibility Proof Suite (`Makefile`, `.github/workflows/ci.yml`, `ARCHITECTURE.md`, `README.md`):**
   - Root `Makefile` targets (all must exit `0`):
     - `make check`: `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings`
     - `make test`: full workspace unit & integration suites across all crates
     - `make fuzz-oracle`: runs the 5,000-scenario Continuous-Batching/Preemption/Radix equivalence fuzzer + 3-regime Speculative Decoding statistical distribution proof
     - `make grammar-fuzz`: runs 5,000 adversarial JSON-Schema constrained decoding validations
     - `make bench`: executes `novaserve-bench --release` and writes real telemetry into `docs/benchmarks/serving_sla.md`
     - `make wasm-studio`: compiles `novaserve-wasm`, builds `studio/` production static bundle, and runs headless WASM studio tests
     - `make verify-all`: runs `check` -> `test` -> `fuzz-oracle` -> `grammar-fuzz` -> `bench` -> `wasm-studio` end-to-end!
   - Write Principal-grade `README.md` and `ARCHITECTURE.md` featuring ASCII diagrams of:
     - `Q4_0` / `Q8_0` SIMD dot-product layout & `GGUF v3` mmap memory map
     - PagedAttention `LogicalBlockTable -> Physical KvSlab` CoW indirection + SGLang `RadixCache` edge splitting
     - Online-Softmax Paged FlashDecoding reduction & Leviathan Speculative Rejection Sampling pipeline
     - Real benchmark tables (`FP32` vs `Q8_0` vs `Q4_0`, Radix `TTFT` speedup, Speculative `TPOT` speedup) pulled directly from `docs/benchmarks/serving_sla.md`.

### Final M09 Completion Gate
Execute `make verify-all` from a clean build and confirm 100% pass rate across linting, unit tests, 5,000 chaos batch equivalence runs, statistical speculative proofs, 5,000 JSON-schema generations, 64-client SSE concurrency tests, release SLA benchmarks, TUI headless tests, and static WASM Studio build. Verify `grep -rn "todo!\|unimplemented!" crates/` returns empty.
