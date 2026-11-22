# Module M09 Design Specification: `novaserve-wasm` & Interactive Visual ML-Systems Studio (`studio/`)

## 1. Architectural Goal
Compile the entire `NovaServe` inference engine (`novaserve-tensor` + `novaserve-frontend` + `novaserve-kv` + `novaserve-model` + `novaserve-sched` + `novaserve-sampler`) to `wasm32-unknown-unknown` with zero external servers or downloads, exposing step-by-step Continuous Batching telemetry to a high-density browser workbench (`studio/`).

## 2. WebAssembly Execution & Memory Model (`crates/novaserve-wasm`)
1. **Zero-Thread / Zero-OS-Mmap Portability:**
   - Under `#[cfg(target_arch = "wasm32")]`, `novaserve-tensor` routes `ByteSource` to 64-byte aligned in-memory buffers (`AlignedBuf`) and executes deterministic scalar/auto-vectorized `FP32` / `Q8_0` / `Q4_0` kernels without `rayon` or `memmap2`.
   - Built-in deterministic `ModelFactory` (`Micro-2M` Target + subspace-distilled `Draft-500K` Draft + 512-token Byte-Level BPE `VocabularyBuilder`) initializes in `< 15 ms` inside browser linear memory.
2. **Dual Binding Surface (`#[wasm_bindgen]` + Raw Linear-Memory JSON ABI):**
   - Exports `#[wasm_bindgen] pub struct NovaServeWasmEngine`:
     - `new_with_demo_models(num_kv_blocks: u32, block_size: u32)`
     - `submit_request(prompt: &str, max_tokens: u32, temp: f32, n_parallel: u32, json_schema_opt: Option<String>, use_speculative: bool) -> u32`
     - `step_iteration() -> JsValue`
   - Simultaneously exports safe linear-memory JSON FFI entrypoints (`ns_wasm_engine_new`, `ns_wasm_configure`, `ns_wasm_alloc_input`, `ns_wasm_submit_json`, `ns_wasm_step_json`, `ns_wasm_snapshot_json`, `ns_wasm_output_ptr`, `ns_wasm_output_len`, `ns_wasm_engine_free`) backed by a lightweight `__wbg` import shim in `studio/src/wasm_bridge.ts`, allowing direct `WebAssembly.instantiate` execution in both Vite browsers and Node `v24` Vitest headless suites with zero external CLI tool dependencies.

## 3. Per-Iteration Telemetry Frame Contract (`WasmStepTelemetry`)
Every call to `step_iteration()` executes **one exact `ServingEngine::step()`** and serializes:
- **Physical KV-Slab Matrix (`kv_slab`):**
  - `block_id: u32`
  - `status: "free" | "unique" | "cow_shared" | "radix_pinned" | "radix_shared"`
  - `ref_count: u32`
  - `owner_seq_ids: Vec<String>` (e.g. `["R1.0", "R1.1", "R1.2", "R1.3"]` on `n=4` CoW blocks)
  - `filled_slots: usize` (`0..=block_size`, computed exactly from owner `LogicalBlockTable::num_tokens()`, or `block_size` for Radix-cached full blocks)
  - `is_radix_cached: bool`
- **Radix Prefix Tree (`radix_tree`):**
  - `node_id`, `parent_id`, `token_preview` (decoded readable snippet), `token_ids`, `total_edge_tokens`, `block_ids`, `ref_count`, `lru_age` (`max_tick - node.last_access_tick`).
- **Speculative Decoding Step Trace (`speculative_traces`):**
  - Per-draft position `{ token_id, token_text, draft_prob, target_prob, ratio_p_over_q, accepted }` plus `{ residual_or_bonus_token_id, residual_or_bonus_text, is_bonus }`.
- **JSON-Schema Grammar X-Ray (`grammar_xrays`):**
  - `req_id`, `child_idx`, `schema_json`, `dfa_state`, `total_dfa_states`, `is_accepting`, `allowed_count`, `vocab_size`, `chosen_token`, `forced_jump_tokens`, `top_allowed` (top-10 legal tokens + raw Transformer logits), and `top_masked` (top-10 highest-logit syntactically illegal tokens crossed out to `-∞` before softmax).

## 4. Studio Four Preset Verification Scenarios (`studio/`)
1. **Preset 1 — Shared System-Prompt Radix Hit & CoW `n=4` Forking:**
   - Warms a 48-token system prompt into `RadixCache`, then launches an `n=4` parallel completion sharing the cached prefix blocks (`ref_count >= 5`, `radix_hit_rate_pct > 70%`).
2. **Preset 2 — Chunked Prefill + Memory Squeeze Swap-Out / Swap-In Preemption:**
   - Configures tight `num_kv_blocks = 16`, `max_num_batched_tokens = 12`, and launches competing multi-block requests forcing multi-step `Chunked Prefill` and lossless `Swap-Out -> Swap-In` preemption cycles with `0` leaked blocks on completion.
3. **Preset 3 — Speculative Decoding (`K=4`) Live Rejection Sampling Inspector:**
   - Runs correlated `Draft-500K` + `Micro-2M` Target at `temperature = 0.7`, emitting live green accepted draft tokens ($u \le p/q$), red rejected draft tokens, and gold residual/bonus tokens.
4. **Preset 4 — JSON-Schema Constrained Decoding DFA Masking Live Demo:**
   - Compiles a strict Draft-2020-12 tool-call schema into a minimized Byte-DFA, streaming live `-∞` logit masking X-rays and guaranteeing 100% `JSON.parse` + schema-valid output.
