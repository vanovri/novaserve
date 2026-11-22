//! `novaserve-wasm`: Full In-Browser `wasm32-unknown-unknown` Inference Serving Engine Bridge
//! exporting `NovaServeWasmEngine` (`#[wasm_bindgen]`) and zero-copy linear-memory JSON FFI
//! for the interactive PagedAttention, RadixCache, Speculative & JSON-Schema Studio (`studio/`).

use novaserve_frontend::{
    compile_json_schema_str, BpeTokenizer, GrammarMaskCache, TokenStreamDecoder, VocabularyBuilder,
};
use novaserve_model::{LlamaConfig, ModelFactory};
use novaserve_sampler::SamplingPipeline;
use novaserve_sched::{
    FinishReason, PreemptionMode, RequestParams, SamplerStrategy, SamplingConfig, SchedulerConfig,
    SequenceStatus, ServingEngine, StepSummary,
};
use novaserve_tensor::DType;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};
use wasm_bindgen::prelude::*;

#[derive(Clone, Debug, Deserialize)]
pub struct EngineConfigPayload {
    pub num_kv_blocks: usize,
    pub block_size: usize,
    pub max_num_batched_tokens: usize,
    pub speculative_lookahead: usize,
    pub enable_radix_cache: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SubmitRequestPayload {
    pub prompt: String,
    pub max_tokens: usize,
    pub temp: f32,
    pub n_parallel: usize,
    pub json_schema_opt: Option<String>,
    pub use_speculative: bool,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmKvBlock {
    pub block_id: u32,
    pub status: String,
    pub ref_count: u32,
    pub owner_seq_ids: Vec<String>,
    pub filled_slots: usize,
    pub is_radix_cached: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmRadixNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub token_preview: String,
    pub token_ids: Vec<u32>,
    pub total_edge_tokens: usize,
    pub block_ids: Vec<u32>,
    pub ref_count: u32,
    pub lru_age: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmSpecDraftItem {
    pub token_id: u32,
    pub token_text: String,
    pub draft_prob: f32,
    pub target_prob: f32,
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmSpecTrace {
    pub req_id: u64,
    pub child_idx: usize,
    pub lookahead_k: usize,
    pub draft_items: Vec<WasmSpecDraftItem>,
    pub residual_or_bonus_token_id: u32,
    pub residual_or_bonus_text: String,
    pub is_bonus: bool,
    pub accepted_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmTokenLogitBar {
    pub token_id: u32,
    pub token_text: String,
    pub raw_logit: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmGrammarXray {
    pub req_id: u64,
    pub child_idx: usize,
    pub schema_json: String,
    pub dfa_state: u32,
    pub total_dfa_states: usize,
    pub is_accepting: bool,
    pub allowed_count: u32,
    pub vocab_size: usize,
    pub chosen_token_id: u32,
    pub chosen_token_text: String,
    pub forced_jump_tokens: Vec<String>,
    pub top_allowed: Vec<WasmTokenLogitBar>,
    pub top_masked: Vec<WasmTokenLogitBar>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmSequenceLane {
    pub seq_key: String,
    pub req_id: u64,
    pub child_idx: usize,
    pub status: String,
    pub prompt_len: usize,
    pub kv_tokens: usize,
    pub generated_count: usize,
    pub max_new_tokens: usize,
    pub decoded_text: String,
    pub block_ids: Vec<u32>,
    pub swapped_blocks: usize,
    pub cached_prefix_tokens: usize,
    pub preemptions: u32,
    pub use_speculative: bool,
    pub has_grammar: bool,
    pub dfa_state: Option<u32>,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmCompletedChoice {
    pub child_idx: usize,
    pub decoded_text: String,
    pub token_count: usize,
    pub finish_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmCompletedRequest {
    pub req_id: u64,
    pub prompt_len: usize,
    pub cached_prefix_tokens: usize,
    pub preemptions: u32,
    pub choices: Vec<WasmCompletedChoice>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasmStepTelemetry {
    pub step_index: u64,
    pub has_unfinished_work: bool,
    pub block_size: usize,
    pub total_kv_blocks: usize,
    pub free_kv_blocks: usize,
    pub radix_cached_blocks: usize,
    pub used_swap_blocks: usize,
    pub total_swap_blocks: usize,
    pub scheduled_prefill_tokens: usize,
    pub scheduled_decode_tokens: usize,
    pub preempted_this_step: usize,
    pub swapped_in_this_step: usize,
    pub waiting_groups: usize,
    pub running_groups: usize,
    pub swapped_groups: usize,
    pub radix_hit_rate_pct: f64,
    pub radix_evicted_blocks: u64,
    pub rolling_spec_alpha_pct: f64,
    pub active_spec_k: usize,
    pub kv_slab: Vec<WasmKvBlock>,
    pub radix_tree: Vec<WasmRadixNode>,
    pub sequences: Vec<WasmSequenceLane>,
    pub speculative_traces: Vec<WasmSpecTrace>,
    pub grammar_xrays: Vec<WasmGrammarXray>,
    pub completed_this_step: Vec<WasmCompletedRequest>,
}

struct EngineCore {
    engine: ServingEngine,
    tokenizer: Arc<BpeTokenizer>,
    schema_cache: HashMap<String, Arc<GrammarMaskCache>>,
    req_schemas: HashMap<u64, String>,
    seed_counter: u64,
}

impl EngineCore {
    fn new_configured(cfg: EngineConfigPayload) -> Result<Self, String> {
        let vocab_size = 512usize;
        let tokenizer =
            Arc::new(VocabularyBuilder::build_synthetic(vocab_size).map_err(|e| e.to_string())?);
        let target_cfg = LlamaConfig::micro_2m(vocab_size, DType::Q8_0);
        let draft_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);
        let (target_model, draft_model) =
            ModelFactory::build_correlated_pair(target_cfg, draft_cfg, 2026_0909)
                .map_err(|e| e.to_string())?;

        let block_sz = cfg.block_size.max(4);
        let num_blks = cfg.num_kv_blocks.max(8);
        let sched_cfg = SchedulerConfig {
            max_num_batched_tokens: cfg.max_num_batched_tokens.max(4),
            max_num_seqs: 16,
            num_physical_blocks: num_blks,
            num_swap_blocks: (num_blks * 4).max(64),
            block_size: block_sz,
            kv_dtype: novaserve_kv::KvDType::F32,
            preemption_mode: PreemptionMode::Swap,
            enable_radix_cache: cfg.enable_radix_cache,
            speculative_lookahead: cfg.speculative_lookahead,
            adaptive_speculative: false,
        };

        let sampler: Arc<dyn SamplerStrategy> = Arc::new(SamplingPipeline::new(vocab_size));
        let engine = ServingEngine::with_sampler(
            target_model,
            Some(draft_model),
            Some(Arc::clone(&tokenizer)),
            sched_cfg,
            sampler,
        )
        .map_err(|e| e.to_string())?;

        Ok(Self {
            engine,
            tokenizer,
            schema_cache: HashMap::new(),
            req_schemas: HashMap::new(),
            seed_counter: 1000,
        })
    }

    fn submit(&mut self, payload: SubmitRequestPayload) -> Result<u64, String> {
        let mut prompt_tokens = self.tokenizer.encode(&payload.prompt);
        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token_id().unwrap_or(0));
        }

        let grammar = if let Some(schema_str) = payload.json_schema_opt.as_ref() {
            let trimmed = schema_str.trim();
            if trimmed.is_empty() {
                None
            } else if let Some(existing) = self.schema_cache.get(trimmed) {
                Some(Arc::clone(existing))
            } else {
                let dfa = compile_json_schema_str(trimmed).map_err(|e| e.to_string())?;
                let cache = Arc::new(GrammarMaskCache::new(Arc::new(dfa), &self.tokenizer));
                self.schema_cache
                    .insert(trimmed.to_string(), Arc::clone(&cache));
                Some(cache)
            }
        } else {
            None
        };

        self.seed_counter += 1;
        let seed = payload.seed.unwrap_or(self.seed_counter);
        let has_schema = grammar.is_some();

        let params = RequestParams {
            prompt_tokens,
            max_new_tokens: payload.max_tokens.clamp(1, 512),
            n_parallel: payload.n_parallel.clamp(1, 8),
            sampling: SamplingConfig {
                temperature: payload.temp.max(0.0),
                top_k: if payload.temp > 0.0 { Some(24) } else { None },
                top_p: 0.95,
                min_p: 0.0,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                seed,
                stop_token_ids: Vec::new(),
                ignore_eos: !has_schema,
                logit_bias: Vec::new(),
                top_logprobs: 0,
            },
            grammar,
            use_speculative: payload.use_speculative,
        };

        let req_id = self
            .engine
            .submit_request(params)
            .map_err(|e| e.to_string())?;
        if let Some(s) = payload.json_schema_opt {
            if !s.trim().is_empty() {
                self.req_schemas.insert(req_id, s);
            }
        }
        Ok(req_id)
    }

    fn step_and_capture(&mut self) -> Result<WasmStepTelemetry, String> {
        let summary = if self.engine.has_unfinished_work() {
            self.engine.step().map_err(|e| e.to_string())?
        } else {
            StepSummary {
                step_index: self.engine.step_counter(),
                rolling_spec_alpha: self.engine.lookahead_controller().rolling_alpha(),
                active_spec_k: self.engine.lookahead_controller().current_k(),
                ..StepSummary::default()
            }
        };
        Ok(self.build_telemetry(Some(&summary)))
    }

    fn decode_token_display(&self, tid: u32) -> String {
        if tid == self.tokenizer.eos_token_id() {
            return "<eos>".into();
        }
        String::from_utf8_lossy(self.tokenizer.decode_raw_bytes(tid)).into_owned()
    }

    fn decode_clean(&self, tokens: &[u32]) -> String {
        let mut dec = TokenStreamDecoder::new(true);
        let mut out = String::new();
        for &t in tokens {
            out.push_str(&dec.push_token(&self.tokenizer, t));
        }
        out.push_str(&dec.flush());
        out
    }

    fn build_telemetry(&mut self, step_opt: Option<&StepSummary>) -> WasmStepTelemetry {
        let block_size = self.engine.config().block_size;
        let active_seqs = self.engine.inspect_active_sequences();
        let raw_blocks = self.engine.snapshot_kv_blocks();

        // Map `block_id -> (owners, max_filled_slots)` from active sequence tables
        let mut block_owners: HashMap<u32, Vec<String>> = HashMap::new();
        let mut block_filled: HashMap<u32, usize> = HashMap::new();

        let mut sequences = Vec::with_capacity(active_seqs.len());
        for s in active_seqs {
            let seq_key = format!("R{}.{}", s.req_id, s.child_idx);
            let num_blks = s.block_ids.len();
            for (bi, &bid) in s.block_ids.iter().enumerate() {
                block_owners.entry(bid).or_default().push(seq_key.clone());
                let slots_in_blk = if bi + 1 < num_blks {
                    block_size
                } else {
                    let rem = s.kv_tokens % block_size;
                    if rem == 0 && s.kv_tokens > 0 {
                        block_size
                    } else {
                        rem
                    }
                };
                let entry = block_filled.entry(bid).or_insert(0);
                *entry = (*entry).max(slots_in_blk);
            }

            let status_str = match s.status {
                SequenceStatus::Waiting => "waiting",
                SequenceStatus::Prefilling => "prefill",
                SequenceStatus::Decoding => "decode",
                SequenceStatus::Swapped => "swapped",
            };

            let decoded_text = self.decode_clean(&s.generated_tokens);
            sequences.push(WasmSequenceLane {
                seq_key,
                req_id: s.req_id,
                child_idx: s.child_idx,
                status: status_str.into(),
                prompt_len: s.prompt_len,
                kv_tokens: s.kv_tokens,
                generated_count: s.generated_tokens.len(),
                max_new_tokens: s.max_new_tokens,
                decoded_text,
                block_ids: s.block_ids,
                swapped_blocks: s.swapped_blocks,
                cached_prefix_tokens: s.cached_prefix_tokens,
                preemptions: s.preemptions,
                use_speculative: s.use_speculative,
                has_grammar: s.has_grammar,
                dfa_state: s.dfa_state,
                finish_reason: None,
            });
        }

        let kv_slab: Vec<WasmKvBlock> = raw_blocks
            .into_iter()
            .map(|b| {
                let owners = block_owners.remove(&b.block_id).unwrap_or_default();
                let filled = if b.ref_count == 0 {
                    0
                } else if b.is_radix_cached {
                    block_size
                } else {
                    block_filled.get(&b.block_id).copied().unwrap_or(block_size)
                };
                let status = if b.ref_count == 0 {
                    "free"
                } else if b.is_radix_cached && b.ref_count > 1 {
                    "radix_shared"
                } else if b.is_radix_cached {
                    "radix_pinned"
                } else if b.ref_count > 1 {
                    "cow_shared"
                } else {
                    "unique"
                };
                WasmKvBlock {
                    block_id: b.block_id,
                    status: status.into(),
                    ref_count: b.ref_count,
                    owner_seq_ids: owners,
                    filled_slots: filled,
                    is_radix_cached: b.is_radix_cached,
                }
            })
            .collect();

        let raw_radix = self.engine.snapshot_radix_tree();
        let max_tick = raw_radix
            .iter()
            .map(|n| n.last_access_tick)
            .max()
            .unwrap_or(0);
        let radix_tree: Vec<WasmRadixNode> = raw_radix
            .into_iter()
            .map(|n| {
                let preview_str = self.decode_clean(&n.token_preview).replace('\n', "↵");
                WasmRadixNode {
                    node_id: n.node_id,
                    parent_id: n.parent_id,
                    token_preview: preview_str,
                    token_ids: n.token_preview,
                    total_edge_tokens: n.total_edge_tokens,
                    block_ids: n.block_ids,
                    ref_count: n.min_ref_count,
                    lru_age: max_tick.saturating_sub(n.last_access_tick),
                }
            })
            .collect();

        let mut speculative_traces = Vec::new();
        let mut grammar_xrays = Vec::new();
        let mut completed_this_step = Vec::new();

        if let Some(st) = step_opt {
            for tr in &st.speculative_traces {
                let draft_items = (0..tr.lookahead_k)
                    .map(|i| WasmSpecDraftItem {
                        token_id: tr.draft_tokens[i],
                        token_text: self.decode_token_display(tr.draft_tokens[i]),
                        draft_prob: tr.draft_probs[i],
                        target_prob: tr.target_probs[i],
                        accepted: tr.accepted_mask[i],
                    })
                    .collect();
                speculative_traces.push(WasmSpecTrace {
                    req_id: tr.req_id,
                    child_idx: tr.child_idx,
                    lookahead_k: tr.lookahead_k,
                    draft_items,
                    residual_or_bonus_token_id: tr.bonus_or_residual_token,
                    residual_or_bonus_text: self.decode_token_display(tr.bonus_or_residual_token),
                    is_bonus: tr.accepted_count == tr.lookahead_k,
                    accepted_count: tr.accepted_count,
                });
            }

            for gx in &st.grammar_xrays {
                let schema_json = self
                    .req_schemas
                    .get(&gx.req_id)
                    .cloned()
                    .unwrap_or_default();
                grammar_xrays.push(WasmGrammarXray {
                    req_id: gx.req_id,
                    child_idx: gx.child_idx,
                    schema_json,
                    dfa_state: gx.dfa_state,
                    total_dfa_states: gx.total_dfa_states,
                    is_accepting: gx.is_accepting,
                    allowed_count: gx.allowed_count,
                    vocab_size: gx.vocab_size,
                    chosen_token_id: gx.chosen_token_id,
                    chosen_token_text: self.decode_token_display(gx.chosen_token_id),
                    forced_jump_tokens: gx
                        .forced_jump_tokens
                        .iter()
                        .map(|&t| self.decode_token_display(t))
                        .collect(),
                    top_allowed: gx
                        .top_allowed
                        .iter()
                        .map(|c| WasmTokenLogitBar {
                            token_id: c.token_id,
                            token_text: self.decode_token_display(c.token_id),
                            raw_logit: c.raw_logit,
                        })
                        .collect(),
                    top_masked: gx
                        .top_masked
                        .iter()
                        .map(|c| WasmTokenLogitBar {
                            token_id: c.token_id,
                            token_text: self.decode_token_display(c.token_id),
                            raw_logit: c.raw_logit,
                        })
                        .collect(),
                });
            }

            for comp in &st.completed {
                let _ = self.engine.take_completed(comp.req_id);
                let choices = comp
                    .outputs
                    .iter()
                    .map(|o| WasmCompletedChoice {
                        child_idx: o.child_idx,
                        decoded_text: self.decode_clean(&o.token_ids),
                        token_count: o.token_ids.len(),
                        finish_reason: match o.finish_reason {
                            FinishReason::Stop => "stop".into(),
                            FinishReason::Length => "length".into(),
                            FinishReason::Aborted => "abort".into(),
                        },
                    })
                    .collect();
                completed_this_step.push(WasmCompletedRequest {
                    req_id: comp.req_id,
                    prompt_len: comp.prompt_len,
                    cached_prefix_tokens: comp.cached_prefix_tokens,
                    preemptions: comp.preemptions,
                    choices,
                });
            }
        }

        let (waiting_groups, running_groups, swapped_groups) = self.engine.queue_depths();

        WasmStepTelemetry {
            step_index: self.engine.step_counter(),
            has_unfinished_work: self.engine.has_unfinished_work(),
            block_size,
            total_kv_blocks: self.engine.allocator().total_blocks(),
            free_kv_blocks: self.engine.allocator().free_blocks(),
            radix_cached_blocks: self.engine.radix_cache().cached_blocks_count(),
            used_swap_blocks: self.engine.swap_engine().total_swap_blocks()
                - self.engine.swap_engine().free_swap_blocks(),
            total_swap_blocks: self.engine.swap_engine().total_swap_blocks(),
            scheduled_prefill_tokens: step_opt.map_or(0, |s| s.scheduled_prefill_tokens),
            scheduled_decode_tokens: step_opt.map_or(0, |s| s.scheduled_decode_tokens),
            preempted_this_step: step_opt.map_or(0, |s| s.preempted_this_step),
            swapped_in_this_step: step_opt.map_or(0, |s| s.swapped_in_this_step),
            waiting_groups,
            running_groups,
            swapped_groups,
            radix_hit_rate_pct: self.engine.radix_cache().hit_rate() * 100.0,
            radix_evicted_blocks: self.engine.radix_cache().total_evicted_blocks(),
            rolling_spec_alpha_pct: (self.engine.lookahead_controller().rolling_alpha() as f64)
                * 100.0,
            active_spec_k: self.engine.lookahead_controller().current_k(),
            kv_slab,
            radix_tree,
            sequences,
            speculative_traces,
            grammar_xrays,
            completed_this_step,
        }
    }
}

/// High-level `#[wasm_bindgen]` WebAssembly Engine interface (`NovaServeWasmEngine`).
#[wasm_bindgen]
pub struct NovaServeWasmEngine {
    core: EngineCore,
}

#[wasm_bindgen]
impl NovaServeWasmEngine {
    #[wasm_bindgen]
    pub fn new_with_demo_models(
        num_kv_blocks: u32,
        block_size: u32,
    ) -> Result<NovaServeWasmEngine, JsValue> {
        let core = EngineCore::new_configured(EngineConfigPayload {
            num_kv_blocks: num_kv_blocks as usize,
            block_size: block_size as usize,
            max_num_batched_tokens: 128,
            speculative_lookahead: 4,
            enable_radix_cache: true,
        })
        .map_err(|e| JsValue::from_str(&e))?;
        Ok(Self { core })
    }

    #[wasm_bindgen]
    pub fn submit_request(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temp: f32,
        n_parallel: u32,
        json_schema_opt: Option<String>,
        use_speculative: bool,
    ) -> Result<u32, JsValue> {
        let rid = self
            .core
            .submit(SubmitRequestPayload {
                prompt: prompt.to_string(),
                max_tokens: max_tokens as usize,
                temp,
                n_parallel: n_parallel as usize,
                json_schema_opt,
                use_speculative,
                seed: None,
            })
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(rid as u32)
    }

    #[wasm_bindgen]
    pub fn step_iteration(&mut self) -> JsValue {
        match self.core.step_and_capture() {
            Ok(snap) => {
                if let Ok(json_str) = serde_json::to_string(&snap) {
                    if let Ok(parsed) = js_sys::JSON::parse(&json_str) {
                        return parsed;
                    }
                }
                serde_wasm_bindgen::to_value(&snap).unwrap_or(JsValue::NULL)
            }
            Err(err) => JsValue::from_str(&err),
        }
    }
}

// ============================================================================
// Safe Linear-Memory JSON FFI Registry (Zero `unsafe` pointer dereferencing!)
// ============================================================================

struct FfiRegistry {
    next_handle: u32,
    engines: BTreeMap<u32, EngineCore>,
    input_buf: Vec<u8>,
    output_buf: Vec<u8>,
}

impl FfiRegistry {
    fn new() -> Self {
        Self {
            next_handle: 1,
            engines: BTreeMap::new(),
            input_buf: Vec::with_capacity(4096),
            output_buf: Vec::with_capacity(16_384),
        }
    }

    fn set_output_str(&mut self, s: &str) {
        self.output_buf.clear();
        self.output_buf.extend_from_slice(s.as_bytes());
    }
}

fn registry() -> &'static Mutex<FfiRegistry> {
    static REG: OnceLock<Mutex<FfiRegistry>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(FfiRegistry::new()))
}

#[no_mangle]
pub extern "C" fn ns_wasm_prepare_input(byte_len: usize) -> *mut u8 {
    let mut reg = registry().lock();
    reg.input_buf.resize(byte_len, 0);
    reg.input_buf.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn ns_wasm_output_ptr() -> *const u8 {
    let reg = registry().lock();
    reg.output_buf.as_ptr()
}

#[no_mangle]
pub extern "C" fn ns_wasm_output_len() -> usize {
    let reg = registry().lock();
    reg.output_buf.len()
}

#[no_mangle]
pub extern "C" fn ns_wasm_engine_new_configured() -> u32 {
    let mut reg = registry().lock();
    let cfg: EngineConfigPayload = match serde_json::from_slice(&reg.input_buf) {
        Ok(c) => c,
        Err(e) => {
            reg.set_output_str(&format!("invalid EngineConfigPayload JSON: {e}"));
            return 0;
        }
    };
    match EngineCore::new_configured(cfg) {
        Ok(core) => {
            let h = reg.next_handle;
            reg.next_handle += 1;
            reg.engines.insert(h, core);
            reg.set_output_str("ok");
            h
        }
        Err(e) => {
            reg.set_output_str(&e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn ns_wasm_engine_free(handle: u32) {
    let mut reg = registry().lock();
    reg.engines.remove(&handle);
}

#[no_mangle]
pub extern "C" fn ns_wasm_submit_json(handle: u32) -> u32 {
    let mut reg = registry().lock();
    let payload: SubmitRequestPayload = match serde_json::from_slice(&reg.input_buf) {
        Ok(p) => p,
        Err(e) => {
            reg.set_output_str(&format!("invalid SubmitRequestPayload JSON: {e}"));
            return 0;
        }
    };
    let res = match reg.engines.get_mut(&handle) {
        Some(core) => core.submit(payload),
        None => Err(format!("unknown engine handle {handle}")),
    };
    match res {
        Ok(rid) => {
            reg.set_output_str("ok");
            rid as u32
        }
        Err(e) => {
            reg.set_output_str(&e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn ns_wasm_step_json(handle: u32) -> u32 {
    let mut reg = registry().lock();
    let res = match reg.engines.get_mut(&handle) {
        Some(core) => core.step_and_capture(),
        None => Err(format!("unknown engine handle {handle}")),
    };
    match res {
        Ok(snap) => match serde_json::to_string(&snap) {
            Ok(json) => {
                reg.set_output_str(&json);
                1
            }
            Err(e) => {
                reg.set_output_str(&e.to_string());
                0
            }
        },
        Err(e) => {
            reg.set_output_str(&e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn ns_wasm_snapshot_json(handle: u32) -> u32 {
    let mut reg = registry().lock();
    let snap_opt = reg
        .engines
        .get_mut(&handle)
        .map(|core| core.build_telemetry(None));
    match snap_opt {
        Some(snap) => match serde_json::to_string(&snap) {
            Ok(json) => {
                reg.set_output_str(&json);
                1
            }
            Err(e) => {
                reg.set_output_str(&e.to_string());
                0
            }
        },
        None => {
            reg.set_output_str(&format!("unknown engine handle {handle}"));
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn ns_wasm_clear_radix(handle: u32) -> u32 {
    let mut reg = registry().lock();
    if let Some(core) = reg.engines.get_mut(&handle) {
        core.engine.clear_radix_cache();
        1
    } else {
        0
    }
}
