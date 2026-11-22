//! Iteration-Level Continuous Batching Scheduler (`ServingEngine`) unifying `RadixCache`
//! prefix matching, `ChunkedPrefill`, `Swap` & `Recompute` preemption, CoW `n`-parallel
//! completions, and Leviathan Exact Speculative Decoding.

use crate::error::SchedError;
use crate::speculative::{leviathan_rejection_sample, AdaptiveLookaheadController};
use crate::types::{
    collect_top_logprobs, ActiveSequenceSnapshot, BuiltinSampler, CompletedRequest, FinishReason,
    GrammarCandidateLogit, GrammarStepXray, PreemptionMode, RequestParams, SampledToken,
    SamplerStrategy, SamplingConfig, SchedulerConfig, SequenceOutput, SequenceStatus,
    SequenceStepDelta, SpeculativeTrace, StepSummary,
};
use novaserve_frontend::{AllowedTokenMask, BpeTokenizer, GrammarMatcher};
use novaserve_kv::{
    BlockAllocator, KvSlab, LogicalBlockTable, PhysicalBlockSnapshot, RadixCache,
    RadixNodeSnapshot, SwapEngine, SwappedHandle,
};
use novaserve_model::{BatchInput, LlamaModel};
use novaserve_tensor::{AlignedScratchArena, ChaCha8Rng};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

struct ChildSeq {
    child_idx: usize,
    /// Full sequence history `prompt_tokens + generated_tokens`.
    /// Invariant while active: `0 <= table.num_tokens() < full_tokens.len()`.
    /// When `table.num_tokens() == full_tokens.len() - 1`, the sequence is in single-token Decode mode!
    full_tokens: Vec<u32>,
    prompt_len: usize,
    step_logprobs: Vec<SampledToken>,
    table: LogicalBlockTable,
    draft_table: Option<LogicalBlockTable>,
    rng: ChaCha8Rng,
    grammar: Option<GrammarMatcher>,
    finish_reason: Option<FinishReason>,
}

struct SwappedChild {
    child_idx: usize,
    full_tokens: Vec<u32>,
    prompt_len: usize,
    step_logprobs: Vec<SampledToken>,
    handle: SwappedHandle,
    rng: ChaCha8Rng,
    grammar: Option<GrammarMatcher>,
    finish_reason: Option<FinishReason>,
}

struct SequenceGroup {
    req_id: u64,
    arrival_order: u64,
    max_new_tokens: usize,
    n_parallel: usize,
    has_forked: bool,
    cached_prefix_tokens: usize,
    preemptions: u32,
    sampling: SamplingConfig,
    use_speculative: bool,
    children: Vec<ChildSeq>,
}

struct SwappedGroup {
    req_id: u64,
    arrival_order: u64,
    max_new_tokens: usize,
    n_parallel: usize,
    has_forked: bool,
    cached_prefix_tokens: usize,
    preemptions: u32,
    sampling: SamplingConfig,
    use_speculative: bool,
    children: Vec<SwappedChild>,
}

struct DraftRuntime {
    model: LlamaModel,
    slab: KvSlab,
    allocator: BlockAllocator,
    arena: AlignedScratchArena,
}

pub struct ServingEngine {
    cfg: SchedulerConfig,
    target_model: LlamaModel,
    kv_slab: KvSlab,
    allocator: BlockAllocator,
    radix_cache: RadixCache,
    swap_engine: SwapEngine,
    arena: AlignedScratchArena,
    draft: Option<DraftRuntime>,
    lookahead_ctrl: AdaptiveLookaheadController,
    tokenizer: Option<Arc<BpeTokenizer>>,
    sampler: Arc<dyn SamplerStrategy>,

    next_req_id: u64,
    next_arrival_order: u64,
    step_counter: u64,

    waiting: VecDeque<SequenceGroup>,
    running: Vec<SequenceGroup>,
    swapped: VecDeque<SwappedGroup>,
    completed_ledger: BTreeMap<u64, CompletedRequest>,
}

impl ServingEngine {
    pub fn new(
        target_model: LlamaModel,
        draft_model: Option<LlamaModel>,
        tokenizer: Option<Arc<BpeTokenizer>>,
        cfg: SchedulerConfig,
    ) -> Result<Self, SchedError> {
        Self::with_sampler(
            target_model,
            draft_model,
            tokenizer,
            cfg,
            Arc::new(BuiltinSampler),
        )
    }

    pub fn with_sampler(
        target_model: LlamaModel,
        draft_model: Option<LlamaModel>,
        tokenizer: Option<Arc<BpeTokenizer>>,
        cfg: SchedulerConfig,
        sampler: Arc<dyn SamplerStrategy>,
    ) -> Result<Self, SchedError> {
        if cfg.max_num_batched_tokens == 0 || cfg.max_num_seqs == 0 || cfg.num_physical_blocks == 0
        {
            return Err(SchedError::InvalidRequest(
                "max_num_batched_tokens, max_num_seqs, and num_physical_blocks must be > 0".into(),
            ));
        }

        let slab_cfg = target_model.config().kv_slab_config(
            cfg.num_physical_blocks,
            cfg.block_size,
            cfg.kv_dtype,
        );
        let kv_slab = KvSlab::new(slab_cfg)?;
        let allocator = BlockAllocator::new(cfg.num_physical_blocks, cfg.block_size);
        let radix_cache = RadixCache::new(cfg.block_size);
        let swap_engine = SwapEngine::new(&kv_slab, cfg.num_swap_blocks)?;
        let arena = AlignedScratchArena::new(
            target_model
                .config()
                .recommended_scratch_bytes(cfg.max_num_batched_tokens),
        )?;

        let draft = if let Some(dm) = draft_model {
            let d_blocks = cfg.num_physical_blocks.max(64);
            let d_slab_cfg =
                dm.config()
                    .kv_slab_config(d_blocks, cfg.block_size, novaserve_kv::KvDType::F32);
            let d_slab = KvSlab::new(d_slab_cfg)?;
            let d_alloc = BlockAllocator::new(d_blocks, cfg.block_size);
            let d_arena = AlignedScratchArena::new(
                dm.config()
                    .recommended_scratch_bytes(cfg.max_num_batched_tokens),
            )?;
            Some(DraftRuntime {
                model: dm,
                slab: d_slab,
                allocator: d_alloc,
                arena: d_arena,
            })
        } else {
            None
        };

        let lookahead_ctrl = AdaptiveLookaheadController::new(
            if draft.is_some() {
                cfg.speculative_lookahead
            } else {
                0
            },
            cfg.adaptive_speculative,
        );

        Ok(Self {
            cfg,
            target_model,
            kv_slab,
            allocator,
            radix_cache,
            swap_engine,
            arena,
            draft,
            lookahead_ctrl,
            tokenizer,
            sampler,
            next_req_id: 1,
            next_arrival_order: 1,
            step_counter: 0,
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            completed_ledger: BTreeMap::new(),
        })
    }

    pub fn set_sampler(&mut self, sampler: Arc<dyn SamplerStrategy>) {
        self.sampler = sampler;
    }

    #[inline]
    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    #[inline]
    pub fn target_model(&self) -> &LlamaModel {
        &self.target_model
    }

    #[inline]
    pub fn allocator(&self) -> &BlockAllocator {
        &self.allocator
    }

    #[inline]
    pub fn allocator_mut(&mut self) -> &mut BlockAllocator {
        &mut self.allocator
    }

    #[inline]
    pub fn radix_cache(&self) -> &RadixCache {
        &self.radix_cache
    }

    #[inline]
    pub fn clear_radix_cache(&mut self) {
        self.radix_cache.clear(&mut self.allocator);
        if let Some(d) = &mut self.draft {
            // Verify draft allocator is also clean when engine is idle
            let _ = d.allocator.free_blocks();
        }
    }

    #[inline]
    pub fn swap_engine(&self) -> &SwapEngine {
        &self.swap_engine
    }

    #[inline]
    pub fn lookahead_controller(&self) -> &AdaptiveLookaheadController {
        &self.lookahead_ctrl
    }

    #[inline]
    pub fn has_unfinished_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }

    #[inline]
    pub fn queue_depths(&self) -> (usize, usize, usize) {
        (self.waiting.len(), self.running.len(), self.swapped.len())
    }

    pub fn take_completed(&mut self, req_id: u64) -> Option<CompletedRequest> {
        self.completed_ledger.remove(&req_id)
    }

    pub fn snapshot_kv_blocks(&self) -> Vec<PhysicalBlockSnapshot> {
        self.allocator.snapshot_blocks()
    }

    pub fn snapshot_radix_tree(&self) -> Vec<RadixNodeSnapshot> {
        self.radix_cache.snapshot_tree(&self.allocator)
    }

    #[inline]
    pub fn tokenizer(&self) -> Option<&Arc<BpeTokenizer>> {
        self.tokenizer.as_ref()
    }

    #[inline]
    pub fn step_counter(&self) -> u64 {
        self.step_counter
    }

    /// Exports live per-sequence KV block ownership, prefill/decode progress, and grammar FSM state
    /// for `novaserve-server`, `novaserve-top` TUI, and `novaserve-wasm` Studio.
    pub fn inspect_active_sequences(&self) -> Vec<ActiveSequenceSnapshot> {
        let mut out =
            Vec::with_capacity(self.running.len() * 2 + self.swapped.len() + self.waiting.len());

        for g in &self.running {
            for c in &g.children {
                let status = if c.table.num_tokens() + 1 < c.full_tokens.len() {
                    SequenceStatus::Prefilling
                } else {
                    SequenceStatus::Decoding
                };
                out.push(ActiveSequenceSnapshot {
                    req_id: g.req_id,
                    child_idx: c.child_idx,
                    status,
                    prompt_len: c.prompt_len,
                    kv_tokens: c.table.num_tokens(),
                    generated_tokens: c.full_tokens[c.prompt_len..].to_vec(),
                    max_new_tokens: g.max_new_tokens,
                    block_ids: c.table.blocks().iter().map(|b| b.0).collect(),
                    swapped_blocks: 0,
                    cached_prefix_tokens: g.cached_prefix_tokens,
                    preemptions: g.preemptions,
                    use_speculative: g.use_speculative,
                    has_grammar: c.grammar.is_some(),
                    dfa_state: c.grammar.as_ref().map(GrammarMatcher::state),
                    total_dfa_states: c.grammar.as_ref().map(|gm| gm.cache().dfa().num_states()),
                });
            }
        }

        for g in &self.swapped {
            for c in &g.children {
                out.push(ActiveSequenceSnapshot {
                    req_id: g.req_id,
                    child_idx: c.child_idx,
                    status: SequenceStatus::Swapped,
                    prompt_len: c.prompt_len,
                    kv_tokens: c.handle.num_tokens,
                    generated_tokens: c.full_tokens[c.prompt_len..].to_vec(),
                    max_new_tokens: g.max_new_tokens,
                    block_ids: Vec::new(),
                    swapped_blocks: c.handle.swap_slots.len(),
                    cached_prefix_tokens: g.cached_prefix_tokens,
                    preemptions: g.preemptions,
                    use_speculative: g.use_speculative,
                    has_grammar: c.grammar.is_some(),
                    dfa_state: c.grammar.as_ref().map(GrammarMatcher::state),
                    total_dfa_states: c.grammar.as_ref().map(|gm| gm.cache().dfa().num_states()),
                });
            }
        }

        for g in &self.waiting {
            for c in &g.children {
                out.push(ActiveSequenceSnapshot {
                    req_id: g.req_id,
                    child_idx: c.child_idx,
                    status: SequenceStatus::Waiting,
                    prompt_len: c.prompt_len,
                    kv_tokens: c.table.num_tokens(),
                    generated_tokens: c.full_tokens[c.prompt_len..].to_vec(),
                    max_new_tokens: g.max_new_tokens,
                    block_ids: c.table.blocks().iter().map(|b| b.0).collect(),
                    swapped_blocks: 0,
                    cached_prefix_tokens: g.cached_prefix_tokens,
                    preemptions: g.preemptions,
                    use_speculative: g.use_speculative,
                    has_grammar: c.grammar.is_some(),
                    dfa_state: c.grammar.as_ref().map(GrammarMatcher::state),
                    total_dfa_states: c.grammar.as_ref().map(|gm| gm.cache().dfa().num_states()),
                });
            }
        }

        out
    }

    /// Submits a new inference request into `waiting`, returning its `req_id`.
    pub fn submit_request(&mut self, params: RequestParams) -> Result<u64, SchedError> {
        if params.prompt_tokens.is_empty() {
            return Err(SchedError::InvalidRequest(
                "prompt_tokens cannot be empty".into(),
            ));
        }
        let vocab_sz = self.target_model.config().vocab_size;
        for &tok in &params.prompt_tokens {
            if (tok as usize) >= vocab_sz {
                return Err(SchedError::InvalidRequest(format!(
                    "prompt token {tok} >= vocab_size {vocab_sz}"
                )));
            }
        }

        let req_id = self.next_req_id;
        self.next_req_id += 1;
        let arrival = self.next_arrival_order;
        self.next_arrival_order += 1;
        let n_parallel = params.n_parallel.max(1);
        let prompt_len = params.prompt_tokens.len();

        if params.max_new_tokens == 0 {
            let outputs = (0..n_parallel)
                .map(|c| SequenceOutput {
                    child_idx: c,
                    token_ids: Vec::new(),
                    step_logprobs: Vec::new(),
                    finish_reason: FinishReason::Length,
                })
                .collect();
            self.completed_ledger.insert(
                req_id,
                CompletedRequest {
                    req_id,
                    prompt_len,
                    cached_prefix_tokens: 0,
                    outputs,
                    preemptions: 0,
                },
            );
            return Ok(req_id);
        }

        let initial_grammar = params
            .grammar
            .as_ref()
            .map(|cache| GrammarMatcher::new(Arc::clone(cache)));

        let child0 = ChildSeq {
            child_idx: 0,
            full_tokens: params.prompt_tokens,
            prompt_len,
            step_logprobs: Vec::new(),
            table: LogicalBlockTable::new(self.cfg.block_size),
            draft_table: if self.draft.is_some() && params.use_speculative && n_parallel == 1 {
                Some(LogicalBlockTable::new(self.cfg.block_size))
            } else {
                None
            },
            rng: ChaCha8Rng::from_seed_u64(params.sampling.seed),
            grammar: initial_grammar,
            finish_reason: None,
        };

        self.waiting.push_back(SequenceGroup {
            req_id,
            arrival_order: arrival,
            max_new_tokens: params.max_new_tokens,
            n_parallel,
            has_forked: n_parallel == 1,
            cached_prefix_tokens: 0,
            preemptions: 0,
            sampling: params.sampling,
            use_speculative: params.use_speculative && self.draft.is_some() && n_parallel == 1,
            children: vec![child0],
        });

        Ok(req_id)
    }

    /// Immediately cancels an active, waiting, or swapped request (`abort_request`),
    /// reclaiming all unshared primary KV and swap blocks instantaneously.
    pub fn abort_request(&mut self, req_id: u64) -> bool {
        if let Some(pos) = self.waiting.iter().position(|g| g.req_id == req_id) {
            let mut g = self.waiting.remove(pos).unwrap();
            self.release_group_tables(&mut g);
            return true;
        }
        if let Some(pos) = self.running.iter().position(|g| g.req_id == req_id) {
            let mut g = self.running.remove(pos);
            self.release_group_tables(&mut g);
            return true;
        }
        if let Some(pos) = self.swapped.iter().position(|g| g.req_id == req_id) {
            let g = self.swapped.remove(pos).unwrap();
            for sc in g.children {
                self.swap_engine.discard_swapped(sc.handle);
            }
            return true;
        }
        false
    }

    fn release_group_tables(&mut self, group: &mut SequenceGroup) {
        for c in &mut group.children {
            self.allocator.release_table(&mut c.table);
            if let (Some(dt), Some(d_rt)) = (&mut c.draft_table, &mut self.draft) {
                d_rt.allocator.release_table(dt);
            }
        }
    }

    #[inline]
    fn evictable_plus_free_blocks(&self) -> usize {
        // Upper bound on blocks available without preempting active sequences
        self.allocator.free_blocks() + self.radix_cache.cached_blocks_count()
    }

    fn ensure_free_blocks(&mut self, needed: usize) -> bool {
        if self.allocator.free_blocks() >= needed {
            return true;
        }
        if self.cfg.enable_radix_cache {
            let deficit = needed - self.allocator.free_blocks();
            self.radix_cache.evict_lru(deficit, &mut self.allocator);
        }
        self.allocator.free_blocks() >= needed
    }

    /// Preempts the lowest-priority (`running.last()`) sequence group via `Swap` or `Recompute`.
    fn preempt_lowest_priority(&mut self) -> Result<bool, SchedError> {
        if self.running.len() <= 1 {
            return Ok(false);
        }
        let mut victim = self.running.pop().expect("len > 1");
        victim.preemptions += 1;

        for c in &mut victim.children {
            if let (Some(dt), Some(d_rt)) = (&mut c.draft_table, &mut self.draft) {
                d_rt.allocator.release_table(dt);
            }
        }

        if self.cfg.preemption_mode == PreemptionMode::Swap {
            let total_swap_needed: usize =
                victim.children.iter().map(|c| c.table.num_blocks()).sum();
            if self.swap_engine.free_swap_blocks() >= total_swap_needed {
                let mut swapped_children = Vec::with_capacity(victim.children.len());
                for mut c in victim.children {
                    let sh = self.swap_engine.swap_out(
                        (victim.req_id << 8) | (c.child_idx as u64),
                        &mut c.table,
                        &self.kv_slab,
                        &mut self.allocator,
                    )?;
                    swapped_children.push(SwappedChild {
                        child_idx: c.child_idx,
                        full_tokens: c.full_tokens,
                        prompt_len: c.prompt_len,
                        step_logprobs: c.step_logprobs,
                        handle: sh,
                        rng: c.rng,
                        grammar: c.grammar,
                        finish_reason: c.finish_reason,
                    });
                }
                self.swapped.push_front(SwappedGroup {
                    req_id: victim.req_id,
                    arrival_order: victim.arrival_order,
                    max_new_tokens: victim.max_new_tokens,
                    n_parallel: victim.n_parallel,
                    has_forked: victim.has_forked,
                    cached_prefix_tokens: victim.cached_prefix_tokens,
                    preemptions: victim.preemptions,
                    sampling: victim.sampling,
                    use_speculative: victim.use_speculative,
                    children: swapped_children,
                });
                return Ok(true);
            }
        }

        // `PreemptionMode::Recompute` (or fallback when secondary swap pool is full):
        // Release primary blocks and re-queue at the front of `waiting` ordered by `arrival_order`!
        for c in &mut victim.children {
            self.allocator.release_table(&mut c.table);
        }
        let insert_idx = self
            .waiting
            .iter()
            .position(|w| w.arrival_order > victim.arrival_order)
            .unwrap_or(self.waiting.len());
        self.waiting.insert(insert_idx, victim);
        Ok(true)
    }

    fn insert_running_sorted(&mut self, group: SequenceGroup) {
        let pos = self
            .running
            .binary_search_by_key(&group.arrival_order, |g| g.arrival_order)
            .unwrap_or_else(|i| i);
        self.running.insert(pos, group);
    }

    /// Executes **one exact iteration** of Continuous Batching (Preemption -> Swap-In ->
    /// Radix-Accelerated Chunked Prefill Admission -> Batched Forward / Speculative Verify -> Sampling).
    pub fn step(&mut self) -> Result<StepSummary, SchedError> {
        self.step_counter += 1;
        let mut summary = StepSummary {
            step_index: self.step_counter,
            ..StepSummary::default()
        };

        if !self.has_unfinished_work() {
            return Ok(summary);
        }

        let block_size = self.cfg.block_size;

        // -------------------------------------------------------------------------
        // PHASE 1: Ensure KV capacity for active `running` groups (Preempt newest if needed!)
        // -------------------------------------------------------------------------
        loop {
            let mut worst_case_blocks_for_running = 0usize;
            for g in &self.running {
                for c in &g.children {
                    if c.finish_reason.is_some() {
                        continue;
                    }
                    let cur_toks = c.table.num_tokens();
                    let rem = c.full_tokens.len() - cur_toks;
                    if rem == 1 {
                        // Decode token needs at most 1 block (new boundary or CoW tail split)
                        let needs_blk = cur_toks.is_multiple_of(block_size)
                            || c.table
                                .blocks()
                                .last()
                                .is_some_and(|&b| self.allocator.ref_count(b) > 1);
                        if needs_blk {
                            worst_case_blocks_for_running += 1;
                        }
                    } else {
                        // Continuing chunked prefill: needs at least 1 block to advance
                        worst_case_blocks_for_running += 1;
                    }
                }
            }

            if self.ensure_free_blocks(worst_case_blocks_for_running) {
                break;
            }
            if self.preempt_lowest_priority()? {
                summary.preempted_this_step += 1;
            } else {
                break;
            }
        }

        // -------------------------------------------------------------------------
        // PHASE 2: Swap-In before admitting new `waiting` requests (with +1 hysteresis)
        // -------------------------------------------------------------------------
        while !self.swapped.is_empty() && self.running.len() < self.cfg.max_num_seqs {
            let blocks_needed: usize = self.swapped[0]
                .children
                .iter()
                .map(|c| c.handle.swap_slots.len() + usize::from(c.finish_reason.is_none()))
                .sum();
            let headroom = if self.running.is_empty() {
                0
            } else {
                self.running.len()
            };
            if !self.ensure_free_blocks(blocks_needed + headroom) {
                break;
            }

            let sg = self.swapped.pop_front().unwrap();
            let mut restored_children = Vec::with_capacity(sg.children.len());
            let mut failed_mid = false;

            let mut temp_swapped = VecDeque::from(sg.children);
            while let Some(sc) = temp_swapped.pop_front() {
                let radix_opt = if self.cfg.enable_radix_cache {
                    Some(&mut self.radix_cache)
                } else {
                    None
                };
                match self.swap_engine.swap_in(
                    sc.handle,
                    &mut self.kv_slab,
                    &mut self.allocator,
                    radix_opt,
                ) {
                    Ok(tbl) => {
                        restored_children.push(ChildSeq {
                            child_idx: sc.child_idx,
                            full_tokens: sc.full_tokens,
                            prompt_len: sc.prompt_len,
                            step_logprobs: sc.step_logprobs,
                            table: tbl,
                            draft_table: if sg.use_speculative && self.draft.is_some() {
                                Some(LogicalBlockTable::new(block_size))
                            } else {
                                None
                            },
                            rng: sc.rng,
                            grammar: sc.grammar,
                            finish_reason: sc.finish_reason,
                        });
                    }
                    Err((h_back, _)) => {
                        // Roll back any already swapped-in sibling children cleanly
                        temp_swapped.push_front(SwappedChild {
                            child_idx: sc.child_idx,
                            full_tokens: sc.full_tokens,
                            prompt_len: sc.prompt_len,
                            step_logprobs: sc.step_logprobs,
                            handle: h_back,
                            rng: sc.rng,
                            grammar: sc.grammar,
                            finish_reason: sc.finish_reason,
                        });
                        for mut rc in restored_children.drain(..) {
                            let h = self.swap_engine.swap_out(
                                (sg.req_id << 8) | (rc.child_idx as u64),
                                &mut rc.table,
                                &self.kv_slab,
                                &mut self.allocator,
                            )?;
                            temp_swapped.push_front(SwappedChild {
                                child_idx: rc.child_idx,
                                full_tokens: rc.full_tokens,
                                prompt_len: rc.prompt_len,
                                step_logprobs: rc.step_logprobs,
                                handle: h,
                                rng: rc.rng,
                                grammar: rc.grammar,
                                finish_reason: rc.finish_reason,
                            });
                        }
                        self.swapped.push_front(SwappedGroup {
                            req_id: sg.req_id,
                            arrival_order: sg.arrival_order,
                            max_new_tokens: sg.max_new_tokens,
                            n_parallel: sg.n_parallel,
                            has_forked: sg.has_forked,
                            cached_prefix_tokens: sg.cached_prefix_tokens,
                            preemptions: sg.preemptions,
                            sampling: sg.sampling.clone(),
                            use_speculative: sg.use_speculative,
                            children: Vec::from(temp_swapped),
                        });
                        failed_mid = true;
                        break;
                    }
                }
            }

            if failed_mid {
                break;
            }
            summary.swapped_in_this_step += 1;
            self.insert_running_sorted(SequenceGroup {
                req_id: sg.req_id,
                arrival_order: sg.arrival_order,
                max_new_tokens: sg.max_new_tokens,
                n_parallel: sg.n_parallel,
                has_forked: sg.has_forked,
                cached_prefix_tokens: sg.cached_prefix_tokens,
                preemptions: sg.preemptions,
                sampling: sg.sampling,
                use_speculative: sg.use_speculative,
                children: restored_children,
            });
        }

        // -------------------------------------------------------------------------
        // PHASE 3: Admit from `waiting` (Radix Prefix Match + Chunked Prefill multiplexing)
        // -------------------------------------------------------------------------
        let decode_tokens_reserved: usize = self
            .running
            .iter()
            .flat_map(|g| g.children.iter())
            .filter(|c| {
                c.finish_reason.is_none() && c.full_tokens.len() - c.table.num_tokens() == 1
            })
            .count();
        let mut token_budget = self
            .cfg
            .max_num_batched_tokens
            .saturating_sub(decode_tokens_reserved)
            .max(1);

        if self.swapped.is_empty() {
            while !self.waiting.is_empty()
                && self.running.len() < self.cfg.max_num_seqs
                && token_budget > 0
            {
                let headroom = if self.running.is_empty() {
                    0
                } else {
                    (self.running.len() * 2).min(self.cfg.num_physical_blocks / 3)
                };
                if self.evictable_plus_free_blocks() <= headroom + 1 && !self.running.is_empty() {
                    break;
                }

                let mut wg = self.waiting.pop_front().unwrap();
                for c in &mut wg.children {
                    if c.finish_reason.is_none()
                        && c.table.num_tokens() == 0
                        && self.cfg.enable_radix_cache
                    {
                        let m = self
                            .radix_cache
                            .match_prefix(&c.full_tokens, &mut self.allocator);
                        if m.matched_tokens > 0 {
                            if c.child_idx == 0 && wg.cached_prefix_tokens == 0 {
                                wg.cached_prefix_tokens = m.matched_tokens;
                            }
                            c.table = LogicalBlockTable::from_cached_prefix(
                                m.cached_block_ids,
                                m.matched_tokens,
                                block_size,
                            );
                        }
                    }
                }
                let rem_first = wg.children[0]
                    .full_tokens
                    .len()
                    .saturating_sub(wg.children[0].table.num_tokens());
                token_budget = token_budget.saturating_sub(rem_first.min(token_budget));
                self.insert_running_sorted(wg);
            }
        }

        // -------------------------------------------------------------------------
        // PHASE 4: Execute Speculative Decode groups & build unified `BatchInput` for the rest!
        // -------------------------------------------------------------------------
        let eos_id = self.tokenizer.as_ref().map(|t| t.eos_token_id());
        let active_k = self.lookahead_ctrl.current_k();

        // Separate speculative-ready single-sequence decode groups when `active_k >= 1`
        let mut spec_group_indices: Vec<usize> = Vec::new();
        if active_k >= 1 && self.draft.is_some() {
            for g_idx in 0..self.running.len() {
                let eligible = {
                    let g = &self.running[g_idx];
                    g.use_speculative
                        && g.children.len() == 1
                        && g.children[0].finish_reason.is_none()
                        && g.children[0].full_tokens.len() - g.children[0].table.num_tokens() == 1
                };
                if eligible && self.ensure_free_blocks(2) {
                    spec_group_indices.push(g_idx);
                }
            }
        }

        for &g_idx in &spec_group_indices {
            self.run_speculative_group_step(g_idx, active_k, eos_id, &mut summary)?;
        }

        // Build unified heterogeneous `BatchInput` for all remaining `running` groups!
        let mut batch = BatchInput::with_capacity(self.cfg.max_num_batched_tokens);
        // Track `(group_idx, child_idx, is_final_prefill_or_decode_logit)` for each emitted logit row!
        let mut logit_targets: Vec<(usize, usize)> = Vec::new();
        // Track which `(group_idx, child_idx)` seats are included in `seq_tables`
        let mut seat_list: Vec<(usize, usize)> = Vec::new();

        let mut remaining_budget = self.cfg.max_num_batched_tokens;

        for (g_idx, g) in self.running.iter_mut().enumerate() {
            if spec_group_indices.contains(&g_idx) {
                continue;
            }
            for c_idx in 0..g.children.len() {
                if g.children[c_idx].finish_reason.is_some() || remaining_budget == 0 {
                    continue;
                }

                let cur_kv = g.children[c_idx].table.num_tokens();
                let full_len = g.children[c_idx].full_tokens.len();
                let uncached_rem = full_len - cur_kv;
                // When Speculative Decoding is active for this sequence (`active_k >= 1`),
                // prefill up to `full_len - 1` so the final prompt token immediately enters
                // `run_speculative_group_step` and generates token 0..K speculatively!
                let target_rem = if g.use_speculative
                    && active_k >= 1
                    && self.draft.is_some()
                    && uncached_rem > 1
                    && g.max_new_tokens >= 2
                {
                    uncached_rem - 1
                } else {
                    uncached_rem
                };
                let chunk_target = target_rem.min(remaining_budget);

                let seat_idx = seat_list.len();
                let mut scheduled_for_child = 0usize;

                for step_off in 0..chunk_target {
                    let pos = cur_kv + step_off;
                    let tok = g.children[c_idx].full_tokens[pos];
                    let radix_opt = if self.cfg.enable_radix_cache {
                        Some(&mut self.radix_cache)
                    } else {
                        None
                    };
                    match self.allocator.cow_append_slot(
                        &mut self.kv_slab,
                        radix_opt,
                        &mut g.children[c_idx].table,
                    ) {
                        Ok((bid, off)) => {
                            let is_last_of_full = pos + 1 == full_len;
                            batch.push_token(tok, pos as u32, seat_idx, bid, off, is_last_of_full);
                            scheduled_for_child += 1;
                            if is_last_of_full {
                                logit_targets.push((g_idx, c_idx));
                            }
                        }
                        Err(_) => {
                            if pos + 1 >= self.cfg.num_physical_blocks * block_size {
                                // Hit absolute physical KV ceiling for single sequence
                                g.children[c_idx].finish_reason = Some(FinishReason::Length);
                            }
                            break;
                        }
                    }
                }

                if scheduled_for_child > 0 {
                    seat_list.push((g_idx, c_idx));
                    remaining_budget -= scheduled_for_child;
                    if uncached_rem == 1 {
                        summary.scheduled_decode_tokens += 1;
                    } else {
                        summary.scheduled_prefill_tokens += scheduled_for_child;
                    }
                }
            }
        }

        if !batch.is_empty() {
            let logits = {
                let table_refs: Vec<&LogicalBlockTable> = seat_list
                    .iter()
                    .map(|&(gi, ci)| &self.running[gi].children[ci].table)
                    .collect();
                self.target_model.forward_batch(
                    &batch,
                    &table_refs,
                    &mut self.kv_slab,
                    &mut self.arena,
                )?
            };

            let vocab_sz = self.target_model.config().vocab_size;
            assert_eq!(logits.len(), logit_targets.len() * vocab_sz);

            for (row_idx, &(g_idx, c_idx)) in logit_targets.iter().enumerate() {
                let row_logits = &logits[row_idx * vocab_sz..(row_idx + 1) * vocab_sz];
                self.post_forward_sample_and_fork(g_idx, c_idx, row_logits, eos_id, &mut summary)?;
            }
        }

        // -------------------------------------------------------------------------
        // PHASE 5: Retire finished `SequenceGroup`s, checkpoint prefixes into `RadixCache`!
        // -------------------------------------------------------------------------
        let mut i = 0usize;
        while i < self.running.len() {
            let all_done = self.running[i]
                .children
                .iter()
                .all(|c| c.finish_reason.is_some());
            if all_done {
                let done_group = self.running.remove(i);
                if self.cfg.enable_radix_cache {
                    for c in &done_group.children {
                        let valid_kv_toks = &c.full_tokens[..c.table.num_tokens()];
                        self.radix_cache.insert_prefix(
                            valid_kv_toks,
                            c.table.blocks(),
                            &mut self.allocator,
                        );
                    }
                }

                let mut outputs = Vec::with_capacity(done_group.children.len());
                let prompt_len = done_group.children[0].prompt_len;
                for mut c in done_group.children {
                    self.allocator.release_table(&mut c.table);
                    if let (Some(mut dt), Some(d_rt)) = (c.draft_table.take(), &mut self.draft) {
                        d_rt.allocator.release_table(&mut dt);
                    }
                    outputs.push(SequenceOutput {
                        child_idx: c.child_idx,
                        token_ids: c.full_tokens[c.prompt_len..].to_vec(),
                        step_logprobs: c.step_logprobs,
                        finish_reason: c.finish_reason.unwrap_or(FinishReason::Length),
                    });
                }
                outputs.sort_by_key(|o| o.child_idx);

                let comp = CompletedRequest {
                    req_id: done_group.req_id,
                    prompt_len,
                    cached_prefix_tokens: done_group.cached_prefix_tokens,
                    outputs,
                    preemptions: done_group.preemptions,
                };
                self.completed_ledger.insert(comp.req_id, comp.clone());
                summary.completed.push(comp);
            } else {
                i += 1;
            }
        }

        summary.waiting_groups = self.waiting.len();
        summary.running_groups = self.running.len();
        summary.swapped_groups = self.swapped.len();
        summary.rolling_spec_alpha = self.lookahead_ctrl.rolling_alpha();
        summary.active_spec_k = self.lookahead_ctrl.current_k();
        Ok(summary)
    }

    /// Samples next token(s) after `TargetModel` produces `row_logits`, handling `n_parallel > 1`
    /// zero-copy CoW forking on prompt completion AND grammar jump-token fast-forwarding!
    fn post_forward_sample_and_fork(
        &mut self,
        g_idx: usize,
        c_idx: usize,
        row_logits: &[f32],
        eos_id: Option<u32>,
        summary: &mut StepSummary,
    ) -> Result<(), SchedError> {
        // Checkpoint completed prompt into `RadixCache` immediately so co-running requests benefit!
        if self.cfg.enable_radix_cache
            && c_idx == 0
            && (!self.running[g_idx].has_forked
                || self.running[g_idx].children[c_idx].step_logprobs.is_empty())
        {
            let p_len = self.running[g_idx].children[0].prompt_len;
            let p_toks = self.running[g_idx].children[0].full_tokens[..p_len].to_vec();
            let p_blks = self.running[g_idx].children[0].table.blocks().to_vec();
            self.radix_cache
                .insert_prefix(&p_toks, &p_blks, &mut self.allocator);
        }

        // If `n_parallel > 1` and this group just finished its shared prompt prefill, CoW fork now!
        if !self.running[g_idx].has_forked {
            let n_par = self.running[g_idx].n_parallel;
            let base_seed = self.running[g_idx].sampling.seed;
            for sibling_idx in 1..n_par {
                let forked_tbl = self
                    .allocator
                    .fork_sequence(&self.running[g_idx].children[0].table);
                let sib_rng = ChaCha8Rng::from_seed_u64(
                    base_seed
                        .wrapping_add((sibling_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let sib_grammar = self.running[g_idx].children[0].grammar.clone();
                let sib_tokens = self.running[g_idx].children[0].full_tokens.clone();
                let p_len = self.running[g_idx].children[0].prompt_len;
                self.running[g_idx].children.push(ChildSeq {
                    child_idx: sibling_idx,
                    full_tokens: sib_tokens,
                    prompt_len: p_len,
                    step_logprobs: Vec::new(),
                    table: forked_tbl,
                    draft_table: None,
                    rng: sib_rng,
                    grammar: sib_grammar,
                    finish_reason: None,
                });
            }
            self.running[g_idx].has_forked = true;

            for sib_i in 0..n_par {
                self.sample_single_child(g_idx, sib_i, row_logits, eos_id, summary)?;
            }
            return Ok(());
        }

        self.sample_single_child(g_idx, c_idx, row_logits, eos_id, summary)
    }

    fn sample_single_child(
        &mut self,
        g_idx: usize,
        c_idx: usize,
        row_logits: &[f32],
        eos_id: Option<u32>,
        summary: &mut StepSummary,
    ) -> Result<(), SchedError> {
        let req_id = self.running[g_idx].req_id;
        let max_new = self.running[g_idx].max_new_tokens;
        let sampling = self.running[g_idx].sampling.clone();

        let child = &mut self.running[g_idx].children[c_idx];
        let pre_grammar_meta = child
            .grammar
            .as_ref()
            .map(|gm| (gm.state(), gm.cache().dfa().num_states(), gm.is_accepting()));
        let mask_arc = match (&child.grammar, &self.tokenizer) {
            (Some(gm), Some(tok)) => Some(gm.current_mask(tok)?),
            _ => None,
        };

        let sampled = self.sampler.sample(
            row_logits,
            &child.full_tokens,
            &sampling,
            mask_arc.as_deref(),
            eos_id,
            &mut child.rng,
        )?;
        let chosen_token_id = sampled.token_id;

        let mut new_tokens = Vec::with_capacity(4);
        let mut sampled_infos = Vec::with_capacity(4);
        let mut forced_emitted = Vec::new();

        Self::commit_emitted_token(
            child,
            sampled,
            &sampling,
            max_new,
            eos_id,
            self.tokenizer.as_deref(),
            &mut new_tokens,
            &mut sampled_infos,
        );

        // Grammar deterministic jump-token fast-forwarding (`forced_tokens()`)!
        if child.finish_reason.is_none() {
            if let (Some(gm), Some(tok)) = (&child.grammar, self.tokenizer.as_deref()) {
                let forced = gm.forced_tokens(tok)?;
                for ft in forced {
                    if child.finish_reason.is_some() {
                        break;
                    }
                    forced_emitted.push(ft);
                    let info = SampledToken {
                        token_id: ft,
                        logprob: 0.0,
                        top_logprobs: Vec::new(),
                    };
                    Self::commit_emitted_token(
                        child,
                        info,
                        &sampling,
                        max_new,
                        eos_id,
                        Some(tok),
                        &mut new_tokens,
                        &mut sampled_infos,
                    );
                }
            }
        }

        if let (Some((dfa_state, total_dfa_states, is_accepting)), Some(mask)) =
            (pre_grammar_meta, mask_arc.as_deref())
        {
            summary.grammar_xrays.push(Self::capture_grammar_xray(
                req_id,
                child.child_idx,
                dfa_state,
                total_dfa_states,
                is_accepting,
                mask,
                row_logits,
                chosen_token_id,
                forced_emitted,
            ));
        }

        summary.deltas.push(SequenceStepDelta {
            req_id,
            child_idx: child.child_idx,
            new_tokens,
            sampled_infos,
            finished: child.finish_reason,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_grammar_xray(
        req_id: u64,
        child_idx: usize,
        dfa_state: u32,
        total_dfa_states: usize,
        is_accepting: bool,
        mask: &AllowedTokenMask,
        raw_logits: &[f32],
        chosen_token_id: u32,
        forced_jump_tokens: Vec<u32>,
    ) -> GrammarStepXray {
        let mut allowed_pairs = Vec::new();
        let mut masked_pairs = Vec::new();
        let limit = raw_logits.len().min(mask.vocab_size());
        for (i, &val) in raw_logits[..limit].iter().enumerate() {
            let tid = i as u32;
            if mask.is_allowed(tid) {
                allowed_pairs.push(GrammarCandidateLogit {
                    token_id: tid,
                    raw_logit: val,
                });
            } else {
                masked_pairs.push(GrammarCandidateLogit {
                    token_id: tid,
                    raw_logit: val,
                });
            }
        }
        allowed_pairs.sort_by(|a, b| {
            b.raw_logit
                .total_cmp(&a.raw_logit)
                .then_with(|| a.token_id.cmp(&b.token_id))
        });
        masked_pairs.sort_by(|a, b| {
            b.raw_logit
                .total_cmp(&a.raw_logit)
                .then_with(|| a.token_id.cmp(&b.token_id))
        });
        allowed_pairs.truncate(10);
        masked_pairs.truncate(10);

        GrammarStepXray {
            req_id,
            child_idx,
            dfa_state,
            total_dfa_states,
            is_accepting,
            allowed_count: mask.allowed_count(),
            vocab_size: mask.vocab_size(),
            top_allowed: allowed_pairs,
            top_masked: masked_pairs,
            chosen_token_id,
            forced_jump_tokens,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_emitted_token(
        child: &mut ChildSeq,
        sampled: SampledToken,
        sampling: &SamplingConfig,
        max_new_tokens: usize,
        eos_id: Option<u32>,
        tokenizer: Option<&BpeTokenizer>,
        new_tokens: &mut Vec<u32>,
        sampled_infos: &mut Vec<SampledToken>,
    ) {
        let tid = sampled.token_id;
        if let (Some(gm), Some(tok)) = (&mut child.grammar, tokenizer) {
            gm.accept_token(tok, tid);
        }

        let is_eos = !sampling.ignore_eos && eos_id == Some(tid);
        let is_stop = sampling.stop_token_ids.contains(&tid);

        if is_eos || is_stop {
            child.step_logprobs.push(sampled.clone());
            sampled_infos.push(sampled);
            child.finish_reason = Some(FinishReason::Stop);
            return;
        }

        child.full_tokens.push(tid);
        child.step_logprobs.push(sampled.clone());
        new_tokens.push(tid);
        sampled_infos.push(sampled);

        let generated_len = child.full_tokens.len() - child.prompt_len;
        if generated_len >= max_new_tokens {
            child.finish_reason = Some(FinishReason::Length);
        }
    }

    /// Executes exact Leviathan Speculative Decoding ($K$ Draft steps + 1 batched $(K+1)$-token
    /// Target verification pass + rejection/residual/bonus sampling + KV rollback).
    fn run_speculative_group_step(
        &mut self,
        g_idx: usize,
        lookahead_k: usize,
        eos_id: Option<u32>,
        summary: &mut StepSummary,
    ) -> Result<(), SchedError> {
        let req_id = self.running[g_idx].req_id;
        let max_new = self.running[g_idx].max_new_tokens;
        let sampling = self.running[g_idx].sampling.clone();
        let vocab_sz = self.target_model.config().vocab_size;

        let d_rt = self.draft.as_mut().expect("draft runtime verified");
        let child = &mut self.running[g_idx].children[0];
        if child.draft_table.is_none() {
            child.draft_table = Some(LogicalBlockTable::new(self.cfg.block_size));
        }
        let m = child.table.num_tokens();
        debug_assert_eq!(m + 1, child.full_tokens.len());

        // 1. Sync `draft_table` up to `m` if behind (e.g. initial prompt or previous bonus step `d_K`)
        {
            let d_tbl = child.draft_table.as_mut().unwrap();
            if d_tbl.num_tokens() < m {
                let missing = m - d_tbl.num_tokens();
                let start_p = d_tbl.num_tokens();
                let mut sync_batch = BatchInput::with_capacity(missing);
                for pos in start_p..m {
                    let (bid, off) = d_rt
                        .allocator
                        .cow_append_slot(&mut d_rt.slab, None, d_tbl)?;
                    sync_batch.push_token(child.full_tokens[pos], pos as u32, 0, bid, off, false);
                }
                let refs = [&*d_tbl];
                let _ = d_rt.model.forward_batch(
                    &sync_batch,
                    &refs,
                    &mut d_rt.slab,
                    &mut d_rt.arena,
                )?;
            }
        }

        // Cap lookahead so we don't overshoot `max_new_tokens`
        let already_gen = child.full_tokens.len() - child.prompt_len;
        let rem_gen_budget = max_new.saturating_sub(already_gen);
        let max_draft_steps = lookahead_k.min(rem_gen_budget.saturating_sub(1));
        if max_draft_steps == 0 {
            // Only 1 token remaining until `max_new_tokens`: run 1 exact target step directly!
            let tok_in = child.full_tokens[m];
            let radix_opt = if self.cfg.enable_radix_cache {
                Some(&mut self.radix_cache)
            } else {
                None
            };
            let (bid, off) =
                self.allocator
                    .cow_append_slot(&mut self.kv_slab, radix_opt, &mut child.table)?;
            let mut b1 = BatchInput::with_capacity(1);
            b1.push_token(tok_in, m as u32, 0, bid, off, true);
            let logits = {
                let refs = [&child.table];
                self.target_model
                    .forward_batch(&b1, &refs, &mut self.kv_slab, &mut self.arena)?
            };
            summary.scheduled_decode_tokens += 1;
            return self.sample_single_child(g_idx, 0, &logits, eos_id, summary);
        }

        // 2. Autoregressive Draft Phase (`1 ..= max_draft_steps`)
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(max_draft_steps);
        let mut draft_dists: Vec<Vec<f32>> = Vec::with_capacity(max_draft_steps);
        let mut draft_grammar = child.grammar.clone();
        let mut draft_history = child.full_tokens.clone();
        let mut step_masks: Vec<Option<Arc<novaserve_frontend::AllowedTokenMask>>> =
            Vec::with_capacity(max_draft_steps + 1);

        let mut cur_input_tok = child.full_tokens[m];
        for step_i in 0..max_draft_steps {
            let d_pos = m + step_i;
            let d_tbl = child.draft_table.as_mut().unwrap();
            let Ok((dbid, doff)) = d_rt.allocator.cow_append_slot(&mut d_rt.slab, None, d_tbl)
            else {
                break;
            };
            let mut dbatch = BatchInput::with_capacity(1);
            dbatch.push_token(cur_input_tok, d_pos as u32, 0, dbid, doff, true);
            let d_logits = {
                let refs = [&*d_tbl];
                d_rt.model
                    .forward_batch(&dbatch, &refs, &mut d_rt.slab, &mut d_rt.arena)?
            };

            let mask_opt = match (&draft_grammar, self.tokenizer.as_deref()) {
                (Some(gm), Some(tok)) => Some(gm.current_mask(tok)?),
                _ => None,
            };
            let mut q_vec = vec![0.0f32; vocab_sz];
            if self
                .sampler
                .compute_probs(
                    &d_logits,
                    &draft_history,
                    &sampling,
                    mask_opt.as_deref(),
                    eos_id,
                    &mut q_vec,
                )
                .is_err()
            {
                d_rt.allocator.truncate_table(d_tbl, d_pos);
                break;
            }

            let d_tok = crate::types::sample_categorical(&q_vec, &mut child.rng) as u32;
            step_masks.push(mask_opt);
            draft_tokens.push(d_tok);
            draft_dists.push(q_vec);

            let is_stop_or_eos = (!sampling.ignore_eos && eos_id == Some(d_tok))
                || sampling.stop_token_ids.contains(&d_tok);
            if let (Some(gm), Some(tok)) = (&mut draft_grammar, self.tokenizer.as_deref()) {
                gm.accept_token(tok, d_tok);
            }
            draft_history.push(d_tok);
            cur_input_tok = d_tok;

            if is_stop_or_eos {
                break;
            }
        }

        let k_eff = draft_tokens.len();
        if k_eff == 0 {
            // Fallback to 1-token Target step
            let tok_in = child.full_tokens[m];
            let radix_opt = if self.cfg.enable_radix_cache {
                Some(&mut self.radix_cache)
            } else {
                None
            };
            let (bid, off) =
                self.allocator
                    .cow_append_slot(&mut self.kv_slab, radix_opt, &mut child.table)?;
            let mut b1 = BatchInput::with_capacity(1);
            b1.push_token(tok_in, m as u32, 0, bid, off, true);
            let logits = {
                let refs = [&child.table];
                self.target_model
                    .forward_batch(&b1, &refs, &mut self.kv_slab, &mut self.arena)?
            };
            summary.scheduled_decode_tokens += 1;
            return self.sample_single_child(g_idx, 0, &logits, eos_id, summary);
        }

        // Mask for bonus position `k_eff` (`p_{k_eff + 1}`), if last drafted token didn't terminate grammar
        let bonus_mask = match (&draft_grammar, self.tokenizer.as_deref()) {
            (Some(gm), Some(tok)) if !gm.is_terminated() => gm.current_mask(tok).ok(),
            _ => None,
        };

        // 3. Single-Pass Parallel Target Verification (`k_eff + 1` tokens: `[x_last, d_1, ..., d_{k_eff}]`)
        let mut verify_batch = BatchInput::with_capacity(k_eff + 1);
        for step_i in 0..=k_eff {
            let v_tok = if step_i == 0 {
                child.full_tokens[m]
            } else {
                draft_tokens[step_i - 1]
            };
            let pos = m + step_i;
            let radix_opt = if self.cfg.enable_radix_cache {
                Some(&mut self.radix_cache)
            } else {
                None
            };
            match self
                .allocator
                .cow_append_slot(&mut self.kv_slab, radix_opt, &mut child.table)
            {
                Ok((bid, off)) => {
                    verify_batch.push_token(v_tok, pos as u32, 0, bid, off, true);
                }
                Err(e) => {
                    // Rollback partial verify allocation cleanly
                    self.allocator.truncate_table(&mut child.table, m);
                    if let Some(dt) = &mut child.draft_table {
                        d_rt.allocator.truncate_table(dt, m);
                    }
                    return Err(SchedError::Kv(e));
                }
            }
        }

        let target_logits = {
            let refs = [&child.table];
            self.target_model.forward_batch(
                &verify_batch,
                &refs,
                &mut self.kv_slab,
                &mut self.arena,
            )?
        };
        summary.scheduled_decode_tokens += k_eff + 1;

        let mut target_dists: Vec<Vec<f32>> = Vec::with_capacity(k_eff + 1);
        let mut hist_prefix = child.full_tokens.clone();
        for step_i in 0..=k_eff {
            let row_l = &target_logits[step_i * vocab_sz..(step_i + 1) * vocab_sz];
            let mut p_vec = vec![0.0f32; vocab_sz];
            let mask_ref = if step_i < k_eff {
                step_masks[step_i].as_deref()
            } else {
                bonus_mask.as_deref()
            };

            if step_i == k_eff
                && draft_grammar
                    .as_ref()
                    .is_some_and(GrammarMatcher::is_terminated)
            {
                // Last drafted token was `<eos>`, so bonus distribution won't be consumed if `<eos>` stops;
                // populate valid dummy 1-hot on `<eos>` so `target_dists` has length `k_eff + 1`.
                p_vec[eos_id.unwrap_or(0) as usize] = 1.0;
            } else {
                self.sampler.compute_probs(
                    row_l,
                    &hist_prefix,
                    &sampling,
                    mask_ref,
                    eos_id,
                    &mut p_vec,
                )?;
            }
            target_dists.push(p_vec);
            if step_i < k_eff {
                hist_prefix.push(draft_tokens[step_i]);
            }
        }

        // 4. Leviathan Exact Rejection + Residual / Bonus Sampling!
        let outcome =
            leviathan_rejection_sample(&draft_tokens, &draft_dists, &target_dists, &mut child.rng);

        self.lookahead_ctrl
            .record_step(k_eff, outcome.accepted_draft_count);

        let draft_chosen_probs: Vec<f32> = (0..k_eff)
            .map(|i| draft_dists[i][draft_tokens[i] as usize])
            .collect();
        let target_chosen_probs: Vec<f32> = (0..k_eff)
            .map(|i| target_dists[i][draft_tokens[i] as usize])
            .collect();

        summary.speculative_traces.push(SpeculativeTrace {
            req_id,
            child_idx: child.child_idx,
            lookahead_k: k_eff,
            draft_tokens: draft_tokens.clone(),
            draft_probs: draft_chosen_probs,
            target_probs: target_chosen_probs,
            accepted_mask: outcome.accepted_mask.clone(),
            bonus_or_residual_token: outcome.final_token,
            accepted_count: outcome.accepted_draft_count,
        });

        if let (Some(gm), Some(Some(mask0))) = (&child.grammar, step_masks.first()) {
            let first_chosen = if outcome.accepted_draft_count > 0 {
                draft_tokens[0]
            } else {
                outcome.final_token
            };
            summary.grammar_xrays.push(Self::capture_grammar_xray(
                req_id,
                child.child_idx,
                gm.state(),
                gm.cache().dfa().num_states(),
                gm.is_accepting(),
                mask0,
                &target_logits[..vocab_sz],
                first_chosen,
                Vec::new(),
            ));
        }

        // 5. Emit accepted draft tokens (`0 .. accepted_draft_count`) + `final_token` (stopping early on EOS/max_new)
        let mut new_tokens = Vec::with_capacity(outcome.accepted_draft_count + 1);
        let mut sampled_infos = Vec::with_capacity(outcome.accepted_draft_count + 1);
        let mut accepted_emitted_drafts = 0usize;

        for i in 0..outcome.accepted_draft_count {
            let dt = draft_tokens[i];
            let p_vec = &target_dists[i];
            let info = SampledToken {
                token_id: dt,
                logprob: p_vec[dt as usize].max(1e-38).ln(),
                top_logprobs: collect_top_logprobs(p_vec, sampling.top_logprobs),
            };
            Self::commit_emitted_token(
                child,
                info,
                &sampling,
                max_new,
                eos_id,
                self.tokenizer.as_deref(),
                &mut new_tokens,
                &mut sampled_infos,
            );
            if child.finish_reason.is_some() {
                break;
            }
            accepted_emitted_drafts += 1;
        }

        if child.finish_reason.is_none() {
            let final_idx = outcome.accepted_draft_count;
            let p_vec = &target_dists[final_idx];
            let ft = outcome.final_token;
            let info = SampledToken {
                token_id: ft,
                logprob: p_vec[ft as usize].max(1e-38).ln(),
                top_logprobs: collect_top_logprobs(p_vec, sampling.top_logprobs),
            };
            Self::commit_emitted_token(
                child,
                info,
                &sampling,
                max_new,
                eos_id,
                self.tokenizer.as_deref(),
                &mut new_tokens,
                &mut sampled_infos,
            );
        }

        // Exact number of tokens in `child.table` whose K/V in `KvSlab` matches `child.full_tokens`
        // is `m + 1 + accepted_emitted_drafts` (positions `0 ..= m + accepted_emitted_drafts`).
        let valid_kv_in_slab = (m + 1 + accepted_emitted_drafts).min(child.full_tokens.len());
        let desired_kv_len = if child.finish_reason.is_some() {
            valid_kv_in_slab
        } else {
            child.full_tokens.len() - 1
        };

        self.allocator
            .truncate_table(&mut child.table, desired_kv_len);
        if let Some(dt) = &mut child.draft_table {
            let d_keep = desired_kv_len.min(dt.num_tokens());
            d_rt.allocator.truncate_table(dt, d_keep);
        }

        summary.deltas.push(SequenceStepDelta {
            req_id,
            child_idx: child.child_idx,
            new_tokens,
            sampled_infos,
            finished: child.finish_reason,
        });
        Ok(())
    }

    /// Runs scheduler steps until all submitted requests complete, returning sorted `CompletedRequest`s.
    pub fn run_to_completion(&mut self) -> Result<Vec<CompletedRequest>, SchedError> {
        let mut all_done = Vec::new();
        while self.has_unfinished_work() {
            let mut s = self.step()?;
            all_done.append(&mut s.completed);
        }
        all_done.sort_by_key(|r| r.req_id);
        Ok(all_done)
    }
}
