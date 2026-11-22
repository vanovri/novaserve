use crate::api_types::{
    ActiveLaneTelemetry, ApiChoiceLogprobs, ApiTokenLogprobInfo, ApiTopLogprobEntry, ChatChoice,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatMessage,
    ChatStreamChoice, CompletionChoice, CompletionChunk, CompletionRequest, CompletionResponse,
    CompletionStreamChoice, ResponseFormatSpec, TelemetrySnapshot, UsageStats,
};
use novaserve_frontend::{
    compile_json_schema, BpeTokenizer, GrammarMaskCache, TokenStreamDecoder, VocabularyBuilder,
};
use novaserve_model::{LlamaConfig, LlamaModel, ModelFactory};
use novaserve_sampler::SamplingPipeline;
use novaserve_sched::{
    FinishReason, RequestParams, SampledToken, SamplingConfig, SchedulerConfig, SequenceStatus,
    ServingEngine,
};
use novaserve_tensor::DType;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamEndpointKind {
    Chat,
    Completion,
}

#[derive(Clone)]
pub struct BuiltSubmission {
    pub params: RequestParams,
    pub stop_strings: Vec<String>,
    pub return_logprobs: bool,
    pub has_json_schema: bool,
    pub model_name: String,
}

pub enum UnaryResponsePayload {
    Chat(ChatCompletionResponse),
    Completion(CompletionResponse),
}

pub enum ResponseMode {
    Unary {
        endpoint: StreamEndpointKind,
        reply_tx: oneshot::Sender<Result<UnaryResponsePayload, String>>,
    },
    Stream {
        endpoint: StreamEndpointKind,
        req_id_tx: oneshot::Sender<Result<u64, String>>,
        chunk_tx: tokio_mpsc::UnboundedSender<String>,
    },
}

pub enum EngineCommand {
    Submit {
        sub: Box<BuiltSubmission>,
        mode: ResponseMode,
    },
    Abort {
        req_id: u64,
    },
    ClearRadix {
        reply_tx: oneshot::Sender<(usize, usize)>,
    },
    Shutdown,
}

struct ChildTracker {
    decoder: TokenStreamDecoder,
    accum_text: String,
    logprobs: Vec<ApiTokenLogprobInfo>,
    stopped_by_string: bool,
    emitted_finish: bool,
    last_spec_burst: usize,
}

struct RequestTracker {
    req_id: u64,
    submitted_at: Instant,
    first_token_recorded: bool,
    last_token_at: Option<Instant>,
    created_unix: u64,
    model_name: String,
    stop_strings: Vec<String>,
    return_logprobs: bool,
    has_json_schema: bool,
    children: Vec<ChildTracker>,
    delivery: DeliveryTarget,
}

enum DeliveryTarget {
    Unary {
        endpoint: StreamEndpointKind,
        reply_tx: Option<oneshot::Sender<Result<UnaryResponsePayload, String>>>,
    },
    Stream {
        endpoint: StreamEndpointKind,
        chunk_tx: tokio_mpsc::UnboundedSender<String>,
        sent_role_header: Vec<bool>,
    },
}

/// Cloneable thread-safe handle to the background `ServingEngine` actor thread.
#[derive(Clone)]
pub struct EngineActorHandle {
    cmd_tx: Sender<EngineCommand>,
    telemetry: Arc<RwLock<TelemetrySnapshot>>,
    tokenizer: Arc<BpeTokenizer>,
    schema_cache: Arc<RwLock<HashMap<String, Arc<GrammarMaskCache>>>>,
    model_name: String,
}

pub struct EngineActorGuard {
    pub handle: EngineActorHandle,
    join: Option<JoinHandle<()>>,
}

impl Drop for EngineActorGuard {
    fn drop(&mut self) {
        let _ = self.handle.cmd_tx.send(EngineCommand::Shutdown);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl EngineActorHandle {
    pub fn spawn(
        model_name: impl Into<String>,
        target_model: LlamaModel,
        draft_model: Option<LlamaModel>,
        tokenizer: Arc<BpeTokenizer>,
        sched_cfg: SchedulerConfig,
    ) -> Result<EngineActorGuard, String> {
        let model_name_str = model_name.into();
        let vocab_sz = target_model.config().vocab_size;
        let sampler = Arc::new(SamplingPipeline::new(vocab_sz));
        let engine = ServingEngine::with_sampler(
            target_model,
            draft_model,
            Some(Arc::clone(&tokenizer)),
            sched_cfg,
            sampler,
        )
        .map_err(|e| e.to_string())?;

        let initial_snap = build_initial_snapshot(&model_name_str, &engine);
        let telemetry = Arc::new(RwLock::new(initial_snap));
        let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>();

        let handle = Self {
            cmd_tx,
            telemetry: Arc::clone(&telemetry),
            tokenizer,
            schema_cache: Arc::new(RwLock::new(HashMap::new())),
            model_name: model_name_str.clone(),
        };

        let actor_telemetry = Arc::clone(&telemetry);
        let join = thread::Builder::new()
            .name("novaserve-engine-actor".into())
            .spawn(move || {
                run_engine_actor_loop(model_name_str, engine, cmd_rx, actor_telemetry);
            })
            .map_err(|e| format!("failed to spawn engine actor thread: {e}"))?;

        Ok(EngineActorGuard {
            handle,
            join: Some(join),
        })
    }

    /// Spawns a self-contained demo/test engine backed by deterministic `ModelFactory` weights.
    pub fn spawn_demo(
        dtype: DType,
        num_kv_blocks: usize,
        lookahead_k: usize,
    ) -> Result<EngineActorGuard, String> {
        let vocab_size = 512;
        let tokenizer =
            Arc::new(VocabularyBuilder::build_synthetic(vocab_size).map_err(|e| e.to_string())?);
        let target_cfg = LlamaConfig::micro_2m(vocab_size, dtype);
        let (target_model, draft_opt) = if lookahead_k > 0 {
            let draft_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);
            let (tm, dm) = ModelFactory::build_correlated_pair(target_cfg, draft_cfg, 2026_0808)
                .map_err(|e| e.to_string())?;
            (tm, Some(dm))
        } else {
            let tm = ModelFactory::build_random_model(target_cfg, 2026_0808)
                .map_err(|e| e.to_string())?;
            (tm, None)
        };

        let sched_cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 64,
            num_physical_blocks: num_kv_blocks.max(32),
            num_swap_blocks: num_kv_blocks.max(64),
            speculative_lookahead: lookahead_k,
            adaptive_speculative: false,
            ..SchedulerConfig::default()
        };

        Self::spawn(
            "novaserve-micro-2m",
            target_model,
            draft_opt,
            tokenizer,
            sched_cfg,
        )
    }

    #[inline]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[inline]
    pub fn tokenizer(&self) -> &Arc<BpeTokenizer> {
        &self.tokenizer
    }

    pub fn snapshot_telemetry(&self) -> TelemetrySnapshot {
        self.telemetry.read().clone()
    }

    pub fn abort_request(&self, req_id: u64) {
        let _ = self.cmd_tx.send(EngineCommand::Abort { req_id });
    }

    pub async fn clear_radix_cache(&self) -> Result<(usize, usize), String> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::ClearRadix { reply_tx: tx })
            .map_err(|_| "engine actor channel closed".to_string())?;
        rx.await
            .map_err(|_| "engine actor dropped reply".to_string())
    }

    pub fn compile_or_get_grammar(
        &self,
        spec: Option<&ResponseFormatSpec>,
    ) -> Result<Option<Arc<GrammarMaskCache>>, String> {
        let Some(rf) = spec else {
            return Ok(None);
        };
        if rf.kind.is_empty() || rf.kind == "text" {
            return Ok(None);
        }
        if rf.kind == "json_object" {
            let generic_obj = serde_json::json!({ "type": "object" });
            return self.get_cached_schema(&generic_obj);
        }
        if rf.kind != "json_schema" {
            return Err(format!("unsupported response_format.type '{}'", rf.kind));
        }
        let schema_val = rf
            .extract_schema()
            .ok_or_else(|| "missing json_schema definition in response_format".to_string())?;
        self.get_cached_schema(schema_val)
    }

    fn get_cached_schema(
        &self,
        schema_val: &serde_json::Value,
    ) -> Result<Option<Arc<GrammarMaskCache>>, String> {
        let key = serde_json::to_string(schema_val).map_err(|e| e.to_string())?;
        if let Some(existing) = self.schema_cache.read().get(&key) {
            return Ok(Some(Arc::clone(existing)));
        }
        let dfa =
            compile_json_schema(schema_val).map_err(|e| format!("invalid json_schema: {e}"))?;
        let cache = Arc::new(GrammarMaskCache::new(Arc::new(dfa), &self.tokenizer));
        self.schema_cache.write().insert(key, Arc::clone(&cache));
        Ok(Some(cache))
    }

    pub fn build_chat_submission(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<BuiltSubmission, String> {
        if req.messages.is_empty() {
            return Err("messages array must not be empty".into());
        }
        if req.n == 0 || req.n > 16 {
            return Err(format!("n must be in 1..=16, got {}", req.n));
        }
        let mut formatted = String::from("<|begin_of_text|>");
        for msg in &req.messages {
            formatted.push_str("<|im_start|>");
            formatted.push_str(&msg.role);
            formatted.push('\n');
            formatted.push_str(&msg.content);
            formatted.push_str("<|im_end|>\n");
        }
        formatted.push_str("<|im_start|>assistant\n");

        let mut prompt_tokens = self.tokenizer.encode(&formatted);
        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token_id().unwrap_or(0));
        }

        let grammar = self.compile_or_get_grammar(req.response_format.as_ref())?;
        let has_json_schema = grammar.is_some();
        let stop_strings = req
            .stop
            .clone()
            .map(|s| s.into_vec())
            .unwrap_or_default()
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        let sampling = build_sampling_config(
            &self.tokenizer,
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.repetition_penalty,
            req.frequency_penalty,
            req.presence_penalty,
            req.seed,
            &stop_strings,
            &req.logit_bias,
            req.logprobs,
            req.top_logprobs,
        )?;

        Ok(BuiltSubmission {
            params: RequestParams {
                prompt_tokens,
                max_new_tokens: req.max_tokens.clamp(1, 2048),
                n_parallel: req.n,
                sampling,
                grammar,
                use_speculative: req.use_speculative,
            },
            stop_strings,
            return_logprobs: req.logprobs,
            has_json_schema,
            model_name: if req.model.is_empty() {
                self.model_name.clone()
            } else {
                req.model.clone()
            },
        })
    }

    pub fn build_completion_submission(
        &self,
        req: &CompletionRequest,
    ) -> Result<BuiltSubmission, String> {
        if req.prompt.is_empty() {
            return Err("prompt must not be empty".into());
        }
        if req.n == 0 || req.n > 16 {
            return Err(format!("n must be in 1..=16, got {}", req.n));
        }
        let mut prompt_tokens = Vec::new();
        if let Some(bos) = self.tokenizer.bos_token_id() {
            prompt_tokens.push(bos);
        }
        prompt_tokens.extend(self.tokenizer.encode(&req.prompt));
        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token_id().unwrap_or(0));
        }

        let grammar = self.compile_or_get_grammar(req.response_format.as_ref())?;
        let has_json_schema = grammar.is_some();
        let stop_strings = req
            .stop
            .clone()
            .map(|s| s.into_vec())
            .unwrap_or_default()
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        let sampling = build_sampling_config(
            &self.tokenizer,
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.repetition_penalty,
            req.frequency_penalty,
            req.presence_penalty,
            req.seed,
            &stop_strings,
            &req.logit_bias,
            req.logprobs,
            req.top_logprobs,
        )?;

        Ok(BuiltSubmission {
            params: RequestParams {
                prompt_tokens,
                max_new_tokens: req.max_tokens.clamp(1, 2048),
                n_parallel: req.n,
                sampling,
                grammar,
                use_speculative: req.use_speculative,
            },
            stop_strings,
            return_logprobs: req.logprobs,
            has_json_schema,
            model_name: if req.model.is_empty() {
                self.model_name.clone()
            } else {
                req.model.clone()
            },
        })
    }

    pub async fn execute_unary(
        &self,
        sub: BuiltSubmission,
        endpoint: StreamEndpointKind,
    ) -> Result<UnaryResponsePayload, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Submit {
                sub: Box::new(sub),
                mode: ResponseMode::Unary { endpoint, reply_tx },
            })
            .map_err(|_| "engine actor channel closed".to_string())?;
        reply_rx
            .await
            .map_err(|_| "engine actor dropped unary response channel".to_string())?
    }

    pub async fn execute_stream(
        &self,
        sub: BuiltSubmission,
        endpoint: StreamEndpointKind,
    ) -> Result<(u64, tokio_mpsc::UnboundedReceiver<String>), String> {
        let (req_id_tx, req_id_rx) = oneshot::channel();
        let (chunk_tx, chunk_rx) = tokio_mpsc::unbounded_channel();
        self.cmd_tx
            .send(EngineCommand::Submit {
                sub: Box::new(sub),
                mode: ResponseMode::Stream {
                    endpoint,
                    req_id_tx,
                    chunk_tx,
                },
            })
            .map_err(|_| "engine actor channel closed".to_string())?;
        let req_id = req_id_rx
            .await
            .map_err(|_| "engine actor dropped stream registration".to_string())??;
        Ok((req_id, chunk_rx))
    }
}

#[allow(clippy::too_many_arguments)]
fn build_sampling_config(
    tokenizer: &BpeTokenizer,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    seed: Option<u64>,
    stop_strings: &[String],
    logit_bias_map: &BTreeMap<String, f32>,
    logprobs: bool,
    top_logprobs: Option<usize>,
) -> Result<SamplingConfig, String> {
    let mut stop_token_ids = Vec::new();
    for s in stop_strings {
        let enc = tokenizer.encode(s);
        if enc.len() == 1 {
            stop_token_ids.push(enc[0]);
        }
    }

    let mut logit_bias = Vec::with_capacity(logit_bias_map.len());
    for (k, &v) in logit_bias_map {
        let tid: u32 = k
            .parse()
            .map_err(|_| format!("invalid token ID '{}' in logit_bias", k))?;
        if !v.is_finite() {
            return Err(format!("non-finite bias for token {tid}"));
        }
        logit_bias.push((tid, v));
    }

    let top_lp = if logprobs {
        top_logprobs.unwrap_or(5).clamp(1, 20)
    } else {
        0
    };

    let cfg = SamplingConfig {
        temperature: temperature.unwrap_or(0.0),
        top_k,
        top_p: top_p.unwrap_or(1.0),
        min_p: min_p.unwrap_or(0.0),
        repetition_penalty: if repetition_penalty <= 0.0 {
            1.0
        } else {
            repetition_penalty
        },
        frequency_penalty,
        presence_penalty,
        seed: seed.unwrap_or(42),
        stop_token_ids,
        ignore_eos: false,
        logit_bias,
        top_logprobs: top_lp,
    };

    SamplingPipeline::validate_params(&cfg).map_err(|e| e.to_string())?;
    Ok(cfg)
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn format_finish_reason(fr: FinishReason, stopped_by_str: bool) -> String {
    if stopped_by_str {
        return "stop".into();
    }
    match fr {
        FinishReason::Stop => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::Aborted => "abort".into(),
    }
}

fn convert_sampled_logprob(tokenizer: &BpeTokenizer, st: &SampledToken) -> ApiTokenLogprobInfo {
    let tok_str = String::from_utf8_lossy(tokenizer.decode_raw_bytes(st.token_id)).into_owned();
    let top = st
        .top_logprobs
        .iter()
        .map(|e| ApiTopLogprobEntry {
            token: String::from_utf8_lossy(tokenizer.decode_raw_bytes(e.token_id)).into_owned(),
            token_id: e.token_id,
            logprob: e.logprob,
        })
        .collect();
    ApiTokenLogprobInfo {
        token: tok_str,
        token_id: st.token_id,
        logprob: st.logprob,
        top_logprobs: top,
    }
}

fn build_initial_snapshot(model_name: &str, engine: &ServingEngine) -> TelemetrySnapshot {
    TelemetrySnapshot {
        step_index: 0,
        model_name: model_name.to_string(),
        total_kv_blocks: engine.allocator().total_blocks(),
        free_kv_blocks: engine.allocator().free_blocks(),
        radix_cached_blocks: engine.radix_cache().cached_blocks_count(),
        total_swap_blocks: engine.swap_engine().total_swap_blocks(),
        used_swap_blocks: engine.swap_engine().total_swap_blocks()
            - engine.swap_engine().free_swap_blocks(),
        external_fragmentation_pct: 0.0,
        waiting_groups: 0,
        running_groups: 0,
        swapped_groups: 0,
        completed_requests_total: 0,
        aborted_requests_total: 0,
        tokens_generated_total: 0,
        tokens_per_sec: 0.0,
        ttft_p50_ms: 0.0,
        ttft_p99_ms: 0.0,
        itl_p50_ms: 0.0,
        itl_p99_ms: 0.0,
        radix_hit_rate_pct: 0.0,
        radix_evicted_blocks: 0,
        radix_hit_sparkline: Vec::new(),
        speculative_alpha_pct: 0.0,
        active_spec_k: engine.lookahead_controller().current_k(),
        blocks: engine.snapshot_kv_blocks(),
        radix_tree: engine.snapshot_radix_tree(),
        active_lanes: Vec::new(),
    }
}

fn percentile_sorted(samples: &VecDeque<f64>, quantile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = samples.iter().copied().collect();
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = (((v.len() - 1) as f64) * quantile).round() as usize;
    v[idx.min(v.len() - 1)]
}

struct ActorMetrics {
    started_at: Instant,
    completed_requests_total: u64,
    aborted_requests_total: u64,
    tokens_generated_total: u64,
    token_rate_window: VecDeque<(Instant, usize)>,
    ttft_ms_window: VecDeque<f64>,
    itl_ms_window: VecDeque<f64>,
    radix_sparkline: VecDeque<u64>,
}

impl ActorMetrics {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            completed_requests_total: 0,
            aborted_requests_total: 0,
            tokens_generated_total: 0,
            token_rate_window: VecDeque::with_capacity(512),
            ttft_ms_window: VecDeque::with_capacity(2048),
            itl_ms_window: VecDeque::with_capacity(4096),
            radix_sparkline: VecDeque::with_capacity(64),
        }
    }

    fn record_ttft(&mut self, ms: f64) {
        if self.ttft_ms_window.len() >= 2048 {
            self.ttft_ms_window.pop_front();
        }
        self.ttft_ms_window.push_back(ms);
    }

    fn record_itl(&mut self, ms: f64) {
        if self.itl_ms_window.len() >= 4096 {
            self.itl_ms_window.pop_front();
        }
        self.itl_ms_window.push_back(ms);
    }

    fn record_tokens(&mut self, now: Instant, count: usize) {
        if count == 0 {
            return;
        }
        self.tokens_generated_total += count as u64;
        self.token_rate_window.push_back((now, count));
        while let Some(&(ts, _)) = self.token_rate_window.front() {
            if now.duration_since(ts).as_secs_f64() > 5.0 {
                self.token_rate_window.pop_front();
            } else {
                break;
            }
        }
    }

    fn current_tok_per_sec(&self, now: Instant) -> f64 {
        let mut recent_toks = 0usize;
        let mut oldest = now;
        for &(ts, c) in &self.token_rate_window {
            if now.duration_since(ts).as_secs_f64() <= 5.0 {
                recent_toks += c;
                if ts < oldest {
                    oldest = ts;
                }
            }
        }
        let span = now.duration_since(oldest).as_secs_f64();
        if recent_toks > 0 && span >= 0.005 {
            (recent_toks as f64) / span
        } else {
            let up = now.duration_since(self.started_at).as_secs_f64().max(0.001);
            (self.tokens_generated_total as f64) / up
        }
    }
}

fn run_engine_actor_loop(
    model_name: String,
    mut engine: ServingEngine,
    cmd_rx: Receiver<EngineCommand>,
    telemetry: Arc<RwLock<TelemetrySnapshot>>,
) {
    let tokenizer = Arc::clone(
        engine
            .tokenizer()
            .expect("engine initialized with tokenizer"),
    );
    let mut trackers: HashMap<u64, RequestTracker> = HashMap::new();
    let mut metrics = ActorMetrics::new();

    loop {
        // Block cleanly when no active requests exist; drain non-blockingly when busy!
        if !engine.has_unfinished_work() {
            match cmd_rx.recv() {
                Ok(cmd) => {
                    if !handle_actor_command(
                        cmd,
                        &mut engine,
                        &mut trackers,
                        &mut metrics,
                        &model_name,
                        &telemetry,
                    ) {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        while let Ok(cmd) = cmd_rx.try_recv() {
            if !handle_actor_command(
                cmd,
                &mut engine,
                &mut trackers,
                &mut metrics,
                &model_name,
                &telemetry,
            ) {
                return;
            }
        }

        if !engine.has_unfinished_work() {
            publish_telemetry(&model_name, &engine, &trackers, &mut metrics, &telemetry);
            continue;
        }

        let step_summary = match engine.step() {
            Ok(s) => s,
            Err(err) => {
                // Fail active trackers gracefully if an unrecoverable step error occurs
                let err_msg = format!("engine step error: {err}");
                let active_ids: Vec<u64> = trackers.keys().copied().collect();
                for rid in active_ids {
                    let _ = engine.abort_request(rid);
                    if let Some(mut tr) = trackers.remove(&rid) {
                        if let DeliveryTarget::Unary { reply_tx, .. } = &mut tr.delivery {
                            if let Some(tx) = reply_tx.take() {
                                let _ = tx.send(Err(err_msg.clone()));
                            }
                        }
                    }
                }
                publish_telemetry(&model_name, &engine, &trackers, &mut metrics, &telemetry);
                continue;
            }
        };

        let now = Instant::now();
        let mut spec_burst_by_seq: HashMap<(u64, usize), usize> = HashMap::new();
        for st in &step_summary.speculative_traces {
            spec_burst_by_seq.insert((st.req_id, st.child_idx), st.accepted_count + 1);
        }

        let mut disconnected_reqs: Vec<u64> = Vec::new();
        let mut string_stopped_reqs: Vec<u64> = Vec::new();

        for delta in &step_summary.deltas {
            let Some(tr) = trackers.get_mut(&delta.req_id) else {
                continue;
            };
            let n_new = delta.new_tokens.len();
            if n_new > 0 {
                metrics.record_tokens(now, n_new);
                if !tr.first_token_recorded {
                    tr.first_token_recorded = true;
                    metrics.record_ttft(now.duration_since(tr.submitted_at).as_secs_f64() * 1000.0);
                    tr.last_token_at = Some(now);
                } else if let Some(prev) = tr.last_token_at {
                    let itl = (now.duration_since(prev).as_secs_f64() * 1000.0) / (n_new as f64);
                    metrics.record_itl(itl);
                    tr.last_token_at = Some(now);
                }
            }

            if delta.child_idx >= tr.children.len() {
                continue;
            }
            let child_tr = &mut tr.children[delta.child_idx];
            child_tr.last_spec_burst = spec_burst_by_seq
                .get(&(delta.req_id, delta.child_idx))
                .copied()
                .unwrap_or(n_new);

            if tr.return_logprobs {
                for si in &delta.sampled_infos {
                    child_tr
                        .logprobs
                        .push(convert_sampled_logprob(&tokenizer, si));
                }
            }

            let mut newly_decoded = String::new();
            if !child_tr.stopped_by_string {
                for &tid in &delta.new_tokens {
                    let piece = child_tr.decoder.push_token(&tokenizer, tid);
                    newly_decoded.push_str(&piece);
                }
                if delta.finished.is_some() {
                    newly_decoded.push_str(&child_tr.decoder.flush());
                }
                child_tr.accum_text.push_str(&newly_decoded);

                // Check multi-token string stop sequences
                for stop_s in &tr.stop_strings {
                    if let Some(idx) = child_tr.accum_text.find(stop_s.as_str()) {
                        let prev_emitted_len = child_tr
                            .accum_text
                            .len()
                            .saturating_sub(newly_decoded.len());
                        child_tr.accum_text.truncate(idx);
                        newly_decoded = if idx > prev_emitted_len {
                            child_tr.accum_text[prev_emitted_len..idx].to_string()
                        } else {
                            String::new()
                        };
                        child_tr.stopped_by_string = true;
                        break;
                    }
                }
            }

            if let DeliveryTarget::Stream {
                endpoint,
                chunk_tx,
                sent_role_header,
            } = &mut tr.delivery
            {
                let chunk_id = format!("chatcmpl-{}", tr.req_id);
                let mut stream_dead = false;

                if *endpoint == StreamEndpointKind::Chat
                    && delta.child_idx < sent_role_header.len()
                    && !sent_role_header[delta.child_idx]
                {
                    sent_role_header[delta.child_idx] = true;
                    let role_chunk = ChatCompletionChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created: tr.created_unix,
                        model: tr.model_name.clone(),
                        choices: vec![ChatStreamChoice {
                            index: delta.child_idx,
                            delta: ChatDelta {
                                role: Some("assistant".into()),
                                content: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    if send_sse_json(chunk_tx, &role_chunk).is_err() {
                        stream_dead = true;
                    }
                }

                if !stream_dead && !newly_decoded.is_empty() && !child_tr.emitted_finish {
                    let send_res = match *endpoint {
                        StreamEndpointKind::Chat => {
                            let c = ChatCompletionChunk {
                                id: chunk_id.clone(),
                                object: "chat.completion.chunk".into(),
                                created: tr.created_unix,
                                model: tr.model_name.clone(),
                                choices: vec![ChatStreamChoice {
                                    index: delta.child_idx,
                                    delta: ChatDelta {
                                        role: None,
                                        content: Some(newly_decoded),
                                    },
                                    finish_reason: None,
                                }],
                            };
                            send_sse_json(chunk_tx, &c)
                        }
                        StreamEndpointKind::Completion => {
                            let c = CompletionChunk {
                                id: format!("cmpl-{}", tr.req_id),
                                object: "text_completion".into(),
                                created: tr.created_unix,
                                model: tr.model_name.clone(),
                                choices: vec![CompletionStreamChoice {
                                    index: delta.child_idx,
                                    text: newly_decoded,
                                    finish_reason: None,
                                }],
                            };
                            send_sse_json(chunk_tx, &c)
                        }
                    };
                    if send_res.is_err() {
                        stream_dead = true;
                    }
                }

                let fin_opt = if child_tr.stopped_by_string {
                    Some(FinishReason::Stop)
                } else {
                    delta.finished
                };
                if !stream_dead && !child_tr.emitted_finish {
                    if let Some(fr) = fin_opt {
                        child_tr.emitted_finish = true;
                        let reason_str = format_finish_reason(fr, child_tr.stopped_by_string);
                        let send_res = match *endpoint {
                            StreamEndpointKind::Chat => {
                                let c = ChatCompletionChunk {
                                    id: chunk_id,
                                    object: "chat.completion.chunk".into(),
                                    created: tr.created_unix,
                                    model: tr.model_name.clone(),
                                    choices: vec![ChatStreamChoice {
                                        index: delta.child_idx,
                                        delta: ChatDelta {
                                            role: None,
                                            content: None,
                                        },
                                        finish_reason: Some(reason_str),
                                    }],
                                };
                                send_sse_json(chunk_tx, &c)
                            }
                            StreamEndpointKind::Completion => {
                                let c = CompletionChunk {
                                    id: format!("cmpl-{}", tr.req_id),
                                    object: "text_completion".into(),
                                    created: tr.created_unix,
                                    model: tr.model_name.clone(),
                                    choices: vec![CompletionStreamChoice {
                                        index: delta.child_idx,
                                        text: String::new(),
                                        finish_reason: Some(reason_str),
                                    }],
                                };
                                send_sse_json(chunk_tx, &c)
                            }
                        };
                        if send_res.is_err() {
                            stream_dead = true;
                        }
                    }
                }

                if stream_dead {
                    disconnected_reqs.push(delta.req_id);
                }
            }

            if tr.children.iter().all(|c| c.stopped_by_string) && delta.finished.is_none() {
                string_stopped_reqs.push(delta.req_id);
            }
        }

        // Cancel any streams whose TCP client dropped mid-step!
        for rid in disconnected_reqs {
            if engine.abort_request(rid) {
                metrics.aborted_requests_total += 1;
            }
            trackers.remove(&rid);
        }

        // Finalize requests where all children hit a multi-token string stop early
        for rid in string_stopped_reqs {
            let _ = engine.abort_request(rid);
            if let Some(tr) = trackers.remove(&rid) {
                metrics.completed_requests_total += 1;
                finalize_string_stopped_tracker(tr);
            }
        }

        // Finalize completed requests from `step_summary.completed`
        for comp in step_summary.completed {
            let _ = engine.take_completed(comp.req_id);
            let Some(mut tr) = trackers.remove(&comp.req_id) else {
                continue;
            };
            metrics.completed_requests_total += 1;

            let completion_tokens: usize = comp.outputs.iter().map(|o| o.token_ids.len()).sum();
            let usage = UsageStats {
                prompt_tokens: comp.prompt_len,
                completion_tokens,
                total_tokens: comp.prompt_len + completion_tokens,
                cached_prefix_tokens: comp.cached_prefix_tokens,
            };

            match &mut tr.delivery {
                DeliveryTarget::Unary { endpoint, reply_tx } => {
                    for out in &comp.outputs {
                        if out.child_idx < tr.children.len() {
                            let ctr = &mut tr.children[out.child_idx];
                            if !ctr.stopped_by_string {
                                ctr.accum_text.push_str(&ctr.decoder.flush());
                            }
                        }
                    }

                    let payload = match *endpoint {
                        StreamEndpointKind::Chat => {
                            let choices = comp
                                .outputs
                                .iter()
                                .map(|out| {
                                    let ctr = &tr.children[out.child_idx];
                                    ChatChoice {
                                        index: out.child_idx,
                                        message: ChatMessage {
                                            role: "assistant".into(),
                                            content: ctr.accum_text.clone(),
                                        },
                                        finish_reason: format_finish_reason(
                                            out.finish_reason,
                                            ctr.stopped_by_string,
                                        ),
                                        logprobs: if tr.return_logprobs {
                                            Some(ApiChoiceLogprobs {
                                                content: ctr.logprobs.clone(),
                                            })
                                        } else {
                                            None
                                        },
                                    }
                                })
                                .collect();
                            UnaryResponsePayload::Chat(ChatCompletionResponse {
                                id: format!("chatcmpl-{}", comp.req_id),
                                object: "chat.completion".into(),
                                created: tr.created_unix,
                                model: tr.model_name.clone(),
                                choices,
                                usage,
                            })
                        }
                        StreamEndpointKind::Completion => {
                            let choices = comp
                                .outputs
                                .iter()
                                .map(|out| {
                                    let ctr = &tr.children[out.child_idx];
                                    CompletionChoice {
                                        index: out.child_idx,
                                        text: ctr.accum_text.clone(),
                                        finish_reason: format_finish_reason(
                                            out.finish_reason,
                                            ctr.stopped_by_string,
                                        ),
                                        logprobs: if tr.return_logprobs {
                                            Some(ApiChoiceLogprobs {
                                                content: ctr.logprobs.clone(),
                                            })
                                        } else {
                                            None
                                        },
                                    }
                                })
                                .collect();
                            UnaryResponsePayload::Completion(CompletionResponse {
                                id: format!("cmpl-{}", comp.req_id),
                                object: "text_completion".into(),
                                created: tr.created_unix,
                                model: tr.model_name.clone(),
                                choices,
                                usage,
                            })
                        }
                    };
                    if let Some(tx) = reply_tx.take() {
                        let _ = tx.send(Ok(payload));
                    }
                }
                DeliveryTarget::Stream { chunk_tx, .. } => {
                    let _ = chunk_tx.send("data: [DONE]\n\n".to_string());
                }
            }
        }

        publish_telemetry(&model_name, &engine, &trackers, &mut metrics, &telemetry);
    }
}

fn finalize_string_stopped_tracker(mut tr: RequestTracker) {
    match &mut tr.delivery {
        DeliveryTarget::Unary { endpoint, reply_tx } => {
            let comp_toks: usize = tr.children.iter().map(|c| c.logprobs.len().max(1)).sum();
            let usage = UsageStats {
                prompt_tokens: 0,
                completion_tokens: comp_toks,
                total_tokens: comp_toks,
                cached_prefix_tokens: 0,
            };
            let payload = match *endpoint {
                StreamEndpointKind::Chat => UnaryResponsePayload::Chat(ChatCompletionResponse {
                    id: format!("chatcmpl-{}", tr.req_id),
                    object: "chat.completion".into(),
                    created: tr.created_unix,
                    model: tr.model_name.clone(),
                    choices: tr
                        .children
                        .iter()
                        .enumerate()
                        .map(|(idx, ctr)| ChatChoice {
                            index: idx,
                            message: ChatMessage {
                                role: "assistant".into(),
                                content: ctr.accum_text.clone(),
                            },
                            finish_reason: "stop".into(),
                            logprobs: if tr.return_logprobs {
                                Some(ApiChoiceLogprobs {
                                    content: ctr.logprobs.clone(),
                                })
                            } else {
                                None
                            },
                        })
                        .collect(),
                    usage,
                }),
                StreamEndpointKind::Completion => {
                    UnaryResponsePayload::Completion(CompletionResponse {
                        id: format!("cmpl-{}", tr.req_id),
                        object: "text_completion".into(),
                        created: tr.created_unix,
                        model: tr.model_name.clone(),
                        choices: tr
                            .children
                            .iter()
                            .enumerate()
                            .map(|(idx, ctr)| CompletionChoice {
                                index: idx,
                                text: ctr.accum_text.clone(),
                                finish_reason: "stop".into(),
                                logprobs: if tr.return_logprobs {
                                    Some(ApiChoiceLogprobs {
                                        content: ctr.logprobs.clone(),
                                    })
                                } else {
                                    None
                                },
                            })
                            .collect(),
                        usage,
                    })
                }
            };
            if let Some(tx) = reply_tx.take() {
                let _ = tx.send(Ok(payload));
            }
        }
        DeliveryTarget::Stream { chunk_tx, .. } => {
            let _ = chunk_tx.send("data: [DONE]\n\n".to_string());
        }
    }
}

fn send_sse_json<T: serde::Serialize>(
    tx: &tokio_mpsc::UnboundedSender<String>,
    val: &T,
) -> Result<(), ()> {
    let json = serde_json::to_string(val).map_err(|_| ())?;
    tx.send(format!("data: {json}\n\n")).map_err(|_| ())
}

fn handle_actor_command(
    cmd: EngineCommand,
    engine: &mut ServingEngine,
    trackers: &mut HashMap<u64, RequestTracker>,
    metrics: &mut ActorMetrics,
    model_name: &str,
    telemetry: &Arc<RwLock<TelemetrySnapshot>>,
) -> bool {
    match cmd {
        EngineCommand::Submit { sub, mode } => {
            let n_parallel = sub.params.n_parallel.max(1);
            match engine.submit_request(sub.params) {
                Ok(req_id) => {
                    let children = (0..n_parallel)
                        .map(|_| ChildTracker {
                            decoder: TokenStreamDecoder::new(true),
                            accum_text: String::new(),
                            logprobs: Vec::new(),
                            stopped_by_string: false,
                            emitted_finish: false,
                            last_spec_burst: 0,
                        })
                        .collect();

                    let delivery = match mode {
                        ResponseMode::Unary { endpoint, reply_tx } => DeliveryTarget::Unary {
                            endpoint,
                            reply_tx: Some(reply_tx),
                        },
                        ResponseMode::Stream {
                            endpoint,
                            req_id_tx,
                            chunk_tx,
                        } => {
                            if req_id_tx.send(Ok(req_id)).is_err() {
                                let _ = engine.abort_request(req_id);
                                metrics.aborted_requests_total += 1;
                                return true;
                            }
                            DeliveryTarget::Stream {
                                endpoint,
                                chunk_tx,
                                sent_role_header: vec![false; n_parallel],
                            }
                        }
                    };

                    trackers.insert(
                        req_id,
                        RequestTracker {
                            req_id,
                            submitted_at: Instant::now(),
                            first_token_recorded: false,
                            last_token_at: None,
                            created_unix: unix_timestamp_secs(),
                            model_name: sub.model_name,
                            stop_strings: sub.stop_strings,
                            return_logprobs: sub.return_logprobs,
                            has_json_schema: sub.has_json_schema,
                            children,
                            delivery,
                        },
                    );
                }
                Err(e) => {
                    let msg = e.to_string();
                    match mode {
                        ResponseMode::Unary { reply_tx, .. } => {
                            let _ = reply_tx.send(Err(msg));
                        }
                        ResponseMode::Stream { req_id_tx, .. } => {
                            let _ = req_id_tx.send(Err(msg));
                        }
                    }
                }
            }
            true
        }
        EngineCommand::Abort { req_id } => {
            if engine.abort_request(req_id) {
                metrics.aborted_requests_total += 1;
            }
            trackers.remove(&req_id);
            publish_telemetry(model_name, engine, trackers, metrics, telemetry);
            true
        }
        EngineCommand::ClearRadix { reply_tx } => {
            engine.clear_radix_cache();
            let free = engine.allocator().free_blocks();
            let total = engine.allocator().total_blocks();
            publish_telemetry(model_name, engine, trackers, metrics, telemetry);
            let _ = reply_tx.send((free, total));
            true
        }
        EngineCommand::Shutdown => false,
    }
}

fn publish_telemetry(
    model_name: &str,
    engine: &ServingEngine,
    trackers: &HashMap<u64, RequestTracker>,
    metrics: &mut ActorMetrics,
    telemetry: &Arc<RwLock<TelemetrySnapshot>>,
) {
    let now = Instant::now();
    let (waiting_groups, running_groups, swapped_groups) = engine.queue_depths();
    let hit_pct = engine.radix_cache().hit_rate() * 100.0;
    if metrics.radix_sparkline.len() >= 64 {
        metrics.radix_sparkline.pop_front();
    }
    metrics
        .radix_sparkline
        .push_back(hit_pct.round().clamp(0.0, 100.0) as u64);

    let seqs = engine.inspect_active_sequences();
    let mut active_lanes = Vec::with_capacity(seqs.len());
    for s in seqs {
        let (preview, burst, has_schema) = trackers
            .get(&s.req_id)
            .and_then(|tr| {
                tr.children.get(s.child_idx).map(|ctr| {
                    let chars: Vec<char> = ctr.accum_text.chars().collect();
                    let tail: String = if chars.len() > 36 {
                        chars[chars.len() - 36..].iter().collect()
                    } else {
                        ctr.accum_text.clone()
                    };
                    (
                        tail.replace('\n', "↵"),
                        ctr.last_spec_burst,
                        tr.has_json_schema,
                    )
                })
            })
            .unwrap_or_else(|| (String::new(), 0, s.has_grammar));

        let status_str = match s.status {
            SequenceStatus::Waiting => "waiting",
            SequenceStatus::Prefilling => "prefill",
            SequenceStatus::Decoding => "decode",
            SequenceStatus::Swapped => "swapped",
        };

        active_lanes.push(ActiveLaneTelemetry {
            req_id: s.req_id,
            child_idx: s.child_idx,
            status: status_str.into(),
            prompt_tokens: s.prompt_len,
            kv_tokens: s.kv_tokens,
            generated_tokens: s.generated_tokens.len(),
            max_tokens: s.max_new_tokens,
            text_preview: preview,
            last_spec_burst: burst,
            has_json_schema: has_schema,
            dfa_state: s.dfa_state,
            total_dfa_states: s.total_dfa_states,
        });
    }

    let snap = TelemetrySnapshot {
        step_index: engine.step_counter(),
        model_name: model_name.to_string(),
        total_kv_blocks: engine.allocator().total_blocks(),
        free_kv_blocks: engine.allocator().free_blocks(),
        radix_cached_blocks: engine.radix_cache().cached_blocks_count(),
        total_swap_blocks: engine.swap_engine().total_swap_blocks(),
        used_swap_blocks: engine.swap_engine().total_swap_blocks()
            - engine.swap_engine().free_swap_blocks(),
        external_fragmentation_pct: 0.0,
        waiting_groups,
        running_groups,
        swapped_groups,
        completed_requests_total: metrics.completed_requests_total,
        aborted_requests_total: metrics.aborted_requests_total,
        tokens_generated_total: metrics.tokens_generated_total,
        tokens_per_sec: metrics.current_tok_per_sec(now),
        ttft_p50_ms: percentile_sorted(&metrics.ttft_ms_window, 0.50),
        ttft_p99_ms: percentile_sorted(&metrics.ttft_ms_window, 0.99),
        itl_p50_ms: percentile_sorted(&metrics.itl_ms_window, 0.50),
        itl_p99_ms: percentile_sorted(&metrics.itl_ms_window, 0.99),
        radix_hit_rate_pct: hit_pct,
        radix_evicted_blocks: engine.radix_cache().total_evicted_blocks(),
        radix_hit_sparkline: metrics.radix_sparkline.iter().copied().collect(),
        speculative_alpha_pct: (engine.lookahead_controller().rolling_alpha() as f64) * 100.0,
        active_spec_k: engine.lookahead_controller().current_k(),
        blocks: engine.snapshot_kv_blocks(),
        radix_tree: engine.snapshot_radix_tree(),
        active_lanes,
    };

    *telemetry.write() = snap;
}
