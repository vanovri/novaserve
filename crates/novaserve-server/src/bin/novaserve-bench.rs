#![deny(unsafe_code)]

//! `novaserve-bench`: Real End-to-End Serving SLA & Systems Ablation Suite (`--release`).
//!
//! Measures and writes verified telemetry to `docs/benchmarks/serving_sla.md` and
//! `docs/benchmarks/serving_sla.json`:
//! 1. Kernel Quantization Speedup (`Tiny-16M`: `FP32` vs `Q8_0` vs `Q4_0`)
//! 2. Continuous Batching vs Naive Sequential Locking (`32` Poisson clients) + Chunked Prefill Jitter
//! 3. SGLang-Style Radix Prefix Cache Ablation (`512`-token Shared System Prompt)
//! 4. Leviathan Speculative Decoding Speedup (`K = 4`, Correlated Draft-500K + Tiny-16M Target)

use novaserve_frontend::VocabularyBuilder;
use novaserve_model::{LlamaConfig, LlamaModel, ModelFactory};
use novaserve_sampler::SamplingPipeline;
use novaserve_sched::{
    RequestParams, SamplerStrategy, SamplingConfig, SchedulerConfig, ServingEngine,
};
use novaserve_tensor::{simd, DType, SplitMix64};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
struct QuantBenchmarkRow {
    dtype: String,
    weight_megabytes: f64,
    compression_ratio_vs_fp32: f64,
    single_stream_tok_per_sec: f64,
    single_stream_tpot_ms: f64,
    batched_16_tok_per_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct BatchingBenchmarkRow {
    mode: String,
    clients: usize,
    aggregate_tok_per_sec: f64,
    ttft_p50_ms: f64,
    ttft_p99_ms: f64,
    itl_p50_ms: f64,
    itl_p99_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ChunkedJitterReport {
    unchunked_midflight_decode_itl_p99_ms: f64,
    chunked_64_midflight_decode_itl_p99_ms: f64,
    jitter_reduction_factor: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RadixAblationReport {
    shared_system_prompt_tokens: usize,
    requests: usize,
    radix_off_ttft_p50_ms: f64,
    radix_off_ttft_p99_ms: f64,
    radix_off_total_ms: f64,
    radix_on_ttft_p50_ms: f64,
    radix_on_ttft_p99_ms: f64,
    radix_on_total_ms: f64,
    ttft_p50_speedup_x: f64,
    total_wall_speedup_x: f64,
    kv_block_hit_rate_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
struct SpeculativeAblationReport {
    lookahead_k: usize,
    acceptance_alpha_pct: f64,
    mean_tokens_per_target_pass: f64,
    baseline_tok_per_sec: f64,
    baseline_tpot_ms: f64,
    speculative_tok_per_sec: f64,
    speculative_tpot_ms: f64,
    tpot_speedup_x: f64,
}

#[derive(Clone, Debug, Serialize)]
struct FullSlaReport {
    simd_backend: String,
    model_arch: String,
    quantization: Vec<QuantBenchmarkRow>,
    batching: Vec<BatchingBenchmarkRow>,
    chunked_prefill_jitter: ChunkedJitterReport,
    radix_ablation: RadixAblationReport,
    speculative_ablation: SpeculativeAblationReport,
}

fn percentile(vals: &[f64], q: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = (((sorted.len() - 1) as f64) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn estimate_gguf_weight_mb(model: &LlamaModel) -> f64 {
    let bytes = model.to_gguf_bytes(None).expect("gguf serialize").len();
    (bytes as f64) / (1024.0 * 1024.0)
}

fn bench_quantization(vocab_size: usize) -> Vec<QuantBenchmarkRow> {
    let tok = Arc::new(VocabularyBuilder::build_synthetic(vocab_size).unwrap());
    let sampler: Arc<dyn SamplerStrategy> = Arc::new(SamplingPipeline::new(vocab_size));
    let mut rows = Vec::new();
    let mut fp32_mb = 0.0f64;

    for (label, dt) in [
        ("FP32", DType::F32),
        ("Q8_0", DType::Q8_0),
        ("Q4_0", DType::Q4_0),
    ] {
        let cfg = LlamaConfig::tiny_16m(vocab_size, dt);
        let model_single = ModelFactory::build_random_model(cfg.clone(), 4242).unwrap();
        let mb = estimate_gguf_weight_mb(&model_single);
        if dt == DType::F32 {
            fp32_mb = mb;
        }

        // 1. Single-stream decode (`prompt = 32`, `gen = 48`)
        let mut eng1 = ServingEngine::with_sampler(
            model_single,
            None,
            Some(Arc::clone(&tok)),
            SchedulerConfig {
                max_num_batched_tokens: 256,
                max_num_seqs: 16,
                num_physical_blocks: 256,
                num_swap_blocks: 128,
                enable_radix_cache: false,
                ..SchedulerConfig::default()
            },
            Arc::clone(&sampler),
        )
        .unwrap();

        // Warmup
        let warm_p: Vec<u32> = (10..26).collect();
        eng1.submit_request(RequestParams::greedy(warm_p, 8))
            .unwrap();
        eng1.run_to_completion().unwrap();

        let prompt: Vec<u32> = (20..52).collect();
        let gen_tokens = 48usize;
        eng1.submit_request(RequestParams::greedy(prompt.clone(), gen_tokens))
            .unwrap();
        // Prefill step first so decode TPOT isolates pure autoregressive token generation
        let _ = eng1.step().unwrap();
        let t0 = Instant::now();
        while eng1.has_unfinished_work() {
            let _ = eng1.step().unwrap();
        }
        let decode_secs = t0.elapsed().as_secs_f64();
        let single_tps = ((gen_tokens - 1) as f64) / decode_secs;
        let single_tpot_ms = (decode_secs * 1000.0) / ((gen_tokens - 1) as f64);

        // 2. 16-sequence Continuous-Batched throughput (`16 x 32` generated tokens = `512` tokens)
        let model_batch = ModelFactory::build_random_model(cfg, 4242).unwrap();
        let mut eng_b = ServingEngine::with_sampler(
            model_batch,
            None,
            Some(Arc::clone(&tok)),
            SchedulerConfig {
                max_num_batched_tokens: 512,
                max_num_seqs: 32,
                num_physical_blocks: 512,
                num_swap_blocks: 128,
                enable_radix_cache: false,
                ..SchedulerConfig::default()
            },
            Arc::clone(&sampler),
        )
        .unwrap();

        let batch_n = 16usize;
        let batch_gen = 32usize;
        for i in 0..batch_n {
            let p: Vec<u32> = (0..24)
                .map(|j| ((30 + i * 7 + j) % (vocab_size - 10)) as u32)
                .collect();
            eng_b
                .submit_request(RequestParams::greedy(p, batch_gen))
                .unwrap();
        }
        let tb = Instant::now();
        eng_b.run_to_completion().unwrap();
        let batched_secs = tb.elapsed().as_secs_f64();
        let batched_tps = ((batch_n * batch_gen) as f64) / batched_secs;

        rows.push(QuantBenchmarkRow {
            dtype: label.into(),
            weight_megabytes: mb,
            compression_ratio_vs_fp32: fp32_mb / mb,
            single_stream_tok_per_sec: single_tps,
            single_stream_tpot_ms: single_tpot_ms,
            batched_16_tok_per_sec: batched_tps,
        });
    }

    rows
}

fn bench_continuous_batching_and_chunked_prefill(
    vocab_size: usize,
) -> (Vec<BatchingBenchmarkRow>, ChunkedJitterReport) {
    let tok = Arc::new(VocabularyBuilder::build_synthetic(vocab_size).unwrap());
    let sampler: Arc<dyn SamplerStrategy> = Arc::new(SamplingPipeline::new(vocab_size));
    let cfg = LlamaConfig::tiny_16m(vocab_size, DType::Q8_0);

    // Deterministic 32-client Poisson arrival trace (arrival offsets in ms)
    let mut rng = SplitMix64::new(2026_0508);
    let num_clients = 32usize;
    let gen_len = 24usize;
    let mut arrival_offsets_ms = Vec::with_capacity(num_clients);
    let mut cur_ms = 0.0f64;
    let mut prompts = Vec::with_capacity(num_clients);
    for i in 0..num_clients {
        if i > 0 {
            let u = rng.range_f32(0.05, 0.99) as f64;
            cur_ms += -u.ln() * 2.5;
        }
        arrival_offsets_ms.push(cur_ms);
        let p_len = 32 + ((rng.next_u32() as usize) % 48);
        let p: Vec<u32> = (0..p_len)
            .map(|j| ((16 + i * 11 + j * 3) % (vocab_size - 10)) as u32)
            .collect();
        prompts.push(p);
    }

    // Mode A: Naive Sequential Mutex-Locked Single-Sequence Execution (`max_num_seqs = 1`)
    // Mode B: Continuous Batching (`max_num_seqs = 32, max_num_batched_tokens = 2048`)
    let mut rows = Vec::new();
    for (mode_name, max_seqs, max_batch_toks) in [
        ("Sequential Locking (batch=1)", 1usize, 2048usize),
        ("Continuous Batching (32 seqs)", 32usize, 2048usize),
        ("Continuous + Chunked Prefill (B=64)", 32usize, 64usize),
    ] {
        let model = ModelFactory::build_random_model(cfg.clone(), 9001).unwrap();
        let mut eng = ServingEngine::with_sampler(
            model,
            None,
            Some(Arc::clone(&tok)),
            SchedulerConfig {
                max_num_batched_tokens: max_batch_toks,
                max_num_seqs: max_seqs,
                num_physical_blocks: 512,
                num_swap_blocks: 256,
                enable_radix_cache: false,
                ..SchedulerConfig::default()
            },
            Arc::clone(&sampler),
        )
        .unwrap();

        let start = Instant::now();
        let mut next_client = 0usize;
        let mut submit_time = vec![Instant::now(); num_clients];
        let mut first_tok_seen = vec![false; num_clients + 1];
        let mut last_tok_time: Vec<Option<Instant>> = vec![None; num_clients + 1];
        let mut ttfts = Vec::with_capacity(num_clients);
        let mut itls = Vec::with_capacity(num_clients * gen_len);

        while next_client < num_clients || eng.has_unfinished_work() {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            while next_client < num_clients && arrival_offsets_ms[next_client] <= elapsed_ms {
                let now_sub = Instant::now();
                let rid = eng
                    .submit_request(RequestParams {
                        prompt_tokens: prompts[next_client].clone(),
                        max_new_tokens: gen_len,
                        n_parallel: 1,
                        sampling: SamplingConfig {
                            ignore_eos: true,
                            ..SamplingConfig::default()
                        },
                        grammar: None,
                        use_speculative: false,
                    })
                    .unwrap();
                submit_time[rid as usize - 1] = now_sub;
                next_client += 1;
            }

            if !eng.has_unfinished_work() && next_client < num_clients {
                // Advance to next Poisson arrival if engine momentarily drained
                let now_sub = Instant::now();
                let rid = eng
                    .submit_request(RequestParams {
                        prompt_tokens: prompts[next_client].clone(),
                        max_new_tokens: gen_len,
                        n_parallel: 1,
                        sampling: SamplingConfig {
                            ignore_eos: true,
                            ..SamplingConfig::default()
                        },
                        grammar: None,
                        use_speculative: false,
                    })
                    .unwrap();
                submit_time[rid as usize - 1] = now_sub;
                next_client += 1;
            }

            if eng.has_unfinished_work() {
                let step = eng.step().unwrap();
                let now = Instant::now();
                for d in &step.deltas {
                    if d.new_tokens.is_empty() {
                        continue;
                    }
                    let idx = d.req_id as usize;
                    if !first_tok_seen[idx] {
                        first_tok_seen[idx] = true;
                        ttfts.push(now.duration_since(submit_time[idx - 1]).as_secs_f64() * 1000.0);
                        last_tok_time[idx] = Some(now);
                    } else if let Some(prev) = last_tok_time[idx] {
                        itls.push(now.duration_since(prev).as_secs_f64() * 1000.0);
                        last_tok_time[idx] = Some(now);
                    }
                }
            }
        }

        let total_secs = start.elapsed().as_secs_f64();
        let total_toks = num_clients * gen_len;
        rows.push(BatchingBenchmarkRow {
            mode: mode_name.into(),
            clients: num_clients,
            aggregate_tok_per_sec: (total_toks as f64) / total_secs,
            ttft_p50_ms: percentile(&ttfts, 0.50),
            ttft_p99_ms: percentile(&ttfts, 0.99),
            itl_p50_ms: percentile(&itls, 0.50),
            itl_p99_ms: percentile(&itls, 0.99),
        });
    }

    // Dedicated Mid-Flight Long-Prompt (512-token) Injection Tail-Jitter Measurement
    let measure_spike = |chunk_budget: usize| -> f64 {
        let model = ModelFactory::build_random_model(cfg.clone(), 9002).unwrap();
        let mut eng = ServingEngine::with_sampler(
            model,
            None,
            Some(Arc::clone(&tok)),
            SchedulerConfig {
                max_num_batched_tokens: chunk_budget,
                max_num_seqs: 16,
                num_physical_blocks: 512,
                num_swap_blocks: 128,
                enable_radix_cache: false,
                ..SchedulerConfig::default()
            },
            Arc::clone(&sampler),
        )
        .unwrap();

        // Start 8 steady decode streams (`prompt=16, gen=28`)
        for i in 0..8usize {
            let p: Vec<u32> = (0..16).map(|j| (20 + i * 5 + j) as u32).collect();
            eng.submit_request(RequestParams {
                prompt_tokens: p,
                max_new_tokens: 28,
                n_parallel: 1,
                sampling: SamplingConfig {
                    ignore_eos: true,
                    ..SamplingConfig::default()
                },
                grammar: None,
                use_speculative: false,
            })
            .unwrap();
        }
        // Advance 4 steps into steady decode
        for _ in 0..4 {
            let _ = eng.step().unwrap();
        }

        let mut last_ts: Vec<Instant> = vec![Instant::now(); 10];
        let mut decode_itls = Vec::new();
        // Inject a 512-token long prompt mid-flight!
        let long_prompt: Vec<u32> = (0..512).map(|j| ((j * 7 + 13) % 500) as u32).collect();
        eng.submit_request(RequestParams {
            prompt_tokens: long_prompt,
            max_new_tokens: 4,
            n_parallel: 1,
            sampling: SamplingConfig {
                ignore_eos: true,
                ..SamplingConfig::default()
            },
            grammar: None,
            use_speculative: false,
        })
        .unwrap();

        while eng.has_unfinished_work() {
            let step = eng.step().unwrap();
            let now = Instant::now();
            for d in &step.deltas {
                if d.req_id <= 8 && !d.new_tokens.is_empty() {
                    let r = d.req_id as usize;
                    decode_itls.push(now.duration_since(last_ts[r]).as_secs_f64() * 1000.0);
                    last_ts[r] = now;
                }
            }
        }
        percentile(&decode_itls, 0.99)
    };

    let unchunked_p99 = measure_spike(2048);
    let chunked_p99 = measure_spike(64);
    let jitter = ChunkedJitterReport {
        unchunked_midflight_decode_itl_p99_ms: unchunked_p99,
        chunked_64_midflight_decode_itl_p99_ms: chunked_p99,
        jitter_reduction_factor: unchunked_p99 / chunked_p99.max(0.01),
    };

    (rows, jitter)
}

fn bench_radix_prefix_cache(vocab_size: usize) -> RadixAblationReport {
    let tok = Arc::new(VocabularyBuilder::build_synthetic(vocab_size).unwrap());
    let sampler: Arc<dyn SamplerStrategy> = Arc::new(SamplingPipeline::new(vocab_size));
    let cfg = LlamaConfig::tiny_16m(vocab_size, DType::Q8_0);
    let shared_sys_len = 512usize;
    let num_requests = 24usize;

    let sys_prefix: Vec<u32> = (0..shared_sys_len)
        .map(|i| ((i * 5 + 17) % (vocab_size - 16)) as u32)
        .collect();

    let run_mode = |enable_radix: bool| -> (Vec<f64>, f64, f64) {
        let model = ModelFactory::build_random_model(cfg.clone(), 7777).unwrap();
        let mut eng = ServingEngine::with_sampler(
            model,
            None,
            Some(Arc::clone(&tok)),
            SchedulerConfig {
                max_num_batched_tokens: 1024,
                max_num_seqs: 16,
                num_physical_blocks: 512,
                num_swap_blocks: 128,
                enable_radix_cache: enable_radix,
                ..SchedulerConfig::default()
            },
            Arc::clone(&sampler),
        )
        .unwrap();

        let mut ttfts = Vec::with_capacity(num_requests);
        let wall_start = Instant::now();
        for r in 0..num_requests {
            let mut full_prompt = sys_prefix.clone();
            for j in 0..16usize {
                full_prompt.push(((100 + r * 13 + j) % (vocab_size - 16)) as u32);
            }
            let sub_t = Instant::now();
            eng.submit_request(RequestParams {
                prompt_tokens: full_prompt,
                max_new_tokens: 8,
                n_parallel: 1,
                sampling: SamplingConfig {
                    ignore_eos: true,
                    ..SamplingConfig::default()
                },
                grammar: None,
                use_speculative: false,
            })
            .unwrap();

            let mut recorded_first = false;
            while eng.has_unfinished_work() {
                let s = eng.step().unwrap();
                if !recorded_first && s.deltas.iter().any(|d| !d.new_tokens.is_empty()) {
                    ttfts.push(sub_t.elapsed().as_secs_f64() * 1000.0);
                    recorded_first = true;
                }
            }
        }
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let hit_pct = eng.radix_cache().hit_rate() * 100.0;
        (ttfts, wall_ms, hit_pct)
    };

    let (off_ttfts, off_wall_ms, _) = run_mode(false);
    let (on_ttfts, on_wall_ms, on_hit_pct) = run_mode(true);

    let off_p50 = percentile(&off_ttfts, 0.50);
    let off_p99 = percentile(&off_ttfts, 0.99);
    let on_p50 = percentile(&on_ttfts, 0.50);
    let on_p99 = percentile(&on_ttfts, 0.99);

    RadixAblationReport {
        shared_system_prompt_tokens: shared_sys_len,
        requests: num_requests,
        radix_off_ttft_p50_ms: off_p50,
        radix_off_ttft_p99_ms: off_p99,
        radix_off_total_ms: off_wall_ms,
        radix_on_ttft_p50_ms: on_p50,
        radix_on_ttft_p99_ms: on_p99,
        radix_on_total_ms: on_wall_ms,
        ttft_p50_speedup_x: off_p50 / on_p50.max(0.001),
        total_wall_speedup_x: off_wall_ms / on_wall_ms.max(0.001),
        kv_block_hit_rate_pct: on_hit_pct,
    }
}

fn bench_speculative_decoding(vocab_size: usize) -> SpeculativeAblationReport {
    let tok = Arc::new(VocabularyBuilder::build_synthetic(vocab_size).unwrap());
    let sampler: Arc<dyn SamplerStrategy> = Arc::new(SamplingPipeline::new(vocab_size));
    let target_cfg = LlamaConfig::tiny_16m(vocab_size, DType::F32);
    let draft_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);

    let (t_base, _) =
        ModelFactory::build_correlated_pair(target_cfg.clone(), draft_cfg.clone(), 2026_0999)
            .unwrap();
    let (t_spec, d_spec) =
        ModelFactory::build_correlated_pair(target_cfg, draft_cfg, 2026_0999).unwrap();

    let prompts: Vec<Vec<u32>> = (0..6usize)
        .map(|i| (0..24).map(|j| (12 + i * 9 + j) as u32).collect())
        .collect();
    let gen_tokens = 64usize;
    let total_gen = prompts.len() * gen_tokens;

    // 1. Pure Target Baseline (`K = 0`)
    let mut eng_base = ServingEngine::with_sampler(
        t_base,
        None,
        Some(Arc::clone(&tok)),
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 8,
            num_physical_blocks: 256,
            num_swap_blocks: 64,
            enable_radix_cache: false,
            speculative_lookahead: 0,
            adaptive_speculative: false,
            ..SchedulerConfig::default()
        },
        Arc::clone(&sampler),
    )
    .unwrap();

    let t0 = Instant::now();
    for p in &prompts {
        eng_base
            .submit_request(RequestParams {
                prompt_tokens: p.clone(),
                max_new_tokens: gen_tokens,
                n_parallel: 1,
                sampling: SamplingConfig {
                    temperature: 0.0,
                    ignore_eos: true,
                    ..SamplingConfig::default()
                },
                grammar: None,
                use_speculative: false,
            })
            .unwrap();
        while eng_base.has_unfinished_work() {
            let _ = eng_base.step().unwrap();
        }
    }
    let base_secs = t0.elapsed().as_secs_f64();
    let base_tps = (total_gen as f64) / base_secs;
    let base_tpot_ms = (base_secs * 1000.0) / (total_gen as f64);

    // 2. Speculative Decoding (`K = 4`)
    let lookahead_k = 4usize;
    let mut eng_spec = ServingEngine::with_sampler(
        t_spec,
        Some(d_spec),
        Some(Arc::clone(&tok)),
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 8,
            num_physical_blocks: 256,
            num_swap_blocks: 64,
            enable_radix_cache: false,
            speculative_lookahead: lookahead_k,
            adaptive_speculative: false,
            ..SchedulerConfig::default()
        },
        Arc::clone(&sampler),
    )
    .unwrap();

    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut total_spec_steps = 0usize;
    let mut total_spec_emitted = 0usize;

    let t1 = Instant::now();
    for p in &prompts {
        eng_spec
            .submit_request(RequestParams {
                prompt_tokens: p.clone(),
                max_new_tokens: gen_tokens,
                n_parallel: 1,
                sampling: SamplingConfig {
                    temperature: 0.0,
                    ignore_eos: true,
                    ..SamplingConfig::default()
                },
                grammar: None,
                use_speculative: true,
            })
            .unwrap();
        while eng_spec.has_unfinished_work() {
            let s = eng_spec.step().unwrap();
            for tr in &s.speculative_traces {
                total_drafted += tr.lookahead_k;
                total_accepted += tr.accepted_count;
                total_spec_steps += 1;
                total_spec_emitted += tr.accepted_count + 1;
            }
        }
    }
    let spec_secs = t1.elapsed().as_secs_f64();
    let spec_tps = (total_gen as f64) / spec_secs;
    let spec_tpot_ms = (spec_secs * 1000.0) / (total_gen as f64);
    let alpha_pct = if total_drafted > 0 {
        (total_accepted as f64 / total_drafted as f64) * 100.0
    } else {
        0.0
    };
    let mean_emit = if total_spec_steps > 0 {
        (total_spec_emitted as f64) / (total_spec_steps as f64)
    } else {
        1.0
    };

    SpeculativeAblationReport {
        lookahead_k,
        acceptance_alpha_pct: alpha_pct,
        mean_tokens_per_target_pass: mean_emit,
        baseline_tok_per_sec: base_tps,
        baseline_tpot_ms: base_tpot_ms,
        speculative_tok_per_sec: spec_tps,
        speculative_tpot_ms: spec_tpot_ms,
        tpot_speedup_x: base_tpot_ms / spec_tpot_ms.max(0.001),
    }
}

fn render_markdown(r: &FullSlaReport) -> String {
    let mut md = String::new();
    md.push_str("# NovaServe Production Serving SLA & Systems Ablation Report\n\n");
    md.push_str(&format!(
        "- **Active SIMD Kernel Backend:** `{}`\n",
        r.simd_backend
    ));
    md.push_str(&format!("- **Benchmark Model:** `{}`\n\n", r.model_arch));

    md.push_str("## 1. Kernel Quantization Speedup (`FP32` vs `Q8_0` vs `Q4_0`)\n\n");
    md.push_str("| Weight DType | Footprint (MB) | Compression vs FP32 | Single-Stream (`tok/s`) | Single-Stream `TPOT` (`ms`) | Continuous Batch-16 (`tok/s`) |\n");
    md.push_str("| --- | ---: | ---: | ---: | ---: | ---: |\n");
    for q in &r.quantization {
        md.push_str(&format!(
            "| `{}` | {:.2} | {:.2}x | {:.1} | {:.3} | {:.1} |\n",
            q.dtype,
            q.weight_megabytes,
            q.compression_ratio_vs_fp32,
            q.single_stream_tok_per_sec,
            q.single_stream_tpot_ms,
            q.batched_16_tok_per_sec
        ));
    }

    md.push_str("\n## 2. Continuous Batching vs Sequential Locking (`32` Poisson Clients) & Chunked Prefill\n\n");
    md.push_str("| Scheduler Policy | Clients | Throughput (`tok/s`) | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | `ITL p50` (`ms`) | `ITL p99` (`ms`) |\n");
    md.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for b in &r.batching {
        md.push_str(&format!(
            "| {} | {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            b.mode,
            b.clients,
            b.aggregate_tok_per_sec,
            b.ttft_p50_ms,
            b.ttft_p99_ms,
            b.itl_p50_ms,
            b.itl_p99_ms
        ));
    }
    md.push_str(&format!(
        "\n- **Mid-Flight 512-Token Prompt Decode Tail-Jitter (`ITL p99`):** Unchunked `{:.2} ms` -> Chunked (`B=64`) `{:.2} ms` (**`{:.2}x` tail-spike reduction**)\n",
        r.chunked_prefill_jitter.unchunked_midflight_decode_itl_p99_ms,
        r.chunked_prefill_jitter.chunked_64_midflight_decode_itl_p99_ms,
        r.chunked_prefill_jitter.jitter_reduction_factor
    ));

    md.push_str(
        "\n## 3. SGLang-Style Radix Prefix Cache Ablation (`512`-Token Shared System Prompt)\n\n",
    );
    md.push_str("| Configuration | Requests | `TTFT p50` (`ms`) | `TTFT p99` (`ms`) | Total Wall Time (`ms`) | KV-Block Hit Rate |\n");
    md.push_str("| --- | ---: | ---: | ---: | ---: | ---: |\n");
    let rx = &r.radix_ablation;
    md.push_str(&format!(
        "| `Radix OFF` | {} | {:.2} | {:.2} | {:.1} | 0.0% |\n",
        rx.requests, rx.radix_off_ttft_p50_ms, rx.radix_off_ttft_p99_ms, rx.radix_off_total_ms
    ));
    md.push_str(&format!(
        "| `Radix ON` | {} | {:.2} | {:.2} | {:.1} | **{:.1}%** |\n",
        rx.requests,
        rx.radix_on_ttft_p50_ms,
        rx.radix_on_ttft_p99_ms,
        rx.radix_on_total_ms,
        rx.kv_block_hit_rate_pct
    ));
    md.push_str(&format!(
        "\n- **Radix `TTFT p50` Speedup:** **`{:.2}x` faster** (`{:.2}x` end-to-end wall speedup)\n",
        rx.ttft_p50_speedup_x, rx.total_wall_speedup_x
    ));

    md.push_str("\n## 4. Exact Leviathan Speculative Decoding Speedup (`K = 4`)\n\n");
    let sp = &r.speculative_ablation;
    md.push_str("| Decoding Engine | Lookahead `K` | Acceptance `α` | Mean Tokens / Target Pass | Throughput (`tok/s`) | `TPOT` (`ms/tok`) | Speedup |\n");
    md.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    md.push_str(&format!(
        "| Pure Target (`Tiny-16M`) | 0 | — | 1.00 | {:.1} | {:.3} | 1.00x |\n",
        sp.baseline_tok_per_sec, sp.baseline_tpot_ms
    ));
    md.push_str(&format!(
        "| Speculative (`Draft-500K` + `Tiny-16M`) | {} | **{:.1}%** | **{:.2}** | **{:.1}** | **{:.3}** | **{:.2}x** |\n",
        sp.lookahead_k,
        sp.acceptance_alpha_pct,
        sp.mean_tokens_per_target_pass,
        sp.speculative_tok_per_sec,
        sp.speculative_tpot_ms,
        sp.tpot_speedup_x
    ));

    md
}

fn main() {
    let vocab_size = 512usize;
    println!("[novaserve-bench] Running M08 Release Serving SLA & Ablation Suite...");
    println!(
        "[novaserve-bench] Active SIMD Backend: {:?}",
        simd::active_backend()
    );

    let quantization = bench_quantization(vocab_size);
    println!("[novaserve-bench] 1/4 Quantization ablation complete.");

    let (batching, chunked_prefill_jitter) =
        bench_continuous_batching_and_chunked_prefill(vocab_size);
    println!("[novaserve-bench] 2/4 Continuous Batching & Chunked Prefill ablation complete.");

    let radix_ablation = bench_radix_prefix_cache(vocab_size);
    println!(
        "[novaserve-bench] 3/4 RadixCache ablation complete (hit_rate={:.1}%, TTFT speedup={:.2}x).",
        radix_ablation.kv_block_hit_rate_pct, radix_ablation.ttft_p50_speedup_x
    );

    let speculative_ablation = bench_speculative_decoding(vocab_size);
    println!(
        "[novaserve-bench] 4/4 Speculative K=4 ablation complete (alpha={:.1}%, TPOT speedup={:.2}x).",
        speculative_ablation.acceptance_alpha_pct, speculative_ablation.tpot_speedup_x
    );

    let report = FullSlaReport {
        simd_backend: format!("{:?}", simd::active_backend()),
        model_arch: "Tiny-16M GQA (dim=256, layers=4, n_heads=8, n_kv_heads=2, vocab=512)".into(),
        quantization,
        batching,
        chunked_prefill_jitter,
        radix_ablation,
        speculative_ablation,
    };

    let out_dir = Path::new("docs/benchmarks");
    fs::create_dir_all(out_dir).expect("create docs/benchmarks");
    let md_path = out_dir.join("serving_sla.md");
    let json_path = out_dir.join("serving_sla.json");

    fs::write(&md_path, render_markdown(&report)).expect("write serving_sla.md");
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&report).expect("serialize serving_sla.json"),
    )
    .expect("write serving_sla.json");

    println!(
        "[novaserve-bench] Wrote {} and {}",
        md_path.display(),
        json_path.display()
    );
}
