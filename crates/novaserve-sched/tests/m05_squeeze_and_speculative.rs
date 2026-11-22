use novaserve_kv::DEFAULT_BLOCK_SIZE;
use novaserve_model::{LlamaConfig, ModelFactory};
use novaserve_sched::{
    AdaptiveLookaheadController, PreemptionMode, RequestParams, SamplingConfig, SchedulerConfig,
    ServingEngine,
};
use novaserve_tensor::{DType, SplitMix64};

#[test]
fn extreme_memory_squeeze_32_requests_on_32_blocks_swap_and_recompute_parity() {
    let vocab_size = 512usize;
    let cfg = LlamaConfig::micro_2m(vocab_size, DType::F32);
    let model = ModelFactory::build_random_model(cfg, 0x0505_2026_ABCD_0032).unwrap();

    // Generate 32 multi-hundred-token requests:
    // e.g. 68..84 prompt tokens + 40 generated tokens = 108..124 tokens (`7..8` physical 16-token blocks each!).
    // With `32` total physical blocks, only `4` sequences (`4 * 8 = 32` blocks) can fit simultaneously!
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
    let shared_sys: Vec<u32> = (10..42).collect(); // 32 shared system tokens (2 Radix blocks)
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(32);
    for r_idx in 0..32usize {
        let mut p = if r_idx.is_multiple_of(3) {
            shared_sys.clone()
        } else {
            Vec::new()
        };
        let target_len = 68 + (r_idx % 17);
        while p.len() < target_len {
            p.push(50 + (rng.next_u32() % 400));
        }
        prompts.push(p);
    }
    let gen_len = 40usize;

    // 1. Unlimited-memory baseline (`1024` physical blocks)
    let unlimited_sched_cfg = SchedulerConfig {
        max_num_batched_tokens: 128,
        max_num_seqs: 32,
        num_physical_blocks: 1024,
        num_swap_blocks: 256,
        block_size: DEFAULT_BLOCK_SIZE,
        preemption_mode: PreemptionMode::Swap,
        enable_radix_cache: true,
        ..SchedulerConfig::default()
    };
    let mut baseline_engine =
        ServingEngine::new(model.clone(), None, None, unlimited_sched_cfg).unwrap();
    for p in &prompts {
        let mut req = RequestParams::greedy(p.clone(), gen_len);
        req.sampling.ignore_eos = true;
        baseline_engine.submit_request(req).unwrap();
    }
    let baseline_results = baseline_engine.run_to_completion().unwrap();
    assert_eq!(baseline_results.len(), 32);

    // 2. Extreme squeeze (`32` total physical blocks!) under BOTH `Swap` and `Recompute`!
    for &mode in &[PreemptionMode::Swap, PreemptionMode::Recompute] {
        // Admit up to 8 concurrent sequences so early prefill packs >4 sequences into 32 blocks,
        // guaranteeing mid-decode KV exhaustion and forced preemptions as sequences cross block boundaries!
        let squeezed_cfg = SchedulerConfig {
            max_num_batched_tokens: 48, // also exercises ChunkedPrefill!
            max_num_seqs: 8,
            num_physical_blocks: 32,
            num_swap_blocks: 256,
            block_size: DEFAULT_BLOCK_SIZE,
            preemption_mode: mode,
            enable_radix_cache: true,
            ..SchedulerConfig::default()
        };

        let mut engine = ServingEngine::new(model.clone(), None, None, squeezed_cfg).unwrap();
        for p in &prompts {
            let mut req = RequestParams::greedy(p.clone(), gen_len);
            req.sampling.ignore_eos = true;
            engine.submit_request(req).unwrap();
        }

        let mut results = Vec::with_capacity(32);
        let mut total_steps = 0usize;
        let mut observed_preemptions = 0usize;
        while engine.has_unfinished_work() {
            total_steps += 1;
            assert!(
                total_steps < 20_000,
                "deadlock/starvation detected under {mode:?} at step {total_steps}"
            );
            let mut s = engine.step().unwrap();
            observed_preemptions += s.preempted_this_step;
            results.append(&mut s.completed);
        }
        results.sort_by_key(|r| r.req_id);
        assert_eq!(
            results.len(),
            32,
            "{mode:?} did not complete all 32 requests"
        );
        assert!(
            observed_preemptions > 0,
            "{mode:?} test must trigger real KV preemptions"
        );

        // Verify 100% exact greedy token parity against unlimited-memory baseline!
        for (got, expected) in results.iter().zip(baseline_results.iter()) {
            assert_eq!(got.req_id, expected.req_id);
            assert_eq!(got.outputs.len(), 1);
            assert_eq!(got.outputs[0].token_ids.len(), gen_len);
            assert_eq!(
                got.outputs[0].token_ids, expected.outputs[0].token_ids,
                "{mode:?} preemption altered greedy output on req_id={}!",
                got.req_id
            );
        }

        engine.clear_radix_cache();
        engine.allocator().verify_invariants();
        assert_eq!(engine.allocator().free_blocks(), 32);
        assert_eq!(
            engine.swap_engine().free_swap_blocks(),
            engine.swap_engine().total_swap_blocks()
        );

        println!(
            "[M05 Telemetry] Mode={mode:?} | 32/32 requests completed on 32 physical blocks in {total_steps} steps (preemptions={observed_preemptions}, 100% greedy parity, 0 leaks)"
        );
    }
}

#[test]
fn speculative_decoding_greedy_parity_abort_reclamation_and_adaptive_k() {
    let vocab_size = 256usize;
    let target_cfg = LlamaConfig::micro_2m(vocab_size, DType::F32);
    let draft_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);
    let (target_model, draft_model) =
        ModelFactory::build_correlated_pair(target_cfg, draft_cfg, 0x5EC0_2026_7788).unwrap();

    let prompt: Vec<u32> = (5..29).collect();

    // Pure Target greedy run
    let mut pure_engine =
        ServingEngine::new(target_model.clone(), None, None, SchedulerConfig::default()).unwrap();
    let mut r_pure = RequestParams::greedy(prompt.clone(), 36);
    r_pure.sampling.ignore_eos = true;
    pure_engine.submit_request(r_pure).unwrap();
    let pure_out = pure_engine.run_to_completion().unwrap();

    // Speculative (`K = 4`) greedy run on correlated Draft+Target pair!
    let spec_cfg = SchedulerConfig {
        speculative_lookahead: 4,
        adaptive_speculative: false,
        ..SchedulerConfig::default()
    };
    let mut spec_engine =
        ServingEngine::new(target_model.clone(), Some(draft_model), None, spec_cfg).unwrap();
    let mut r_spec = RequestParams::greedy(prompt.clone(), 36);
    r_spec.sampling.ignore_eos = true;
    r_spec.use_speculative = true;
    spec_engine.submit_request(r_spec).unwrap();

    let mut spec_steps = 0usize;
    let mut spec_done = Vec::new();
    while spec_engine.has_unfinished_work() {
        spec_steps += 1;
        let mut s = spec_engine.step().unwrap();
        spec_done.append(&mut s.completed);
    }

    assert_eq!(
        spec_done[0].outputs[0].token_ids,
        pure_out[0].outputs[0].token_ids
    );
    let alpha = spec_engine.lookahead_controller().empirical_alpha();
    println!(
        "[M05 Telemetry] Speculative K=4 greedy parity 100% | steps={spec_steps} (vs 37 pure) | acceptance alpha={:.1}%",
        alpha * 100.0
    );
    assert!(
        spec_steps < 36,
        "correlated speculative decoding should take fewer steps than 1-tok/step: got {spec_steps}"
    );

    // Test mid-flight `abort_request` across running, waiting, and parallel `n = 3` CoW forks
    let mut abort_engine = ServingEngine::new(
        target_model,
        None,
        None,
        SchedulerConfig {
            num_physical_blocks: 64,
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let mut ids = Vec::new();
    for i in 0..6u64 {
        let req = RequestParams {
            prompt_tokens: prompt.clone(),
            max_new_tokens: 30,
            n_parallel: 3,
            sampling: SamplingConfig {
                temperature: 0.8,
                seed: 100 + i,
                ignore_eos: true,
                ..SamplingConfig::default()
            },
            grammar: None,
            use_speculative: false,
        };
        ids.push(abort_engine.submit_request(req).unwrap());
    }
    for _ in 0..4 {
        abort_engine.step().unwrap();
    }
    // Abort 1 running and 1 waiting request mid-flight!
    assert!(abort_engine.abort_request(ids[1]));
    assert!(abort_engine.abort_request(ids[5]));
    let rem = abort_engine.run_to_completion().unwrap();
    assert_eq!(rem.len(), 4);
    abort_engine.clear_radix_cache();
    assert_eq!(abort_engine.allocator().free_blocks(), 64);

    // Verify AdaptiveLookaheadController disables lookahead when rolling alpha < 0.25
    let mut ctrl = AdaptiveLookaheadController::new(4, true);
    assert_eq!(ctrl.current_k(), 4);
    for _ in 0..20 {
        ctrl.record_step(4, 0);
    }
    assert_eq!(ctrl.current_k(), 0);
}
