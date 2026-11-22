//! 5,000-Scenario Continuous-Batching, RadixCache, CoW Forking, ChunkedPrefill, and
//! Preemption (`Swap` & `Recompute`) Equivalence Fuzzer against `EagerReferenceEngine`.

use crate::eager_ref::EagerReferenceEngine;
use novaserve_kv::{KvDType, DEFAULT_BLOCK_SIZE};
use novaserve_model::{LlamaConfig, ModelFactory};
use novaserve_sampler::SamplingPipeline;
use novaserve_sched::{
    PreemptionMode, RequestParams, SamplingConfig, SchedulerConfig, ServingEngine, TokenLogprob,
};
use novaserve_tensor::{DType, SplitMix64};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChaosFuzzReport {
    pub scenarios_verified: usize,
    pub total_sequences_checked: usize,
    pub total_preemptions_triggered: usize,
    pub total_aborts_injected: usize,
    pub max_logprob_abs_diff: f32,
}

struct PromptCase {
    prompt: Vec<u32>,
    gen_len: usize,
    expected_tokens: Vec<u32>,
    expected_top5: Vec<Vec<TokenLogprob>>,
}

pub fn run_chaos_batch_equivalence_campaign(num_scenarios: usize) -> ChaosFuzzReport {
    let vocab_size = 128usize;
    let model_cfg = LlamaConfig {
        arch_name: "llama".into(),
        vocab_size,
        dim: 32,
        intermediate_size: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: 1.0,
        max_seq_len: 512,
        tie_word_embeddings: false,
        weight_dtype: DType::F32,
    };

    let model = ModelFactory::build_random_model(model_cfg, 0x0707_2026_CAFE_F00D)
        .expect("build oracle model");
    let eager_ref = EagerReferenceEngine::from_model(&model);

    // Build 96 diverse multi-turn prompts sharing 16-token and 32-token system prefixes
    // (hitting RadixCache both on exact 16-token block boundaries and mid-edge splits!)
    let mut seed_rng = SplitMix64::new(0x1111_2222_3333_4444);
    let sys_16: Vec<u32> = (1..=16).collect();
    let sys_32: Vec<u32> = (1..=32).collect();
    let alt_32: Vec<u32> = (1..=16).chain(40..56).collect();

    let mut prompt_pool = Vec::with_capacity(96);
    for idx in 0..96usize {
        let mut p = match idx % 4 {
            0 => sys_16.clone(),
            1 => sys_32.clone(),
            2 => alt_32.clone(),
            _ => Vec::new(),
        };
        let extra = 3 + (idx % 14); // total lengths span 3..46 tokens (> 19 chunk budget!)
        for _ in 0..extra {
            p.push((seed_rng.next_u32() as usize % vocab_size) as u32);
        }
        let gen_len = 10 + (idx % 7); // 10..16 generated tokens crossing 16-token boundaries!
        let (expected_tokens, expected_top5) = eager_ref.greedy_trajectory_and_top5(&p, gen_len);
        prompt_pool.push(PromptCase {
            prompt: p,
            gen_len,
            expected_tokens,
            expected_top5,
        });
    }

    let sampler: Arc<dyn novaserve_sched::SamplerStrategy> =
        Arc::new(SamplingPipeline::new(vocab_size));

    let eval_single_scenario = |scen_idx: usize| -> (usize, usize, usize, f32) {
        let mut rng = SplitMix64::new(0x5CE4_0000_0000_0000u64.wrapping_add(scen_idx as u64));
        let mode = if scen_idx.is_multiple_of(2) {
            PreemptionMode::Swap
        } else {
            PreemptionMode::Recompute
        };

        // Severe physical KV slab scarcity (`8..12` blocks) + tiny `max_num_batched_tokens = 19`
        // forcing multi-step ChunkedPrefill + relentless mid-decode Swap / Recompute preemptions!
        let phys_blocks = 8 + (scen_idx % 5);
        let swap_blocks = 96usize;
        let sched_cfg = SchedulerConfig {
            max_num_batched_tokens: 19,
            max_num_seqs: 6,
            num_physical_blocks: phys_blocks,
            num_swap_blocks: swap_blocks,
            block_size: DEFAULT_BLOCK_SIZE,
            kv_dtype: KvDType::F32,
            preemption_mode: mode,
            enable_radix_cache: true,
            speculative_lookahead: 0,
            adaptive_speculative: false,
        };

        let mut engine =
            ServingEngine::with_sampler(model.clone(), None, None, sched_cfg, Arc::clone(&sampler))
                .expect("init engine");
        engine.allocator_mut().shuffle_free_list(&mut rng);

        let num_reqs = 3 + (rng.next_u32() as usize % 3); // 3..=5 requests
        let mut submitted: Vec<(u64, usize, usize, bool)> = Vec::with_capacity(num_reqs);

        // Submit initial wave (`num_reqs - 1` requests), including `n = 3` CoW parallel completions!
        for r_i in 0..(num_reqs - 1) {
            let case_id = (rng.next_u32() as usize) % prompt_pool.len();
            let n_par = if r_i == 0 && scen_idx.is_multiple_of(3) {
                3
            } else {
                1
            };
            let req = RequestParams {
                prompt_tokens: prompt_pool[case_id].prompt.clone(),
                max_new_tokens: prompt_pool[case_id].gen_len,
                n_parallel: n_par,
                sampling: SamplingConfig {
                    temperature: 0.0,
                    ignore_eos: true,
                    top_logprobs: 5,
                    seed: scen_idx as u64 + r_i as u64,
                    ..SamplingConfig::default()
                },
                grammar: None,
                use_speculative: false,
            };
            let rid = engine.submit_request(req).unwrap();
            submitted.push((rid, case_id, n_par, false));
        }

        let mut preemptions = 0usize;
        let mut aborts = 0usize;
        let mut completed = Vec::new();
        let mut step_num = 0usize;

        while engine.has_unfinished_work() || step_num == 2 {
            if step_num == 2 {
                // Inject staggered mid-flight arrival hitting live RadixCache!
                let case_id = submitted[0].1; // shares prefix with req 0!
                let req = RequestParams {
                    prompt_tokens: prompt_pool[case_id].prompt.clone(),
                    max_new_tokens: prompt_pool[case_id].gen_len,
                    n_parallel: 1,
                    sampling: SamplingConfig {
                        temperature: 0.0,
                        ignore_eos: true,
                        top_logprobs: 5,
                        ..SamplingConfig::default()
                    },
                    grammar: None,
                    use_speculative: false,
                };
                let rid = engine.submit_request(req).unwrap();
                submitted.push((rid, case_id, 1, false));
            }

            if step_num == 3 && scen_idx.is_multiple_of(2) && submitted.len() >= 3 {
                // Sudden client disconnect cancellation mid-stream!
                let abort_slot = 1;
                let abort_rid = submitted[abort_slot].0;
                if engine.abort_request(abort_rid) {
                    submitted[abort_slot].3 = true;
                    aborts += 1;
                }
            }

            if !engine.has_unfinished_work() {
                break;
            }
            step_num += 1;
            assert!(
                step_num < 5_000,
                "chaos scenario {scen_idx} exceeded 5,000 steps (starvation!)"
            );
            let mut s = engine.step().expect("engine step");
            preemptions += s.preempted_this_step;
            completed.append(&mut s.completed);
        }

        let mut seqs_checked = 0usize;
        let mut max_lp_diff = 0.0f32;

        for (rid, case_id, n_par, was_aborted) in &submitted {
            if *was_aborted {
                continue;
            }
            let comp = completed
                .iter()
                .find(|c| c.req_id == *rid)
                .unwrap_or_else(|| panic!("scenario {scen_idx} missing completed req_id={rid}"));
            assert_eq!(comp.outputs.len(), *n_par);

            let expected = &prompt_pool[*case_id];
            for child_out in &comp.outputs {
                assert_eq!(
                    child_out.token_ids, expected.expected_tokens,
                    "greedy token mismatch in scenario {scen_idx} (mode={mode:?}) req={rid} child={}",
                    child_out.child_idx
                );
                assert_eq!(child_out.step_logprobs.len(), expected.gen_len);

                for (step_k, (got_step, exp_top5)) in child_out
                    .step_logprobs
                    .iter()
                    .zip(expected.expected_top5.iter())
                    .enumerate()
                {
                    assert_eq!(got_step.top_logprobs.len(), 5);
                    for (rank_i, (got_lp, exp_lp)) in got_step
                        .top_logprobs
                        .iter()
                        .zip(exp_top5.iter())
                        .enumerate()
                    {
                        let d = (got_lp.logprob - exp_lp.logprob).abs();
                        max_lp_diff = max_lp_diff.max(d);
                        assert!(
                            d <= 1e-4,
                            "top-5 logprob drift at scen={scen_idx} step={step_k} rank={rank_i}: got={} vs f64_ref={} (diff={d:.3e})",
                            got_lp.logprob,
                            exp_lp.logprob
                        );
                        // Unless two tokens had near-identical logits (`<= 1e-5`), token IDs also match
                        if rank_i == 0 {
                            assert_eq!(got_lp.token_id, exp_lp.token_id);
                        }
                    }
                }
                seqs_checked += 1;
            }
        }

        // Assert 100% physical KV block and swap block reclamation!
        engine.clear_radix_cache();
        engine.allocator().verify_invariants();
        assert_eq!(
            engine.allocator().free_blocks(),
            phys_blocks,
            "physical KV block leak in scenario {scen_idx}"
        );
        assert_eq!(
            engine.swap_engine().free_swap_blocks(),
            swap_blocks,
            "swap slot leak in scenario {scen_idx}"
        );

        (seqs_checked, preemptions, aborts, max_lp_diff)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let partials: Vec<(usize, usize, usize, f32)> = (0..num_scenarios)
        .into_par_iter()
        .map(eval_single_scenario)
        .collect();

    #[cfg(target_arch = "wasm32")]
    let partials: Vec<(usize, usize, usize, f32)> =
        (0..num_scenarios).map(eval_single_scenario).collect();

    let mut total_sequences_checked = 0usize;
    let mut total_preemptions_triggered = 0usize;
    let mut total_aborts_injected = 0usize;
    let mut max_logprob_abs_diff = 0.0f32;

    for (sc, pr, ab, diff) in partials {
        total_sequences_checked += sc;
        total_preemptions_triggered += pr;
        total_aborts_injected += ab;
        max_logprob_abs_diff = max_logprob_abs_diff.max(diff);
    }

    ChaosFuzzReport {
        scenarios_verified: num_scenarios,
        total_sequences_checked,
        total_preemptions_triggered,
        total_aborts_injected,
        max_logprob_abs_diff,
    }
}
