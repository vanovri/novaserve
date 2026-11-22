//! Statistical Proof of Exact Speculative Decoding (`Leviathan et al., ICML 2023`)
//! across Aligned ($\alpha \approx 0.70$), Mismatched ($\alpha \approx 0.15$), and
//! Adversarial (~95% probability mass on Target's least-likely tokens!) regimes.

use novaserve_model::{LlamaConfig, LlamaModel, ModelFactory};
use novaserve_sampler::SamplingPipeline;
use novaserve_sched::{
    RequestParams, SamplerStrategy, SamplingConfig, SchedulerConfig, ServingEngine,
};
use novaserve_tensor::DType;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeProofStats {
    pub regime_name: String,
    pub total_transitions: usize,
    pub empirical_alpha: f64,
    pub tvd_pos0_vs_exact: f64,
    pub tvd_bigram_pos1_vs_exact: f64,
    pub chi_square_pos0: f64,
    pub degrees_of_freedom: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeculativeProofReport {
    pub regimes: Vec<RegimeProofStats>,
}

pub fn run_3_regime_speculative_proof(transitions_per_regime: usize) -> SpeculativeProofReport {
    let vocab_size = 64usize;
    let target_cfg = LlamaConfig {
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
        max_seq_len: 256,
        tie_word_embeddings: false,
        weight_dtype: DType::F32,
    };
    let draft_cfg = LlamaConfig {
        arch_name: "llama".into(),
        vocab_size,
        dim: 32,
        intermediate_size: 48,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: 1.0,
        max_seq_len: 256,
        tie_word_embeddings: false,
        weight_dtype: DType::F32,
    };

    // 1. Aligned Regime (`dim_d == dim_t = 32`, 1-layer distilled subspace of 2-layer Target -> alpha ~ 0.70!)
    let (target_model, aligned_draft) =
        ModelFactory::build_correlated_pair(target_cfg.clone(), draft_cfg.clone(), 0xA119_4ED0)
            .expect("aligned pair");

    // 2. Mismatched Regime (completely unrelated random weights -> alpha ~ 0.15!)
    let mismatched_draft =
        ModelFactory::build_random_model(draft_cfg.clone(), 0xDEAD_BEEF_9999_1111)
            .expect("mismatched draft");

    // 3. Adversarial Regime (anti-correlated `-3x` lm_head placing ~95% mass on least-likely tokens!)
    let (_, adversarial_draft) =
        ModelFactory::build_adversarial_draft(target_cfg, draft_cfg, 0xA119_4ED0)
            .expect("adversarial draft");

    let context_prompt: Vec<u32> = (2..20).collect(); // 18-token fixed context state (> 1 Radix block)
    let base_sampling = SamplingConfig {
        temperature: 0.9,
        top_k: Some(16),
        top_p: 1.0,
        min_p: 0.0,
        ignore_eos: true,
        ..SamplingConfig::default()
    };

    // Compute exact analytical first-step target distribution `P_target(x_0 | C)`
    let sampler = Arc::new(SamplingPipeline::new(vocab_size));
    let raw_pos0_logits = target_model
        .forward_eager_contiguous(&context_prompt, false)
        .expect("eager target pos0");
    let mut exact_pos0_probs = vec![0.0f32; vocab_size];
    sampler
        .compute_probs(
            &raw_pos0_logits,
            &context_prompt,
            &base_sampling,
            None,
            None,
            &mut exact_pos0_probs,
        )
        .expect("exact pos0 probs");

    // Find top-3 most probable tokens `x_0` so we ALSO compute the exact conditional 2-gram
    // mixture distribution `P_target(x_1 | C, x_0 in Top3)` inside the K=4 multi-token batch!
    let mut ranked_pos0: Vec<(usize, f32)> = exact_pos0_probs.iter().copied().enumerate().collect();
    ranked_pos0.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top3_x0: Vec<usize> = ranked_pos0.iter().take(3).map(|&(v, _)| v).collect();
    let top3_mass: f64 = ranked_pos0.iter().take(3).map(|&(_, p)| p as f64).sum();

    let mut exact_pos1_cond_top3 = vec![0.0f64; vocab_size];
    for &x0 in &top3_x0 {
        let mut ext_prompt = context_prompt.clone();
        ext_prompt.push(x0 as u32);
        let raw_p1 = target_model
            .forward_eager_contiguous(&ext_prompt, false)
            .unwrap();
        let mut p1_vec = vec![0.0f32; vocab_size];
        sampler
            .compute_probs(
                &raw_p1,
                &ext_prompt,
                &base_sampling,
                None,
                None,
                &mut p1_vec,
            )
            .unwrap();
        let w = (exact_pos0_probs[x0] as f64) / top3_mass;
        for v in 0..vocab_size {
            exact_pos1_cond_top3[v] += w * (p1_vec[v] as f64);
        }
    }

    // Verify adversarial draft genuinely places >90% probability mass outside Target's top-likely tokens!
    {
        let adv_logits = adversarial_draft
            .forward_eager_contiguous(&context_prompt, false)
            .unwrap();
        let mut adv_probs = vec![0.0f32; vocab_size];
        sampler
            .compute_probs(
                &adv_logits,
                &context_prompt,
                &base_sampling,
                None,
                None,
                &mut adv_probs,
            )
            .unwrap();
        let overlap: f64 = exact_pos0_probs
            .iter()
            .zip(adv_probs.iter())
            .map(|(&p, &q)| (p as f64).min(q as f64))
            .sum();
        assert!(
            overlap < 0.15,
            "adversarial draft overlap too high ({overlap:.3}); must strongly oppose target"
        );
    }

    let regimes_input = vec![
        ("Aligned", aligned_draft, 0x1000_0000_0000_0000u64),
        ("Mismatched", mismatched_draft, 0x2000_0000_0000_0000u64),
        ("Adversarial", adversarial_draft, 0x3000_0000_0000_0000u64),
    ];

    // Each rollout runs `ServingEngine(K = 4)` for `5` generated tokens (`[t_0, t_1, t_2, t_3, t_4]`):
    // Because prefill stops at `P - 1`, ALL `5` tokens (`t_0` through bonus `t_4`) are generated
    // directly by `K = 4` Speculative Decoding!
    // `num_rollouts = transitions_per_regime` (`35,000` rollouts -> `175,000` speculative tokens/regime!)
    let mut stats_out = Vec::with_capacity(3);
    for (name, draft_m, seed_base) in regimes_input {
        let (pos0_hist, pos1_top3_hist, pos1_top3_n, alpha) = sample_speculative_k4(
            &target_model,
            &draft_m,
            &context_prompt,
            &base_sampling,
            &top3_x0,
            transitions_per_regime,
            seed_base,
        );

        let n_f64 = transitions_per_regime as f64;
        let mut tvd_pos0 = 0.0f64;
        let mut chi_sq = 0.0f64;
        let mut support_count = 0usize;

        for v in 0..vocab_size {
            let p_emp = (pos0_hist[v] as f64) / n_f64;
            let p_true = exact_pos0_probs[v] as f64;
            tvd_pos0 += 0.5 * (p_emp - p_true).abs();
            if p_true > 1e-6 {
                support_count += 1;
                let expected = n_f64 * p_true;
                let diff = (pos0_hist[v] as f64) - expected;
                chi_sq += (diff * diff) / expected;
            } else {
                assert_eq!(
                    pos0_hist[v], 0,
                    "regime {name} sampled zero-probability masked token {v}!"
                );
            }
        }

        let mut tvd_bigram_pos1 = 0.0f64;
        let n_p1 = (pos1_top3_n.max(1)) as f64;
        for v in 0..vocab_size {
            let p_emp1 = (pos1_top3_hist[v] as f64) / n_p1;
            tvd_bigram_pos1 += 0.5 * (p_emp1 - exact_pos1_cond_top3[v]).abs();
        }

        let df = support_count.saturating_sub(1);
        stats_out.push(RegimeProofStats {
            regime_name: name.into(),
            total_transitions: transitions_per_regime,
            empirical_alpha: alpha,
            tvd_pos0_vs_exact: tvd_pos0,
            tvd_bigram_pos1_vs_exact: tvd_bigram_pos1,
            chi_square_pos0: chi_sq,
            degrees_of_freedom: df,
        });
    }

    SpeculativeProofReport { regimes: stats_out }
}

fn sample_speculative_k4(
    target_model: &LlamaModel,
    draft_model: &LlamaModel,
    context_prompt: &[u32],
    base_sampling: &SamplingConfig,
    top3_x0: &[usize],
    num_runs: usize,
    seed_base: u64,
) -> (Vec<u64>, Vec<u64>, u64, f64) {
    let vocab_sz = target_model.config().vocab_size;
    let num_shards = 16usize;
    let runs_per_shard = num_runs.div_ceil(num_shards);

    struct ShardStats {
        pos0_counts: Vec<u64>,
        pos1_top3_counts: Vec<u64>,
        pos1_top3_total: u64,
        drafted_tot: u64,
        accepted_tot: u64,
    }

    let run_shard = |shard_idx: usize| -> ShardStats {
        let start = shard_idx * runs_per_shard;
        let end = ((shard_idx + 1) * runs_per_shard).min(num_runs);
        if start >= end {
            return ShardStats {
                pos0_counts: vec![0; vocab_sz],
                pos1_top3_counts: vec![0; vocab_sz],
                pos1_top3_total: 0,
                drafted_tot: 0,
                accepted_tot: 0,
            };
        }

        let sampler = Arc::new(SamplingPipeline::new(vocab_sz));
        let sched_cfg = SchedulerConfig {
            max_num_batched_tokens: 64,
            max_num_seqs: 4,
            num_physical_blocks: 64,
            num_swap_blocks: 16,
            enable_radix_cache: true,
            speculative_lookahead: 4,
            adaptive_speculative: false, // keep K=4 pinned across all 3 regimes!
            ..SchedulerConfig::default()
        };

        let mut engine = ServingEngine::with_sampler(
            target_model.clone(),
            Some(draft_model.clone()),
            None,
            sched_cfg,
            sampler,
        )
        .expect("init speculative shard engine");

        let mut pos0_counts = vec![0u64; vocab_sz];
        let mut pos1_top3_counts = vec![0u64; vocab_sz];
        let mut pos1_top3_total = 0u64;
        let mut drafted_tot = 0u64;
        let mut accepted_tot = 0u64;

        for run_i in start..end {
            let mut s_cfg = base_sampling.clone();
            s_cfg.seed = seed_base.wrapping_add((run_i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            // Request `5` tokens (`K + 1 = 5`) so step 1 runs full `K = 4` draft + 5-token verify!
            let req = RequestParams {
                prompt_tokens: context_prompt.to_vec(),
                max_new_tokens: 5,
                n_parallel: 1,
                sampling: s_cfg,
                grammar: None,
                use_speculative: true,
            };
            let rid = engine.submit_request(req).unwrap();
            while engine.has_unfinished_work() {
                let step_s = engine.step().unwrap();
                for tr in &step_s.speculative_traces {
                    drafted_tot += tr.lookahead_k as u64;
                    accepted_tot += tr.accepted_count as u64;
                }
            }
            let comp = engine.take_completed(rid).expect("completed");
            let toks = &comp.outputs[0].token_ids;
            let t0 = toks[0] as usize;
            let t1 = toks[1] as usize;
            pos0_counts[t0] += 1;
            if top3_x0.contains(&t0) {
                pos1_top3_counts[t1] += 1;
                pos1_top3_total += 1;
            }
        }

        ShardStats {
            pos0_counts,
            pos1_top3_counts,
            pos1_top3_total,
            drafted_tot,
            accepted_tot,
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let parts: Vec<ShardStats> = (0..num_shards).into_par_iter().map(run_shard).collect();

    #[cfg(target_arch = "wasm32")]
    let parts: Vec<ShardStats> = (0..num_shards).map(run_shard).collect();

    let mut pos0_total = vec![0u64; vocab_sz];
    let mut pos1_total = vec![0u64; vocab_sz];
    let mut pos1_n = 0u64;
    let mut drafted = 0u64;
    let mut accepted = 0u64;

    for part in parts {
        for v in 0..vocab_sz {
            pos0_total[v] += part.pos0_counts[v];
            pos1_total[v] += part.pos1_top3_counts[v];
        }
        pos1_n += part.pos1_top3_total;
        drafted += part.drafted_tot;
        accepted += part.accepted_tot;
    }

    let alpha = if drafted == 0 {
        0.0
    } else {
        (accepted as f64) / (drafted as f64)
    };
    (pos0_total, pos1_total, pos1_n, alpha)
}
