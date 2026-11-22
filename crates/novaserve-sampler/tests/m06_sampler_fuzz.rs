use novaserve_frontend::AllowedTokenMask;
use novaserve_sampler::{SamplerError, SamplingPipeline};
use novaserve_sched::SamplingConfig;
use novaserve_tensor::{ChaCha8Rng, SplitMix64};

#[test]
fn pathological_10000_logits_fuzz_zero_nan_inf_and_seed_determinism() {
    let pipeline = SamplingPipeline::new(2048);
    let mut rng = SplitMix64::new(0x0606_2026_FA57_0001);

    let mut clean_empty_grammar = 0usize;

    for case_idx in 0..10_000usize {
        let vocab_sz = match case_idx % 5 {
            0 => 17,
            1 => 64,
            2 => 257,
            3 => 512,
            _ => 1024,
        };

        // Construct pathological logit patterns
        let mut logits = vec![0.0f32; vocab_sz];
        match case_idx % 7 {
            0 => {
                // Extreme positive & negative spikes (`+88`, `-88`, `1e4`, `-1e10`, `1e30`)
                for slot in &mut logits {
                    *slot = rng.range_f32(-1e10, -1e6);
                }
                logits[case_idx % vocab_sz] = 88.0;
                logits[(case_idx + 1) % vocab_sz] = -88.0;
                logits[(case_idx + 2) % vocab_sz] = 1e4;
                if vocab_sz > 5 {
                    logits[5] = 1e30;
                }
            }
            1 => {
                // Exact all-equal ties!
                let tie_val = rng.range_f32(-100.0, 100.0);
                logits.fill(tie_val);
            }
            2 => {
                // Nearly all `-inf` except 1..3 entries
                logits.fill(f32::NEG_INFINITY);
                let live = 1 + (case_idx % 3).min(vocab_sz - 1);
                for k in 0..live {
                    logits[(case_idx + k * 7) % vocab_sz] = rng.range_f32(-500.0, 500.0);
                }
            }
            3 => {
                // Steep linear ramp
                for (i, slot) in logits.iter_mut().enumerate() {
                    *slot = (i as f32) * 5.0 - 1000.0;
                }
            }
            _ => {
                // Dense noisy logits with occasional `-inf`
                for slot in &mut logits {
                    if rng.next_u32().is_multiple_of(11) {
                        *slot = f32::NEG_INFINITY;
                    } else {
                        *slot = rng.range_f32(-200.0, 200.0);
                    }
                }
                logits[0] = 0.0;
            }
        }

        // Optional 1-hot or sparse grammar bitmask (plus empty mask error verification!)
        let mask_opt = if case_idx.is_multiple_of(25) {
            // Completely zeroed mask -> must cleanly return `SamplerError::EmptyGrammarSupport`!
            let words = vec![0u64; vocab_sz.div_ceil(64)];
            Some(AllowedTokenMask::from_words(
                words, vocab_sz, 0, false, None,
            ))
        } else if case_idx.is_multiple_of(4) {
            // 1-hot grammar mask at `hot_id`
            let hot_id = (rng.next_u32() as usize) % vocab_sz;
            logits[hot_id] = rng.range_f32(-50.0, 50.0);
            let mut words = vec![0u64; vocab_sz.div_ceil(64)];
            words[hot_id >> 6] |= 1u64 << (hot_id & 63);
            Some(AllowedTokenMask::from_words(
                words,
                vocab_sz,
                1,
                false,
                Some(hot_id as u32),
            ))
        } else {
            None
        };

        let temp = match case_idx % 6 {
            0 => 0.0,  // greedy
            1 => 1e-5, // ultra-sharp near-zero temp
            2 => 0.05, // sharp
            3 => 0.85, // standard
            4 => 2.5,  // high entropy
            _ => 50.0, // extreme flat temp
        };
        let top_k = match case_idx % 4 {
            0 => Some(1),
            1 => Some(8),
            2 => Some(32),
            _ => None,
        };
        let top_p = match case_idx % 4 {
            0 => 0.0001,
            1 => 0.5,
            2 => 0.92,
            _ => 1.0,
        };
        let min_p = match case_idx % 4 {
            0 => 0.99,
            1 => 0.15,
            _ => 0.0,
        };

        let params = SamplingConfig {
            temperature: temp,
            top_k,
            top_p,
            min_p,
            repetition_penalty: if case_idx.is_multiple_of(3) {
                1.25
            } else {
                1.0
            },
            frequency_penalty: if case_idx.is_multiple_of(5) { 0.4 } else { 0.0 },
            presence_penalty: if case_idx.is_multiple_of(7) { 0.6 } else { 0.0 },
            seed: 0xABCD_0000 + (case_idx as u64),
            stop_token_ids: Vec::new(),
            ignore_eos: case_idx.is_multiple_of(9),
            logit_bias: vec![(0, 1.5), (3, -2.5)],
            top_logprobs: 5,
        };

        let history = [0u32, 1, 1, 3, 7, 7, 7];
        match pipeline.compute_distribution(&logits, &history, &params, mask_opt.as_ref(), Some(2))
        {
            Ok(dist) => {
                let sum = dist.sum_probs();
                assert!(
                    sum.is_finite() && (sum - 1.0).abs() <= 1e-5,
                    "prob sum drift at case {case_idx}: sum={sum}"
                );

                // Verify 100% deterministic `(seed, params)` reproducibility!
                let mut r1 = ChaCha8Rng::from_seed_u64(params.seed);
                let mut r2 = ChaCha8Rng::from_seed_u64(params.seed);
                let s1 = pipeline
                    .sample_token(
                        &logits,
                        &history,
                        &params,
                        mask_opt.as_ref(),
                        Some(2),
                        &mut r1,
                    )
                    .unwrap();
                let s2 = pipeline
                    .sample_token(
                        &logits,
                        &history,
                        &params,
                        mask_opt.as_ref(),
                        Some(2),
                        &mut r2,
                    )
                    .unwrap();
                assert_eq!(s1.token_id, s2.token_id);
                assert!(s1.logprob.is_finite());
                assert!(s1.top_logprobs.iter().all(|lp| lp.logprob.is_finite()));
            }
            Err(SamplerError::EmptyGrammarSupport) => {
                clean_empty_grammar += 1;
            }
            Err(other) => panic!("unexpected sampler error at case {case_idx}: {other}"),
        }
    }

    assert!(clean_empty_grammar >= 400);
    println!(
        "[M06 Telemetry] 10,000 pathological logits vectors verified (0 NaN/Inf, clean EmptyGrammarSupport={clean_empty_grammar})"
    );
}

#[test]
fn shannon_entropy_monotonicity_across_temperature_topk_topp_minp() {
    let pipeline = SamplingPipeline::new(512);
    let mut rng = SplitMix64::new(0xE472_0911);
    let logits: Vec<f32> = (0..256).map(|_| rng.range_f32(-4.0, 4.0)).collect();

    // 1. Entropy strictly non-decreasing as `temperature` increases!
    let temps = [0.0f32, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0];
    let mut prev_h = -1.0f64;
    for &t in &temps {
        let p = SamplingConfig {
            temperature: t,
            ..SamplingConfig::default()
        };
        let h = pipeline
            .compute_distribution(&logits, &[], &p, None, None)
            .unwrap()
            .shannon_entropy();
        assert!(
            h + 1e-6 >= prev_h,
            "entropy decreased with rising temperature at T={t}: {h} < {prev_h}"
        );
        prev_h = h;
    }

    // 2. Entropy non-decreasing as `top_k` widens (`1 < 2 < 4 < 8 < 16 < 64 < 256`)!
    prev_h = -1.0;
    for &k in &[1usize, 2, 4, 8, 16, 64, 256] {
        let p = SamplingConfig {
            temperature: 1.0,
            top_k: Some(k),
            ..SamplingConfig::default()
        };
        let h = pipeline
            .compute_distribution(&logits, &[], &p, None, None)
            .unwrap()
            .shannon_entropy();
        assert!(h + 1e-6 >= prev_h, "entropy decreased as top_k grew to {k}");
        prev_h = h;
    }

    // 3. Entropy non-decreasing as `top_p` widens (`0.05 < 0.2 < 0.5 < 0.8 < 0.95 < 1.0`)!
    prev_h = -1.0;
    for &tp in &[0.05f32, 0.2, 0.5, 0.8, 0.95, 1.0] {
        let p = SamplingConfig {
            temperature: 1.0,
            top_p: tp,
            ..SamplingConfig::default()
        };
        let h = pipeline
            .compute_distribution(&logits, &[], &p, None, None)
            .unwrap()
            .shannon_entropy();
        assert!(
            h + 1e-6 >= prev_h,
            "entropy decreased as top_p grew to {tp}"
        );
        prev_h = h;
    }

    // 4. Entropy non-increasing as `min_p` tightens (`0.0 -> 0.05 -> 0.2 -> 0.5 -> 0.95`)!
    prev_h = f64::INFINITY;
    for &mp in &[0.0f32, 0.05, 0.2, 0.5, 0.95] {
        let p = SamplingConfig {
            temperature: 1.0,
            min_p: mp,
            ..SamplingConfig::default()
        };
        let h = pipeline
            .compute_distribution(&logits, &[], &p, None, None)
            .unwrap()
            .shannon_entropy();
        assert!(
            h <= prev_h + 1e-6,
            "entropy increased as min_p tightened to {mp}"
        );
        prev_h = h;
    }

    // 5. Tie-breaking determinism: all-equal logits at `temp = 0` MUST pick token ID 0 (or lowest unmasked ID)!
    let ties = vec![7.0f32; 64];
    let greedy = SamplingConfig::default();
    let mut chacha = ChaCha8Rng::from_seed_u64(1);
    let picked = pipeline
        .sample_token(&ties, &[], &greedy, None, None, &mut chacha)
        .unwrap();
    assert_eq!(picked.token_id, 0);
}
