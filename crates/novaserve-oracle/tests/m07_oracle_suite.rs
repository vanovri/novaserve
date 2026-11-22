use novaserve_oracle::{run_3_regime_speculative_proof, run_chaos_batch_equivalence_campaign};

#[test]
fn chaos_5000_continuous_batching_preemption_and_radix_equivalence_fuzzer() {
    let report = run_chaos_batch_equivalence_campaign(5_000);
    assert_eq!(report.scenarios_verified, 5_000);
    assert!(
        report.total_preemptions_triggered > 1_000,
        "expected heavy Swap/Recompute preemptions across 5,000 scenarios, got {}",
        report.total_preemptions_triggered
    );
    assert!(report.total_aborts_injected > 1_000);
    assert!(report.max_logprob_abs_diff <= 1e-4);

    println!(
        "[M07 Telemetry] 5,000 Chaos Scenarios PASS | sequences_checked={} | preemptions={} | aborts={} | max_logprob_diff={:.3e} (<= 1e-4)",
        report.total_sequences_checked,
        report.total_preemptions_triggered,
        report.total_aborts_injected,
        report.max_logprob_abs_diff
    );
}

#[test]
fn speculative_decoding_3_regime_statistical_distribution_proof() {
    // 32,000 K=4 speculative rollouts per regime (160,000 speculative tokens per regime!)
    let report = run_3_regime_speculative_proof(32_000);
    assert_eq!(report.regimes.len(), 3);

    for r in &report.regimes {
        println!(
            "[M07 Telemetry] Regime={:<11} | N={} | alpha={:5.1}% | TVD_pos0={:.5} (< 0.015) | TVD_2gram_pos1={:.5} (< 0.015) | chi2={:.2} (df={})",
            r.regime_name,
            r.total_transitions,
            r.empirical_alpha * 100.0,
            r.tvd_pos0_vs_exact,
            r.tvd_bigram_pos1_vs_exact,
            r.chi_square_pos0,
            r.degrees_of_freedom
        );

        assert!(
            r.tvd_pos0_vs_exact < 0.015,
            "Regime {} failed exact pos0 TVD gate: {:.5} >= 0.015",
            r.regime_name,
            r.tvd_pos0_vs_exact
        );
        assert!(
            r.tvd_bigram_pos1_vs_exact < 0.015,
            "Regime {} failed conditional 2-gram pos1 TVD gate: {:.5} >= 0.015",
            r.regime_name,
            r.tvd_bigram_pos1_vs_exact
        );
        // For df = 15 (`top_k = 16`), chi2 critical value at alpha = 1e-4 is ~44.26
        assert!(
            r.chi_square_pos0 < 45.0,
            "Regime {} failed Pearson chi-square goodness-of-fit: chi2={:.2} (df={})",
            r.regime_name,
            r.chi_square_pos0,
            r.degrees_of_freedom
        );
    }

    let alpha_aligned = report.regimes[0].empirical_alpha;
    let alpha_mismatch = report.regimes[1].empirical_alpha;
    let alpha_adv = report.regimes[2].empirical_alpha;
    assert!(
        alpha_aligned > 0.55 && alpha_aligned > alpha_mismatch && alpha_mismatch > alpha_adv,
        "unexpected regime acceptance ordering: aligned={alpha_aligned:.3}, mismatch={alpha_mismatch:.3}, adv={alpha_adv:.3}"
    );
}
