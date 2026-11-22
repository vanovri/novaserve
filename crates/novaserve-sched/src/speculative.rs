//! Leviathan-Kalman-Matias (ICML 2023) Exact Speculative Rejection Sampling &
//! Rolling-Acceptance ($\alpha$) Adaptive Lookahead Controller.

use crate::types::sample_categorical;
use novaserve_tensor::ChaCha8Rng;

/// Rolling EWMA controller dynamically scaling speculative lookahead $K \in [0, K_{\max}]$,
/// automatically disabling draft overhead when $\hat{\alpha} < 0.25$ on high-entropy prompts.
#[derive(Clone, Debug)]
pub struct AdaptiveLookaheadController {
    max_k: usize,
    adaptive: bool,
    rolling_alpha: f32,
    total_drafted: u64,
    total_accepted: u64,
}

impl AdaptiveLookaheadController {
    pub fn new(max_k: usize, adaptive: bool) -> Self {
        Self {
            max_k,
            adaptive,
            rolling_alpha: 0.70,
            total_drafted: 0,
            total_accepted: 0,
        }
    }

    #[inline]
    pub fn current_k(&self) -> usize {
        if self.max_k == 0 {
            return 0;
        }
        if !self.adaptive {
            return self.max_k;
        }
        if self.rolling_alpha < 0.25 && self.total_drafted >= 16 {
            0
        } else if self.rolling_alpha < 0.45 && self.total_drafted >= 16 {
            (self.max_k / 2).max(1)
        } else {
            self.max_k
        }
    }

    #[inline]
    pub fn record_step(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        self.total_drafted += drafted as u64;
        self.total_accepted += accepted as u64;
        let step_alpha = (accepted as f32) / (drafted as f32);
        self.rolling_alpha = 0.85 * self.rolling_alpha + 0.15 * step_alpha;
    }

    #[inline]
    pub fn rolling_alpha(&self) -> f32 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.rolling_alpha
        }
    }

    #[inline]
    pub fn empirical_alpha(&self) -> f64 {
        if self.total_drafted == 0 {
            0.0
        } else {
            (self.total_accepted as f64) / (self.total_drafted as f64)
        }
    }
}

#[derive(Clone, Debug)]
pub struct RejectionOutcome {
    /// Number of draft tokens accepted (`0 ..= K`).
    pub accepted_draft_count: usize,
    /// Per-position acceptance boolean (`len == K`).
    pub accepted_mask: Vec<bool>,
    /// Either the residual replacement token $t_i \sim \text{norm}(\max(0, p_i - q_i))$
    /// when rejected at $i$, OR the bonus token $t_{K+1} \sim p_{K+1}$ when all $K$ accepted!
    pub final_token: u32,
    pub final_token_target_prob: f32,
}

/// Exact Leviathan et al. rejection + residual sampler over $K$ drafted steps.
///
/// - `draft_tokens`: `[d_1, ..., d_K]`
/// - `draft_dists`: `K` probability vectors `[q_1(·), ..., q_K(·)]` each of length `V`
/// - `target_dists`: `K + 1` probability vectors `[p_1(·), ..., p_K(·), p_{K+1}(·)]` each of length `V`
///
/// Preserves the **exact mathematical target distribution $p(x)$** for ANY $q(x)$,
/// including adversarial or uniform noise!
pub fn leviathan_rejection_sample(
    draft_tokens: &[u32],
    draft_dists: &[Vec<f32>],
    target_dists: &[Vec<f32>],
    rng: &mut ChaCha8Rng,
) -> RejectionOutcome {
    let k = draft_tokens.len();
    assert_eq!(draft_dists.len(), k);
    assert_eq!(target_dists.len(), k + 1);

    let mut accepted_mask = vec![false; k];

    for i in 0..k {
        let d_tok = draft_tokens[i] as usize;
        let q_vec = &draft_dists[i];
        let p_vec = &target_dists[i];
        let q_d = q_vec[d_tok] as f64;
        let p_d = p_vec[d_tok] as f64;

        let accept_prob = if q_d <= 0.0 {
            0.0
        } else {
            (p_d / q_d).min(1.0)
        };

        let u = rng.next_f64();
        if u <= accept_prob && accept_prob > 0.0 {
            accepted_mask[i] = true;
        } else {
            // Rejected at index `i`: sample replacement token from residual `norm(max(0, p_i - q_i))`!
            let residual_tok = sample_residual_distribution(p_vec, q_vec, rng);
            let tp = p_vec[residual_tok];
            return RejectionOutcome {
                accepted_draft_count: i,
                accepted_mask,
                final_token: residual_tok as u32,
                final_token_target_prob: tp,
            };
        }
    }

    // All `K` draft tokens accepted! Sample bonus token `t_{K+1} ~ p_{K+1}`!
    let p_bonus = &target_dists[k];
    let bonus_tok = sample_categorical(p_bonus, rng);
    let tp = p_bonus[bonus_tok];
    RejectionOutcome {
        accepted_draft_count: k,
        accepted_mask,
        final_token: bonus_tok as u32,
        final_token_target_prob: tp,
    }
}

/// Samples strictly from $p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$
/// using `f64` accumulation to eliminate floating-point bias.
pub fn sample_residual_distribution(p: &[f32], q: &[f32], rng: &mut ChaCha8Rng) -> usize {
    debug_assert_eq!(p.len(), q.len());
    let mut sum_pos = 0.0f64;
    for (&pv, &qv) in p.iter().zip(q.iter()) {
        let diff = (pv as f64) - (qv as f64);
        if diff > 0.0 {
            sum_pos += diff;
        }
    }

    if sum_pos <= 1e-15 {
        // Only reachable if `p == q` identically everywhere (where accept_prob == 1.0)
        return sample_categorical(p, rng);
    }

    let threshold = rng.next_f64() * sum_pos;
    let mut cum = 0.0f64;
    let mut last_pos = 0usize;
    for (idx, (&pv, &qv)) in p.iter().zip(q.iter()).enumerate() {
        let diff = (pv as f64) - (qv as f64);
        if diff > 0.0 {
            last_pos = idx;
            cum += diff;
            if threshold < cum {
                return idx;
            }
        }
    }
    last_pos
}
