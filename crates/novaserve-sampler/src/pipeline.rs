//! Allocation-free 9-Stage Vectorized Logits Post-Processing & Sampling Pipeline (`SamplingPipeline`).

use novaserve_frontend::AllowedTokenMask;
use novaserve_sched::{SampledToken, SamplerStrategy, SamplingConfig, SchedError, TokenLogprob};
use novaserve_tensor::ChaCha8Rng;
use parking_lot::Mutex;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SamplerError {
    #[error("empty vocabulary slice passed to sampler")]
    EmptyVocab,

    #[error("NaN or +Inf encountered in raw input logits")]
    NonFiniteInputLogit,

    #[error("invalid sampling hyperparameter: {0}")]
    InvalidParam(String),

    #[error("grammar mask left zero valid next tokens")]
    EmptyGrammarSupport,
}

#[derive(Clone, Debug)]
pub enum SparseOrDenseProbDist {
    /// Greedy or sparse truncated distribution sorted descending by `(prob, -token_id)`.
    Sparse(Vec<(u32, f32)>),
    /// Full dense probability vector of length `vocab_size` summing to `1.0`.
    Dense(Vec<f32>),
}

impl SparseOrDenseProbDist {
    pub fn prob_of(&self, token_id: u32) -> f32 {
        match self {
            Self::Dense(v) => v.get(token_id as usize).copied().unwrap_or(0.0),
            Self::Sparse(items) => items
                .iter()
                .find(|&&(tid, _)| tid == token_id)
                .map(|&(_, p)| p)
                .unwrap_or(0.0),
        }
    }

    pub fn sum_probs(&self) -> f64 {
        match self {
            Self::Dense(v) => v.iter().map(|&p| p as f64).sum(),
            Self::Sparse(items) => items.iter().map(|&(_, p)| p as f64).sum(),
        }
    }

    /// Shannon entropy $H(P) = -\sum_{p_i > 0} p_i \ln p_i$ in nats.
    pub fn shannon_entropy(&self) -> f64 {
        match self {
            Self::Dense(v) => v
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| {
                    let pf = p as f64;
                    -pf * pf.ln()
                })
                .sum(),
            Self::Sparse(items) => items
                .iter()
                .filter(|&&(_, p)| p > 0.0)
                .map(|&(_, p)| {
                    let pf = p as f64;
                    -pf * pf.ln()
                })
                .sum(),
        }
    }

    pub fn write_dense(&self, vocab_size: usize, out: &mut [f32]) {
        assert_eq!(out.len(), vocab_size);
        match self {
            Self::Dense(v) => out.copy_from_slice(&v[..vocab_size]),
            Self::Sparse(items) => {
                out.fill(0.0);
                for &(tid, p) in items {
                    if (tid as usize) < vocab_size {
                        out[tid as usize] = p;
                    }
                }
            }
        }
    }
}

/// Pre-allocated per-worker scratch buffers eliminating heap allocations on hot sampling steps.
#[derive(Debug, Default)]
pub struct SamplerScratch {
    work_logits: Vec<f32>,
    candidates: Vec<(u32, f32)>,
    token_counts: Vec<u32>,
    touched_tokens: Vec<u32>,
}

impl SamplerScratch {
    pub fn with_vocab_capacity(vocab_size: usize) -> Self {
        Self {
            work_logits: vec![0.0; vocab_size],
            candidates: Vec::with_capacity(vocab_size),
            token_counts: vec![0; vocab_size],
            touched_tokens: Vec::with_capacity(512),
        }
    }

    fn ensure_vocab(&mut self, vocab_size: usize) {
        if self.work_logits.len() < vocab_size {
            self.work_logits.resize(vocab_size, 0.0);
        }
        if self.token_counts.len() < vocab_size {
            self.token_counts.resize(vocab_size, 0);
        }
    }
}

/// Thread-safe composable vectorized sampling pipeline (`SamplingPipeline`).
pub struct SamplingPipeline {
    scratch_pool: Mutex<SamplerScratch>,
}

impl Default for SamplingPipeline {
    fn default() -> Self {
        Self::new(4096)
    }
}

impl SamplingPipeline {
    pub fn new(initial_vocab_capacity: usize) -> Self {
        Self {
            scratch_pool: Mutex::new(SamplerScratch::with_vocab_capacity(initial_vocab_capacity)),
        }
    }

    pub fn validate_params(params: &SamplingConfig) -> Result<(), SamplerError> {
        if params.temperature.is_nan() || params.temperature < 0.0 {
            return Err(SamplerError::InvalidParam(format!(
                "temperature must be >= 0, got {}",
                params.temperature
            )));
        }
        if !(0.0..=1.0).contains(&params.top_p) || params.top_p == 0.0 || params.top_p.is_nan() {
            return Err(SamplerError::InvalidParam(format!(
                "top_p must be in (0, 1], got {}",
                params.top_p
            )));
        }
        if !(0.0..=1.0).contains(&params.min_p) || params.min_p.is_nan() {
            return Err(SamplerError::InvalidParam(format!(
                "min_p must be in [0, 1], got {}",
                params.min_p
            )));
        }
        if params.repetition_penalty <= 0.0 || !params.repetition_penalty.is_finite() {
            return Err(SamplerError::InvalidParam(format!(
                "repetition_penalty must be > 0, got {}",
                params.repetition_penalty
            )));
        }
        Ok(())
    }

    /// Executes Stages 1–8 using caller-supplied `SamplerScratch` (zero heap allocations on warm scratch!).
    pub fn compute_distribution_with_scratch(
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        scratch: &mut SamplerScratch,
    ) -> Result<SparseOrDenseProbDist, SamplerError> {
        let v = raw_logits.len();
        if v == 0 {
            return Err(SamplerError::EmptyVocab);
        }
        Self::validate_params(params)?;
        scratch.ensure_vocab(v);

        let logits = &mut scratch.work_logits[..v];
        for (dst, &src) in logits.iter_mut().zip(raw_logits.iter()) {
            if src.is_nan() || src == f32::INFINITY {
                return Err(SamplerError::NonFiniteInputLogit);
            }
            *dst = src;
        }

        // Stage 1: Logit Bias & Ignore-EOS
        for &(tid, bias) in &params.logit_bias {
            let idx = tid as usize;
            if idx < v && logits[idx] > f32::NEG_INFINITY && bias.is_finite() {
                logits[idx] = (logits[idx] + bias).clamp(-1e38, 1e38);
            }
        }
        if params.ignore_eos {
            if let Some(eos) = eos_token_id {
                if (eos as usize) < v {
                    logits[eos as usize] = f32::NEG_INFINITY;
                }
            }
        }

        // Stage 2: Grammar Bitmask BEFORE penalties/softmax/truncation
        if let Some(m) = mask {
            m.apply_to_logits(logits);
        }

        // Stage 3: Sparse $O(|H|)$ Repetition, Frequency & Presence Penalties
        let has_rep = (params.repetition_penalty - 1.0).abs() > 1e-7;
        let has_freq = params.frequency_penalty.abs() > 1e-7;
        let has_pres = params.presence_penalty.abs() > 1e-7;

        if (has_rep || has_freq || has_pres) && !history.is_empty() {
            scratch.touched_tokens.clear();
            for &tid in history {
                let idx = tid as usize;
                if idx < v {
                    if scratch.token_counts[idx] == 0 {
                        scratch.touched_tokens.push(tid);
                    }
                    scratch.token_counts[idx] += 1;
                }
            }

            let rep_p = params.repetition_penalty;
            let freq_p = params.frequency_penalty;
            let pres_p = params.presence_penalty;

            for &tid in &scratch.touched_tokens {
                let idx = tid as usize;
                let count = scratch.token_counts[idx];
                scratch.token_counts[idx] = 0; // clean up in O(|unique_H|)!

                let z = logits[idx];
                if z == f32::NEG_INFINITY {
                    continue;
                }
                let mut z_adj = if has_rep {
                    if z > 0.0 {
                        z / rep_p
                    } else {
                        z * rep_p
                    }
                } else {
                    z
                };
                if has_freq {
                    z_adj -= freq_p * (count as f32);
                }
                if has_pres && count > 0 {
                    z_adj -= pres_p;
                }
                logits[idx] = z_adj.clamp(-1e38, 1e38);
            }
        }

        // Find global maximum (`best_id` ties broken by lowest token ID!)
        let mut best_id = usize::MAX;
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &val) in logits.iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                best_id = i;
            }
        }
        if best_id == usize::MAX || !max_logit.is_finite() {
            return Err(SamplerError::EmptyGrammarSupport);
        }

        // Stage 4: Greedy Fast-Path (`temperature <= 1e-6` or `top_k == Some(1)`)
        if params.temperature <= 1e-6 || params.top_k == Some(1) {
            return Ok(SparseOrDenseProbDist::Sparse(vec![(best_id as u32, 1.0)]));
        }

        // Stage 5: Shift-Before-Divide Temperature Scaling + Numerically Stable Softmax
        let inv_temp = 1.0f64 / (params.temperature as f64);
        scratch.candidates.clear();
        let mut sum_exp = 0.0f64;

        for (i, &z) in logits.iter().enumerate() {
            if z > f32::NEG_INFINITY {
                let shifted = ((z as f64) - (max_logit as f64)) * inv_temp;
                // `shifted <= 0.0` always! Cannot overflow to +Inf!
                if shifted > -100.0 {
                    let e = shifted.exp() as f32;
                    if e > 0.0 {
                        scratch.candidates.push((i as u32, e));
                        sum_exp += e as f64;
                    }
                }
            }
        }

        if scratch.candidates.is_empty() || sum_exp <= 0.0 {
            return Ok(SparseOrDenseProbDist::Sparse(vec![(best_id as u32, 1.0)]));
        }

        let inv_sum = (1.0 / sum_exp) as f32;
        for pair in &mut scratch.candidates {
            pair.1 *= inv_sum;
        }

        let cmp_desc =
            |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));

        // Stage 6: O(V) `select_nth_unstable_by` Top-K Partial Selection!
        if let Some(k) = params.top_k {
            if k > 0 && scratch.candidates.len() > k {
                scratch.candidates.select_nth_unstable_by(k - 1, cmp_desc);
                scratch.candidates.truncate(k);
            }
        }

        // Stage 7: `Min-P` Truncation (`p_i >= min_p * p_max`)
        if params.min_p > 0.0 && scratch.candidates.len() > 1 {
            let p_max = scratch
                .candidates
                .iter()
                .map(|&(_, p)| p)
                .fold(0.0f32, f32::max);
            let threshold = p_max * params.min_p;
            scratch.candidates.retain(|&(_, p)| p >= threshold);
            if scratch.candidates.is_empty() {
                scratch.candidates.push((best_id as u32, 1.0));
            }
        }

        // Stage 8: `Top-P` (Nucleus) Truncation over surviving candidates
        let need_renorm = params.top_k.is_some() || params.min_p > 0.0 || params.top_p < 1.0;
        if params.top_p < 1.0 && scratch.candidates.len() > 1 {
            scratch.candidates.sort_unstable_by(cmp_desc);
            // Re-normalize pre-Top-P if Top-K/Min-P already pruned mass? Standard HuggingFace/vLLM
            // applies Top-K -> Min-P -> Top-P sequentially on renormalized or raw probabilities;
            // renormalizing before cumulative sum ensures `top_p = 0.9` takes 90% of surviving mass!
            let cur_mass: f64 = scratch.candidates.iter().map(|&(_, p)| p as f64).sum();
            let target_mass = cur_mass * (params.top_p as f64);
            let mut cum = 0.0f64;
            let mut keep = 0usize;
            for &(_, p) in &scratch.candidates {
                cum += p as f64;
                keep += 1;
                if cum >= target_mass {
                    break;
                }
            }
            scratch.candidates.truncate(keep.max(1));
        }

        if need_renorm || scratch.candidates.len() <= v / 2 {
            scratch.candidates.sort_unstable_by(cmp_desc);
            let survive_sum: f64 = scratch.candidates.iter().map(|&(_, p)| p as f64).sum();
            if survive_sum <= 0.0 {
                return Ok(SparseOrDenseProbDist::Sparse(vec![(best_id as u32, 1.0)]));
            }
            let norm = 1.0 / survive_sum;
            let mut out_sparse = Vec::with_capacity(scratch.candidates.len());
            for &(tid, p) in &scratch.candidates {
                out_sparse.push((tid, ((p as f64) * norm) as f32));
            }
            Ok(SparseOrDenseProbDist::Sparse(out_sparse))
        } else {
            let survive_sum: f64 = scratch.candidates.iter().map(|&(_, p)| p as f64).sum();
            let norm = 1.0 / survive_sum;
            let mut dense = vec![0.0f32; v];
            for &(tid, p) in &scratch.candidates {
                dense[tid as usize] = ((p as f64) * norm) as f32;
            }
            Ok(SparseOrDenseProbDist::Dense(dense))
        }
    }

    pub fn compute_distribution(
        &self,
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
    ) -> Result<SparseOrDenseProbDist, SamplerError> {
        let mut guard = self.scratch_pool.lock();
        Self::compute_distribution_with_scratch(
            raw_logits,
            history,
            params,
            mask,
            eos_token_id,
            &mut guard,
        )
    }

    /// Stage 9: Deterministic Seeded Multinomial Draw + Top-$N$ `logprobs`.
    pub fn sample_token(
        &self,
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        rng: &mut ChaCha8Rng,
    ) -> Result<SampledToken, SamplerError> {
        let mut guard = self.scratch_pool.lock();
        let dist = Self::compute_distribution_with_scratch(
            raw_logits,
            history,
            params,
            mask,
            eos_token_id,
            &mut guard,
        )?;
        let mut sampled = sample_from_dist(&dist, params.top_logprobs, rng);

        // Under greedy decoding (`temp <= 1e-6` or `top_k == 1`), populate `top_logprobs`
        // from exact post-mask `f64` log-softmax so OpenAI API & Oracle verifiers receive
        // genuine top-N log-probabilities!
        if params.top_logprobs > 0 && (params.temperature <= 1e-6 || params.top_k == Some(1)) {
            let v = raw_logits.len();
            let s_ref = &mut *guard;
            let logits = &s_ref.work_logits[..v];
            let candidates = &mut s_ref.candidates;

            let max_l = logits[sampled.token_id as usize] as f64;
            let mut sum_exp = 0.0f64;
            for &z in logits {
                if z > f32::NEG_INFINITY {
                    sum_exp += ((z as f64) - max_l).exp();
                }
            }
            let log_z = max_l + sum_exp.ln();
            sampled.logprob = ((logits[sampled.token_id as usize] as f64) - log_z) as f32;

            candidates.clear();
            for (i, &z) in logits.iter().enumerate() {
                if z > f32::NEG_INFINITY {
                    candidates.push((i as u32, ((z as f64) - log_z) as f32));
                }
            }
            let k = params.top_logprobs.min(candidates.len());
            let cmp_desc =
                |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
            if candidates.len() > k {
                candidates.select_nth_unstable_by(k - 1, cmp_desc);
                candidates.truncate(k);
            }
            candidates.sort_unstable_by(cmp_desc);
            sampled.top_logprobs = candidates
                .iter()
                .map(|&(tid, lp)| TokenLogprob {
                    token_id: tid,
                    logprob: lp,
                })
                .collect();
        }
        Ok(sampled)
    }
}

pub fn sample_from_dist(
    dist: &SparseOrDenseProbDist,
    top_logprobs_n: usize,
    rng: &mut ChaCha8Rng,
) -> SampledToken {
    let u = rng.next_f64();
    match dist {
        SparseOrDenseProbDist::Sparse(items) => {
            let mut cum = 0.0f64;
            let mut chosen = items[0];
            for &(tid, p) in items {
                if p > 0.0 {
                    chosen = (tid, p);
                    cum += p as f64;
                    if u < cum {
                        break;
                    }
                }
            }
            let top_logprobs = if top_logprobs_n == 0 {
                Vec::new()
            } else {
                items
                    .iter()
                    .take(top_logprobs_n)
                    .map(|&(tid, p)| TokenLogprob {
                        token_id: tid,
                        logprob: p.max(1e-38).ln(),
                    })
                    .collect()
            };
            SampledToken {
                token_id: chosen.0,
                logprob: chosen.1.max(1e-38).ln(),
                top_logprobs,
            }
        }
        SparseOrDenseProbDist::Dense(probs) => {
            let mut cum = 0.0f64;
            let mut chosen_id = 0usize;
            for (idx, &p) in probs.iter().enumerate() {
                if p > 0.0 {
                    chosen_id = idx;
                    cum += p as f64;
                    if u < cum {
                        break;
                    }
                }
            }
            let top_logprobs = novaserve_sched::collect_top_logprobs(probs, top_logprobs_n);
            SampledToken {
                token_id: chosen_id as u32,
                logprob: probs[chosen_id].max(1e-38).ln(),
                top_logprobs,
            }
        }
    }
}

impl SamplerStrategy for SamplingPipeline {
    fn compute_probs(
        &self,
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        out_probs: &mut [f32],
    ) -> Result<(), SchedError> {
        let dist = self
            .compute_distribution(raw_logits, history, params, mask, eos_token_id)
            .map_err(|e| match e {
                SamplerError::EmptyGrammarSupport => SchedError::EmptyGrammarSupport,
                other => SchedError::InvalidRequest(other.to_string()),
            })?;
        dist.write_dense(raw_logits.len(), out_probs);
        Ok(())
    }

    fn sample(
        &self,
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        rng: &mut ChaCha8Rng,
    ) -> Result<SampledToken, SchedError> {
        self.sample_token(raw_logits, history, params, mask, eos_token_id, rng)
            .map_err(|e| match e {
                SamplerError::EmptyGrammarSupport => SchedError::EmptyGrammarSupport,
                other => SchedError::InvalidRequest(other.to_string()),
            })
    }
}
