use crate::error::SchedError;
use novaserve_frontend::{AllowedTokenMask, GrammarMaskCache};
use novaserve_kv::{KvDType, DEFAULT_BLOCK_SIZE};
use novaserve_tensor::{softmax_inplace_f32, ChaCha8Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreemptionMode {
    Swap,
    Recompute,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub num_physical_blocks: usize,
    pub num_swap_blocks: usize,
    pub block_size: usize,
    pub kv_dtype: KvDType,
    pub preemption_mode: PreemptionMode,
    pub enable_radix_cache: bool,
    pub speculative_lookahead: usize,
    pub adaptive_speculative: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 512,
            max_num_seqs: 64,
            num_physical_blocks: 256,
            num_swap_blocks: 256,
            block_size: DEFAULT_BLOCK_SIZE,
            kv_dtype: KvDType::F32,
            preemption_mode: PreemptionMode::Swap,
            enable_radix_cache: true,
            speculative_lookahead: 0,
            adaptive_speculative: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    pub logit_bias: Vec<(u32, f32)>,
    pub top_logprobs: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 42,
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            logit_bias: Vec::new(),
            top_logprobs: 0,
        }
    }
}

#[derive(Clone)]
pub struct RequestParams {
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub n_parallel: usize,
    pub sampling: SamplingConfig,
    pub grammar: Option<Arc<GrammarMaskCache>>,
    pub use_speculative: bool,
}

impl RequestParams {
    pub fn greedy(prompt_tokens: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            max_new_tokens,
            n_parallel: 1,
            sampling: SamplingConfig::default(),
            grammar: None,
            use_speculative: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampledToken {
    pub token_id: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<TokenLogprob>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Aborted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeculativeTrace {
    pub req_id: u64,
    pub child_idx: usize,
    pub lookahead_k: usize,
    pub draft_tokens: Vec<u32>,
    pub draft_probs: Vec<f32>,
    pub target_probs: Vec<f32>,
    pub accepted_mask: Vec<bool>,
    pub bonus_or_residual_token: u32,
    pub accepted_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceOutput {
    pub child_idx: usize,
    pub token_ids: Vec<u32>,
    pub step_logprobs: Vec<SampledToken>,
    pub finish_reason: FinishReason,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletedRequest {
    pub req_id: u64,
    pub prompt_len: usize,
    pub cached_prefix_tokens: usize,
    pub outputs: Vec<SequenceOutput>,
    pub preemptions: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceStepDelta {
    pub req_id: u64,
    pub child_idx: usize,
    pub new_tokens: Vec<u32>,
    pub sampled_infos: Vec<SampledToken>,
    pub finished: Option<FinishReason>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepSummary {
    pub step_index: u64,
    pub scheduled_prefill_tokens: usize,
    pub scheduled_decode_tokens: usize,
    pub preempted_this_step: usize,
    pub swapped_in_this_step: usize,
    pub waiting_groups: usize,
    pub running_groups: usize,
    pub swapped_groups: usize,
    pub deltas: Vec<SequenceStepDelta>,
    pub speculative_traces: Vec<SpeculativeTrace>,
    pub completed: Vec<CompletedRequest>,
    pub rolling_spec_alpha: f32,
    pub active_spec_k: usize,
}

/// Pluggable logit post-processing & sampling trait (implemented natively here and
/// upgraded with full vectorized Top-K/Min-P/Top-P/Penalties in `novaserve-sampler` M06).
pub trait SamplerStrategy: Send + Sync {
    fn compute_probs(
        &self,
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        out_probs: &mut [f32],
    ) -> Result<(), SchedError>;

    fn sample(
        &self,
        raw_logits: &[f32],
        history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        rng: &mut ChaCha8Rng,
    ) -> Result<SampledToken, SchedError>;
}

/// Built-in exact baseline sampler supporting greedy, temperature, Top-K, Top-P, Min-P,
/// logit bias, and grammar bitmasks.
#[derive(Clone, Debug, Default)]
pub struct BuiltinSampler;

impl SamplerStrategy for BuiltinSampler {
    fn compute_probs(
        &self,
        raw_logits: &[f32],
        _history: &[u32],
        params: &SamplingConfig,
        mask: Option<&AllowedTokenMask>,
        eos_token_id: Option<u32>,
        out_probs: &mut [f32],
    ) -> Result<(), SchedError> {
        let v = raw_logits.len();
        out_probs.copy_from_slice(raw_logits);

        for &(tid, bias) in &params.logit_bias {
            if (tid as usize) < v {
                out_probs[tid as usize] += bias;
            }
        }
        if params.ignore_eos {
            if let Some(eos) = eos_token_id {
                if (eos as usize) < v {
                    out_probs[eos as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(m) = mask {
            m.apply_to_logits(out_probs);
        }

        // Check support non-empty
        let mut best_id = usize::MAX;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &val) in out_probs.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_id = i;
            }
        }
        if best_id == usize::MAX || !best_val.is_finite() {
            return Err(SchedError::EmptyGrammarSupport);
        }

        // Greedy fast-path: exact 1-hot distribution on lowest-token-ID argmax
        if params.temperature <= 1e-6 {
            out_probs.fill(0.0);
            out_probs[best_id] = 1.0;
            return Ok(());
        }

        let inv_temp = 1.0 / params.temperature;
        for slot in out_probs.iter_mut() {
            if slot.is_finite() {
                *slot *= inv_temp;
            }
        }
        softmax_inplace_f32(out_probs)?;

        // Top-K / Min-P / Top-P filtering if active
        let needs_trunc = params.top_k.is_some_and(|k| k > 0 && k < v)
            || params.min_p > 0.0
            || params.top_p < 1.0;
        if needs_trunc {
            let mut cands: Vec<(usize, f32)> = out_probs
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, p)| p > 0.0)
                .collect();
            cands.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            if let Some(k) = params.top_k {
                if k > 0 && cands.len() > k {
                    cands.truncate(k);
                }
            }
            if params.min_p > 0.0 && !cands.is_empty() {
                let thresh = cands[0].1 * params.min_p;
                let keep = cands
                    .iter()
                    .take_while(|&&(_, p)| p >= thresh)
                    .count()
                    .max(1);
                cands.truncate(keep);
            }
            if params.top_p < 1.0 && !cands.is_empty() {
                let mut cum = 0.0f32;
                let mut keep = 0usize;
                for &(_, p) in &cands {
                    cum += p;
                    keep += 1;
                    if cum >= params.top_p {
                        break;
                    }
                }
                cands.truncate(keep.max(1));
            }

            let sum: f64 = cands.iter().map(|&(_, p)| p as f64).sum();
            out_probs.fill(0.0);
            if sum > 0.0 {
                let inv = (1.0 / sum) as f32;
                for (idx, p) in cands {
                    out_probs[idx] = p * inv;
                }
            } else {
                out_probs[best_id] = 1.0;
            }
        }
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
        let mut probs = vec![0.0f32; raw_logits.len()];
        self.compute_probs(raw_logits, history, params, mask, eos_token_id, &mut probs)?;

        // Compute logprobs from raw temperature-scaled softmax (or post-filter probs) for reporting
        let chosen = sample_categorical(&probs, rng);
        let chosen_lp = (probs[chosen].max(1e-38)).ln();

        let top_logprobs = collect_top_logprobs(&probs, params.top_logprobs);
        Ok(SampledToken {
            token_id: chosen as u32,
            logprob: chosen_lp,
            top_logprobs,
        })
    }
}

#[inline]
pub fn sample_categorical(probs: &[f32], rng: &mut ChaCha8Rng) -> usize {
    let u = rng.next_f64();
    let mut cum = 0.0f64;
    let mut last_nonzero = 0usize;
    for (idx, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_nonzero = idx;
            cum += p as f64;
            if u < cum {
                return idx;
            }
        }
    }
    last_nonzero
}

pub fn collect_top_logprobs(probs: &[f32], top_n: usize) -> Vec<TokenLogprob> {
    if top_n == 0 {
        return Vec::new();
    }
    let mut pairs: Vec<(u32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > 0.0)
        .map(|(i, &p)| (i as u32, p))
        .collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(top_n)
        .map(|(tid, p)| TokenLogprob {
            token_id: tid,
            logprob: p.max(1e-38).ln(),
        })
        .collect()
}
