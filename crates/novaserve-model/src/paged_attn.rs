//! Zero-copy Scatter KV Write & Milakov-Gimelshein Online-Softmax Paged FlashDecoding
//! (`paged_attention_gqa`) with parallel block-partition log-sum-exp reduction.

use crate::batch::BatchInput;
use crate::error::ModelError;
use novaserve_kv::{KvDType, KvSlab, LogicalBlockTable};
use novaserve_tensor::simd;

#[cfg(all(not(target_arch = "wasm32"), not(miri)))]
use rayon::prelude::*;

/// Scatter-writes `K` and `V` vectors (`[batch_tokens, n_kv_heads * head_dim]`) directly into
/// each token's assigned `(physical_block_id, offset_in_block)` inside `KvSlab`.
pub fn scatter_write_kv_layer(
    slab: &mut KvSlab,
    layer: usize,
    k_batch: &[f32],
    v_batch: &[f32],
    batch: &BatchInput,
) -> Result<(), ModelError> {
    let kv_dim = slab.config().num_kv_heads * slab.config().head_dim;
    let n_toks = batch.len();
    if k_batch.len() != n_toks * kv_dim || v_batch.len() != n_toks * kv_dim {
        return Err(ModelError::InvalidBatch(
            "scatter_write_kv_layer buffer shape mismatch".into(),
        ));
    }

    for t in 0..n_toks {
        let bid = batch.kv_write_block_id[t];
        let off = batch.kv_write_block_offset[t];
        slab.write_token_kv(
            layer,
            bid,
            off,
            &k_batch[t * kv_dim..(t + 1) * kv_dim],
            &v_batch[t * kv_dim..(t + 1) * kv_dim],
        )?;
    }
    Ok(())
}

#[derive(Clone)]
struct OnlineSoftmaxState {
    max_score: f64,
    sum_exp: f64,
    acc_out: Vec<f64>,
}

impl OnlineSoftmaxState {
    #[inline]
    fn new(head_dim: usize) -> Self {
        Self {
            max_score: f64::NEG_INFINITY,
            sum_exp: 0.0,
            acc_out: vec![0.0; head_dim],
        }
    }

    #[inline]
    fn update(&mut self, score: f64, v_slice: &[f32]) {
        let m_new = self.max_score.max(score);
        let alpha = if self.max_score == f64::NEG_INFINITY {
            0.0
        } else {
            (self.max_score - m_new).exp()
        };
        let beta = (score - m_new).exp();
        self.sum_exp = self.sum_exp * alpha + beta;
        for (acc_d, &vd) in self.acc_out.iter_mut().zip(v_slice.iter()) {
            *acc_d = *acc_d * alpha + beta * (vd as f64);
        }
        self.max_score = m_new;
    }

    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    #[inline]
    fn merge(mut self, other: Self) -> Self {
        if other.sum_exp == 0.0 {
            return self;
        }
        if self.sum_exp == 0.0 {
            return other;
        }
        let m_merged = self.max_score.max(other.max_score);
        let sa = (self.max_score - m_merged).exp();
        let sb = (other.max_score - m_merged).exp();
        self.sum_exp = self.sum_exp * sa + other.sum_exp * sb;
        for (a, &b) in self.acc_out.iter_mut().zip(other.acc_out.iter()) {
            *a = *a * sa + b * sb;
        }
        self.max_score = m_merged;
        self
    }

    #[inline]
    fn write_normalized(&self, dst: &mut [f32]) {
        if self.sum_exp <= 0.0 {
            dst.fill(0.0);
            return;
        }
        let inv = 1.0 / self.sum_exp;
        for (out_d, &acc_d) in dst.iter_mut().zip(self.acc_out.iter()) {
            *out_d = (acc_d * inv) as f32;
        }
    }
}

/// Evaluates online-softmax attention for a single `(q_vec, kv_head)` over logical blocks
/// `[blk_start .. blk_end)` causally bounded by `query_pos`.
#[allow(clippy::too_many_arguments)]
fn eval_block_partition(
    slab: &KvSlab,
    layer: usize,
    q_vec: &[f32],
    kv_head: usize,
    table: &LogicalBlockTable,
    query_pos: usize,
    blk_start: usize,
    blk_end: usize,
    inv_sqrt_d: f64,
) -> OnlineSoftmaxState {
    let head_dim = q_vec.len();
    let block_size = table.block_size();
    let blocks = table.blocks();
    let mut state = OnlineSoftmaxState::new(head_dim);

    match slab.config().kv_dtype {
        KvDType::F32 => {
            for (rel_idx, &phys_bid) in blocks[blk_start..blk_end].iter().enumerate() {
                let b_idx = blk_start + rel_idx;
                let base_pos = b_idx * block_size;
                if base_pos > query_pos {
                    break;
                }
                let valid_slots = (query_pos - base_pos + 1).min(block_size);
                for s in 0..valid_slots {
                    let k_slice = slab
                        .slot_f32(layer, 0, phys_bid, kv_head, s)
                        .expect("F32 KV slab");
                    let v_slice = slab
                        .slot_f32(layer, 1, phys_bid, kv_head, s)
                        .expect("F32 KV slab");
                    let score = (simd::vec_dot_f32(q_vec, k_slice) as f64) * inv_sqrt_d;
                    state.update(score, v_slice);
                }
            }
        }
        KvDType::Q8_0 => {
            let mut k_tmp = vec![0.0f32; head_dim];
            let mut v_tmp = vec![0.0f32; head_dim];
            for (rel_idx, &phys_bid) in blocks[blk_start..blk_end].iter().enumerate() {
                let b_idx = blk_start + rel_idx;
                let base_pos = b_idx * block_size;
                if base_pos > query_pos {
                    break;
                }
                let valid_slots = (query_pos - base_pos + 1).min(block_size);
                for s in 0..valid_slots {
                    slab.read_k_slot(layer, phys_bid, kv_head, s, &mut k_tmp)
                        .expect("Q8_0 K read");
                    slab.read_v_slot(layer, phys_bid, kv_head, s, &mut v_tmp)
                        .expect("Q8_0 V read");
                    let score = (simd::vec_dot_f32(q_vec, &k_tmp) as f64) * inv_sqrt_d;
                    state.update(score, &v_tmp);
                }
            }
        }
    }

    state
}

/// Online-Softmax Grouped-Query PagedAttention + FlashDecoding parallel block-partition reduction.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_gqa(
    slab: &KvSlab,
    layer: usize,
    q_batch: &[f32],
    batch: &BatchInput,
    seq_tables: &[&LogicalBlockTable],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out_attn: &mut [f32],
) -> Result<(), ModelError> {
    if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(ModelError::InvalidConfig(format!(
            "invalid GQA head counts: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )));
    }
    let q_dim = n_heads * head_dim;
    let n_toks = batch.len();
    if q_batch.len() != n_toks * q_dim || out_attn.len() != n_toks * q_dim {
        return Err(ModelError::InvalidBatch(
            "paged_attention_gqa buffer shape mismatch".into(),
        ));
    }

    let gqa_ratio = n_heads / n_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f64).sqrt();

    let compute_single_token = |t: usize, out_tok: &mut [f32]| {
        let query_pos = batch.positions[t] as usize;
        let seq_slot = batch.seq_slot_idx[t];
        let table = seq_tables[seq_slot];
        assert!(
            query_pos < table.num_tokens(),
            "query_pos {query_pos} >= table.num_tokens {}",
            table.num_tokens()
        );

        let block_size = table.block_size();
        let causal_len = query_pos + 1;
        let num_causal_blocks = causal_len.div_ceil(block_size);
        let q_tok = &q_batch[t * q_dim..(t + 1) * q_dim];

        for qh in 0..n_heads {
            let kvh = qh / gqa_ratio;
            let q_vec = &q_tok[qh * head_dim..(qh + 1) * head_dim];
            let out_head = &mut out_tok[qh * head_dim..(qh + 1) * head_dim];

            #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
            if causal_len >= 256 && num_causal_blocks >= 8 {
                // FlashDecoding parallel block-partition reduction (4 blocks = 64 tokens / partition)
                const BLOCKS_PER_PART: usize = 4;
                let num_parts = num_causal_blocks.div_ceil(BLOCKS_PER_PART);
                let final_state = (0..num_parts)
                    .into_par_iter()
                    .map(|part_idx| {
                        let bs = part_idx * BLOCKS_PER_PART;
                        let be = (bs + BLOCKS_PER_PART).min(num_causal_blocks);
                        eval_block_partition(
                            slab, layer, q_vec, kvh, table, query_pos, bs, be, inv_sqrt_d,
                        )
                    })
                    .reduce(
                        || OnlineSoftmaxState::new(head_dim),
                        OnlineSoftmaxState::merge,
                    );
                final_state.write_normalized(out_head);
                continue;
            }

            let state = eval_block_partition(
                slab,
                layer,
                q_vec,
                kvh,
                table,
                query_pos,
                0,
                num_causal_blocks,
                inv_sqrt_d,
            );
            state.write_normalized(out_head);
        }
    };

    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    if n_toks >= 4 {
        out_attn
            .par_chunks_mut(q_dim)
            .enumerate()
            .for_each(|(t, out_tok)| compute_single_token(t, out_tok));
        return Ok(());
    }

    for t in 0..n_toks {
        compute_single_token(t, &mut out_attn[t * q_dim..(t + 1) * q_dim]);
    }
    Ok(())
}
