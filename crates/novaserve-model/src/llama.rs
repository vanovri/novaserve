//! Full Pre-Norm Grouped-Query Attention (`GQA`) Transformer (`Llama` / `Qwen2` / `Mistral`
//! architecture) with `.gguf` (`v3`) and `.safetensors` loaders & exporters.

use crate::batch::BatchInput;
use crate::error::ModelError;
use crate::paged_attn::{paged_attention_gqa, scatter_write_kv_layer};
use novaserve_frontend::BpeTokenizer;
use novaserve_kv::{KvDType, KvSlab, KvSlabConfig, LogicalBlockTable};
use novaserve_tensor::{
    linear_projection, rmsnorm_f32, rope_inplace_f32, serialize_gguf_v3, serialize_safetensors,
    simd, softmax_f32, swiglu_f32, AlignedScratchArena, DType, GgufFile, GgufValue,
    SafetensorsFile, TensorBuf, BYTES_Q8_0, QK8_0,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub arch_name: String,
    pub vocab_size: usize,
    pub dim: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: f32,
    pub max_seq_len: usize,
    pub tie_word_embeddings: bool,
    pub weight_dtype: DType,
}

impl LlamaConfig {
    /// Canonical `Tiny-16M` GQA config (`dim=256, layers=4, n_heads=8, n_kv_heads=2, head_dim=32`).
    pub fn tiny_16m(vocab_size: usize, weight_dtype: DType) -> Self {
        Self {
            arch_name: "llama".into(),
            vocab_size,
            dim: 256,
            intermediate_size: 640,
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: 1.0,
            max_seq_len: 2048,
            tie_word_embeddings: false,
            weight_dtype,
        }
    }

    /// Compact `Micro-2M` GQA config (`dim=64, layers=2, n_heads=4, n_kv_heads=2, head_dim=16`).
    pub fn micro_2m(vocab_size: usize, weight_dtype: DType) -> Self {
        Self {
            arch_name: "llama".into(),
            vocab_size,
            dim: 64,
            intermediate_size: 192,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: 1.0,
            max_seq_len: 2048,
            tie_word_embeddings: false,
            weight_dtype,
        }
    }

    /// Ultra-fast `Draft-500K` speculative draft config (`dim=64, layers=1, n_heads=4, n_kv_heads=2`).
    pub fn draft_500k(vocab_size: usize, weight_dtype: DType) -> Self {
        Self {
            arch_name: "llama".into(),
            vocab_size,
            dim: 64,
            intermediate_size: 128,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: 1.0,
            max_seq_len: 2048,
            tie_word_embeddings: false,
            weight_dtype,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.vocab_size == 0
            || self.dim == 0
            || self.intermediate_size == 0
            || self.n_layers == 0
            || self.n_heads == 0
            || self.n_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(ModelError::InvalidConfig(
                "all LlamaConfig dimensions must be > 0".into(),
            ));
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(ModelError::InvalidConfig(format!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            )));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(ModelError::InvalidConfig(format!(
                "head_dim ({}) must be even for RoPE",
                self.head_dim
            )));
        }
        Ok(())
    }

    pub fn kv_slab_config(
        &self,
        num_physical_blocks: usize,
        block_size: usize,
        kv_dtype: KvDType,
    ) -> KvSlabConfig {
        KvSlabConfig {
            num_layers: self.n_layers,
            num_physical_blocks,
            num_kv_heads: self.n_kv_heads,
            block_size,
            head_dim: self.head_dim,
            kv_dtype,
        }
    }

    /// Computes sufficient 64-byte aligned arena capacity for `max_batch_tokens` across all layers.
    pub fn recommended_scratch_bytes(&self, max_batch_tokens: usize) -> usize {
        let t = max_batch_tokens.max(16);
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let act_floats = t * (self.dim * 4 + q_dim * 2 + kv_dim * 2 + self.intermediate_size * 3);
        let max_k = self.dim.max(self.intermediate_size).max(q_dim);
        let q8_per_proj = t * max_k.div_ceil(QK8_0) * BYTES_Q8_0 + 256;
        let proj_bytes = (self.n_layers * 7 + 2) * q8_per_proj;
        (act_floats * 4 + proj_bytes + 65_536).next_multiple_of(4096)
    }
}

#[derive(Clone, Debug)]
pub struct TransformerLayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: TensorBuf,
    pub wk: TensorBuf,
    pub wv: TensorBuf,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub wo: TensorBuf,
    pub ffn_norm: Vec<f32>,
    pub w_gate: TensorBuf,
    pub w_up: TensorBuf,
    pub w_down: TensorBuf,
}

#[derive(Clone, Debug)]
pub struct LlamaModel {
    cfg: LlamaConfig,
    tok_embeddings_f32: Vec<f32>,
    tok_embeddings_buf: TensorBuf,
    layers: Vec<TransformerLayerWeights>,
    output_norm: Vec<f32>,
    lm_head: Option<TensorBuf>,
}

impl LlamaModel {
    pub fn from_parts(
        cfg: LlamaConfig,
        tok_embeddings_buf: TensorBuf,
        layers: Vec<TransformerLayerWeights>,
        output_norm: Vec<f32>,
        lm_head: Option<TensorBuf>,
    ) -> Result<Self, ModelError> {
        cfg.validate()?;
        if layers.len() != cfg.n_layers {
            return Err(ModelError::InvalidConfig(format!(
                "expected {} layers, got {}",
                cfg.n_layers,
                layers.len()
            )));
        }
        if tok_embeddings_buf.dims() != [cfg.vocab_size, cfg.dim] {
            return Err(ModelError::InvalidConfig(format!(
                "tok_embeddings shape {:?} != [{}, {}]",
                tok_embeddings_buf.dims(),
                cfg.vocab_size,
                cfg.dim
            )));
        }
        let tok_embeddings_f32 = tok_embeddings_buf.view().to_f32_vec()?;
        Ok(Self {
            cfg,
            tok_embeddings_f32,
            tok_embeddings_buf,
            layers,
            output_norm,
            lm_head,
        })
    }

    #[inline]
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    #[inline]
    pub fn layers(&self) -> &[TransformerLayerWeights] {
        &self.layers
    }

    #[inline]
    pub fn tok_embeddings_f32(&self) -> &[f32] {
        &self.tok_embeddings_f32
    }

    #[inline]
    pub fn output_norm(&self) -> &[f32] {
        &self.output_norm
    }

    #[inline]
    pub fn lm_head_view(&self) -> novaserve_tensor::TensorView<'_> {
        match &self.lm_head {
            Some(head) => head.view(),
            None => self.tok_embeddings_buf.view(),
        }
    }

    /// Total weight bytes across embeddings, layers, norms, and `lm_head`.
    pub fn total_weight_bytes(&self) -> usize {
        let mut bytes = self.tok_embeddings_buf.as_bytes().len() + self.output_norm.len() * 4;
        if let Some(h) = &self.lm_head {
            bytes += h.as_bytes().len();
        }
        for l in &self.layers {
            bytes += l.attn_norm.len() * 4
                + l.ffn_norm.len() * 4
                + l.wq.as_bytes().len()
                + l.wk.as_bytes().len()
                + l.wv.as_bytes().len()
                + l.wo.as_bytes().len()
                + l.w_gate.as_bytes().len()
                + l.w_up.as_bytes().len()
                + l.w_down.as_bytes().len();
        }
        bytes
    }

    /// Executes a unified PagedAttention + ChunkedPrefill + Decode + Speculative forward pass.
    /// Returns `[num_logit_outputs, vocab_size]` flattened `f32` logits.
    pub fn forward_batch(
        &self,
        batch: &BatchInput,
        seq_tables: &[&LogicalBlockTable],
        kv_slab: &mut KvSlab,
        arena: &mut AlignedScratchArena,
    ) -> Result<Vec<f32>, ModelError> {
        batch.validate(seq_tables.len(), self.cfg.vocab_size)?;

        let n_toks = batch.len();
        let dim = self.cfg.dim;
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;

        let needed_arena = self.cfg.recommended_scratch_bytes(n_toks);
        if arena.capacity() < needed_arena {
            *arena = AlignedScratchArena::new(needed_arena)?;
        }
        arena.reset();

        let x_cur = arena.alloc_f32(n_toks * dim)?;
        for (t, &tid) in batch.token_ids.iter().enumerate() {
            let emb_off = (tid as usize) * dim;
            x_cur[t * dim..(t + 1) * dim]
                .copy_from_slice(&self.tok_embeddings_f32[emb_off..emb_off + dim]);
        }

        let x_norm = arena.alloc_f32(n_toks * dim)?;
        let q_buf = arena.alloc_f32(n_toks * q_dim)?;
        let k_buf = arena.alloc_f32(n_toks * kv_dim)?;
        let v_buf = arena.alloc_f32(n_toks * kv_dim)?;
        let attn_out = arena.alloc_f32(n_toks * q_dim)?;
        let proj_out = arena.alloc_f32(n_toks * dim)?;
        let gate_buf = arena.alloc_f32(n_toks * inter)?;
        let up_buf = arena.alloc_f32(n_toks * inter)?;
        let swiglu_buf = arena.alloc_f32(n_toks * inter)?;

        for (layer_idx, lw) in self.layers.iter().enumerate() {
            // 1. Pre-Attention RMSNorm
            rmsnorm_f32(x_cur, &lw.attn_norm, self.cfg.rms_norm_eps, x_norm)?;

            // 2. Q, K, V Projections (+ optional Qwen2 biases)
            linear_projection(x_norm, &lw.wq.view(), n_toks, arena, q_buf)?;
            linear_projection(x_norm, &lw.wk.view(), n_toks, arena, k_buf)?;
            linear_projection(x_norm, &lw.wv.view(), n_toks, arena, v_buf)?;

            if let Some(bq) = &lw.bq {
                add_row_bias_inplace(q_buf, bq, n_toks);
            }
            if let Some(bk) = &lw.bk {
                add_row_bias_inplace(k_buf, bk, n_toks);
            }
            if let Some(bv) = &lw.bv {
                add_row_bias_inplace(v_buf, bv, n_toks);
            }

            // 3. Per-Token RoPE on Q and K
            rope_inplace_f32(
                q_buf,
                &batch.positions,
                self.cfg.n_heads,
                self.cfg.head_dim,
                self.cfg.rope_theta,
                self.cfg.rope_scaling,
            )?;
            rope_inplace_f32(
                k_buf,
                &batch.positions,
                self.cfg.n_kv_heads,
                self.cfg.head_dim,
                self.cfg.rope_theta,
                self.cfg.rope_scaling,
            )?;

            // 4. Scatter KV Write into Paged KvSlab
            scatter_write_kv_layer(kv_slab, layer_idx, k_buf, v_buf, batch)?;

            // 5. Online-Softmax Paged GQA / FlashDecoding
            paged_attention_gqa(
                kv_slab,
                layer_idx,
                q_buf,
                batch,
                seq_tables,
                self.cfg.n_heads,
                self.cfg.n_kv_heads,
                self.cfg.head_dim,
                attn_out,
            )?;

            // 6. Output Projection + Residual
            linear_projection(attn_out, &lw.wo.view(), n_toks, arena, proj_out)?;
            for (xc, &po) in x_cur.iter_mut().zip(proj_out.iter()) {
                *xc += po;
            }

            // 7. Pre-FFN RMSNorm + Gate/Up SwiGLU + Down Projection + Residual
            rmsnorm_f32(x_cur, &lw.ffn_norm, self.cfg.rms_norm_eps, x_norm)?;
            linear_projection(x_norm, &lw.w_gate.view(), n_toks, arena, gate_buf)?;
            linear_projection(x_norm, &lw.w_up.view(), n_toks, arena, up_buf)?;
            swiglu_f32(gate_buf, up_buf, swiglu_buf)?;
            linear_projection(swiglu_buf, &lw.w_down.view(), n_toks, arena, proj_out)?;
            for (xc, &po) in x_cur.iter_mut().zip(proj_out.iter()) {
                *xc += po;
            }
        }

        // Gather only tokens flagged `is_logit_output == true` before final norm + `lm_head`!
        let num_out = batch.num_logit_outputs();
        if num_out == 0 {
            return Ok(Vec::new());
        }

        let selected_hidden = arena.alloc_f32(num_out * dim)?;
        let normed_out = arena.alloc_f32(num_out * dim)?;
        let mut out_idx = 0usize;
        for (t, &emit) in batch.is_logit_output.iter().enumerate() {
            if emit {
                selected_hidden[out_idx * dim..(out_idx + 1) * dim]
                    .copy_from_slice(&x_cur[t * dim..(t + 1) * dim]);
                out_idx += 1;
            }
        }

        rmsnorm_f32(
            selected_hidden,
            &self.output_norm,
            self.cfg.rms_norm_eps,
            normed_out,
        )?;

        let mut logits = vec![0.0f32; num_out * self.cfg.vocab_size];
        linear_projection(
            normed_out,
            &self.lm_head_view(),
            num_out,
            arena,
            &mut logits,
        )?;
        Ok(logits)
    }

    /// Simple contiguous `[1, seq_len]` single-sequence eager forward pass (zero paging, zero
    /// chunking, zero batching) used as an immediate single-model reference check.
    pub fn forward_eager_contiguous(
        &self,
        tokens: &[u32],
        emit_all_positions: bool,
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::InvalidBatch("empty token slice".into()));
        }
        let s_len = tokens.len();
        let dim = self.cfg.dim;
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        let hd = self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;
        let gqa_ratio = self.cfg.n_heads / self.cfg.n_kv_heads;
        let inv_sqrt_d = 1.0 / (hd as f32).sqrt();

        let mut arena = AlignedScratchArena::new(self.cfg.recommended_scratch_bytes(s_len))?;
        let positions: Vec<u32> = (0..s_len as u32).collect();

        let mut x = vec![0.0f32; s_len * dim];
        for (t, &tid) in tokens.iter().enumerate() {
            let off = (tid as usize) * dim;
            x[t * dim..(t + 1) * dim].copy_from_slice(&self.tok_embeddings_f32[off..off + dim]);
        }

        let mut x_norm = vec![0.0f32; s_len * dim];
        let mut q = vec![0.0f32; s_len * q_dim];
        let mut k = vec![0.0f32; s_len * kv_dim];
        let mut v = vec![0.0f32; s_len * kv_dim];
        let mut attn_out = vec![0.0f32; s_len * q_dim];
        let mut proj = vec![0.0f32; s_len * dim];
        let mut gate = vec![0.0f32; s_len * inter];
        let mut up = vec![0.0f32; s_len * inter];
        let mut swiglu = vec![0.0f32; s_len * inter];

        for lw in &self.layers {
            arena.reset();
            rmsnorm_f32(&x, &lw.attn_norm, self.cfg.rms_norm_eps, &mut x_norm)?;
            linear_projection(&x_norm, &lw.wq.view(), s_len, &arena, &mut q)?;
            linear_projection(&x_norm, &lw.wk.view(), s_len, &arena, &mut k)?;
            linear_projection(&x_norm, &lw.wv.view(), s_len, &arena, &mut v)?;
            if let Some(bq) = &lw.bq {
                add_row_bias_inplace(&mut q, bq, s_len);
            }
            if let Some(bk) = &lw.bk {
                add_row_bias_inplace(&mut k, bk, s_len);
            }
            if let Some(bv) = &lw.bv {
                add_row_bias_inplace(&mut v, bv, s_len);
            }

            rope_inplace_f32(
                &mut q,
                &positions,
                self.cfg.n_heads,
                hd,
                self.cfg.rope_theta,
                self.cfg.rope_scaling,
            )?;
            rope_inplace_f32(
                &mut k,
                &positions,
                self.cfg.n_kv_heads,
                hd,
                self.cfg.rope_theta,
                self.cfg.rope_scaling,
            )?;

            // Contiguous causal attention
            attn_out.fill(0.0);
            for t in 0..s_len {
                let mut scores = vec![0.0f32; t + 1];
                let mut probs = vec![0.0f32; t + 1];
                for qh in 0..self.cfg.n_heads {
                    let kvh = qh / gqa_ratio;
                    let q_vec = &q[t * q_dim + qh * hd..t * q_dim + (qh + 1) * hd];
                    for (j, sc) in scores.iter_mut().enumerate() {
                        let k_vec = &k[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                        *sc = simd::vec_dot_f32(q_vec, k_vec) * inv_sqrt_d;
                    }
                    softmax_f32(&scores, &mut probs)?;
                    let out_h = &mut attn_out[t * q_dim + qh * hd..t * q_dim + (qh + 1) * hd];
                    for (j, &p) in probs.iter().enumerate() {
                        let v_vec = &v[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                        for d in 0..hd {
                            out_h[d] = p.mul_add(v_vec[d], out_h[d]);
                        }
                    }
                }
            }

            linear_projection(&attn_out, &lw.wo.view(), s_len, &arena, &mut proj)?;
            for (xc, &po) in x.iter_mut().zip(proj.iter()) {
                *xc += po;
            }

            rmsnorm_f32(&x, &lw.ffn_norm, self.cfg.rms_norm_eps, &mut x_norm)?;
            linear_projection(&x_norm, &lw.w_gate.view(), s_len, &arena, &mut gate)?;
            linear_projection(&x_norm, &lw.w_up.view(), s_len, &arena, &mut up)?;
            swiglu_f32(&gate, &up, &mut swiglu)?;
            linear_projection(&swiglu, &lw.w_down.view(), s_len, &arena, &mut proj)?;
            for (xc, &po) in x.iter_mut().zip(proj.iter()) {
                *xc += po;
            }
        }

        arena.reset();
        let target_slice = if emit_all_positions {
            &x[..]
        } else {
            &x[(s_len - 1) * dim..s_len * dim]
        };
        let out_rows = if emit_all_positions { s_len } else { 1 };
        let mut normed = vec![0.0f32; out_rows * dim];
        rmsnorm_f32(
            target_slice,
            &self.output_norm,
            self.cfg.rms_norm_eps,
            &mut normed,
        )?;
        let mut logits = vec![0.0f32; out_rows * self.cfg.vocab_size];
        linear_projection(&normed, &self.lm_head_view(), out_rows, &arena, &mut logits)?;
        Ok(logits)
    }

    /// Loads a `Llama` / `Qwen2` / `SmolLM2` model from a `GGUF v3` file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ModelError> {
        let meta = gguf.metadata();
        let arch = meta
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .unwrap_or("llama")
            .to_string();

        let get_u = |suffix: &str| -> Result<usize, ModelError> {
            let key = format!("{arch}.{suffix}");
            meta.get(&key)
                .and_then(GgufValue::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| ModelError::InvalidConfig(format!("missing GGUF metadata '{key}'")))
        };

        let dim = get_u("embedding_length")?;
        let intermediate_size = get_u("feed_forward_length")?;
        let n_layers = get_u("block_count")?;
        let n_heads = get_u("attention.head_count")?;
        let n_kv_heads = meta
            .get(&format!("{arch}.attention.head_count_kv"))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(n_heads);
        let head_dim = meta
            .get(&format!("{arch}.attention.key_length"))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let rms_norm_eps = meta
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(GgufValue::as_f32)
            .unwrap_or(1e-5);
        let rope_theta = meta
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(GgufValue::as_f32)
            .unwrap_or(10_000.0);
        let max_seq_len = meta
            .get(&format!("{arch}.context_length"))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(2048);

        let emb_view = gguf.tensor("token_embd.weight")?;
        let vocab_size = emb_view.dims()[0];
        let weight_dtype = emb_view.dtype();
        let tok_embeddings_buf = clone_view_to_buf(&emb_view)?;

        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let lm_head = if let Ok(out_v) = gguf.tensor("output.weight") {
            Some(clone_view_to_buf(&out_v)?)
        } else {
            None
        };
        let tie_word_embeddings = lm_head.is_none();

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let attn_norm = gguf
                .tensor(&format!("blk.{l}.attn_norm.weight"))?
                .to_f32_vec()?;
            let wq = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.attn_q.weight"))?)?;
            let wk = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.attn_k.weight"))?)?;
            let wv = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.attn_v.weight"))?)?;
            let bq = gguf
                .tensor(&format!("blk.{l}.attn_q.bias"))
                .ok()
                .and_then(|v| v.to_f32_vec().ok());
            let bk = gguf
                .tensor(&format!("blk.{l}.attn_k.bias"))
                .ok()
                .and_then(|v| v.to_f32_vec().ok());
            let bv = gguf
                .tensor(&format!("blk.{l}.attn_v.bias"))
                .ok()
                .and_then(|v| v.to_f32_vec().ok());
            let wo = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.attn_output.weight"))?)?;
            let ffn_norm = gguf
                .tensor(&format!("blk.{l}.ffn_norm.weight"))?
                .to_f32_vec()?;
            let w_gate = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.ffn_gate.weight"))?)?;
            let w_up = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.ffn_up.weight"))?)?;
            let w_down = clone_view_to_buf(&gguf.tensor(&format!("blk.{l}.ffn_down.weight"))?)?;

            layers.push(TransformerLayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                bq,
                bk,
                bv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        let cfg = LlamaConfig {
            arch_name: arch,
            vocab_size,
            dim,
            intermediate_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            rope_scaling: 1.0,
            max_seq_len,
            tie_word_embeddings,
            weight_dtype,
        };
        Self::from_parts(cfg, tok_embeddings_buf, layers, output_norm, lm_head)
    }

    /// Exports this model (and optional `BpeTokenizer`) into a spec-compliant `GGUF v3` binary image.
    pub fn to_gguf_bytes(&self, tokenizer: Option<&BpeTokenizer>) -> Result<Vec<u8>, ModelError> {
        let arch = &self.cfg.arch_name;
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String(arch.clone()),
        );
        meta.insert("general.alignment".into(), GgufValue::U32(32));
        meta.insert(
            format!("{arch}.context_length"),
            GgufValue::U32(self.cfg.max_seq_len as u32),
        );
        meta.insert(
            format!("{arch}.embedding_length"),
            GgufValue::U32(self.cfg.dim as u32),
        );
        meta.insert(
            format!("{arch}.feed_forward_length"),
            GgufValue::U32(self.cfg.intermediate_size as u32),
        );
        meta.insert(
            format!("{arch}.block_count"),
            GgufValue::U32(self.cfg.n_layers as u32),
        );
        meta.insert(
            format!("{arch}.attention.head_count"),
            GgufValue::U32(self.cfg.n_heads as u32),
        );
        meta.insert(
            format!("{arch}.attention.head_count_kv"),
            GgufValue::U32(self.cfg.n_kv_heads as u32),
        );
        meta.insert(
            format!("{arch}.attention.key_length"),
            GgufValue::U32(self.cfg.head_dim as u32),
        );
        meta.insert(
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(self.cfg.rms_norm_eps),
        );
        meta.insert(
            format!("{arch}.rope.freq_base"),
            GgufValue::F32(self.cfg.rope_theta),
        );

        if let Some(tok) = tokenizer {
            tok.write_gguf_metadata(&mut meta);
        }

        // Own norm TensorBufs so their views stay alive through serialization
        let out_norm_buf = TensorBuf::from_f32_slice(&[self.cfg.dim], &self.output_norm)?;
        let mut layer_norm_bufs = Vec::with_capacity(self.layers.len() * 2);
        for lw in &self.layers {
            layer_norm_bufs.push(TensorBuf::from_f32_slice(&[self.cfg.dim], &lw.attn_norm)?);
            layer_norm_bufs.push(TensorBuf::from_f32_slice(&[self.cfg.dim], &lw.ffn_norm)?);
        }

        let mut named_entries: Vec<(String, novaserve_tensor::TensorView<'_>)> = Vec::new();
        named_entries.push(("token_embd.weight".into(), self.tok_embeddings_buf.view()));
        named_entries.push(("output_norm.weight".into(), out_norm_buf.view()));
        if let Some(head) = &self.lm_head {
            named_entries.push(("output.weight".into(), head.view()));
        }

        for (l, lw) in self.layers.iter().enumerate() {
            named_entries.push((
                format!("blk.{l}.attn_norm.weight"),
                layer_norm_bufs[l * 2].view(),
            ));
            named_entries.push((format!("blk.{l}.attn_q.weight"), lw.wq.view()));
            named_entries.push((format!("blk.{l}.attn_k.weight"), lw.wk.view()));
            named_entries.push((format!("blk.{l}.attn_v.weight"), lw.wv.view()));
            named_entries.push((format!("blk.{l}.attn_output.weight"), lw.wo.view()));
            named_entries.push((
                format!("blk.{l}.ffn_norm.weight"),
                layer_norm_bufs[l * 2 + 1].view(),
            ));
            named_entries.push((format!("blk.{l}.ffn_gate.weight"), lw.w_gate.view()));
            named_entries.push((format!("blk.{l}.ffn_up.weight"), lw.w_up.view()));
            named_entries.push((format!("blk.{l}.ffn_down.weight"), lw.w_down.view()));
        }

        let borrowed: Vec<(&str, novaserve_tensor::TensorView<'_>)> = named_entries
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        Ok(serialize_gguf_v3(&meta, &borrowed)?)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_gguf<P: AsRef<Path>>(
        &self,
        path: P,
        tokenizer: Option<&BpeTokenizer>,
    ) -> Result<(), ModelError> {
        let bytes = self.to_gguf_bytes(tokenizer)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads HuggingFace `Safetensors` weights (`model.layers.{l}.*` & `model.embed_tokens.weight`).
    pub fn from_safetensors(
        st: &SafetensorsFile,
        mut cfg: LlamaConfig,
    ) -> Result<Self, ModelError> {
        let emb_v = st.tensor("model.embed_tokens.weight")?;
        cfg.weight_dtype = emb_v.dtype();
        let tok_embeddings_buf = clone_view_to_buf(&emb_v)?;
        let output_norm = st.tensor("model.norm.weight")?.to_f32_vec()?;
        let lm_head = if let Ok(hv) = st.tensor("lm_head.weight") {
            cfg.tie_word_embeddings = false;
            Some(clone_view_to_buf(&hv)?)
        } else {
            cfg.tie_word_embeddings = true;
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let pfx = format!("model.layers.{l}");
            let attn_norm = st
                .tensor(&format!("{pfx}.input_layernorm.weight"))?
                .to_f32_vec()?;
            let wq = clone_view_to_buf(&st.tensor(&format!("{pfx}.self_attn.q_proj.weight"))?)?;
            let wk = clone_view_to_buf(&st.tensor(&format!("{pfx}.self_attn.k_proj.weight"))?)?;
            let wv = clone_view_to_buf(&st.tensor(&format!("{pfx}.self_attn.v_proj.weight"))?)?;
            let bq = st
                .tensor(&format!("{pfx}.self_attn.q_proj.bias"))
                .ok()
                .and_then(|v| v.to_f32_vec().ok());
            let bk = st
                .tensor(&format!("{pfx}.self_attn.k_proj.bias"))
                .ok()
                .and_then(|v| v.to_f32_vec().ok());
            let bv = st
                .tensor(&format!("{pfx}.self_attn.v_proj.bias"))
                .ok()
                .and_then(|v| v.to_f32_vec().ok());
            let wo = clone_view_to_buf(&st.tensor(&format!("{pfx}.self_attn.o_proj.weight"))?)?;
            let ffn_norm = st
                .tensor(&format!("{pfx}.post_attention_layernorm.weight"))?
                .to_f32_vec()?;
            let w_gate = clone_view_to_buf(&st.tensor(&format!("{pfx}.mlp.gate_proj.weight"))?)?;
            let w_up = clone_view_to_buf(&st.tensor(&format!("{pfx}.mlp.up_proj.weight"))?)?;
            let w_down = clone_view_to_buf(&st.tensor(&format!("{pfx}.mlp.down_proj.weight"))?)?;

            layers.push(TransformerLayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                bq,
                bk,
                bv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        Self::from_parts(cfg, tok_embeddings_buf, layers, output_norm, lm_head)
    }

    /// Exports HuggingFace-compatible `Safetensors` bytes (converting quantized weights to `F32` if needed).
    pub fn to_safetensors_bytes(&self) -> Result<Vec<u8>, ModelError> {
        let to_st_buf = |tb: &TensorBuf| -> Result<TensorBuf, ModelError> {
            if tb.dtype().is_quantized() {
                let f = tb.view().to_f32_vec()?;
                Ok(TensorBuf::from_f32_slice(tb.dims(), &f)?)
            } else {
                Ok(tb.clone())
            }
        };

        let mut owned: Vec<(String, TensorBuf)> = Vec::new();
        owned.push((
            "model.embed_tokens.weight".into(),
            to_st_buf(&self.tok_embeddings_buf)?,
        ));
        owned.push((
            "model.norm.weight".into(),
            TensorBuf::from_f32_slice(&[self.cfg.dim], &self.output_norm)?,
        ));
        if let Some(h) = &self.lm_head {
            owned.push(("lm_head.weight".into(), to_st_buf(h)?));
        }
        for (l, lw) in self.layers.iter().enumerate() {
            let pfx = format!("model.layers.{l}");
            owned.push((
                format!("{pfx}.input_layernorm.weight"),
                TensorBuf::from_f32_slice(&[self.cfg.dim], &lw.attn_norm)?,
            ));
            owned.push((format!("{pfx}.self_attn.q_proj.weight"), to_st_buf(&lw.wq)?));
            owned.push((format!("{pfx}.self_attn.k_proj.weight"), to_st_buf(&lw.wk)?));
            owned.push((format!("{pfx}.self_attn.v_proj.weight"), to_st_buf(&lw.wv)?));
            owned.push((format!("{pfx}.self_attn.o_proj.weight"), to_st_buf(&lw.wo)?));
            owned.push((
                format!("{pfx}.post_attention_layernorm.weight"),
                TensorBuf::from_f32_slice(&[self.cfg.dim], &lw.ffn_norm)?,
            ));
            owned.push((
                format!("{pfx}.mlp.gate_proj.weight"),
                to_st_buf(&lw.w_gate)?,
            ));
            owned.push((format!("{pfx}.mlp.up_proj.weight"), to_st_buf(&lw.w_up)?));
            owned.push((
                format!("{pfx}.mlp.down_proj.weight"),
                to_st_buf(&lw.w_down)?,
            ));
        }

        let mut meta = BTreeMap::new();
        meta.insert("format".into(), "pt".into());
        meta.insert("novaserve_config".into(), serde_json::to_string(&self.cfg)?);

        let views: Vec<(&str, novaserve_tensor::TensorView<'_>)> =
            owned.iter().map(|(k, b)| (k.as_str(), b.view())).collect();
        Ok(serialize_safetensors(&views, Some(&meta))?)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        let bytes = self.to_safetensors_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

fn clone_view_to_buf(view: &novaserve_tensor::TensorView<'_>) -> Result<TensorBuf, ModelError> {
    let raw = novaserve_tensor::AlignedBuf::from_bytes(view.raw_bytes())?;
    Ok(TensorBuf::from_raw_aligned(raw, view.dtype(), view.dims())?)
}

#[inline]
fn add_row_bias_inplace(mat: &mut [f32], bias: &[f32], rows: usize) {
    let cols = bias.len();
    for r in 0..rows {
        let row = &mut mat[r * cols..(r + 1) * cols];
        for (dst, &b) in row.iter_mut().zip(bias.iter()) {
            *dst += b;
        }
    }
}
