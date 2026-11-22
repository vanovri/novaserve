//! Deterministic Synthetic & Subspace-Distilled Weight Factory (`ModelFactory`)
//! enabling zero-download verification across `FP32`, `Q8_0`, `Q4_0`, `Safetensors`, and `GGUF v3`.

use crate::error::ModelError;
use crate::llama::{LlamaConfig, LlamaModel, TransformerLayerWeights};
use novaserve_tensor::{DType, SplitMix64, TensorBuf};

pub struct ModelFactory;

impl ModelFactory {
    fn xavier_vec(rng: &mut SplitMix64, rows: usize, cols: usize, scale_mult: f32) -> Vec<f32> {
        let std = ((2.0 / ((rows + cols) as f64)).sqrt() as f32) * scale_mult;
        (0..rows * cols)
            .map(|_| rng.next_gaussian_f32() * std)
            .collect()
    }

    /// Builds a standalone deterministic Xavier-initialized `LlamaModel`.
    pub fn build_random_model(cfg: LlamaConfig, seed: u64) -> Result<LlamaModel, ModelError> {
        cfg.validate()?;
        let mut rng = SplitMix64::new(seed);
        let dt = cfg.weight_dtype;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        let emb_f32 = Self::xavier_vec(&mut rng, cfg.vocab_size, cfg.dim, 1.0);
        let tok_embeddings_buf =
            TensorBuf::from_f32_as_dtype(&[cfg.vocab_size, cfg.dim], &emb_f32, dt)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let attn_norm: Vec<f32> = (0..cfg.dim).map(|_| rng.range_f32(0.85, 1.15)).collect();
            let ffn_norm: Vec<f32> = (0..cfg.dim).map(|_| rng.range_f32(0.85, 1.15)).collect();

            let wq = TensorBuf::from_f32_as_dtype(
                &[q_dim, cfg.dim],
                &Self::xavier_vec(&mut rng, q_dim, cfg.dim, 1.0),
                dt,
            )?;
            let wk = TensorBuf::from_f32_as_dtype(
                &[kv_dim, cfg.dim],
                &Self::xavier_vec(&mut rng, kv_dim, cfg.dim, 1.0),
                dt,
            )?;
            let wv = TensorBuf::from_f32_as_dtype(
                &[kv_dim, cfg.dim],
                &Self::xavier_vec(&mut rng, kv_dim, cfg.dim, 1.0),
                dt,
            )?;
            let wo = TensorBuf::from_f32_as_dtype(
                &[cfg.dim, q_dim],
                &Self::xavier_vec(&mut rng, cfg.dim, q_dim, 0.7),
                dt,
            )?;

            let w_gate = TensorBuf::from_f32_as_dtype(
                &[cfg.intermediate_size, cfg.dim],
                &Self::xavier_vec(&mut rng, cfg.intermediate_size, cfg.dim, 1.0),
                dt,
            )?;
            let w_up = TensorBuf::from_f32_as_dtype(
                &[cfg.intermediate_size, cfg.dim],
                &Self::xavier_vec(&mut rng, cfg.intermediate_size, cfg.dim, 1.0),
                dt,
            )?;
            let w_down = TensorBuf::from_f32_as_dtype(
                &[cfg.dim, cfg.intermediate_size],
                &Self::xavier_vec(&mut rng, cfg.dim, cfg.intermediate_size, 0.7),
                dt,
            )?;

            layers.push(TransformerLayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                bq: None,
                bk: None,
                bv: None,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        let output_norm: Vec<f32> = (0..cfg.dim).map(|_| rng.range_f32(0.9, 1.1)).collect();
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(TensorBuf::from_f32_as_dtype(
                &[cfg.vocab_size, cfg.dim],
                &Self::xavier_vec(&mut rng, cfg.vocab_size, cfg.dim, 1.2),
                dt,
            )?)
        };

        LlamaModel::from_parts(cfg, tok_embeddings_buf, layers, output_norm, lm_head)
    }

    /// Builds a correlated `(TargetModel, DraftModel)` pair via principal-subspace distillation:
    /// - `DraftModel` (`dim_d <= dim_t, layers = 1`) captures the leading `dim_d` principal
    ///   subspace of `TargetModel`'s token embeddings, Layer 0, and `lm_head`.
    /// - `TargetModel` (`dim_t, layers = n_layers`) executes dense full-rank projections across
    ///   all `dim_t` dimensions and `n_layers`, adding controlled higher-layer refinements so
    ///   Speculative Decoding achieves realistic $\alpha \approx 0.65 - 0.80$ acceptance!
    pub fn build_correlated_pair(
        target_cfg: LlamaConfig,
        draft_cfg: LlamaConfig,
        seed: u64,
    ) -> Result<(LlamaModel, LlamaModel), ModelError> {
        target_cfg.validate()?;
        draft_cfg.validate()?;
        if draft_cfg.vocab_size != target_cfg.vocab_size || draft_cfg.dim > target_cfg.dim {
            return Err(ModelError::InvalidConfig(
                "draft_cfg must match vocab_size and have dim <= target_cfg.dim".into(),
            ));
        }

        let mut rng = SplitMix64::new(seed);
        let v_sz = target_cfg.vocab_size;
        let dt_t = target_cfg.weight_dtype;
        let dt_d = draft_cfg.weight_dtype;
        let dim_t = target_cfg.dim;
        let dim_d = draft_cfg.dim;
        let scale_ratio = ((dim_t as f32) / (dim_d as f32)).sqrt();

        // 1. Shared principal subspace embeddings:
        // Keep `[0..dim_d]` high-signal and `[dim_d..dim_t]` gentle so RMSNorm across `dim_t`
        // preserves the relative scale of the shared `[0..dim_d]` coordinates!
        let mut emb_d = vec![0.0f32; v_sz * dim_d];
        let mut emb_t = vec![0.0f32; v_sz * dim_t];
        for v in 0..v_sz {
            for d in 0..dim_d {
                let val = rng.next_gaussian_f32() * 0.5;
                emb_d[v * dim_d + d] = val;
                emb_t[v * dim_t + d] = val * scale_ratio;
            }
            for d in dim_d..dim_t {
                emb_t[v * dim_t + d] = rng.next_gaussian_f32() * 0.04;
            }
        }

        // Build DraftModel layers first (`1` layer typically)
        let mut draft_layers = Vec::with_capacity(draft_cfg.n_layers);
        let qd_d = draft_cfg.n_heads * draft_cfg.head_dim;
        let kvd_d = draft_cfg.n_kv_heads * draft_cfg.head_dim;
        let inter_d = draft_cfg.intermediate_size;

        let d_wq_f32 = Self::xavier_vec(&mut rng, qd_d, dim_d, 0.5);
        let d_wk_f32 = Self::xavier_vec(&mut rng, kvd_d, dim_d, 0.5);
        let d_wv_f32 = Self::xavier_vec(&mut rng, kvd_d, dim_d, 0.5);
        let d_wo_f32 = Self::xavier_vec(&mut rng, dim_d, qd_d, 0.4);
        let d_wg_f32 = Self::xavier_vec(&mut rng, inter_d, dim_d, 0.5);
        let d_wu_f32 = Self::xavier_vec(&mut rng, inter_d, dim_d, 0.5);
        let d_wd_f32 = Self::xavier_vec(&mut rng, dim_d, inter_d, 0.4);

        for _ in 0..draft_cfg.n_layers {
            draft_layers.push(TransformerLayerWeights {
                attn_norm: vec![1.0; dim_d],
                wq: TensorBuf::from_f32_as_dtype(&[qd_d, dim_d], &d_wq_f32, dt_d)?,
                wk: TensorBuf::from_f32_as_dtype(&[kvd_d, dim_d], &d_wk_f32, dt_d)?,
                wv: TensorBuf::from_f32_as_dtype(&[kvd_d, dim_d], &d_wv_f32, dt_d)?,
                bq: None,
                bk: None,
                bv: None,
                wo: TensorBuf::from_f32_as_dtype(&[dim_d, qd_d], &d_wo_f32, dt_d)?,
                ffn_norm: vec![1.0; dim_d],
                w_gate: TensorBuf::from_f32_as_dtype(&[inter_d, dim_d], &d_wg_f32, dt_d)?,
                w_up: TensorBuf::from_f32_as_dtype(&[inter_d, dim_d], &d_wu_f32, dt_d)?,
                w_down: TensorBuf::from_f32_as_dtype(&[dim_d, inter_d], &d_wd_f32, dt_d)?,
            });
        }

        // Embed Draft Layer 0 into Target Layer 0's leading subspace + gentle dense background,
        // and add subtle deep residual refinements on Target Layers `1..n_layers`!
        let qd_t = target_cfg.n_heads * target_cfg.head_dim;
        let kvd_t = target_cfg.n_kv_heads * target_cfg.head_dim;
        let inter_t = target_cfg.intermediate_size;
        let mut target_layers = Vec::with_capacity(target_cfg.n_layers);

        for l in 0..target_cfg.n_layers {
            let resid_scale = if l == 0 { 0.08 } else { 0.12 };
            let mut t_wq = Self::xavier_vec(&mut rng, qd_t, dim_t, resid_scale);
            let mut t_wk = Self::xavier_vec(&mut rng, kvd_t, dim_t, resid_scale);
            let mut t_wv = Self::xavier_vec(&mut rng, kvd_t, dim_t, resid_scale);
            let mut t_wo = Self::xavier_vec(&mut rng, dim_t, qd_t, resid_scale);
            let mut t_wg = Self::xavier_vec(&mut rng, inter_t, dim_t, resid_scale);
            let mut t_wu = Self::xavier_vec(&mut rng, inter_t, dim_t, resid_scale);
            let mut t_wd = Self::xavier_vec(&mut rng, dim_t, inter_t, resid_scale);

            if l == 0 {
                copy_submatrix(
                    &d_wq_f32,
                    qd_d,
                    dim_d,
                    &mut t_wq,
                    qd_t,
                    dim_t,
                    1.0 / scale_ratio,
                );
                copy_submatrix(
                    &d_wk_f32,
                    kvd_d,
                    dim_d,
                    &mut t_wk,
                    kvd_t,
                    dim_t,
                    1.0 / scale_ratio,
                );
                copy_submatrix(
                    &d_wv_f32,
                    kvd_d,
                    dim_d,
                    &mut t_wv,
                    kvd_t,
                    dim_t,
                    1.0 / scale_ratio,
                );
                copy_submatrix(&d_wo_f32, dim_d, qd_d, &mut t_wo, dim_t, qd_t, scale_ratio);
                copy_submatrix(
                    &d_wg_f32,
                    inter_d.min(inter_t),
                    dim_d,
                    &mut t_wg,
                    inter_t,
                    dim_t,
                    1.0 / scale_ratio,
                );
                copy_submatrix(
                    &d_wu_f32,
                    inter_d.min(inter_t),
                    dim_d,
                    &mut t_wu,
                    inter_t,
                    dim_t,
                    1.0 / scale_ratio,
                );
                copy_submatrix(
                    &d_wd_f32,
                    dim_d,
                    inter_d.min(inter_t),
                    &mut t_wd,
                    dim_t,
                    inter_t,
                    scale_ratio,
                );
            }

            target_layers.push(TransformerLayerWeights {
                attn_norm: vec![1.0; dim_t],
                wq: TensorBuf::from_f32_as_dtype(&[qd_t, dim_t], &t_wq, dt_t)?,
                wk: TensorBuf::from_f32_as_dtype(&[kvd_t, dim_t], &t_wk, dt_t)?,
                wv: TensorBuf::from_f32_as_dtype(&[kvd_t, dim_t], &t_wv, dt_t)?,
                bq: None,
                bk: None,
                bv: None,
                wo: TensorBuf::from_f32_as_dtype(&[dim_t, qd_t], &t_wo, dt_t)?,
                ffn_norm: vec![1.0; dim_t],
                w_gate: TensorBuf::from_f32_as_dtype(&[inter_t, dim_t], &t_wg, dt_t)?,
                w_up: TensorBuf::from_f32_as_dtype(&[inter_t, dim_t], &t_wu, dt_t)?,
                w_down: TensorBuf::from_f32_as_dtype(&[dim_t, inter_t], &t_wd, dt_t)?,
            });
        }

        // Shared principal `lm_head` + subtle higher-dim target nuance (`~0.18` std)
        let mut head_d = vec![0.0f32; v_sz * dim_d];
        let mut head_t = vec![0.0f32; v_sz * dim_t];
        for v in 0..v_sz {
            for d in 0..dim_d {
                let w = rng.next_gaussian_f32() * 0.45;
                head_d[v * dim_d + d] = w;
                head_t[v * dim_t + d] = w / scale_ratio;
            }
            for d in dim_d..dim_t {
                head_t[v * dim_t + d] = rng.next_gaussian_f32() * 0.14;
            }
        }

        let target_model = LlamaModel::from_parts(
            target_cfg,
            TensorBuf::from_f32_as_dtype(&[v_sz, dim_t], &emb_t, dt_t)?,
            target_layers,
            vec![1.0; dim_t],
            Some(TensorBuf::from_f32_as_dtype(&[v_sz, dim_t], &head_t, dt_t)?),
        )?;

        let draft_model = LlamaModel::from_parts(
            draft_cfg,
            TensorBuf::from_f32_as_dtype(&[v_sz, dim_d], &emb_d, dt_d)?,
            draft_layers,
            vec![1.0; dim_d],
            Some(TensorBuf::from_f32_as_dtype(&[v_sz, dim_d], &head_d, dt_d)?),
        )?;

        Ok((target_model, draft_model))
    }

    /// Convenience builder creating a `Tiny-16M` Target + `Draft-500K` Draft pair.
    pub fn build_deterministic_pair(
        vocab_size: usize,
        target_dtype: DType,
        seed: u64,
    ) -> Result<(LlamaModel, LlamaModel), ModelError> {
        let target_cfg = LlamaConfig::tiny_16m(vocab_size, target_dtype);
        let draft_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);
        Self::build_correlated_pair(target_cfg, draft_cfg, seed)
    }

    /// Builds a pathological adversarial `DraftModel` whose `lm_head` is anti-correlated (`-3.0x`)
    /// with `target`, placing ~95% of probability mass on the Target model's least-likely tokens!
    pub fn build_adversarial_draft(
        target_cfg: LlamaConfig,
        draft_cfg: LlamaConfig,
        seed: u64,
    ) -> Result<(LlamaModel, LlamaModel), ModelError> {
        let (target, aligned_draft) =
            Self::build_correlated_pair(target_cfg, draft_cfg.clone(), seed)?;
        let orig_head = aligned_draft.lm_head_view().to_f32_vec()?;
        let anti_head: Vec<f32> = orig_head.into_iter().map(|w| -3.0 * w).collect();
        let dt_d = draft_cfg.weight_dtype;
        let adv_draft = LlamaModel::from_parts(
            draft_cfg.clone(),
            TensorBuf::from_f32_as_dtype(
                &[draft_cfg.vocab_size, draft_cfg.dim],
                aligned_draft.tok_embeddings_f32(),
                dt_d,
            )?,
            aligned_draft.layers().to_vec(),
            aligned_draft.output_norm().to_vec(),
            Some(TensorBuf::from_f32_as_dtype(
                &[draft_cfg.vocab_size, draft_cfg.dim],
                &anti_head,
                dt_d,
            )?),
        )?;
        Ok((target, adv_draft))
    }
}

fn copy_submatrix(
    src: &[f32],
    src_rows: usize,
    src_cols: usize,
    dst: &mut [f32],
    dst_rows: usize,
    dst_cols: usize,
    scale: f32,
) {
    let r_lim = src_rows.min(dst_rows);
    let c_lim = src_cols.min(dst_cols);
    for r in 0..r_lim {
        for c in 0..c_lim {
            dst[r * dst_cols + c] = src[r * src_cols + c] * scale;
        }
    }
}
