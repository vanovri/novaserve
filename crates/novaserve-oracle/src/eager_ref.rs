//! Crystal-clear pure-`f64` single-sequence eager Transformer reference engine (`EagerReferenceEngine`).
//! Zero SIMD, zero paging, zero batching, zero caching—processes `[t_0, ..., t_n]` sequentially
//! from scratch on every call using textbook `f64` equations.

use novaserve_model::{LlamaConfig, LlamaModel};
use novaserve_sched::TokenLogprob;

struct EagerLayerF64 {
    attn_norm: Vec<f64>,
    wq: Vec<f64>,
    wk: Vec<f64>,
    wv: Vec<f64>,
    wo: Vec<f64>,
    ffn_norm: Vec<f64>,
    w_gate: Vec<f64>,
    w_up: Vec<f64>,
    w_down: Vec<f64>,
}

pub struct EagerReferenceEngine {
    cfg: LlamaConfig,
    tok_emb: Vec<f64>,
    layers: Vec<EagerLayerF64>,
    output_norm: Vec<f64>,
    lm_head: Vec<f64>,
}

impl EagerReferenceEngine {
    pub fn from_model(model: &LlamaModel) -> Self {
        let cfg = model.config().clone();
        let to_f64 = |v: Vec<f32>| -> Vec<f64> { v.into_iter().map(|x| x as f64).collect() };

        let tok_emb = model
            .tok_embeddings_f32()
            .iter()
            .map(|&x| x as f64)
            .collect();
        let output_norm = model.output_norm().iter().map(|&x| x as f64).collect();
        let lm_head = to_f64(model.lm_head_view().to_f32_vec().expect("lm_head to_f32"));

        let layers = model
            .layers()
            .iter()
            .map(|lw| EagerLayerF64 {
                attn_norm: lw.attn_norm.iter().map(|&x| x as f64).collect(),
                wq: to_f64(lw.wq.view().to_f32_vec().unwrap()),
                wk: to_f64(lw.wk.view().to_f32_vec().unwrap()),
                wv: to_f64(lw.wv.view().to_f32_vec().unwrap()),
                wo: to_f64(lw.wo.view().to_f32_vec().unwrap()),
                ffn_norm: lw.ffn_norm.iter().map(|&x| x as f64).collect(),
                w_gate: to_f64(lw.w_gate.view().to_f32_vec().unwrap()),
                w_up: to_f64(lw.w_up.view().to_f32_vec().unwrap()),
                w_down: to_f64(lw.w_down.view().to_f32_vec().unwrap()),
            })
            .collect();

        Self {
            cfg,
            tok_emb,
            layers,
            output_norm,
            lm_head,
        }
    }

    #[inline]
    fn rmsnorm_f64(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
        let d = weight.len();
        let mean_sq: f64 = x.iter().map(|&v| v * v).sum::<f64>() / (d as f64);
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .map(|(&xi, &wi)| xi * inv * wi)
            .collect()
    }

    #[inline]
    fn matvec_f64(w_row_major: &[f64], x: &[f64], out_dim: usize, in_dim: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; out_dim];
        for r in 0..out_dim {
            let mut sum = 0.0f64;
            let row = &w_row_major[r * in_dim..(r + 1) * in_dim];
            for c in 0..in_dim {
                sum += row[c] * x[c];
            }
            y[r] = sum;
        }
        y
    }

    #[inline]
    fn rope_vec_f64(
        vec: &mut [f64],
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        theta_base: f64,
        scaling: f64,
    ) {
        let p = (pos as f64) / scaling;
        for h in 0..n_heads {
            for i in 0..(head_dim / 2) {
                let exponent = -((2 * i) as f64) / (head_dim as f64);
                let angle = p * theta_base.powf(exponent);
                let (s, c) = angle.sin_cos();
                let idx = h * head_dim + 2 * i;
                let x0 = vec[idx];
                let x1 = vec[idx + 1];
                vec[idx] = x0 * c - x1 * s;
                vec[idx + 1] = x0 * s + x1 * c;
            }
        }
    }

    /// Computes exact `f64` final-position logits from scratch over `tokens`.
    pub fn forward_last_logits_f64(&self, tokens: &[u32]) -> Vec<f64> {
        assert!(!tokens.is_empty());
        let seq_len = tokens.len();
        let dim = self.cfg.dim;
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        let hd = self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;
        let eps = self.cfg.rms_norm_eps as f64;
        let gqa_ratio = self.cfg.n_heads / self.cfg.n_kv_heads;
        let inv_sqrt_d = 1.0 / (hd as f64).sqrt();

        let mut hidden: Vec<Vec<f64>> = tokens
            .iter()
            .map(|&tid| {
                let off = (tid as usize) * dim;
                self.tok_emb[off..off + dim].to_vec()
            })
            .collect();

        for layer in &self.layers {
            let mut q_seq = Vec::with_capacity(seq_len);
            let mut k_seq = Vec::with_capacity(seq_len);
            let mut v_seq = Vec::with_capacity(seq_len);

            for (pos, h_vec) in hidden.iter().enumerate() {
                let normed = Self::rmsnorm_f64(h_vec, &layer.attn_norm, eps);
                let mut q = Self::matvec_f64(&layer.wq, &normed, q_dim, dim);
                let mut k = Self::matvec_f64(&layer.wk, &normed, kv_dim, dim);
                let v = Self::matvec_f64(&layer.wv, &normed, kv_dim, dim);
                Self::rope_vec_f64(
                    &mut q,
                    pos,
                    self.cfg.n_heads,
                    hd,
                    self.cfg.rope_theta as f64,
                    self.cfg.rope_scaling as f64,
                );
                Self::rope_vec_f64(
                    &mut k,
                    pos,
                    self.cfg.n_kv_heads,
                    hd,
                    self.cfg.rope_theta as f64,
                    self.cfg.rope_scaling as f64,
                );
                q_seq.push(q);
                k_seq.push(k);
                v_seq.push(v);
            }

            for t in 0..seq_len {
                let mut attn_concat = vec![0.0f64; q_dim];
                for qh in 0..self.cfg.n_heads {
                    let kvh = qh / gqa_ratio;
                    let q_h = &q_seq[t][qh * hd..(qh + 1) * hd];
                    let mut scores = vec![0.0f64; t + 1];
                    let mut max_s = f64::NEG_INFINITY;
                    for (j, sc) in scores.iter_mut().enumerate() {
                        let k_h = &k_seq[j][kvh * hd..(kvh + 1) * hd];
                        let dot: f64 = q_h.iter().zip(k_h.iter()).map(|(&a, &b)| a * b).sum();
                        let s = dot * inv_sqrt_d;
                        *sc = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut sum_exp = 0.0f64;
                    for sc in &mut scores {
                        *sc = (*sc - max_s).exp();
                        sum_exp += *sc;
                    }
                    for (j, &w_p) in scores.iter().enumerate() {
                        let prob = w_p / sum_exp;
                        let v_h = &v_seq[j][kvh * hd..(kvh + 1) * hd];
                        for d in 0..hd {
                            attn_concat[qh * hd + d] += prob * v_h[d];
                        }
                    }
                }

                let o_proj = Self::matvec_f64(&layer.wo, &attn_concat, dim, q_dim);
                for d in 0..dim {
                    hidden[t][d] += o_proj[d];
                }

                let ffn_in = Self::rmsnorm_f64(&hidden[t], &layer.ffn_norm, eps);
                let gate = Self::matvec_f64(&layer.w_gate, &ffn_in, inter, dim);
                let up = Self::matvec_f64(&layer.w_up, &ffn_in, inter, dim);
                let mut swiglu = vec![0.0f64; inter];
                for i in 0..inter {
                    let g = gate[i];
                    swiglu[i] = (g / (1.0 + (-g).exp())) * up[i];
                }
                let down = Self::matvec_f64(&layer.w_down, &swiglu, dim, inter);
                for d in 0..dim {
                    hidden[t][d] += down[d];
                }
            }
        }

        let final_norm = Self::rmsnorm_f64(&hidden[seq_len - 1], &self.output_norm, eps);
        Self::matvec_f64(&self.lm_head, &final_norm, self.cfg.vocab_size, dim)
    }

    /// Runs textbook `f64` greedy generation from scratch at every step, returning
    /// `(generated_tokens, per_step_top5_logprobs)`.
    pub fn greedy_trajectory_and_top5(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> (Vec<u32>, Vec<Vec<TokenLogprob>>) {
        let mut seq = prompt.to_vec();
        let mut gen_tokens = Vec::with_capacity(max_new_tokens);
        let mut all_top5 = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let logits = self.forward_last_logits_f64(&seq);
            let mut max_l = f64::NEG_INFINITY;
            let mut best_id = 0usize;
            for (i, &z) in logits.iter().enumerate() {
                if z > max_l {
                    max_l = z;
                    best_id = i;
                }
            }

            let sum_exp: f64 = logits.iter().map(|&z| (z - max_l).exp()).sum();
            let log_z = max_l + sum_exp.ln();

            let mut indexed: Vec<(u32, f64)> = logits
                .iter()
                .enumerate()
                .map(|(i, &z)| (i as u32, z - log_z))
                .collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let top5: Vec<TokenLogprob> = indexed
                .into_iter()
                .take(5)
                .map(|(tid, lp)| TokenLogprob {
                    token_id: tid,
                    logprob: lp as f32,
                })
                .collect();

            gen_tokens.push(best_id as u32);
            all_top5.push(top5);
            seq.push(best_id as u32);
        }

        (gen_tokens, all_top5)
    }
}
