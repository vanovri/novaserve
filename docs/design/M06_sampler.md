# Module M06 Design Specification — `crates/novaserve-sampler`

## 1. Architectural Scope
`novaserve-sampler` implements an allocation-free, numerically bulletproof 9-stage logits post-processing and sampling pipeline (`#![deny(unsafe_code)]`) integrated directly with `novaserve-frontend` (`AllowedTokenMask`) and `novaserve-sched` (`SamplerStrategy`).

---

## 2. Canonical 9-Stage Pipeline & Numerical Protections

1. **Logit Bias & Stop/EOS Filtering:** Applies user `(token_id, bias)` offsets and masks `eos_token_id` to $-\infty$ when `ignore_eos == true`.
2. **Pre-Softmax Grammar Bitmask (`AllowedTokenMask`):** Applied strictly *before* temperature/Top-K/Top-P so probability mass concentrates exclusively over syntactically valid next tokens. Returns `SamplerError::EmptyGrammarSupport` cleanly if no finite token survives.
3. **$O(|H|)$ Sparse Repetition, Frequency & Presence Penalties:**
   - Accumulates token counts into a reusable scratch array and resets only touched indices (`seen_ids`) in $O(|H|)$.
   - Applies Keskar sign-aware `repetition_penalty` ($z > 0 \implies z / r$, $z < 0 \implies z \cdot r$) followed by linear `frequency_penalty * count + presence_penalty`.
4. **Deterministic Greedy Fast-Path (`temperature <= 1e-6`):** Single-pass `argmax` with lowest-token-ID tie-breaking (`if z > best`), emitting an exact 1-hot distribution.
5. **Shift-Before-Divide Stable Temperature Softmax:**
   - Computes $m = \max_j z_j$ *prior* to dividing by $T$:
     $$u_i = \frac{z_i - m}{T} \le 0 \implies e^{u_i} \in [0, 1]$$
   - Eliminates `f32::INFINITY` / `NaN` overflow even for pathological inputs ($z = 10^{30}, T = 10^{-5}$).
6. **$O(V)$ Partial-Selection `Top-K`:** Uses `select_nth_unstable_by` in linear $O(V)$ time instead of full $O(V \log V)$ sorting.
7. **`Min-P` Truncation (Nguyen et al., 2024):** Retains $p_i \ge \text{min\_p} \cdot p_{\max}$ (at least 1 survivor).
8. **`Top-P` Nucleus Truncation:** Sorts only the surviving $\le K$ candidates descending (`total_cmp` + token-ID tie-break), keeps minimal prefix $\sum p_i \ge \text{top\_p}$, and renormalizes in `f64`.
9. **Seeded `ChaCha8Rng` Multinomial Draw + `TopLogprobs`:** Exact `f64` CDF draw plus top-$N$ $\ln p_i$ alternatives.
