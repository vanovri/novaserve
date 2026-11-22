# Module M07 Design Specification — `crates/novaserve-oracle`

## 1. Architectural Scope
`novaserve-oracle` is the mathematical ground-truth and chaos verification harness of **NovaServe** (`#![deny(unsafe_code)]`):
1. **Pure-`f64` Single-Sequence Eager Reference (`eager_ref.rs`):** Zero SIMD, zero paging, zero caching, zero batching textbook `f64` implementation of `RMSNorm`, `RoPE`, causal `GQA`, `SwiGLU`, and `log_softmax`.
2. **5,000-Scenario Continuous-Batching & Preemption Chaos Fuzzer (`batch_fuzzer.rs`):** Exercises scrambled physical blocks, shared/mid-block `RadixCache` hits, `n = 3` CoW forks, `max_num_batched_tokens = 19` `ChunkedPrefill`, severe block scarcity (`Swap` & `Recompute` preemptions), and mid-flight `abort_request` cancellations—verifying greedy token trajectories, `top-5 logprobs` (`<= 1e-4`), and 100% block/swap reclamation across all 5,000 workloads.
3. **3-Regime Statistical Speculative Decoding Verifier (`spec_proof.rs`):** Samples `30,000+` transitions per regime at `temperature = 0.9, top_k = 16` across *Aligned* ($\alpha \approx 0.70$), *Mismatched* ($\alpha \approx 0.15$), and *Adversarial* (~95% mass on Target's least-likely tokens) Draft models, proving $\text{TVD}(P_{\text{spec}}, P_{\text{target}}) < 0.015$ and Pearson $\chi^2$ goodness-of-fit pass.

---

## 2. Statistical Proof Equations

### 2.1 Leviathan Marginal Invariance Identity
For any target distribution $p(x)$ and arbitrary draft distribution $q(x)$:
$$\mathbb{P}(X_{\text{spec}} = x) = q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_{y} \min(p(y), q(y))\right) \frac{\max(0, p(x) - q(x))}{\sum_z \max(0, p(z) - q(z))} = \min(p(x), q(x)) + \max(0, p(x) - q(x)) \equiv p(x)$$

### 2.2 Total Variation Distance & $\chi^2$ Gate
$$\text{TVD}(\hat{P}_{\text{spec}}, P_{\text{target}}) = \frac{1}{2}\sum_{x=0}^{V-1} \left|\hat{P}_{\text{spec}}(x) - P_{\text{target}}(x)\right| < 0.015, \qquad \chi^2 = \sum_{x \in \text{supp}(P)} \frac{(N_x - N P_{\text{target}}(x))^2}{N P_{\text{target}}(x)}$$
