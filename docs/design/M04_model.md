# Module M04 Design Specification — `crates/novaserve-model`

## 1. Architectural Scope
`novaserve-model` implements a full decoder-only **Grouped-Query Attention (`GQA`) Transformer** (`Llama` / `Qwen2` / `Mistral` compatible, `#![deny(unsafe_code)]`) executing natively over `novaserve-kv` **Paged Block Tables**:
1. **Unified Heterogeneous Batch Descriptor (`BatchInput`):** Combines `1`-token Decode steps, arbitrary `C`-token `ChunkedPrefill` slices (including post-`RadixCache` non-zero `start_pos`), and `K+1`-token Speculative Verification passes in a single forward invocation.
2. **Scatter Paged KV Write + Online-Softmax Paged FlashDecoding (`paged_attn.rs`):** Never materializes contiguous `[seq_len, head_dim]` KV tensors; streams directly over `16`-token physical `BlockId`s with Milakov-Gimelshein running `(m, l, acc)` log-sum-exp state and `rayon` parallel block-partition reduction for long contexts (`context_len >= 256`).
3. **Full Pre-Norm Residual Stack & Subspace-Distilled `ModelFactory` (`llama.rs`, `factory.rs`):** Supports `FP32`, `F16`, `BF16`, `Q8_0`, and `Q4_0` weights, loads/exports canonical `.gguf` (`v3`) and `.safetensors` models, and deterministically constructs correlated `(TargetModel, DraftModel)` pairs.

---

## 2. Online-Softmax Paged Attention & FlashDecoding Reduction

### 2.1 Single-Pass Block Streaming
Given query head $q \in \mathbb{R}^d$ at causal position $p$, mapped to KV head $h_{kv} = \lfloor h_q / g \rfloor$ ($g = n_{\text{heads}} / n_{\text{kv\_heads}}$), and logical block table $[b_0, b_1, \dots, b_M]$:
- Number of causal tokens is $L = p + 1$, spanning $B = \lceil L / 16 \rceil$ blocks (with final block bounded to $(p \bmod 16) + 1$ valid slots).
- Within any block partition $[B_{\text{start}}, B_{\text{end}})$, maintain `f64` running state $(m, \ell, o \in \mathbb{R}^d)$ initialized to $(-\infty, 0, \mathbf{0})$:
  $$s_j = \frac{q \cdot k_j}{\sqrt{d}}, \quad m' = \max(m, s_j), \quad \alpha = e^{m - m'}, \quad \beta = e^{s_j - m'}$$
  $$\ell \leftarrow \ell \cdot \alpha + \beta, \quad o \leftarrow o \cdot \alpha + \beta v_j, \quad m \leftarrow m'$$

### 2.2 Parallel Block-Partition Merge (`context_len >= 256`)
For $L \ge 256$ ($\ge 16$ physical blocks), blocks are partitioned across `rayon` worker threads into 4-block (`64`-token) chunks producing partial states $(m_r, \ell_r, o_r)$ merged associatively:
$$m_{AB} = \max(m_A, m_B), \quad \ell_{AB} = \ell_A e^{m_A - m_{AB}} + \ell_B e^{m_B - m_{AB}}$$
$$o_{AB} = o_A e^{m_A - m_{AB}} + o_B e^{m_B - m_{AB}}, \quad \text{out} = \frac{o_{\text{total}}}{\ell_{\text{total}}}$$

---

## 3. Exact Paged & Chunked Invariance Law
Because:
1. `RoPE` applies exact per-token `positions[t]` (`f64` phase computation),
2. `RMSNorm` and `SwiGLU` are strictly token-wise independent,
3. All `BatchInput` tokens scatter their `K, V` slots into `KvSlab` prior to attention, and
4. `paged_attention_gqa` restricts attention strictly to $0 \le j \le \text{positions}[t]$ regardless of physical `BlockId` permutation or tail tokens belonging to later positions in the same prefill chunk,

the resulting logits at any position $t$ are invariant within `< 1e-5` (`FP32`) across:
- Contiguous `[1, N]` eager execution,
- Scrambled `BlockId` permutations,
- `RadixCache` CoW shared prefix blocks,
- Arbitrary `ChunkedPrefill` splits $C \in [1, N]$, and
- Interleaved multi-sequence `ContinuousBatch` execution.
