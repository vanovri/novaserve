# Module M05 Design Specification — `crates/novaserve-sched`

## 1. Architectural Scope
`novaserve-sched` orchestrates iteration-level LLM serving (`#![deny(unsafe_code)]`), combining:
1. **Continuous Batching Lifecycle (`scheduler.rs`):** Dynamic multiplexing of `waiting`, `running`, and `swapped` sequence groups under `max_num_batched_tokens` and `max_num_seqs` budgets, supporting `n >= 1` Copy-on-Write parallel completions, `RadixCache` prefix reuse, `ChunkedPrefill`, and immediate `abort_request(req_id)` reclamation.
2. **Starvation-Free Preemption (`PreemptionMode::Swap` & `PreemptionMode::Recompute`):** Strict FIFO arrival-priority protection—on KV block exhaustion after `RadixCache::evict_lru`, the most-recently-arrived (`lowest-priority`) running group is preempted so the oldest request always makes forward progress.
3. **Exact Speculative Decoding (`speculative.rs`):** Leviathan-Kalman-Matias (ICML 2023) $K$-step draft + single-pass $(K+1)$-token target verification with exact rejection probability $\min(1, p_i(d_i)/q_i(d_i))$, residual replacement sampling $\text{norm}(\max(0, p_i - q_i))$, exact Paged KV-cache rollback (`truncate_table`), bonus token $t_{K+1} \sim p_{K+1}$, and rolling-$\alpha$ adaptive lookahead control.

---

## 2. Scheduler State Machine & Preemption Guarantees

```text
  submit_request ──▶ [ WAITING ] ──(Radix match + ChunkedPrefill)──▶ [ RUNNING ] ──▶ [ FINISHED / ABORTED ]
                         ▲                                              │   ▲
                         │          (PreemptionMode::Recompute)         │   │
                         └──────────────────────────────────────────────┤   │ (Swap-In + Hysteresis)
                                                                        ▼   │
                                           (PreemptionMode::Swap) ──▶ [ SWAPPED ]
```

- **Priority Invariant:** `running` stays sorted ascending by `arrival_order`. Preemption always evicts `running.pop()` (largest `arrival_order`), mathematically preventing cyclic livelock or starvation of the head-of-line request.
- **`Recompute`Determinism:** Preempted sequences retain `prompt_tokens + generated_tokens`, RNG state, and `GrammarMatcher`. Upon re-admission, they prefill up to `last_emitted_token` via `RadixCache` + `ChunkedPrefill` and resume sampling from the exact next position—producing identical greedy outputs to unlimited-memory execution.

---

## 3. Exact Speculative Decoding Synchronization
For sequence prefix $T = [t_0, \dots, t_{m-1}, x_{\text{last}}]$ (`len = m + 1`), both `Target` and `Draft` KV caches hold positions $0 \dots m-1$ (`m` tokens) at step entry:
1. **Draft ($K$ single-token steps):** Feeds $x_{\text{last}}, d_1, \dots, d_{K-1}$ at positions $m \dots m+K-1$ into `DraftModel`, recording $[d_1, \dots, d_K]$ and $[q_1, \dots, q_K]$. (`Draft KV` len $= m + K$).
2. **Target Verify ($1$ batched pass of $K+1$ tokens):** Feeds $[x_{\text{last}}, d_1, \dots, d_K]$ at positions $m \dots m+K$ into `TargetModel`, producing $[p_1, \dots, p_K, p_{K+1}]$. (`Target KV` len $= m + K + 1$).
3. **Rejection / Bonus Resolution:**
   - **First rejection at $i \in \{1..K\}$ ($u_i > p_i(d_i)/q_i(d_i)$):** Truncate **both** `Target KV` and `Draft KV` to `m + i`, sample $t_i \sim \text{norm}(\max(0, p_i - q_i))$, set $x_{\text{last}} \leftarrow t_i$ (`i` verified tokens emitted).
   - **All $K$ accepted:** Sync $d_K$ at pos $m + K$ into `Draft KV`, sample bonus $t_{K+1} \sim p_{K+1}$, set $x_{\text{last}} \leftarrow t_{K+1}$ (`K + 1` verified tokens emitted in 1 Target pass!).
4. **Adaptive Lookahead Controller:** Exponentially weighted moving average $\hat{\alpha} \leftarrow 0.85 \hat{\alpha} + 0.15 \alpha_{\text{step}}$ dynamically scales active $K \in [0, K_{\max}]$ (disabling draft overhead when $\hat{\alpha} < 0.25$).
