# Module M08 Design Specification — `crates/novaserve-server` & `crates/novaserve-tui`

## 1. Architectural Scope
Module M08 exposes **NovaServe** as a production async OpenAI-compatible HTTP/SSE gateway (`novaserve-server`), an end-to-end SLA & ablation profiler (`novaserve-bench`), and an interactive 4-quadrant `ratatui` operator cockpit (`novaserve-tui` / `novaserve-top`), all under `#![deny(unsafe_code)]`.

---

## 2. Async Gateway & Instantaneous TCP-Drop Cancellation

```text
  64 Async HTTP/SSE Clients ──▶ [ Tokio / Axum Router ]
                                        │
                   EngineCommand MPSC   │   SseChannelBody (HttpBody + Drop Guard)
                                        ▼
                       [ Dedicated Engine OS Thread ]
                         ├─ ServingEngine (Paged KV + RadixCache + Speculative)
                         ├─ Schema DFA Cache (response_format: json_schema)
                         ├─ TokenStreamDecoder (per-child incremental UTF-8)
                         └─ Live TelemetrySnapshot (TTFT / ITL p50 & p99)
```

- **Zero-Blocking Compute Isolation:** `ServingEngine` runs on a dedicated OS worker thread polling an MPSC command queue between continuous-batching `step()` calls, preventing SIMD matrix multiplications from starving Tokio's epoll/IOCP reactor.
- **Instantaneous Disconnect Reclamation:** `SseChannelBody` implements `hyper::body::Body` directly over `tokio::sync::mpsc::UnboundedReceiver` (`0` extra stream dependencies). Dropping the TCP socket drops `SseChannelBody`, whose `Drop` implementation immediately dispatches `EngineCommand::Abort { req_id }`, reclaiming all unshared Paged KV blocks on the next iteration.

---

## 3. SLA Profiler (`novaserve-bench`) & `novaserve-top` Cockpit
- **`novaserve-bench`** runs four reproducible `--release` serving experiments on `Tiny-16M` / `Micro-2M` + `Draft-500K` and writes `docs/benchmarks/serving_sla.md` and `docs/benchmarks/serving_sla.json`:
  1. `FP32` vs `Q8_0` vs `Q4_0` throughput & weight memory compression.
  2. `32`-client Continuous Batching + Chunked Prefill vs Naive Sequential Locking (`Tok/s`, `TTFT p50/p99`, `ITL p50/p99`).
  3. Shared 512-token system prompt `RadixCache OFF` vs `RadixCache ON` (`TTFT` speedup & KV block hit rate).
  4. Correlated `Draft + Target` Speculative Decoding (`K = 4`) acceptance $\alpha$ & `TPOT` wall-clock speedup.
- **`novaserve-top`** renders 4 live quadrants (`Physical KV Block Map`, `Continuous Batch & Speculative Lanes`, `Radix Tree & JSON-Schema FSM Inspector`, `SLA Latency Gauges`) in both `--demo` embedded mode and `--url` HTTP polling mode, verified headlessly at `120x40` via `ratatui::backend::TestBackend`.
