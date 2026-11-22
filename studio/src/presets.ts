import type { EngineConfigPayload, SubmitRequestPayload } from "./types";
import type { NovaServeStudioClient } from "./wasm_bridge";

export interface StudioPreset {
  id: string;
  title: string;
  badge: string;
  description: string;
  defaultTab: "radix" | "speculative" | "grammar";
  config: EngineConfigPayload;
  setup: (client: NovaServeStudioClient) => void;
}

const SHARED_SYSTEM_PROMPT =
  "SYSTEM: You are NovaServe, a deterministic Rust LLM inference engine combining vLLM PagedAttention non-contiguous KV blocks, SGLang RadixAttention prefix caching, Sarathi Chunked Prefill, and Leviathan Exact Speculative Decoding.\n";

export const TOOL_CALL_JSON_SCHEMA = JSON.stringify(
  {
    type: "object",
    properties: {
      tool: {
        type: "string",
        enum: ["search_docs", "inspect_kv", "route_batch"],
      },
      limit: { type: "integer" },
      strict: { type: "boolean" },
    },
    required: ["tool", "limit", "strict"],
    additionalProperties: false,
  },
  null,
  2
);

export const STUDIO_PRESETS: StudioPreset[] = [
  {
    id: "radix-cow",
    title: "1. Shared System-Prompt Radix Hit & CoW n=4 Forking",
    badge: "RADIX + CoW x4",
    description:
      "Warms a multi-block system prompt into RadixCache, then forks an n=4 parallel completion sharing physical KV blocks with zero copy.",
    defaultTab: "radix",
    config: {
      num_kv_blocks: 36,
      block_size: 8,
      max_num_batched_tokens: 128,
      speculative_lookahead: 0,
      enable_radix_cache: true,
    },
    setup: (client) => {
      const warmReq: SubmitRequestPayload = {
        prompt: `${SHARED_SYSTEM_PROMPT}USER: Summarize KV block conservation.\nASSISTANT:`,
        max_tokens: 4,
        temp: 0.6,
        n_parallel: 1,
        json_schema_opt: null,
        use_speculative: false,
        seed: 101,
      };
      client.submitRequest(warmReq);
      // Step warmup request so shared prefix blocks populate RadixCache immediately
      for (let i = 0; i < 8; i++) {
        const s = client.stepIteration();
        if (!s.has_unfinished_work) break;
      }

      client.submitRequest({
        prompt: `${SHARED_SYSTEM_PROMPT}USER: Generate 4 parallel rollout hypotheses.\nASSISTANT:`,
        max_tokens: 16,
        temp: 0.85,
        n_parallel: 4,
        json_schema_opt: null,
        use_speculative: false,
        seed: 202,
      });
      client.submitRequest({
        prompt: `${SHARED_SYSTEM_PROMPT}USER: Analyze prefix divergence branch.\nASSISTANT:`,
        max_tokens: 14,
        temp: 0.75,
        n_parallel: 1,
        json_schema_opt: null,
        use_speculative: false,
        seed: 303,
      });
    },
  },
  {
    id: "chunked-swap",
    title: "2. Chunked Prefill + Memory Squeeze Swap Preemption",
    badge: "SWAP-OUT / IN",
    description:
      "Runs 4 competing long sequences on a scarce 14-block KV slab with chunk budget B=10, triggering multi-step Chunked Prefill and lossless CPU Swap-Out/Swap-In.",
    defaultTab: "radix",
    config: {
      num_kv_blocks: 14,
      block_size: 8,
      max_num_batched_tokens: 10,
      speculative_lookahead: 0,
      enable_radix_cache: false,
    },
    setup: (client) => {
      for (let i = 0; i < 4; i++) {
        client.submitRequest({
          prompt: `Sequence #${i + 1}: High-pressure PagedAttention memory squeeze stress workload segment alpha beta gamma delta.`,
          max_tokens: 18,
          temp: 0.5,
          n_parallel: 1,
          json_schema_opt: null,
          use_speculative: false,
          seed: 400 + i,
        });
      }
    },
  },
  {
    id: "speculative-k4",
    title: "3. Speculative Decoding (K=4) Live Rejection Inspector",
    badge: "LEVIATHAN K=4",
    description:
      "Pairs subspace-distilled Draft-500K with Micro-2M Target (K=4, temp=0.7), visualizing u <= p(x)/q(x) acceptances and residual norm(max(0, p-q)) resampling.",
    defaultTab: "speculative",
    config: {
      num_kv_blocks: 36,
      block_size: 8,
      max_num_batched_tokens: 128,
      speculative_lookahead: 4,
      enable_radix_cache: true,
    },
    setup: (client) => {
      client.submitRequest({
        prompt:
          "Explain exact speculative rejection sampling and KV rollback invariants:",
        max_tokens: 28,
        temp: 0.7,
        n_parallel: 1,
        json_schema_opt: null,
        use_speculative: true,
        seed: 777,
      });
    },
  },
  {
    id: "grammar-dfa",
    title: "4. JSON-Schema Constrained Decoding DFA X-Ray",
    badge: "JSON-DFA -∞ MASK",
    description:
      "Compiles a strict Draft-2020-12 tool-call JSON Schema into a minimized Byte-DFA, crossing out syntactically illegal tokens (-∞) prior to softmax.",
    defaultTab: "grammar",
    config: {
      num_kv_blocks: 32,
      block_size: 8,
      max_num_batched_tokens: 128,
      speculative_lookahead: 0,
      enable_radix_cache: true,
    },
    setup: (client) => {
      client.submitRequest({
        prompt: "Emit a valid JSON tool call for documentation search:",
        max_tokens: 64,
        temp: 0.8,
        n_parallel: 1,
        json_schema_opt: TOOL_CALL_JSON_SCHEMA,
        use_speculative: false,
        seed: 2026,
      });
    },
  },
];
