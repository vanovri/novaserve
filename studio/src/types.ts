export type KvBlockStatus =
  | "free"
  | "unique"
  | "cow_shared"
  | "radix_pinned"
  | "radix_shared";

export interface WasmKvBlock {
  block_id: number;
  status: KvBlockStatus;
  ref_count: number;
  owner_seq_ids: string[];
  filled_slots: number;
  is_radix_cached: boolean;
}

export interface WasmRadixNode {
  node_id: number;
  parent_id: number | null;
  token_preview: string;
  token_ids: number[];
  total_edge_tokens: number;
  block_ids: number[];
  ref_count: number;
  lru_age: number;
}

export interface WasmSpecDraftItem {
  token_id: number;
  token_text: string;
  draft_prob: number;
  target_prob: number;
  accepted: boolean;
}

export interface WasmSpecTrace {
  req_id: number;
  child_idx: number;
  lookahead_k: number;
  draft_items: WasmSpecDraftItem[];
  residual_or_bonus_token_id: number;
  residual_or_bonus_text: string;
  is_bonus: boolean;
  accepted_count: number;
}

export interface WasmTokenLogitBar {
  token_id: number;
  token_text: string;
  raw_logit: number;
}

export interface WasmGrammarXray {
  req_id: number;
  child_idx: number;
  schema_json: string;
  dfa_state: number;
  total_dfa_states: number;
  is_accepting: boolean;
  allowed_count: number;
  vocab_size: number;
  chosen_token_id: number;
  chosen_token_text: string;
  forced_jump_tokens: string[];
  top_allowed: WasmTokenLogitBar[];
  top_masked: WasmTokenLogitBar[];
}

export interface WasmSequenceLane {
  seq_key: string;
  req_id: number;
  child_idx: number;
  status: "waiting" | "prefill" | "decode" | "swapped" | "completed";
  prompt_len: number;
  kv_tokens: number;
  generated_count: number;
  max_new_tokens: number;
  decoded_text: string;
  block_ids: number[];
  swapped_blocks: number;
  cached_prefix_tokens: number;
  preemptions: number;
  use_speculative: boolean;
  has_grammar: boolean;
  dfa_state: number | null;
  finish_reason: string | null;
}

export interface WasmCompletedChoice {
  child_idx: number;
  decoded_text: string;
  token_count: number;
  finish_reason: string;
}

export interface WasmCompletedRequest {
  req_id: number;
  prompt_len: number;
  cached_prefix_tokens: number;
  preemptions: number;
  choices: WasmCompletedChoice[];
}

export interface WasmStepTelemetry {
  step_index: number;
  has_unfinished_work: boolean;
  block_size: number;
  total_kv_blocks: number;
  free_kv_blocks: number;
  radix_cached_blocks: number;
  used_swap_blocks: number;
  total_swap_blocks: number;
  scheduled_prefill_tokens: number;
  scheduled_decode_tokens: number;
  preempted_this_step: number;
  swapped_in_this_step: number;
  waiting_groups: number;
  running_groups: number;
  swapped_groups: number;
  radix_hit_rate_pct: number;
  radix_evicted_blocks: number;
  rolling_spec_alpha_pct: number;
  active_spec_k: number;
  kv_slab: WasmKvBlock[];
  radix_tree: WasmRadixNode[];
  sequences: WasmSequenceLane[];
  speculative_traces: WasmSpecTrace[];
  grammar_xrays: WasmGrammarXray[];
  completed_this_step: WasmCompletedRequest[];
}

export interface SubmitRequestPayload {
  prompt: string;
  max_tokens: number;
  temp: number;
  n_parallel: number;
  json_schema_opt: string | null;
  use_speculative: boolean;
  seed?: number;
}

export interface EngineConfigPayload {
  num_kv_blocks: number;
  block_size: number;
  max_num_batched_tokens: number;
  speculative_lookahead: number;
  enable_radix_cache: boolean;
}
