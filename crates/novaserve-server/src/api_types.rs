use novaserve_kv::{PhysicalBlockSnapshot, RadixNodeSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopSpec {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSpec {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResponseFormatSpec {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub json_schema: Option<Value>,
}

impl ResponseFormatSpec {
    /// Extracts the underlying JSON Schema object whether wrapped in OpenAI's
    /// `{"name": "...", "schema": {...}}` envelope or passed directly as `{ "type": "object", ... }`.
    pub fn extract_schema(&self) -> Option<&Value> {
        if self.kind != "json_schema" {
            return None;
        }
        let js = self.json_schema.as_ref()?;
        if let Some(inner) = js.get("schema") {
            if inner.is_object() {
                return Some(inner);
            }
        }
        Some(js)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model_name")]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default = "default_one")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default = "default_one_f32")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub logit_bias: BTreeMap<String, f32>,
    #[serde(default)]
    pub response_format: Option<ResponseFormatSpec>,
    #[serde(default = "default_true")]
    pub use_speculative: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionRequest {
    #[serde(default = "default_model_name")]
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default = "default_one")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default = "default_one_f32")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub logit_bias: BTreeMap<String, f32>,
    #[serde(default)]
    pub response_format: Option<ResponseFormatSpec>,
    #[serde(default = "default_true")]
    pub use_speculative: bool,
}

fn default_model_name() -> String {
    "novaserve-tiny-16m".into()
}
const fn default_max_tokens() -> usize {
    64
}
const fn default_one() -> usize {
    1
}
const fn default_one_f32() -> f32 {
    1.0
}
const fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiTopLogprobEntry {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiTokenLogprobInfo {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<ApiTopLogprobEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiChoiceLogprobs {
    pub content: Vec<ApiTokenLogprobInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ApiChoiceLogprobs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cached_prefix_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ApiChoiceLogprobs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatStreamChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionStreamChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveLaneTelemetry {
    pub req_id: u64,
    pub child_idx: usize,
    pub status: String,
    pub prompt_tokens: usize,
    pub kv_tokens: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub text_preview: String,
    pub last_spec_burst: usize,
    pub has_json_schema: bool,
    pub dfa_state: Option<u32>,
    pub total_dfa_states: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub step_index: u64,
    pub model_name: String,
    pub total_kv_blocks: usize,
    pub free_kv_blocks: usize,
    pub radix_cached_blocks: usize,
    pub total_swap_blocks: usize,
    pub used_swap_blocks: usize,
    pub external_fragmentation_pct: f32,
    pub waiting_groups: usize,
    pub running_groups: usize,
    pub swapped_groups: usize,
    pub completed_requests_total: u64,
    pub aborted_requests_total: u64,
    pub tokens_generated_total: u64,
    pub tokens_per_sec: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p99_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p99_ms: f64,
    pub radix_hit_rate_pct: f64,
    pub radix_evicted_blocks: u64,
    pub radix_hit_sparkline: Vec<u64>,
    pub speculative_alpha_pct: f64,
    pub active_spec_k: usize,
    pub blocks: Vec<PhysicalBlockSnapshot>,
    pub radix_tree: Vec<RadixNodeSnapshot>,
    pub active_lanes: Vec<ActiveLaneTelemetry>,
}
