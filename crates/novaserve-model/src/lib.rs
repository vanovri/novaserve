#![deny(unsafe_code)]

//! `novaserve-model`: Grouped-Query Attention (`GQA`) Transformer (`Llama`/`Qwen2`/`Mistral`),
//! Online-Softmax Paged FlashDecoding, Token-Budgeted Chunked Prefill, and `ModelFactory`.

pub mod batch;
pub mod error;
pub mod factory;
pub mod llama;
pub mod paged_attn;

pub use batch::BatchInput;
pub use error::ModelError;
pub use factory::ModelFactory;
pub use llama::{LlamaConfig, LlamaModel, TransformerLayerWeights};
pub use paged_attn::{paged_attention_gqa, scatter_write_kv_layer};
