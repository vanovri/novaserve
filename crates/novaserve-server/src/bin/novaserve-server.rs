#![deny(unsafe_code)]

use clap::Parser;
use novaserve_frontend::{BpeTokenizer, VocabularyBuilder};
use novaserve_model::{LlamaConfig, LlamaModel, ModelFactory};
use novaserve_sched::SchedulerConfig;
use novaserve_server::{build_router, EngineActorHandle};
use novaserve_tensor::{DType, GgufFile, SafetensorsFile};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
#[command(
    name = "novaserve-server",
    about = "NovaServe OpenAI-Compatible High-Throughput LLM Inference Server"
)]
struct ServerArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Optional path to a `.gguf` (v3) or `.safetensors` weights file.
    /// When omitted, boots a deterministic in-memory `Tiny-16M` GQA model via `ModelFactory`.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Optional path to `config.json` (used alongside `.safetensors` weights).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Optional path to HuggingFace `tokenizer.json`.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Weight quantization dtype for built-in `Tiny-16M` (`f32`, `q8_0`, `q4_0`).
    #[arg(long, default_value = "q8_0")]
    dtype: String,

    /// Number of physical 16-token PagedAttention KV blocks.
    #[arg(long, default_value_t = 512)]
    kv_blocks: usize,

    /// Token budget per Continuous Batching iteration (`Chunked Prefill` threshold).
    #[arg(long, default_value_t = 256)]
    max_batch_tokens: usize,

    /// Speculative decoding lookahead `K` (`0` disables draft speculation).
    #[arg(long, default_value_t = 4)]
    lookahead_k: usize,
}

fn parse_dtype(s: &str) -> Result<DType, String> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Ok(DType::F32),
        "f16" | "fp16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "q8_0" | "q8" => Ok(DType::Q8_0),
        "q4_0" | "q4" => Ok(DType::Q4_0),
        other => Err(format!("unsupported --dtype '{other}'")),
    }
}

fn load_hf_llama_config(path: &PathBuf) -> Result<LlamaConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let vocab_size = v
        .get("vocab_size")
        .and_then(|x| x.as_u64())
        .ok_or("missing vocab_size")? as usize;
    let dim = v
        .get("hidden_size")
        .and_then(|x| x.as_u64())
        .ok_or("missing hidden_size")? as usize;
    let intermediate_size = v
        .get("intermediate_size")
        .and_then(|x| x.as_u64())
        .ok_or("missing intermediate_size")? as usize;
    let n_layers = v
        .get("num_hidden_layers")
        .and_then(|x| x.as_u64())
        .ok_or("missing num_hidden_layers")? as usize;
    let n_heads = v
        .get("num_attention_heads")
        .and_then(|x| x.as_u64())
        .ok_or("missing num_attention_heads")? as usize;
    let n_kv_heads = v
        .get("num_key_value_heads")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(n_heads);
    let head_dim = v
        .get("head_dim")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(dim / n_heads);
    let rms_norm_eps = v
        .get("rms_norm_eps")
        .and_then(|x| x.as_f64())
        .unwrap_or(1e-5) as f32;
    let rope_theta = v
        .get("rope_theta")
        .and_then(|x| x.as_f64())
        .unwrap_or(10_000.0) as f32;
    let max_seq_len = v
        .get("max_position_embeddings")
        .and_then(|x| x.as_u64())
        .unwrap_or(2048) as usize;
    let tie_word_embeddings = v
        .get("tie_word_embeddings")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);

    Ok(LlamaConfig {
        arch_name: "llama".into(),
        vocab_size,
        dim,
        intermediate_size,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        rope_scaling: 1.0,
        max_seq_len,
        tie_word_embeddings,
        weight_dtype: DType::F32,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ServerArgs::parse();
    let dtype = parse_dtype(&args.dtype)?;

    let (model_name, target_model, draft_opt, tokenizer) = if let Some(model_path) = &args.model {
        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("novaserve-custom")
            .to_string();
        let ext = model_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if ext == "gguf" {
            let gguf = GgufFile::open(model_path)?;
            let model = LlamaModel::from_gguf(&gguf)?;
            let tok = if let Some(tp) = &args.tokenizer {
                let raw = std::fs::read_to_string(tp)?;
                BpeTokenizer::from_tokenizer_json(&raw)?
            } else {
                BpeTokenizer::from_gguf(&gguf)
                    .or_else(|_| VocabularyBuilder::build_synthetic(model.config().vocab_size))?
            };
            (name, model, None, Arc::new(tok))
        } else if ext == "safetensors" {
            let cfg_path = args
                .config
                .clone()
                .or_else(|| model_path.parent().map(|p| p.join("config.json")))
                .ok_or("Safetensors model requires --config <config.json>")?;
            let cfg = load_hf_llama_config(&cfg_path)?;
            let st = SafetensorsFile::open(model_path)?;
            let model = LlamaModel::from_safetensors(&st, cfg)?;
            let tok = if let Some(tp) = &args.tokenizer {
                let raw = std::fs::read_to_string(tp)?;
                BpeTokenizer::from_tokenizer_json(&raw)?
            } else {
                VocabularyBuilder::build_synthetic(model.config().vocab_size)?
            };
            (name, model, None, Arc::new(tok))
        } else {
            return Err(format!("unsupported model file extension '.{ext}'").into());
        }
    } else {
        let vocab_size = 1024;
        let tok = Arc::new(VocabularyBuilder::build_synthetic(vocab_size)?);
        let t_cfg = LlamaConfig::tiny_16m(vocab_size, dtype);
        let (tm, dm) = if args.lookahead_k > 0 {
            let d_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);
            let (t, d) = ModelFactory::build_correlated_pair(t_cfg, d_cfg, 2026_1122)?;
            (t, Some(d))
        } else {
            (ModelFactory::build_random_model(t_cfg, 2026_1122)?, None)
        };
        (format!("novaserve-tiny-16m-{}", args.dtype), tm, dm, tok)
    };

    let sched_cfg = SchedulerConfig {
        max_num_batched_tokens: args.max_batch_tokens.max(16),
        max_num_seqs: 64,
        num_physical_blocks: args.kv_blocks.max(32),
        num_swap_blocks: args.kv_blocks.max(64),
        speculative_lookahead: if draft_opt.is_some() {
            args.lookahead_k
        } else {
            0
        },
        adaptive_speculative: true,
        ..SchedulerConfig::default()
    };

    let actor_guard =
        EngineActorHandle::spawn(&model_name, target_model, draft_opt, tokenizer, sched_cfg)?;
    let app = build_router(actor_guard.handle.clone());

    let bind_addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&bind_addr).await?;
    let local_addr = listener.local_addr()?;
    println!(
        "NovaServe listening on http://{} (model={}, kv_blocks={}, spec_k={})",
        local_addr, model_name, args.kv_blocks, args.lookahead_k
    );
    axum::serve(listener, app).await?;
    drop(actor_guard);
    Ok(())
}
