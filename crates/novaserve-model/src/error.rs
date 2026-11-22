use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),

    #[error("invalid batch input descriptor: {0}")]
    InvalidBatch(String),

    #[error("tensor error: {0}")]
    Tensor(#[from] novaserve_tensor::TensorError),

    #[error("KV cache error: {0}")]
    Kv(#[from] novaserve_kv::KvError),

    #[error("frontend error: {0}")]
    Frontend(#[from] novaserve_frontend::FrontendError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
