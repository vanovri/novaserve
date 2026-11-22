use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchedError {
    #[error("invalid request or scheduler parameter: {0}")]
    InvalidRequest(String),

    #[error("grammar mask left zero valid next tokens")]
    EmptyGrammarSupport,

    #[error("model error: {0}")]
    Model(#[from] novaserve_model::ModelError),

    #[error("KV subsystem error: {0}")]
    Kv(#[from] novaserve_kv::KvError),

    #[error("frontend error: {0}")]
    Frontend(#[from] novaserve_frontend::FrontendError),

    #[error("tensor error: {0}")]
    Tensor(#[from] novaserve_tensor::TensorError),
}
