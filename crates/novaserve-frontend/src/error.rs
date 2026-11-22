use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrontendError {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("invalid regular expression syntax: {0}")]
    RegexSyntax(String),

    #[error("JSON schema compilation error: {0}")]
    SchemaCompile(String),

    #[error("invalid DFA state id: {0}")]
    InvalidState(u32),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("tensor/GGUF error: {0}")]
    Tensor(#[from] novaserve_tensor::TensorError),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}
