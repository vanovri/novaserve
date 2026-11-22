use thiserror::Error;

#[derive(Debug, Error)]
pub enum KvError {
    #[error("out of physical KV blocks (requested={requested}, free={free})")]
    OutOfKvBlocks { requested: usize, free: usize },

    #[error("out of CPU swap blocks (requested={requested}, free={free})")]
    OutOfSwapBlocks { requested: usize, free: usize },

    #[error("invalid KV configuration or bounds: {0}")]
    InvalidBounds(String),

    #[error("tensor error: {0}")]
    Tensor(#[from] novaserve_tensor::TensorError),
}
