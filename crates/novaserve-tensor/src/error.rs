use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("memory allocation failed (size={size}, align={align})")]
    AllocationFailed { size: usize, align: usize },

    #[error("scratch arena exhausted (requested={requested}, available={available}, capacity={capacity})")]
    ArenaExhausted {
        requested: usize,
        available: usize,
        capacity: usize,
    },

    #[error("shape mismatch in {context}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: &'static str,
    },

    #[error("invalid tensor dimension: {0}")]
    InvalidDimension(String),

    #[error("pointer alignment violation: addr=0x{addr:x} not aligned to {required} bytes")]
    AlignmentError { addr: usize, required: usize },

    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDType(String),

    #[error("corrupt or hostile safetensors payload: {0}")]
    CorruptSafetensors(String),

    #[error("corrupt or hostile GGUF payload: {0}")]
    CorruptGguf(String),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
