#![deny(unsafe_code)]

//! `novaserve-kv`: PagedAttention Non-Contiguous Block Allocator (Copy-on-Write refcounting),
//! SGLang-style Compressed LRU Radix Prefix Tree (`RadixCache`), and Preemption `SwapEngine`.

pub mod block_manager;
pub mod error;
pub mod radix_cache;
pub mod swap_engine;

pub use block_manager::{
    BlockAllocator, BlockId, KvDType, KvSlab, KvSlabConfig, LogicalBlockTable,
    PhysicalBlockSnapshot, DEFAULT_BLOCK_SIZE,
};
pub use error::KvError;
pub use radix_cache::{PrefixMatch, RadixCache, RadixNodeSnapshot};
pub use swap_engine::{SwapEngine, SwapSlotId, SwappedHandle};
