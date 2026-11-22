//! Secondary Host RAM Swap Pool (`SwapEngine`) for graceful lossless preemption
//! (`swap_out` -> `swap_in`) when primary PagedAttention KV blocks are exhausted.

use crate::block_manager::{BlockAllocator, KvSlab, LogicalBlockTable};
use crate::error::KvError;
use crate::radix_cache::RadixCache;
use novaserve_tensor::AlignedBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwapSlotId(pub u32);

#[derive(Clone, Debug)]
pub struct SwappedHandle {
    pub seq_id: u64,
    pub swap_slots: Vec<SwapSlotId>,
    pub num_tokens: usize,
    pub block_size: usize,
}

pub struct SwapEngine {
    num_swap_blocks: usize,
    num_layers: usize,
    block_plane_bytes: usize,
    bytes_per_swap_block: usize,
    pool: AlignedBuf,
    free_slots: Vec<SwapSlotId>,
}

impl SwapEngine {
    pub fn new(primary_slab: &KvSlab, num_swap_blocks: usize) -> Result<Self, KvError> {
        let cfg = primary_slab.config();
        let block_plane_bytes = primary_slab.block_plane_bytes();
        let bytes_per_swap_block = primary_slab.total_bytes_per_block();
        let total_bytes = num_swap_blocks
            .checked_mul(bytes_per_swap_block)
            .ok_or_else(|| KvError::InvalidBounds("swap pool byte size overflow".into()))?;
        let pool = AlignedBuf::new_zeroed(total_bytes.max(64))?;
        let free_slots: Vec<SwapSlotId> =
            (0..num_swap_blocks as u32).rev().map(SwapSlotId).collect();

        Ok(Self {
            num_swap_blocks,
            num_layers: cfg.num_layers,
            block_plane_bytes,
            bytes_per_swap_block,
            pool,
            free_slots,
        })
    }

    #[inline]
    pub fn total_swap_blocks(&self) -> usize {
        self.num_swap_blocks
    }

    #[inline]
    pub fn free_swap_blocks(&self) -> usize {
        self.free_slots.len()
    }

    #[inline]
    pub fn used_swap_blocks(&self) -> usize {
        self.num_swap_blocks - self.free_slots.len()
    }

    /// Copies all logical blocks of `table` into `SwapEngine`'s secondary buffer and releases
    /// the sequence's primary KV block references cleanly.
    pub fn swap_out(
        &mut self,
        seq_id: u64,
        table: &mut LogicalBlockTable,
        primary_slab: &KvSlab,
        allocator: &mut BlockAllocator,
    ) -> Result<SwappedHandle, KvError> {
        let needed = table.num_blocks();
        if self.free_slots.len() < needed {
            return Err(KvError::OutOfSwapBlocks {
                requested: needed,
                free: self.free_slots.len(),
            });
        }

        let mut swap_slots = Vec::with_capacity(needed);
        let pool_bytes = self.pool.as_bytes_mut();
        let plane_sz = self.block_plane_bytes;

        for &phys_bid in table.blocks() {
            let slot_id = self.free_slots.pop().expect("verified swap capacity");
            let swap_base = (slot_id.0 as usize) * self.bytes_per_swap_block;

            for layer in 0..self.num_layers {
                for kv_kind in 0..2 {
                    let plane_idx = layer * 2 + kv_kind;
                    let dst_off = swap_base + plane_idx * plane_sz;
                    let src_slice = primary_slab.block_plane_slice(layer, kv_kind, phys_bid);
                    pool_bytes[dst_off..dst_off + plane_sz].copy_from_slice(src_slice);
                }
            }
            swap_slots.push(slot_id);
        }

        let num_tokens = table.num_tokens();
        let block_size = table.block_size();
        allocator.release_table(table);

        Ok(SwappedHandle {
            seq_id,
            swap_slots,
            num_tokens,
            block_size,
        })
    }

    /// Atomically restores a swapped sequence into newly allocated primary `BlockId`s.
    /// On insufficient primary capacity, returns `Err((handle, KvError))` with swap state untouched.
    pub fn swap_in(
        &mut self,
        handle: SwappedHandle,
        primary_slab: &mut KvSlab,
        allocator: &mut BlockAllocator,
        radix: Option<&mut RadixCache>,
    ) -> Result<LogicalBlockTable, (SwappedHandle, KvError)> {
        let needed = handle.swap_slots.len();
        let new_primary_blocks = match allocator.allocate_n(needed, radix) {
            Ok(blks) => blks,
            Err(e) => return Err((handle, e)),
        };

        let pool_bytes = self.pool.as_bytes();
        let plane_sz = self.block_plane_bytes;

        for (&slot_id, &phys_bid) in handle.swap_slots.iter().zip(new_primary_blocks.iter()) {
            let swap_base = (slot_id.0 as usize) * self.bytes_per_swap_block;
            for layer in 0..self.num_layers {
                for kv_kind in 0..2 {
                    let plane_idx = layer * 2 + kv_kind;
                    let src_off = swap_base + plane_idx * plane_sz;
                    let dst_slice = primary_slab.block_plane_slice_mut(layer, kv_kind, phys_bid);
                    dst_slice.copy_from_slice(&pool_bytes[src_off..src_off + plane_sz]);
                }
            }
            self.free_slots.push(slot_id);
        }

        Ok(LogicalBlockTable::from_raw_parts(
            new_primary_blocks,
            handle.num_tokens,
            handle.block_size,
        ))
    }

    /// Discards a swapped sequence (e.g., on client abort while swapped out), freeing swap slots.
    pub fn discard_swapped(&mut self, handle: SwappedHandle) {
        for slot in handle.swap_slots {
            self.free_slots.push(slot);
        }
    }
}
