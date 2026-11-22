//! Physical KV-Cache Slab (`KvSlab`) and PagedAttention Non-Contiguous Block Allocator
//! (`BlockAllocator`, `LogicalBlockTable`, Copy-on-Write `fork_sequence` & `cow_append_slot`).

use crate::error::KvError;
use crate::radix_cache::RadixCache;
use novaserve_tensor::{
    dequantize_row_q8_0, quantize_row_q8_0, AlignedBuf, SplitMix64, BYTES_Q8_0, QK8_0,
};
use serde::{Deserialize, Serialize};

pub const DEFAULT_BLOCK_SIZE: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvDType {
    F32,
    Q8_0,
}

impl KvDType {
    #[inline]
    pub fn slot_bytes(self, head_dim: usize) -> usize {
        match self {
            Self::F32 => head_dim * 4,
            Self::Q8_0 => head_dim.div_ceil(QK8_0) * BYTES_Q8_0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSlabConfig {
    pub num_layers: usize,
    pub num_physical_blocks: usize,
    pub num_kv_heads: usize,
    pub block_size: usize,
    pub head_dim: usize,
    pub kv_dtype: KvDType,
}

/// Pre-allocated contiguous 64-byte aligned physical KV slab shaped:
/// `[num_layers, 2 (0=K, 1=V), num_physical_blocks, num_kv_heads, block_size, head_dim]`.
pub struct KvSlab {
    cfg: KvSlabConfig,
    buf: AlignedBuf,
    slot_bytes: usize,
    block_plane_bytes: usize,
    kv_plane_bytes: usize,
    layer_bytes: usize,
}

impl KvSlab {
    pub fn new(cfg: KvSlabConfig) -> Result<Self, KvError> {
        if cfg.num_layers == 0
            || cfg.num_physical_blocks == 0
            || cfg.num_kv_heads == 0
            || cfg.block_size == 0
            || cfg.head_dim == 0
        {
            return Err(KvError::InvalidBounds(
                "all KvSlabConfig dimensions must be > 0".into(),
            ));
        }

        let slot_bytes = cfg.kv_dtype.slot_bytes(cfg.head_dim);
        let block_plane_bytes = cfg
            .num_kv_heads
            .checked_mul(cfg.block_size)
            .and_then(|v| v.checked_mul(slot_bytes))
            .ok_or_else(|| KvError::InvalidBounds("block_plane_bytes overflow".into()))?;
        let kv_plane_bytes = cfg
            .num_physical_blocks
            .checked_mul(block_plane_bytes)
            .ok_or_else(|| KvError::InvalidBounds("kv_plane_bytes overflow".into()))?;
        let layer_bytes = kv_plane_bytes
            .checked_mul(2)
            .ok_or_else(|| KvError::InvalidBounds("layer_bytes overflow".into()))?;
        let total_bytes = cfg
            .num_layers
            .checked_mul(layer_bytes)
            .ok_or_else(|| KvError::InvalidBounds("total KvSlab bytes overflow".into()))?;

        let buf = AlignedBuf::new_zeroed(total_bytes)?;
        Ok(Self {
            cfg,
            buf,
            slot_bytes,
            block_plane_bytes,
            kv_plane_bytes,
            layer_bytes,
        })
    }

    #[inline]
    pub fn config(&self) -> &KvSlabConfig {
        &self.cfg
    }

    #[inline]
    pub fn block_plane_bytes(&self) -> usize {
        self.block_plane_bytes
    }

    #[inline]
    pub fn total_bytes_per_block(&self) -> usize {
        self.cfg.num_layers * 2 * self.block_plane_bytes
    }

    #[inline]
    fn slot_byte_offset(
        &self,
        layer: usize,
        kv_kind: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
    ) -> usize {
        debug_assert!(layer < self.cfg.num_layers);
        debug_assert!(kv_kind < 2);
        debug_assert!((block_id.0 as usize) < self.cfg.num_physical_blocks);
        debug_assert!(kv_head < self.cfg.num_kv_heads);
        debug_assert!(slot_in_block < self.cfg.block_size);

        layer * self.layer_bytes
            + kv_kind * self.kv_plane_bytes
            + (block_id.0 as usize) * self.block_plane_bytes
            + (kv_head * self.cfg.block_size + slot_in_block) * self.slot_bytes
    }

    #[inline]
    pub fn block_plane_slice(&self, layer: usize, kv_kind: usize, block_id: BlockId) -> &[u8] {
        let start = layer * self.layer_bytes
            + kv_kind * self.kv_plane_bytes
            + (block_id.0 as usize) * self.block_plane_bytes;
        &self.buf.as_bytes()[start..start + self.block_plane_bytes]
    }

    #[inline]
    pub fn block_plane_slice_mut(
        &mut self,
        layer: usize,
        kv_kind: usize,
        block_id: BlockId,
    ) -> &mut [u8] {
        let start = layer * self.layer_bytes
            + kv_kind * self.kv_plane_bytes
            + (block_id.0 as usize) * self.block_plane_bytes;
        let end = start + self.block_plane_bytes;
        &mut self.buf.as_bytes_mut()[start..end]
    }

    /// Writes `k_vec` and `v_vec` (`[head_dim]`) into `(layer, block_id, kv_head, slot_in_block)`.
    pub fn write_kv_slot(
        &mut self,
        layer: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
        k_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<(), KvError> {
        let hd = self.cfg.head_dim;
        if k_vec.len() != hd || v_vec.len() != hd {
            return Err(KvError::InvalidBounds(format!(
                "write_kv_slot expected head_dim={hd}, got K={}, V={}",
                k_vec.len(),
                v_vec.len()
            )));
        }
        let k_off = self.slot_byte_offset(layer, 0, block_id, kv_head, slot_in_block);
        let v_off = self.slot_byte_offset(layer, 1, block_id, kv_head, slot_in_block);
        let sb = self.slot_bytes;

        match self.cfg.kv_dtype {
            KvDType::F32 => {
                let f32_slab = self.buf.as_f32_slice_mut()?;
                let k_f = k_off / 4;
                let v_f = v_off / 4;
                f32_slab[k_f..k_f + hd].copy_from_slice(k_vec);
                f32_slab[v_f..v_f + hd].copy_from_slice(v_vec);
            }
            KvDType::Q8_0 => {
                let raw = self.buf.as_bytes_mut();
                quantize_row_q8_0(k_vec, &mut raw[k_off..k_off + sb])?;
                quantize_row_q8_0(v_vec, &mut raw[v_off..v_off + sb])?;
            }
        }
        Ok(())
    }

    /// Writes all `num_kv_heads` (`[num_kv_heads * head_dim]`) for a single token at `(layer, block_id, slot_in_block)`.
    pub fn write_token_kv(
        &mut self,
        layer: usize,
        block_id: BlockId,
        slot_in_block: usize,
        k_heads: &[f32],
        v_heads: &[f32],
    ) -> Result<(), KvError> {
        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_kv_heads;
        if k_heads.len() != nh * hd || v_heads.len() != nh * hd {
            return Err(KvError::InvalidBounds(
                "write_token_kv head slice mismatch".into(),
            ));
        }
        for h in 0..nh {
            self.write_kv_slot(
                layer,
                block_id,
                h,
                slot_in_block,
                &k_heads[h * hd..(h + 1) * hd],
                &v_heads[h * hd..(h + 1) * hd],
            )?;
        }
        Ok(())
    }

    /// Reads `K` slot into `out_k` (`[head_dim]`).
    pub fn read_k_slot(
        &self,
        layer: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
        out_k: &mut [f32],
    ) -> Result<(), KvError> {
        self.read_plane_slot(layer, 0, block_id, kv_head, slot_in_block, out_k)
    }

    /// Reads `V` slot into `out_v` (`[head_dim]`).
    pub fn read_v_slot(
        &self,
        layer: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
        out_v: &mut [f32],
    ) -> Result<(), KvError> {
        self.read_plane_slot(layer, 1, block_id, kv_head, slot_in_block, out_v)
    }

    fn read_plane_slot(
        &self,
        layer: usize,
        kv_kind: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
        out: &mut [f32],
    ) -> Result<(), KvError> {
        let hd = self.cfg.head_dim;
        if out.len() != hd {
            return Err(KvError::InvalidBounds(
                "read_plane_slot head_dim mismatch".into(),
            ));
        }
        let off = self.slot_byte_offset(layer, kv_kind, block_id, kv_head, slot_in_block);
        match self.cfg.kv_dtype {
            KvDType::F32 => {
                let f32_slab = self.buf.as_f32_slice()?;
                let f_off = off / 4;
                out.copy_from_slice(&f32_slab[f_off..f_off + hd]);
            }
            KvDType::Q8_0 => {
                let raw = &self.buf.as_bytes()[off..off + self.slot_bytes];
                dequantize_row_q8_0(raw, out)?;
            }
        }
        Ok(())
    }

    /// Zero-copy `F32` slice for `K` (`kv_kind=0`) or `V` (`kv_kind=1`) slot when `kv_dtype == F32`.
    #[inline]
    pub fn slot_f32(
        &self,
        layer: usize,
        kv_kind: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
    ) -> Option<&[f32]> {
        if self.cfg.kv_dtype != KvDType::F32 {
            return None;
        }
        let f32_slab = self.buf.as_f32_slice().ok()?;
        let off = self.slot_byte_offset(layer, kv_kind, block_id, kv_head, slot_in_block) / 4;
        Some(&f32_slab[off..off + self.cfg.head_dim])
    }

    /// Zero-copy `Q8_0` byte slice for `K` or `V` slot when `kv_dtype == Q8_0`.
    #[inline]
    pub fn slot_q8_bytes(
        &self,
        layer: usize,
        kv_kind: usize,
        block_id: BlockId,
        kv_head: usize,
        slot_in_block: usize,
    ) -> Option<&[u8]> {
        if self.cfg.kv_dtype != KvDType::Q8_0 {
            return None;
        }
        let off = self.slot_byte_offset(layer, kv_kind, block_id, kv_head, slot_in_block);
        Some(&self.buf.as_bytes()[off..off + self.slot_bytes])
    }

    /// Copies the first `filled_slots` (`<= block_size`) of `src_block` into `dst_block`
    /// across all layers, K/V planes, and KV heads (used by Copy-on-Write tail splits).
    pub fn copy_block_slots(
        &mut self,
        src_block: BlockId,
        dst_block: BlockId,
        filled_slots: usize,
    ) {
        if src_block == dst_block || filled_slots == 0 {
            return;
        }
        let slots = filled_slots.min(self.cfg.block_size);
        let copy_len = slots * self.slot_bytes;
        let num_layers = self.cfg.num_layers;
        let num_kv_heads = self.cfg.num_kv_heads;

        for layer in 0..num_layers {
            for kv_kind in 0..2 {
                for h in 0..num_kv_heads {
                    let src_off = self.slot_byte_offset(layer, kv_kind, src_block, h, 0);
                    let dst_off = self.slot_byte_offset(layer, kv_kind, dst_block, h, 0);
                    self.buf
                        .as_bytes_mut()
                        .copy_within(src_off..src_off + copy_len, dst_off);
                }
            }
        }
    }
}

/// Per-sequence logical-to-physical KV block table (`LogicalBlockTable`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalBlockTable {
    blocks: Vec<BlockId>,
    num_tokens: usize,
    block_size: usize,
}

impl LogicalBlockTable {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0);
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    pub fn from_cached_prefix(
        cached_blocks: Vec<BlockId>,
        matched_tokens: usize,
        block_size: usize,
    ) -> Self {
        assert_eq!(cached_blocks.len() * block_size, matched_tokens);
        Self {
            blocks: cached_blocks,
            num_tokens: matched_tokens,
            block_size,
        }
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Internal fragmentation waste (unused token slots on the final partially filled block),
    /// strictly bounded to `<= block_size - 1`.
    #[inline]
    pub fn internal_waste_slots(&self) -> usize {
        self.blocks.len() * self.block_size - self.num_tokens
    }

    /// Maps logical token position `pos < num_tokens` to `(BlockId, slot_in_block)`.
    #[inline]
    pub fn resolve_slot(&self, pos: usize) -> Option<(BlockId, usize)> {
        if pos >= self.num_tokens {
            return None;
        }
        let blk_idx = pos / self.block_size;
        let slot = pos % self.block_size;
        Some((self.blocks[blk_idx], slot))
    }

    /// Returns all full `block_size`-aligned blocks (`0 .. num_tokens / block_size`).
    #[inline]
    pub fn full_blocks(&self) -> &[BlockId] {
        let full = self.num_tokens / self.block_size;
        &self.blocks[..full]
    }

    pub(crate) fn clear_after_release(&mut self) {
        self.blocks.clear();
        self.num_tokens = 0;
    }

    pub(crate) fn from_raw_parts(
        blocks: Vec<BlockId>,
        num_tokens: usize,
        block_size: usize,
    ) -> Self {
        Self {
            blocks,
            num_tokens,
            block_size,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysicalBlockSnapshot {
    pub block_id: u32,
    pub ref_count: u32,
    pub is_radix_cached: bool,
    pub content_hash: Option<u64>,
}

/// PagedAttention Physical Block Allocator with CoW reference counting and zero external fragmentation.
pub struct BlockAllocator {
    total_blocks: usize,
    block_size: usize,
    free_list: Vec<BlockId>,
    ref_counts: Vec<u32>,
    radix_cached: Vec<bool>,
    content_hashes: Vec<Option<u64>>,
}

impl BlockAllocator {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        assert!(total_blocks > 0 && block_size > 0);
        // Push in reverse so `pop()` allocates `BlockId(0), BlockId(1), ...` deterministically by default
        let free_list: Vec<BlockId> = (0..total_blocks as u32).rev().map(BlockId).collect();
        Self {
            total_blocks,
            block_size,
            free_list,
            ref_counts: vec![0; total_blocks],
            radix_cached: vec![false; total_blocks],
            content_hashes: vec![None; total_blocks],
        }
    }

    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    #[inline]
    pub fn allocated_blocks(&self) -> usize {
        self.total_blocks - self.free_list.len()
    }

    #[inline]
    pub fn ref_count(&self, block_id: BlockId) -> u32 {
        self.ref_counts[block_id.0 as usize]
    }

    #[inline]
    pub fn is_radix_cached(&self, block_id: BlockId) -> bool {
        self.radix_cached[block_id.0 as usize]
    }

    #[inline]
    pub fn set_radix_cached(&mut self, block_id: BlockId, cached: bool) {
        self.radix_cached[block_id.0 as usize] = cached;
    }

    #[inline]
    pub fn set_content_hash(&mut self, block_id: BlockId, hash: Option<u64>) {
        self.content_hashes[block_id.0 as usize] = hash;
    }

    #[inline]
    pub fn content_hash(&self, block_id: BlockId) -> Option<u64> {
        self.content_hashes[block_id.0 as usize]
    }

    /// Scrambles the physical free-list order using `rng` (used in M04/M07 invariance proofs).
    pub fn shuffle_free_list(&mut self, rng: &mut SplitMix64) {
        let len = self.free_list.len();
        for i in (1..len).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            self.free_list.swap(i, j);
        }
    }

    /// Allocates 1 physical block directly from `free_list` (evicting from `radix` if provided and needed).
    pub fn allocate_block(&mut self, radix: Option<&mut RadixCache>) -> Result<BlockId, KvError> {
        if self.free_list.is_empty() {
            if let Some(rc) = radix {
                rc.evict_lru(1, self);
            }
        }
        let Some(bid) = self.free_list.pop() else {
            return Err(KvError::OutOfKvBlocks {
                requested: 1,
                free: 0,
            });
        };
        let idx = bid.0 as usize;
        debug_assert_eq!(self.ref_counts[idx], 0);
        debug_assert!(!self.radix_cached[idx]);
        self.ref_counts[idx] = 1;
        self.content_hashes[idx] = None;
        Ok(bid)
    }

    /// Atomically allocates `count` physical blocks (evicting unpinned Radix leaves if needed).
    pub fn allocate_n(
        &mut self,
        count: usize,
        radix: Option<&mut RadixCache>,
    ) -> Result<Vec<BlockId>, KvError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.free_list.len() < count {
            let deficit = count - self.free_list.len();
            if let Some(rc) = radix {
                rc.evict_lru(deficit, self);
            }
        }
        if self.free_list.len() < count {
            return Err(KvError::OutOfKvBlocks {
                requested: count,
                free: self.free_list.len(),
            });
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let bid = self
                .free_list
                .pop()
                .expect("verified free_list.len() >= count");
            let idx = bid.0 as usize;
            debug_assert_eq!(self.ref_counts[idx], 0);
            self.ref_counts[idx] = 1;
            self.content_hashes[idx] = None;
            out.push(bid);
        }
        Ok(out)
    }

    #[inline]
    pub fn inc_ref(&mut self, block_id: BlockId) {
        let idx = block_id.0 as usize;
        assert!(
            self.ref_counts[idx] >= 1,
            "cannot inc_ref on free block {block_id:?}"
        );
        self.ref_counts[idx] += 1;
    }

    #[inline]
    pub fn free_block(&mut self, block_id: BlockId) {
        let idx = block_id.0 as usize;
        let rc = self.ref_counts[idx];
        assert!(rc >= 1, "double free detected on {block_id:?}");
        if rc == 1 {
            assert!(
                !self.radix_cached[idx],
                "block {block_id:?} dropped to ref_count 0 while still marked radix_cached"
            );
            self.ref_counts[idx] = 0;
            self.content_hashes[idx] = None;
            self.free_list.push(block_id);
        } else {
            self.ref_counts[idx] = rc - 1;
        }
    }

    /// Releases all block references held by `table`.
    pub fn release_table(&mut self, table: &mut LogicalBlockTable) {
        for &bid in &table.blocks {
            self.free_block(bid);
        }
        table.clear_after_release();
    }

    /// Zero-copy Copy-on-Write sequence fork ($O(\text{blocks})$ refcount increments, 0 bytes copied).
    pub fn fork_sequence(&mut self, src: &LogicalBlockTable) -> LogicalBlockTable {
        for &bid in &src.blocks {
            self.inc_ref(bid);
        }
        src.clone()
    }

    /// Appends 1 token slot to `table`, transparently performing a Copy-on-Write block split
    /// if the tail partially filled block is shared (`ref_count > 1`).
    /// Returns `(physical_block_id, slot_in_block)` where the caller writes the new K/V vectors.
    pub fn cow_append_slot(
        &mut self,
        slab: &mut KvSlab,
        radix: Option<&mut RadixCache>,
        table: &mut LogicalBlockTable,
    ) -> Result<(BlockId, usize), KvError> {
        let pos = table.num_tokens;
        let slot = pos % self.block_size;

        if slot == 0 {
            let new_bid = self.allocate_block(radix)?;
            table.blocks.push(new_bid);
            table.num_tokens += 1;
            return Ok((new_bid, 0));
        }

        let tail_bid = *table
            .blocks
            .last()
            .expect("non-zero slot implies non-empty blocks");
        if self.ref_counts[tail_bid.0 as usize] == 1 {
            // Exclusive ownership: append in-place with 0 copies!
            table.num_tokens += 1;
            return Ok((tail_bid, slot));
        }

        // Shared partial block (`ref_count > 1`): CoW allocate 1 fresh block and copy `[0..slot)`!
        let fresh_bid = self.allocate_block(radix)?;
        slab.copy_block_slots(tail_bid, fresh_bid, slot);
        self.free_block(tail_bid);
        *table.blocks.last_mut().unwrap() = fresh_bid;
        table.num_tokens += 1;
        Ok((fresh_bid, slot))
    }

    /// Truncates `table` back to `new_num_tokens` (used by Speculative Decoding rejection rollback),
    /// freeing any newly crossed physical blocks cleanly!
    pub fn truncate_table(&mut self, table: &mut LogicalBlockTable, new_num_tokens: usize) {
        if new_num_tokens >= table.num_tokens {
            return;
        }
        let needed_blocks = if new_num_tokens == 0 {
            0
        } else {
            new_num_tokens.div_ceil(self.block_size)
        };
        while table.blocks.len() > needed_blocks {
            let popped = table.blocks.pop().unwrap();
            self.free_block(popped);
        }
        table.num_tokens = new_num_tokens;
    }

    /// Verifies global conservation and non-corruption invariants across all physical blocks.
    pub fn verify_invariants(&self) {
        let mut in_free = vec![false; self.total_blocks];
        for &bid in &self.free_list {
            let idx = bid.0 as usize;
            assert!(idx < self.total_blocks, "free_list BlockId out of bounds");
            assert!(!in_free[idx], "duplicate BlockId {bid:?} in free_list");
            assert_eq!(
                self.ref_counts[idx], 0,
                "free block {bid:?} has non-zero ref_count {}",
                self.ref_counts[idx]
            );
            assert!(
                !self.radix_cached[idx],
                "free block {bid:?} still marked radix_cached"
            );
            in_free[idx] = true;
        }

        let mut active_count = 0usize;
        for (idx, &is_free) in in_free.iter().enumerate() {
            if is_free {
                assert_eq!(self.ref_counts[idx], 0);
            } else {
                assert!(
                    self.ref_counts[idx] >= 1,
                    "unlisted block {idx} has ref_count == 0 (leaked!)"
                );
                active_count += 1;
            }
        }
        assert_eq!(
            active_count + self.free_list.len(),
            self.total_blocks,
            "block conservation law violated"
        );
    }

    pub fn snapshot_blocks(&self) -> Vec<PhysicalBlockSnapshot> {
        (0..self.total_blocks)
            .map(|i| PhysicalBlockSnapshot {
                block_id: i as u32,
                ref_count: self.ref_counts[i],
                is_radix_cached: self.radix_cached[i],
                content_hash: self.content_hashes[i],
            })
            .collect()
    }
}
