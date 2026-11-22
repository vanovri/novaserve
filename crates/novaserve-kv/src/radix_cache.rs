//! SGLang-Style Compressed LRU Radix Prefix Tree (`RadixCache`) keyed by `block_size`-token
//! chunks, with mid-edge splitting, longest-prefix matching (`match_prefix`), and
//! demand-driven strict-LRU leaf eviction (`evict_lru`).

use crate::block_manager::{BlockAllocator, BlockId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
struct RadixNode {
    id: usize,
    parent: Option<usize>,
    /// Token sequence along incoming edge (`len == blocks.len() * block_size`; empty for root `0`).
    edge_tokens: Vec<u32>,
    /// Physical `BlockId`s cached along incoming edge (`ref_count >= 1` held by `RadixCache`).
    blocks: Vec<BlockId>,
    /// Keyed by the first `block_size` tokens (`edge_tokens[0..block_size]`) of each child edge,
    /// allowing prompts that diverge inside block 0 to coexist without collision while guaranteeing
    /// `common_blocks >= 1` on every child hit.
    children: HashMap<Vec<u32>, usize>,
    last_access_tick: u64,
}

#[derive(Clone, Debug)]
pub struct PrefixMatch<'a> {
    pub cached_block_ids: Vec<BlockId>,
    pub matched_tokens: usize,
    pub remaining_uncached_suffix: &'a [u32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadixNodeSnapshot {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub token_preview: Vec<u32>,
    pub total_edge_tokens: usize,
    pub block_ids: Vec<u32>,
    pub min_ref_count: u32,
    pub last_access_tick: u64,
}

pub struct RadixCache {
    block_size: usize,
    nodes: BTreeMap<usize, RadixNode>,
    next_node_id: usize,
    clock: u64,
    cached_blocks_count: usize,
    total_queries: u64,
    total_prompt_blocks_queried: u64,
    total_hit_blocks: u64,
    total_evicted_blocks: u64,
}

impl RadixCache {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0);
        let mut nodes = BTreeMap::new();
        nodes.insert(
            0,
            RadixNode {
                id: 0,
                parent: None,
                edge_tokens: Vec::new(),
                blocks: Vec::new(),
                children: HashMap::new(),
                last_access_tick: 0,
            },
        );
        Self {
            block_size,
            nodes,
            next_node_id: 1,
            clock: 1,
            cached_blocks_count: 0,
            total_queries: 0,
            total_prompt_blocks_queried: 0,
            total_hit_blocks: 0,
            total_evicted_blocks: 0,
        }
    }

    #[inline]
    fn tick(&mut self) -> u64 {
        let t = self.clock;
        self.clock += 1;
        t
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn cached_blocks_count(&self) -> usize {
        self.cached_blocks_count
    }

    #[inline]
    pub fn total_evicted_blocks(&self) -> u64 {
        self.total_evicted_blocks
    }

    #[inline]
    pub fn total_hit_blocks(&self) -> u64 {
        self.total_hit_blocks
    }

    #[inline]
    pub fn total_prompt_blocks_queried(&self) -> u64 {
        self.total_prompt_blocks_queried
    }

    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.total_prompt_blocks_queried == 0 {
            0.0
        } else {
            (self.total_hit_blocks as f64) / (self.total_prompt_blocks_queried as f64)
        }
    }

    #[inline]
    pub fn total_queries(&self) -> u64 {
        self.total_queries
    }

    /// Splits `node_id` after `split_blocks` (`1 <= split_blocks < node.blocks.len()`),
    /// keeping `[0 .. split_blocks]` in `node_id` and moving `[split_blocks ..]` + children
    /// into a new child node. Net refcount change = 0!
    fn split_node(&mut self, node_id: usize, split_blocks: usize) {
        let split_toks = split_blocks * self.block_size;
        let new_child_id = self.next_node_id;
        self.next_node_id += 1;

        let parent_node = self.nodes.get_mut(&node_id).expect("node exists");
        debug_assert!(split_blocks > 0 && split_blocks < parent_node.blocks.len());

        let tail_tokens = parent_node.edge_tokens.split_off(split_toks);
        let tail_blocks = parent_node.blocks.split_off(split_blocks);
        let moved_children = std::mem::take(&mut parent_node.children);
        let tick = parent_node.last_access_tick;
        let tail_key = tail_tokens[..self.block_size].to_vec();

        parent_node.children.insert(tail_key, new_child_id);

        let grand_ids: Vec<usize> = moved_children.values().copied().collect();
        for gid in grand_ids {
            if let Some(gnode) = self.nodes.get_mut(&gid) {
                gnode.parent = Some(new_child_id);
            }
        }

        self.nodes.insert(
            new_child_id,
            RadixNode {
                id: new_child_id,
                parent: Some(node_id),
                edge_tokens: tail_tokens,
                blocks: tail_blocks,
                children: moved_children,
                last_access_tick: tick,
            },
        );
    }

    fn touch_to_root(&mut self, mut node_id: usize, tick: u64) {
        while let Some(node) = self.nodes.get_mut(&node_id) {
            node.last_access_tick = tick;
            if let Some(p) = node.parent {
                node_id = p;
            } else {
                break;
            }
        }
    }

    /// Longest-prefix match aligned to `block_size`, automatically splitting compressed edges
    /// mid-way when needed, incrementing `ref_count` on matched `BlockId`s, and ensuring
    /// `remaining_uncached_suffix` retains at least 1 token for next-token logit computation.
    pub fn match_prefix<'a>(
        &mut self,
        prompt_tokens: &'a [u32],
        allocator: &mut BlockAllocator,
    ) -> PrefixMatch<'a> {
        self.total_queries += 1;
        let max_cachable_blocks = prompt_tokens.len().saturating_sub(1) / self.block_size;
        self.total_prompt_blocks_queried += max_cachable_blocks as u64;

        if max_cachable_blocks == 0 {
            return PrefixMatch {
                cached_block_ids: Vec::new(),
                matched_tokens: 0,
                remaining_uncached_suffix: prompt_tokens,
            };
        }

        let max_toks = max_cachable_blocks * self.block_size;
        let query = &prompt_tokens[..max_toks];
        let mut cursor_blk = 0usize;
        let mut cur_node = 0usize;
        let mut matched_blocks = Vec::new();

        while cursor_blk < max_cachable_blocks {
            let q_off = cursor_blk * self.block_size;
            let first_blk_slice = &query[q_off..q_off + self.block_size];
            let Some(&child_id) = self.nodes[&cur_node].children.get(first_blk_slice) else {
                break;
            };

            let edge_blocks = self.nodes[&child_id].blocks.len();
            let rem_blocks = max_cachable_blocks - cursor_blk;
            let max_cmp = edge_blocks.min(rem_blocks);
            let mut common_blocks = 1usize; // block 0 matched via map key!
            for b in 1..max_cmp {
                let es = b * self.block_size;
                let ee = es + self.block_size;
                let qs = (cursor_blk + b) * self.block_size;
                let qe = qs + self.block_size;
                if self.nodes[&child_id].edge_tokens[es..ee] == query[qs..qe] {
                    common_blocks += 1;
                } else {
                    break;
                }
            }

            if common_blocks < edge_blocks {
                self.split_node(child_id, common_blocks);
            }

            for &bid in &self.nodes[&child_id].blocks {
                allocator.inc_ref(bid);
                matched_blocks.push(bid);
            }
            cursor_blk += common_blocks;
            cur_node = child_id;
        }

        if cur_node != 0 {
            let now = self.tick();
            self.touch_to_root(cur_node, now);
        }

        let matched_tokens = cursor_blk * self.block_size;
        self.total_hit_blocks += matched_blocks.len() as u64;
        PrefixMatch {
            cached_block_ids: matched_blocks,
            matched_tokens,
            remaining_uncached_suffix: &prompt_tokens[matched_tokens..],
        }
    }

    /// Inserts all full `block_size`-aligned prefix blocks of a sequence into `RadixCache`.
    pub fn insert_prefix(
        &mut self,
        tokens: &[u32],
        blocks: &[BlockId],
        allocator: &mut BlockAllocator,
    ) -> usize {
        let full_blocks = (tokens.len() / self.block_size).min(blocks.len());
        if full_blocks == 0 {
            return 0;
        }

        let aligned_tokens = &tokens[..full_blocks * self.block_size];
        let aligned_blocks = &blocks[..full_blocks];

        let now = self.tick();
        let mut cursor_blk = 0usize;
        let mut cur_node = 0usize;
        let mut newly_inserted = 0usize;

        while cursor_blk < full_blocks {
            let tok_off = cursor_blk * self.block_size;
            let first_blk_slice = &aligned_tokens[tok_off..tok_off + self.block_size];

            if let Some(&child_id) = self.nodes[&cur_node].children.get(first_blk_slice) {
                let edge_blocks = self.nodes[&child_id].blocks.len();
                let rem_blocks = full_blocks - cursor_blk;
                let max_cmp = edge_blocks.min(rem_blocks);
                let mut common_blocks = 1usize;
                for b in 1..max_cmp {
                    let es = b * self.block_size;
                    let ee = es + self.block_size;
                    let qs = (cursor_blk + b) * self.block_size;
                    let qe = qs + self.block_size;
                    if self.nodes[&child_id].edge_tokens[es..ee] == aligned_tokens[qs..qe] {
                        common_blocks += 1;
                    } else {
                        break;
                    }
                }

                if common_blocks < edge_blocks {
                    self.split_node(child_id, common_blocks);
                }
                cursor_blk += common_blocks;
                cur_node = child_id;
            } else {
                let new_id = self.next_node_id;
                self.next_node_id += 1;
                let new_toks = aligned_tokens[tok_off..].to_vec();
                let new_blks = aligned_blocks[cursor_blk..].to_vec();
                let key = first_blk_slice.to_vec();

                for &bid in &new_blks {
                    allocator.inc_ref(bid);
                    allocator.set_radix_cached(bid, true);
                }
                newly_inserted += new_blks.len();
                self.cached_blocks_count += new_blks.len();

                self.nodes
                    .get_mut(&cur_node)
                    .unwrap()
                    .children
                    .insert(key, new_id);
                self.nodes.insert(
                    new_id,
                    RadixNode {
                        id: new_id,
                        parent: Some(cur_node),
                        edge_tokens: new_toks,
                        blocks: new_blks,
                        children: HashMap::new(),
                        last_access_tick: now,
                    },
                );
                cur_node = new_id;
                break;
            }
        }

        self.touch_to_root(cur_node, now);
        newly_inserted
    }

    /// Evicts unpinned leaf nodes (`active_readers == 0`, i.e., `allocator.ref_count(b) == 1`
    /// for all blocks on the leaf edge) in strict LRU order until at least `needed_blocks`
    /// physical blocks have been reclaimed onto `allocator.free_list` (or no evictable leaves remain).
    pub fn evict_lru(&mut self, needed_blocks: usize, allocator: &mut BlockAllocator) -> usize {
        if needed_blocks == 0 || self.cached_blocks_count == 0 {
            return 0;
        }

        let mut reclaimed = 0usize;
        while reclaimed < needed_blocks {
            // Find unpinned leaf (`id != 0`, `children.is_empty()`, all blocks `ref_count == 1`)
            // with minimum `(last_access_tick, id)`
            let victim = self
                .nodes
                .values()
                .filter(|n| {
                    n.id != 0
                        && n.children.is_empty()
                        && n.blocks.iter().all(|&b| allocator.ref_count(b) == 1)
                })
                .min_by_key(|n| (n.last_access_tick, n.id))
                .map(|n| n.id);

            let Some(vid) = victim else {
                break;
            };

            let removed = self.nodes.remove(&vid).expect("victim exists");
            if let Some(pid) = removed.parent {
                let key = &removed.edge_tokens[..self.block_size];
                if let Some(pnode) = self.nodes.get_mut(&pid) {
                    pnode.children.remove(key);
                }
            }

            let blk_count = removed.blocks.len();
            for bid in removed.blocks {
                allocator.set_radix_cached(bid, false);
                allocator.free_block(bid);
            }
            self.cached_blocks_count -= blk_count;
            self.total_evicted_blocks += blk_count as u64;
            reclaimed += blk_count;
        }
        reclaimed
    }

    /// Evicts all unpinned cached blocks from `RadixCache` (releasing `RadixCache` references
    /// on all nodes whose blocks are not actively held, or dropping `RadixCache`'s reference
    /// across the entire tree so `100%` of blocks free once sequences finish).
    pub fn clear(&mut self, allocator: &mut BlockAllocator) {
        let all_ids: Vec<usize> = self.nodes.keys().copied().filter(|&id| id != 0).collect();
        for nid in all_ids {
            if let Some(node) = self.nodes.remove(&nid) {
                for bid in node.blocks {
                    allocator.set_radix_cached(bid, false);
                    allocator.free_block(bid);
                }
            }
        }
        if let Some(root) = self.nodes.get_mut(&0) {
            root.children.clear();
        }
        self.cached_blocks_count = 0;
    }

    /// Exports tree hierarchy snapshot for TUI & WASM Studio visualization.
    pub fn snapshot_tree(&self, allocator: &BlockAllocator) -> Vec<RadixNodeSnapshot> {
        self.nodes
            .values()
            .filter(|n| n.id != 0)
            .map(|n| {
                let min_rc = n
                    .blocks
                    .iter()
                    .map(|&b| allocator.ref_count(b))
                    .min()
                    .unwrap_or(0);
                RadixNodeSnapshot {
                    node_id: n.id,
                    parent_id: n.parent,
                    token_preview: n.edge_tokens.iter().take(8).copied().collect(),
                    total_edge_tokens: n.edge_tokens.len(),
                    block_ids: n.blocks.iter().map(|b| b.0).collect(),
                    min_ref_count: min_rc,
                    last_access_tick: n.last_access_tick,
                }
            })
            .collect()
    }
}
