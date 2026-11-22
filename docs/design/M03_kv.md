# Module M03 Design Specification — `crates/novaserve-kv`

## 1. Architectural Scope
`novaserve-kv` implements the memory management core of **NovaServe** (`#![deny(unsafe_code)]`):
1. **Contiguous 64B-Aligned Physical KV Slab (`KvSlab`)** shaped `[num_layers, 2 (K, V), num_physical_blocks, num_kv_heads, block_size (16), head_dim]` supporting both `FP32` and `Q8_0` quantized KV-cache storage.
2. **PagedAttention Non-Contiguous Block Allocator (`BlockAllocator` & `LogicalBlockTable`)** achieving **0% external fragmentation**, bounded internal waste ($\le 15$ tokens per sequence), $O(\text{blocks})$ zero-copy Copy-on-Write (`fork_sequence`), and transparent partial-block `cow_append_slot`.
3. **SGLang-Style Compressed LRU Radix Prefix Tree (`RadixCache`)** with automatic mid-edge splitting, longest-prefix matching (`match_prefix`), and demand-driven bottom-up LRU leaf eviction (`evict_lru`).
4. **Atomic Swap-Out / Swap-In Preemption Pool (`SwapEngine`)** reclaiming primary KV blocks under memory pressure without losing sequence K/V state.

---

## 2. Refcount & Eviction Invariants

### 2.1 Conservation of Physical Blocks
Every physical `BlockId(0 .. total_physical_blocks)` is in exactly one state:
- **Free (`ref_count == 0`):** Present once on `BlockAllocator::free_list`.
- **Allocated (`ref_count >= 1`):** Referenced by $\ge 1$ active `LogicalBlockTable` and/or `RadixCache` node.

$$\text{allocated\_blocks} + \text{free\_blocks} \equiv \text{total\_physical\_blocks}$$

### 2.2 Copy-on-Write (`CoW`) Transitions
- **`fork_sequence(src)`:** Clones `src.blocks` and increments `ref_count[b] += 1` for each block (`0` bytes copied).
- **`cow_append_slot(table)`:**
  - `num_tokens % 16 == 0`: Allocates 1 new physical block (`ref_count = 1`).
  - `num_tokens % 16 != 0` and `ref_count[tail] == 1`: Mutates `tail` in place (`0` copies).
  - `num_tokens % 16 != 0` and `ref_count[tail] > 1`: Allocates `fresh` block, copies `0 .. (num_tokens % 16)` filled slots across all `(layer, K/V, kv_head)`, decrements `ref_count[tail] -= 1`, and points `table.blocks.last_mut()` to `fresh`.

### 2.3 Radix Prefix Tree Edge Splitting & LRU Eviction
- Edges store `block_size`-aligned token chunks `edge_tokens` (`16 * m` tokens) paired with `m` physical `BlockId`s (`ref_count >= 1`).
- **`match_prefix(prompt_tokens)`** caps matched prefix length to $\lfloor (|P| - 1) / 16 \rfloor \times 16$ tokens so `remaining_suffix` always retains $\ge 1$ token to compute next-token logits.
- When a new request matches $16k$ tokens ($1 \le k < m$) of an existing $16m$-token edge, the edge splits in-place into parent `[0..16k]` (`blocks[0..k]`) and child `[16k..16m]` (`blocks[k..m]`) with zero net refcount change on the tree.
- A Radix node `u` is **evictable** iff `u != root`, `u.children.is_empty()`, and $\forall b \in u.\text{blocks}: \text{ref\_count}(b) == 1$ (zero active sequence readers). Evicting `u` decrements `ref_count(b)` `1 -> 0` and exposes `u.parent` for subsequent bottom-up eviction if needed.
