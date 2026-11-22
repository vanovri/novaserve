use novaserve_kv::{
    BlockAllocator, KvDType, KvSlab, KvSlabConfig, LogicalBlockTable, RadixCache, SwapEngine,
    SwappedHandle, DEFAULT_BLOCK_SIZE,
};
use novaserve_tensor::SplitMix64;

#[derive(Clone)]
struct ShadowSeq {
    seq_id: u64,
    tokens: Vec<u32>,
    table: LogicalBlockTable,
}

struct SwappedShadow {
    seq_id: u64,
    tokens: Vec<u32>,
    handle: SwappedHandle,
}

#[inline]
fn rolling_prefix_hash(prev_hash: u64, tok: u32) -> u64 {
    let mut z = prev_hash ^ ((tok as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 27)
}

#[inline]
fn canary_kv(prefix_hash: u64, layer: usize, kv_head: usize, is_v: bool, d: usize) -> f32 {
    let tag = prefix_hash
        .wrapping_add((layer as u64) * 131)
        .wrapping_add((kv_head as u64) * 17)
        .wrapping_add((u64::from(is_v)) * 997)
        .wrapping_add(d as u64);
    let bucket = (tag % 20_000) as f32;
    (bucket - 10_000.0) * 0.001
}

fn write_pos_canaries(
    slab: &mut KvSlab,
    block_id: novaserve_kv::BlockId,
    slot: usize,
    prefix_hash: u64,
) {
    let cfg = *slab.config();
    let mut k_vec = vec![0.0f32; cfg.head_dim];
    let mut v_vec = vec![0.0f32; cfg.head_dim];
    for layer in 0..cfg.num_layers {
        for h in 0..cfg.num_kv_heads {
            for d in 0..cfg.head_dim {
                k_vec[d] = canary_kv(prefix_hash, layer, h, false, d);
                v_vec[d] = canary_kv(prefix_hash, layer, h, true, d);
            }
            slab.write_kv_slot(layer, block_id, h, slot, &k_vec, &v_vec)
                .expect("write_kv_slot");
        }
    }
}

fn verify_seq_canaries(slab: &KvSlab, seq: &ShadowSeq) {
    assert_eq!(seq.tokens.len(), seq.table.num_tokens());
    assert!(seq.table.internal_waste_slots() < seq.table.block_size());

    let cfg = *slab.config();
    let mut k_read = vec![0.0f32; cfg.head_dim];
    let mut v_read = vec![0.0f32; cfg.head_dim];
    let mut h_acc = 0xCBF2_9CE4_8422_2325u64;

    for (pos, &tok) in seq.tokens.iter().enumerate() {
        h_acc = rolling_prefix_hash(h_acc, tok);
        let (bid, slot) = seq.table.resolve_slot(pos).expect("resolve_slot");
        for layer in 0..cfg.num_layers {
            for h in 0..cfg.num_kv_heads {
                slab.read_k_slot(layer, bid, h, slot, &mut k_read)
                    .expect("read_k");
                slab.read_v_slot(layer, bid, h, slot, &mut v_read)
                    .expect("read_v");
                for d in 0..cfg.head_dim {
                    let ek = canary_kv(h_acc, layer, h, false, d);
                    let ev = canary_kv(h_acc, layer, h, true, d);
                    assert_eq!(
                        k_read[d], ek,
                        "K canary mismatch on seq={} pos={pos} layer={layer} head={h} d={d}",
                        seq.seq_id
                    );
                    assert_eq!(
                        v_read[d], ev,
                        "V canary mismatch on seq={} pos={pos} layer={layer} head={h} d={d}",
                        seq.seq_id
                    );
                }
            }
        }
    }
}

#[test]
fn radix_mid_edge_split_and_cow_partial_tail_isolation() {
    let cfg = KvSlabConfig {
        num_layers: 2,
        num_physical_blocks: 32,
        num_kv_heads: 2,
        block_size: DEFAULT_BLOCK_SIZE,
        head_dim: 8,
        kv_dtype: KvDType::F32,
    };
    let mut slab = KvSlab::new(cfg).unwrap();
    let mut alloc = BlockAllocator::new(cfg.num_physical_blocks, cfg.block_size);
    let mut radix = RadixCache::new(cfg.block_size);

    // 1. Populate an 80-token sequence (5 full 16-token blocks) and insert into RadixCache
    let tokens_80: Vec<u32> = (100..180).collect();
    let mut seq1 = ShadowSeq {
        seq_id: 1,
        tokens: Vec::new(),
        table: LogicalBlockTable::new(16),
    };
    let mut h_acc = 0xCBF2_9CE4_8422_2325u64;
    for &t in &tokens_80 {
        h_acc = rolling_prefix_hash(h_acc, t);
        let (bid, slot) = alloc
            .cow_append_slot(&mut slab, Some(&mut radix), &mut seq1.table)
            .unwrap();
        write_pos_canaries(&mut slab, bid, slot, h_acc);
        seq1.tokens.push(t);
    }
    radix.insert_prefix(&seq1.tokens, seq1.table.blocks(), &mut alloc);
    alloc.release_table(&mut seq1.table);
    assert_eq!(radix.cached_blocks_count(), 5);

    // 2. New request shares first 48 tokens (3 blocks) of the 80-token edge and diverges at 48..55!
    let mut prompt_55: Vec<u32> = (100..148).collect();
    prompt_55.extend(500..507);

    let m = radix.match_prefix(&prompt_55, &mut alloc);
    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.cached_block_ids.len(), 3);
    assert_eq!(m.remaining_uncached_suffix.len(), 7);

    let mut seq2 = ShadowSeq {
        seq_id: 2,
        tokens: prompt_55[..48].to_vec(),
        table: LogicalBlockTable::from_cached_prefix(m.cached_block_ids, 48, 16),
    };
    let mut h2 = 0xCBF2_9CE4_8422_2325u64;
    for &t in &seq2.tokens {
        h2 = rolling_prefix_hash(h2, t);
    }
    for &t in m.remaining_uncached_suffix {
        h2 = rolling_prefix_hash(h2, t);
        let (bid, slot) = alloc
            .cow_append_slot(&mut slab, Some(&mut radix), &mut seq2.table)
            .unwrap();
        write_pos_canaries(&mut slab, bid, slot, h2);
        seq2.tokens.push(t);
    }

    // 3. Fork seq2 at token 55 (partial tail block has 7/16 slots!) into n=4 parallel completions!
    let mut forks: Vec<ShadowSeq> = (0..4)
        .map(|k| ShadowSeq {
            seq_id: 10 + k,
            tokens: seq2.tokens.clone(),
            table: alloc.fork_sequence(&seq2.table),
        })
        .collect();
    alloc.release_table(&mut seq2.table);

    // Mutate each fork's partial tail block with DIFFERENT tokens and verify zero cross-talk!
    for (idx, fseq) in forks.iter_mut().enumerate() {
        let mut hf = 0xCBF2_9CE4_8422_2325u64;
        for &t in &fseq.tokens {
            hf = rolling_prefix_hash(hf, t);
        }
        for step in 0..12 {
            let unique_tok = 10_000 + (idx as u32) * 100 + step;
            hf = rolling_prefix_hash(hf, unique_tok);
            let (bid, slot) = alloc
                .cow_append_slot(&mut slab, Some(&mut radix), &mut fseq.table)
                .unwrap();
            write_pos_canaries(&mut slab, bid, slot, hf);
            fseq.tokens.push(unique_tok);
        }
    }

    for fseq in &forks {
        verify_seq_canaries(&slab, fseq);
    }
    for mut fseq in forks {
        alloc.release_table(&mut fseq.table);
    }
    radix.clear(&mut alloc);
    alloc.verify_invariants();
    assert_eq!(alloc.free_blocks(), cfg.num_physical_blocks);
}

#[test]
fn q8_0_quantized_kv_slab_storage_roundtrip() {
    let cfg = KvSlabConfig {
        num_layers: 2,
        num_physical_blocks: 8,
        num_kv_heads: 2,
        block_size: 16,
        head_dim: 32,
        kv_dtype: KvDType::Q8_0,
    };
    let mut slab = KvSlab::new(cfg).unwrap();
    let mut alloc = BlockAllocator::new(8, 16);
    let bid = alloc.allocate_block(None).unwrap();

    let k_in: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let v_in: Vec<f32> = (0..32).map(|i| (16.0 - i as f32) * 0.05).collect();
    slab.write_kv_slot(1, bid, 1, 7, &k_in, &v_in).unwrap();

    let mut k_out = vec![0.0f32; 32];
    let mut v_out = vec![0.0f32; 32];
    slab.read_k_slot(1, bid, 1, 7, &mut k_out).unwrap();
    slab.read_v_slot(1, bid, 1, 7, &mut v_out).unwrap();

    for i in 0..32 {
        assert!((k_in[i] - k_out[i]).abs() < 0.02);
        assert!((v_in[i] - v_out[i]).abs() < 0.02);
    }
    alloc.free_block(bid);
    assert_eq!(alloc.free_blocks(), 8);
}

#[test]
fn stateful_10000_op_kv_radix_cow_and_swap_chaos_fuzzer() {
    // Tight physical block budget (48 blocks = 768 tokens total) to force relentless
    // Radix LRU eviction, CoW tail splits, swap-outs, and swap-ins!
    let cfg = KvSlabConfig {
        num_layers: 2,
        num_physical_blocks: 48,
        num_kv_heads: 2,
        block_size: 16,
        head_dim: 8,
        kv_dtype: KvDType::F32,
    };
    let mut slab = KvSlab::new(cfg).unwrap();
    let mut alloc = BlockAllocator::new(cfg.num_physical_blocks, cfg.block_size);
    let mut radix = RadixCache::new(cfg.block_size);
    let mut swap = SwapEngine::new(&slab, 128).unwrap();

    let mut rng = SplitMix64::new(0x03C0_FFEE_1000_0000);
    alloc.shuffle_free_list(&mut rng);

    // Pre-build overlapping multi-turn chat history templates to exercise Radix prefix hits & splits
    let sys_prompt_a: Vec<u32> = (1..=32).collect(); // 2 blocks
    let sys_prompt_b: Vec<u32> = (1..=16).chain(201..=232).collect(); // shares 1st block with A, then 2 blocks
    let turn_ext_1: Vec<u32> = (500..532).collect(); // +2 blocks
    let turn_ext_2: Vec<u32> = (500..516).chain(800..820).collect(); // shares 1 block of turn_ext_1, then splits!

    let mut active: Vec<ShadowSeq> = Vec::new();
    let mut swapped: Vec<SwappedShadow> = Vec::new();
    let mut next_seq_id = 1u64;

    let mut count_spawns = 0usize;
    let mut count_forks = 0usize;
    let mut count_swaps_out = 0usize;
    let mut count_swaps_in = 0usize;
    let mut count_truncates = 0usize;

    for step in 0..10_000usize {
        let action = rng.next_u32() % 100;

        if active.is_empty() || (action < 22 && active.len() < 12) {
            // Action 1: Spawn new multi-turn sequence matching RadixCache prefix
            let mut prompt = if rng.next_u32().is_multiple_of(2) {
                sys_prompt_a.clone()
            } else {
                sys_prompt_b.clone()
            };
            match rng.next_u32() % 3 {
                0 => prompt.extend_from_slice(&turn_ext_1),
                1 => prompt.extend_from_slice(&turn_ext_2),
                _ => {}
            }
            // Add 1..9 request-specific suffix tokens
            let extra = 1 + (rng.next_u32() as usize % 9);
            for _ in 0..extra {
                prompt.push(10_000 + (rng.next_u32() % 500));
            }

            let m = radix.match_prefix(&prompt, &mut alloc);
            assert!(!m.remaining_uncached_suffix.is_empty());

            let mut table =
                LogicalBlockTable::from_cached_prefix(m.cached_block_ids, m.matched_tokens, 16);
            let mut toks = prompt[..m.matched_tokens].to_vec();
            let mut h_acc = 0xCBF2_9CE4_8422_2325u64;
            for &t in &toks {
                h_acc = rolling_prefix_hash(h_acc, t);
            }

            let mut ok = true;
            for &t in m.remaining_uncached_suffix {
                // If blocks scarce, preempt an active sequence to swap before failing!
                if alloc.free_blocks() == 0
                    && radix.cached_blocks_count() == 0
                    && !active.is_empty()
                {
                    let victim = active.pop().unwrap();
                    let mut v_tbl = victim.table;
                    if let Ok(sh) = swap.swap_out(victim.seq_id, &mut v_tbl, &slab, &mut alloc) {
                        swapped.push(SwappedShadow {
                            seq_id: victim.seq_id,
                            tokens: victim.tokens,
                            handle: sh,
                        });
                        count_swaps_out += 1;
                    } else {
                        alloc.release_table(&mut v_tbl);
                    }
                }

                match alloc.cow_append_slot(&mut slab, Some(&mut radix), &mut table) {
                    Ok((bid, slot)) => {
                        h_acc = rolling_prefix_hash(h_acc, t);
                        write_pos_canaries(&mut slab, bid, slot, h_acc);
                        toks.push(t);
                    }
                    Err(_) => {
                        alloc.release_table(&mut table);
                        ok = false;
                        break;
                    }
                }
            }

            if ok {
                let s = ShadowSeq {
                    seq_id: next_seq_id,
                    tokens: toks,
                    table,
                };
                next_seq_id += 1;
                verify_seq_canaries(&slab, &s);
                active.push(s);
                count_spawns += 1;
            }
        } else if action < 55 && !active.is_empty() {
            // Action 2: Append 1..6 decode tokens (exercising CoW partial-block splits!)
            let idx = (rng.next_u32() as usize) % active.len();
            let append_n = 1 + (rng.next_u32() as usize % 6);
            let mut h_acc = 0xCBF2_9CE4_8422_2325u64;
            for &t in &active[idx].tokens {
                h_acc = rolling_prefix_hash(h_acc, t);
            }

            for _ in 0..append_n {
                let new_tok = 20_000 + (rng.next_u32() % 2000);
                match alloc.cow_append_slot(&mut slab, Some(&mut radix), &mut active[idx].table) {
                    Ok((bid, slot)) => {
                        h_acc = rolling_prefix_hash(h_acc, new_tok);
                        write_pos_canaries(&mut slab, bid, slot, h_acc);
                        active[idx].tokens.push(new_tok);
                    }
                    Err(_) => break,
                }
            }
        } else if action < 68 && !active.is_empty() && active.len() <= 10 {
            // Action 3: Parallel CoW Fork (`n = 3` or `n = 4` clones sharing blocks!)
            let idx = (rng.next_u32() as usize) % active.len();
            let clones = 2 + (rng.next_u32() as usize % 2);
            let base_tokens = active[idx].tokens.clone();
            for _ in 0..clones {
                let forked_tbl = alloc.fork_sequence(&active[idx].table);
                active.push(ShadowSeq {
                    seq_id: next_seq_id,
                    tokens: base_tokens.clone(),
                    table: forked_tbl,
                });
                next_seq_id += 1;
                count_forks += 1;
            }
        } else if action < 76 && !active.is_empty() {
            // Action 4: Speculative rejection KV rollback (`truncate_table`)
            let idx = (rng.next_u32() as usize) % active.len();
            if active[idx].tokens.len() > 8 {
                let drop_k = 1 + (rng.next_u32() as usize % 5);
                let new_len = active[idx].tokens.len() - drop_k;
                alloc.truncate_table(&mut active[idx].table, new_len);
                active[idx].tokens.truncate(new_len);
                count_truncates += 1;
            }
        } else if action < 86 && !active.is_empty() {
            // Action 5: Swap-Out preemption
            let idx = (rng.next_u32() as usize) % active.len();
            let mut victim = active.swap_remove(idx);
            match swap.swap_out(victim.seq_id, &mut victim.table, &slab, &mut alloc) {
                Ok(handle) => {
                    swapped.push(SwappedShadow {
                        seq_id: victim.seq_id,
                        tokens: victim.tokens,
                        handle,
                    });
                    count_swaps_out += 1;
                }
                Err(_) => {
                    // Swap pool full: put back into active
                    active.push(victim);
                }
            }
        } else if action < 94 && !swapped.is_empty() {
            // Action 6: Swap-In restoration + immediate canary integrity verification!
            let idx = (rng.next_u32() as usize) % swapped.len();
            let item = swapped.swap_remove(idx);
            match swap.swap_in(item.handle, &mut slab, &mut alloc, Some(&mut radix)) {
                Ok(restored_tbl) => {
                    let restored = ShadowSeq {
                        seq_id: item.seq_id,
                        tokens: item.tokens,
                        table: restored_tbl,
                    };
                    verify_seq_canaries(&slab, &restored);
                    active.push(restored);
                    count_swaps_in += 1;
                }
                Err((handle_back, _)) => {
                    swapped.push(SwappedShadow {
                        seq_id: item.seq_id,
                        tokens: item.tokens,
                        handle: handle_back,
                    });
                }
            }
        } else if !active.is_empty() {
            // Action 7: Finish or abort sequence; insert finished prefix into RadixCache!
            let idx = (rng.next_u32() as usize) % active.len();
            let mut done = active.swap_remove(idx);
            verify_seq_canaries(&slab, &done);
            if !rng.next_u32().is_multiple_of(4) {
                radix.insert_prefix(&done.tokens, done.table.blocks(), &mut alloc);
            }
            alloc.release_table(&mut done.table);
        }

        // Assert global block conservation invariant at EVERY SINGLE STEP!
        assert_eq!(
            alloc.allocated_blocks() + alloc.free_blocks(),
            cfg.num_physical_blocks,
            "block conservation failed at step {step}"
        );
        if step.is_multiple_of(250) {
            alloc.verify_invariants();
            for seq in &active {
                verify_seq_canaries(&slab, seq);
            }
        }
    }

    // Drain all remaining swapped & active sequences and verify 100% canary integrity + 0 leaks!
    for mut seq in active.drain(..) {
        verify_seq_canaries(&slab, &seq);
        radix.insert_prefix(&seq.tokens, seq.table.blocks(), &mut alloc);
        alloc.release_table(&mut seq.table);
    }

    while let Some(item) = swapped.pop() {
        let mut tbl = swap
            .swap_in(item.handle, &mut slab, &mut alloc, Some(&mut radix))
            .expect("swap_in during final drain");
        let restored = ShadowSeq {
            seq_id: item.seq_id,
            tokens: item.tokens,
            table: tbl.clone(),
        };
        verify_seq_canaries(&slab, &restored);
        alloc.release_table(&mut tbl);
    }

    radix.clear(&mut alloc);
    alloc.verify_invariants();
    assert_eq!(alloc.free_blocks(), cfg.num_physical_blocks);
    assert_eq!(swap.free_swap_blocks(), swap.total_swap_blocks());

    println!(
        "[M03 Telemetry] 10,000 chaos ops PASS: spawns={count_spawns}, forks={count_forks}, swaps_out={count_swaps_out}, swaps_in={count_swaps_in}, truncates={count_truncates}, radix_hit_rate={:.1}%, evicted_blocks={}",
        radix.hit_rate() * 100.0,
        radix.total_evicted_blocks()
    );
}
