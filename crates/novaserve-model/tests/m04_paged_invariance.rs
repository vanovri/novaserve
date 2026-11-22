use novaserve_frontend::{BpeTokenizer, VocabularyBuilder};
use novaserve_kv::{
    BlockAllocator, KvDType, KvSlab, LogicalBlockTable, RadixCache, DEFAULT_BLOCK_SIZE,
};
use novaserve_model::{BatchInput, LlamaConfig, LlamaModel, ModelFactory};
use novaserve_tensor::{AlignedScratchArena, DType, GgufFile, SafetensorsFile, SplitMix64};

#[test]
fn exact_paged_scrambled_radix_chunked_and_multiseq_invariance_proof() {
    let vocab_size = 1024usize;
    let cfg = LlamaConfig::tiny_16m(vocab_size, DType::F32);
    assert_eq!(cfg.n_layers, 4);
    assert_eq!(cfg.n_heads, 8);
    assert_eq!(cfg.n_kv_heads, 2);
    assert_eq!(cfg.head_dim, 32);

    let model = ModelFactory::build_random_model(cfg.clone(), 0x0404_2026_ABCD_EF01)
        .expect("build Tiny-16M GQA model");

    let mut rng = SplitMix64::new(0x7777_8888_9999_AAAA);
    let seq_len = 200usize;
    let target_seq: Vec<u32> = (0..seq_len)
        .map(|_| (rng.next_u32() as usize % vocab_size) as u32)
        .collect();

    // 1. Contiguous single-sequence eager reference pass (`[1, 200]` all at once, all positions emitted)
    let eager_all_logits = model
        .forward_eager_contiguous(&target_seq, true)
        .expect("eager contiguous forward");
    assert_eq!(eager_all_logits.len(), seq_len * vocab_size);

    // 2. Scrambled PagedAttention + 64-token Radix CoW shared blocks + irregular ChunkedPrefill
    //    (`23, 31, 32` covering `64..150`) + 7 decoy sequences interleaved on `150..200`!
    let num_blocks = 256usize;
    let mut slab = KvSlab::new(cfg.kv_slab_config(num_blocks, DEFAULT_BLOCK_SIZE, KvDType::F32))
        .expect("kv slab");
    let mut alloc = BlockAllocator::new(num_blocks, DEFAULT_BLOCK_SIZE);
    let mut radix = RadixCache::new(DEFAULT_BLOCK_SIZE);
    let mut arena = AlignedScratchArena::new(cfg.recommended_scratch_bytes(256)).expect("arena");

    // Scramble physical free list completely!
    alloc.shuffle_free_list(&mut rng);

    // Pre-populate a prior 80-token sequence sharing `target_seq[0..64]` (`4` full blocks) into RadixCache
    // AND keep a CoW clone alive so those 4 blocks remain shared (`ref_count >= 3`)!
    let mut prior_prompt = target_seq[..64].to_vec();
    for _ in 0..16 {
        prior_prompt.push((rng.next_u32() as usize % vocab_size) as u32);
    }
    let mut prior_table = LogicalBlockTable::new(DEFAULT_BLOCK_SIZE);
    let mut prior_batch = BatchInput::with_capacity(80);
    for (pos, &tok) in prior_prompt.iter().enumerate() {
        let (bid, off) = alloc
            .cow_append_slot(&mut slab, Some(&mut radix), &mut prior_table)
            .unwrap();
        prior_batch.push_token(tok, pos as u32, 0, bid, off, pos == 79);
    }
    {
        let tables = [&prior_table];
        let _ = model
            .forward_batch(&prior_batch, &tables, &mut slab, &mut arena)
            .expect("prior prefill");
    }
    radix.insert_prefix(&prior_prompt, prior_table.blocks(), &mut alloc);
    let mut cow_observer = alloc.fork_sequence(&prior_table);
    alloc.release_table(&mut prior_table);

    // Match `target_seq` against `RadixCache`: hits exact 64 tokens (`4` scrambled shared blocks)!
    let prefix_hit = radix.match_prefix(&target_seq, &mut alloc);
    assert_eq!(prefix_hit.matched_tokens, 64);
    assert_eq!(prefix_hit.cached_block_ids.len(), 4);

    let mut main_table =
        LogicalBlockTable::from_cached_prefix(prefix_hit.cached_block_ids, 64, DEFAULT_BLOCK_SIZE);

    // Process `64..150` (`86` tokens) via irregular ChunkedPrefill budgets `[23, 31, 32]`!
    let mut worst_diff = 0.0f32;
    let chunk_sizes = [23usize, 31, 32];
    let mut cursor = 64usize;
    for &c_size in &chunk_sizes {
        let end = cursor + c_size;
        let mut batch = BatchInput::with_capacity(c_size);
        for (pos, &tok) in target_seq.iter().enumerate().take(end).skip(cursor) {
            let (bid, off) = alloc
                .cow_append_slot(&mut slab, Some(&mut radix), &mut main_table)
                .unwrap();
            // Emit logits on the last token of each prefill chunk and verify against eager!
            batch.push_token(tok, pos as u32, 0, bid, off, pos == end - 1);
        }
        let tables = [&main_table];
        let chunk_logits = model
            .forward_batch(&batch, &tables, &mut slab, &mut arena)
            .expect("chunked prefill forward");
        assert_eq!(chunk_logits.len(), vocab_size);

        let check_pos = end - 1;
        let ref_slice = &eager_all_logits[check_pos * vocab_size..(check_pos + 1) * vocab_size];
        for v in 0..vocab_size {
            let d = (chunk_logits[v] - ref_slice[v]).abs();
            worst_diff = worst_diff.max(d);
            assert!(
                d < 1e-5,
                "ChunkedPrefill drift at pos={check_pos} v={v}: paged={} vs eager={} (diff={d:.3e})",
                chunk_logits[v],
                ref_slice[v]
            );
        }
        cursor = end;
    }
    assert_eq!(cursor, 150);

    // Initialize 7 decoy sequences of varying lengths (`5..43`) to interleave alongside `main_table`!
    let mut decoy_tables: Vec<LogicalBlockTable> = Vec::with_capacity(7);
    let mut decoy_lens: Vec<usize> = Vec::with_capacity(7);
    for d_idx in 0..7usize {
        let d_len = 5 + d_idx * 6;
        let mut dtbl = LogicalBlockTable::new(DEFAULT_BLOCK_SIZE);
        let mut dbatch = BatchInput::with_capacity(d_len);
        for p in 0..d_len {
            let dtok = (rng.next_u32() as usize % vocab_size) as u32;
            let (bid, off) = alloc
                .cow_append_slot(&mut slab, Some(&mut radix), &mut dtbl)
                .unwrap();
            dbatch.push_token(dtok, p as u32, 0, bid, off, p == d_len - 1);
        }
        let refs = [&dtbl];
        let _ = model
            .forward_batch(&dbatch, &refs, &mut slab, &mut arena)
            .unwrap();
        decoy_tables.push(dtbl);
        decoy_lens.push(d_len);
    }

    // Run tokens `150..200` interleaved in a heterogeneous 8-sequence ContinuousBatch!
    // Place `main_table` at a varying middle slot (`slot = step % 8`) on every iteration!
    for (pos, &main_tok) in target_seq.iter().enumerate().take(200).skip(150) {
        let main_slot = pos % 8;
        let mut batch = BatchInput::with_capacity(16);

        let (m_bid, m_off) = alloc
            .cow_append_slot(&mut slab, Some(&mut radix), &mut main_table)
            .unwrap();

        let mut decoy_writes = Vec::with_capacity(7);
        for d_idx in 0..7 {
            let (dbid, doff) = alloc
                .cow_append_slot(&mut slab, Some(&mut radix), &mut decoy_tables[d_idx])
                .unwrap();
            let dpos = decoy_lens[d_idx] as u32;
            decoy_lens[d_idx] += 1;
            let dtok = (rng.next_u32() as usize % vocab_size) as u32;
            decoy_writes.push((dtok, dpos, dbid, doff));
        }

        // Assemble 8-slot `seq_tables` array and interleaved token entries
        let mut d_cursor = 0usize;
        for slot_i in 0..8 {
            if slot_i == main_slot {
                batch.push_token(main_tok, pos as u32, slot_i, m_bid, m_off, true);
            } else {
                let (dtok, dpos, dbid, doff) = decoy_writes[d_cursor];
                d_cursor += 1;
                batch.push_token(dtok, dpos, slot_i, dbid, doff, true);
            }
        }

        let mut table_refs: Vec<&LogicalBlockTable> = Vec::with_capacity(8);
        let mut dc = 0usize;
        for slot_i in 0..8 {
            if slot_i == main_slot {
                table_refs.push(&main_table);
            } else {
                table_refs.push(&decoy_tables[dc]);
                dc += 1;
            }
        }

        let step_logits = model
            .forward_batch(&batch, &table_refs, &mut slab, &mut arena)
            .expect("multi-seq forward");
        assert_eq!(step_logits.len(), 8 * vocab_size);

        let main_logits = &step_logits[main_slot * vocab_size..(main_slot + 1) * vocab_size];
        let ref_logits = &eager_all_logits[pos * vocab_size..(pos + 1) * vocab_size];
        for v in 0..vocab_size {
            let d = (main_logits[v] - ref_logits[v]).abs();
            worst_diff = worst_diff.max(d);
            assert!(
                d < 1e-5,
                "Decode drift at pos={pos} v={v}: paged={} vs eager={} (diff={d:.3e})",
                main_logits[v],
                ref_logits[v]
            );
        }
    }

    println!(
        "[M04 Telemetry] 200-tok Scrambled Paged + Radix CoW + ChunkedPrefill + 8-Seq Batch max_abs_diff = {worst_diff:.3e} (< 1e-5)"
    );

    // Cleanup and verify 0 leaked blocks
    alloc.release_table(&mut main_table);
    alloc.release_table(&mut cow_observer);
    for mut dt in decoy_tables {
        alloc.release_table(&mut dt);
    }
    radix.clear(&mut alloc);
    alloc.verify_invariants();
    assert_eq!(alloc.free_blocks(), num_blocks);
}

#[test]
fn flash_decoding_long_context_300_tokens_and_gguf_safetensors_roundtrip() {
    let vocab_size = 512usize;
    let tok = VocabularyBuilder::build_synthetic(vocab_size).unwrap();

    // Test long context (`300 >= 256` tokens) activating Rayon FlashDecoding partition reduction!
    let cfg = LlamaConfig::micro_2m(vocab_size, DType::F32);
    let model = ModelFactory::build_random_model(cfg.clone(), 0xF1A5_4DEC_0DE0_0300).unwrap();

    let mut rng = SplitMix64::new(42);
    let long_tokens: Vec<u32> = (0..300)
        .map(|_| (rng.next_u32() as usize % vocab_size) as u32)
        .collect();

    let eager_last = model.forward_eager_contiguous(&long_tokens, false).unwrap();

    let mut slab = KvSlab::new(cfg.kv_slab_config(32, 16, KvDType::F32)).unwrap();
    let mut alloc = BlockAllocator::new(32, 16);
    alloc.shuffle_free_list(&mut rng);
    let mut arena = AlignedScratchArena::new(cfg.recommended_scratch_bytes(300)).unwrap();
    let mut tbl = LogicalBlockTable::new(16);
    let mut batch = BatchInput::with_capacity(300);
    for (p, &tid) in long_tokens.iter().enumerate() {
        let (bid, off) = alloc.cow_append_slot(&mut slab, None, &mut tbl).unwrap();
        batch.push_token(tid, p as u32, 0, bid, off, p == 299);
    }
    let paged_last = model
        .forward_batch(&batch, &[&tbl], &mut slab, &mut arena)
        .unwrap();

    for v in 0..vocab_size {
        let diff = (paged_last[v] - eager_last[v]).abs();
        assert!(
            diff < 1e-5,
            "FlashDecoding 300-token mismatch at v={v}: diff={diff:.3e}"
        );
    }

    // Verify `.gguf` (`Q8_0` & `Q4_0` + embedded tokenizer) and `.safetensors` (`F32`) round-trip!
    for &dt in &[DType::F32, DType::Q8_0, DType::Q4_0] {
        let qcfg = LlamaConfig::micro_2m(vocab_size, dt);
        let qmodel = ModelFactory::build_random_model(qcfg.clone(), 0x9988_1122).unwrap();
        let gguf_bytes = qmodel.to_gguf_bytes(Some(&tok)).expect("export gguf");
        let parsed_gguf = GgufFile::from_bytes(&gguf_bytes).expect("parse gguf");
        let loaded_tok = BpeTokenizer::from_gguf(&parsed_gguf).expect("load tok from gguf");
        assert_eq!(loaded_tok.vocab_size(), vocab_size);

        let loaded_model = LlamaModel::from_gguf(&parsed_gguf).expect("load model from gguf");
        let ref_out = qmodel
            .forward_eager_contiguous(&long_tokens[..24], false)
            .unwrap();
        let got_out = loaded_model
            .forward_eager_contiguous(&long_tokens[..24], false)
            .unwrap();
        for v in 0..vocab_size {
            assert!(
                (ref_out[v] - got_out[v]).abs() < 1e-6,
                "GGUF {dt:?} reload mismatch"
            );
        }
    }

    let st_bytes = model.to_safetensors_bytes().expect("export safetensors");
    let st_file = SafetensorsFile::from_bytes(&st_bytes).expect("parse safetensors");
    let st_loaded = LlamaModel::from_safetensors(&st_file, cfg).expect("load safetensors");
    let st_out = st_loaded
        .forward_eager_contiguous(&long_tokens[..24], false)
        .unwrap();
    let orig_out = model
        .forward_eager_contiguous(&long_tokens[..24], false)
        .unwrap();
    for v in 0..vocab_size {
        assert!((st_out[v] - orig_out[v]).abs() < 1e-6);
    }
}
