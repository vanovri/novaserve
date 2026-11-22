//! Byte-Level BPE Tokenizer (`GPT-4` / `Llama-3` compatible), $O(N \log N)$ priority-queue
//! merge encoder, `TokenStreamDecoder` incremental UTF-8 buffer, and `VocabularyBuilder`.

use crate::error::FrontendError;
use novaserve_tensor::{GgufArray, GgufFile, GgufValue};
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// Builds the canonical 256-byte GPT-2 / Llama-3 byte-to-unicode reversible lookup tables.
pub fn gpt2_byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut is_direct = [false; 256];

    for b in b'!'..=b'~' {
        is_direct[b as usize] = true;
    }
    for b in 0xA1u8..=0xACu8 {
        is_direct[b as usize] = true;
    }
    for b in 0xAEu8..=0xFFu8 {
        is_direct[b as usize] = true;
    }

    let mut extra = 0u32;
    for b in 0usize..256 {
        if is_direct[b] {
            byte_to_char[b] = char::from(b as u8);
        } else {
            byte_to_char[b] = char::from_u32(256 + extra).expect("valid unicode scalar");
            extra += 1;
        }
    }

    let mut char_to_byte = HashMap::with_capacity(256);
    for (b, &ch) in byte_to_char.iter().enumerate() {
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

#[derive(Clone, Debug)]
struct SymbolNode {
    token_id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    version: u32,
    alive: bool,
}

/// Priority-queue entry ordered by lowest `(rank, left_position)` for deterministic BPE merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MergeCandidate {
    rank: u32,
    left_idx: usize,
    left_ver: u32,
    right_ver: u32,
    merged_id: u32,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    vocab_strings: Vec<String>,
    raw_token_bytes: Vec<Vec<u8>>,
    is_special: Vec<bool>,
    token_to_id: HashMap<String, u32>,
    byte_fallback_ids: [u32; 256],
    /// Maps `(left_token_id, right_token_id) -> (rank, merged_token_id)`.
    merge_ranks: HashMap<(u32, u32), (u32, u32)>,
    /// Ordered merge list `(left_id, right_id)` for GGUF / JSON export.
    ordered_merges: Vec<(u32, u32)>,
    /// Special tokens sorted by descending byte length for greedy longest-match splitting.
    special_tokens: Vec<(String, u32)>,
    bos_token_id: Option<u32>,
    eos_token_id: u32,
}

impl BpeTokenizer {
    pub fn from_parts(
        vocab_strings: Vec<String>,
        is_special: Vec<bool>,
        merges: Vec<(String, String)>,
        bos_token_id: Option<u32>,
        eos_token_id: u32,
    ) -> Result<Self, FrontendError> {
        if vocab_strings.is_empty() || vocab_strings.len() != is_special.len() {
            return Err(FrontendError::Tokenizer(
                "vocab_strings and is_special length mismatch or empty".into(),
            ));
        }
        if (eos_token_id as usize) >= vocab_strings.len() {
            return Err(FrontendError::Tokenizer(format!(
                "eos_token_id {eos_token_id} out of range for vocab size {}",
                vocab_strings.len()
            )));
        }

        let (byte_to_char, char_to_byte) = gpt2_byte_to_unicode();
        let mut token_to_id = HashMap::with_capacity(vocab_strings.len());
        let mut raw_token_bytes = Vec::with_capacity(vocab_strings.len());
        let mut special_tokens = Vec::new();
        let mut byte_fallback_ids = [u32::MAX; 256];

        for (idx, (tok_str, &spec)) in vocab_strings.iter().zip(is_special.iter()).enumerate() {
            let tid = idx as u32;
            token_to_id.entry(tok_str.clone()).or_insert(tid);

            if spec {
                raw_token_bytes.push(tok_str.as_bytes().to_vec());
                special_tokens.push((tok_str.clone(), tid));
                continue;
            }

            // Support `<0xXX>` SentencePiece/GGUF byte tokens if present
            if tok_str.len() == 6 && tok_str.starts_with("<0x") && tok_str.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok_str[3..5], 16) {
                    byte_fallback_ids[b as usize] = tid;
                    raw_token_bytes.push(vec![b]);
                    continue;
                }
            }

            // Decode GPT-2 byte-level unicode string into exact raw bytes
            let mut decoded = Vec::with_capacity(tok_str.len());
            let mut all_mapped = true;
            for ch in tok_str.chars() {
                if let Some(&b) = char_to_byte.get(&ch) {
                    decoded.push(b);
                } else {
                    all_mapped = false;
                    break;
                }
            }
            if !all_mapped {
                decoded = tok_str.as_bytes().to_vec();
            }
            if decoded.len() == 1 && byte_fallback_ids[decoded[0] as usize] == u32::MAX {
                byte_fallback_ids[decoded[0] as usize] = tid;
            }
            raw_token_bytes.push(decoded);
        }

        // Also check direct single GPT-2 char lookup for any unpopulated byte fallback
        for b in 0usize..256 {
            if byte_fallback_ids[b] == u32::MAX {
                let s = byte_to_char[b].to_string();
                if let Some(&tid) = token_to_id.get(&s) {
                    byte_fallback_ids[b] = tid;
                } else {
                    return Err(FrontendError::Tokenizer(format!(
                        "missing byte-fallback token for byte 0x{b:02x}"
                    )));
                }
            }
        }

        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        let mut merge_ranks = HashMap::with_capacity(merges.len());
        let mut ordered_merges = Vec::with_capacity(merges.len());
        for (rank, (left_s, right_s)) in merges.into_iter().enumerate() {
            let Some(&left_id) = token_to_id.get(&left_s) else {
                continue;
            };
            let Some(&right_id) = token_to_id.get(&right_s) else {
                continue;
            };
            let joined = format!("{left_s}{right_s}");
            let Some(&merged_id) = token_to_id.get(&joined) else {
                continue;
            };
            if let std::collections::hash_map::Entry::Vacant(e) =
                merge_ranks.entry((left_id, right_id))
            {
                e.insert((rank as u32, merged_id));
                ordered_merges.push((left_id, right_id));
            }
        }

        Ok(Self {
            vocab_strings,
            raw_token_bytes,
            is_special,
            token_to_id,
            byte_fallback_ids,
            merge_ranks,
            ordered_merges,
            special_tokens,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Loads a Byte-Level BPE tokenizer directly from `GGUF v3` metadata (`tokenizer.ggml.*`).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, FrontendError> {
        let meta = gguf.metadata();
        let tokens_val = meta.get("tokenizer.ggml.tokens").ok_or_else(|| {
            FrontendError::Tokenizer("missing tokenizer.ggml.tokens in GGUF".into())
        })?;
        let token_strs = tokens_val.as_string_array().ok_or_else(|| {
            FrontendError::Tokenizer("tokenizer.ggml.tokens must be string array".into())
        })?;
        let vocab_strings: Vec<String> = token_strs.into_iter().map(str::to_string).collect();

        // GGUF token_type: 1 = normal, 2 = unknown, 3 = control/special, 4 = user_defined, 6 = byte
        let mut is_special = vec![false; vocab_strings.len()];
        if let Some(GgufValue::Array(arr)) = meta.get("tokenizer.ggml.token_type") {
            for (i, el) in arr.elements.iter().enumerate().take(vocab_strings.len()) {
                if let Some(t) = el.as_u32() {
                    if t == 3 || t == 4 {
                        is_special[i] = true;
                    }
                }
            }
        } else {
            for (i, s) in vocab_strings.iter().enumerate() {
                if (s.starts_with("<|") && s.ends_with("|>"))
                    || s == "<eos>"
                    || s == "<s>"
                    || s == "</s>"
                {
                    is_special[i] = true;
                }
            }
        }

        let mut merges = Vec::new();
        if let Some(merges_val) = meta.get("tokenizer.ggml.merges") {
            let merge_strs = merges_val.as_string_array().ok_or_else(|| {
                FrontendError::Tokenizer("tokenizer.ggml.merges must be string array".into())
            })?;
            for m in merge_strs {
                if let Some((l, r)) = m.split_once(' ') {
                    merges.push((l.to_string(), r.to_string()));
                }
            }
        }

        let eos_token_id = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_u32)
            .unwrap_or((vocab_strings.len() - 1) as u32);
        let bos_token_id = meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(GgufValue::as_u32);

        Self::from_parts(
            vocab_strings,
            is_special,
            merges,
            bos_token_id,
            eos_token_id,
        )
    }

    /// Exports tokenizer metadata keys (`tokenizer.ggml.*`) into a GGUF metadata map.
    pub fn write_gguf_metadata(&self, meta: &mut BTreeMap<String, GgufValue>) {
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(GgufArray {
                elem_type: 8,
                elements: self
                    .vocab_strings
                    .iter()
                    .cloned()
                    .map(GgufValue::String)
                    .collect(),
            }),
        );
        meta.insert(
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(GgufArray {
                elem_type: 5,
                elements: self
                    .is_special
                    .iter()
                    .map(|&sp| GgufValue::I32(if sp { 3 } else { 1 }))
                    .collect(),
            }),
        );
        let merge_elements: Vec<GgufValue> = self
            .ordered_merges
            .iter()
            .map(|&(l, r)| {
                GgufValue::String(format!(
                    "{} {}",
                    self.vocab_strings[l as usize], self.vocab_strings[r as usize]
                ))
            })
            .collect();
        meta.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(GgufArray {
                elem_type: 8,
                elements: merge_elements,
            }),
        );
        meta.insert(
            "tokenizer.ggml.eos_token_id".into(),
            GgufValue::U32(self.eos_token_id),
        );
        if let Some(bos) = self.bos_token_id {
            meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(bos));
        }
    }

    /// Loads a HuggingFace `tokenizer.json` Byte-Level BPE configuration.
    pub fn from_tokenizer_json(json_str: &str) -> Result<Self, FrontendError> {
        let root: Value = serde_json::from_str(json_str)?;
        let model = root
            .get("model")
            .ok_or_else(|| FrontendError::Tokenizer("missing 'model' object".into()))?;
        let vocab_obj = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| FrontendError::Tokenizer("missing 'model.vocab' map".into()))?;

        let mut max_id = 0usize;
        let mut id_pairs = Vec::with_capacity(vocab_obj.len());
        for (tok, id_v) in vocab_obj {
            let id = id_v.as_u64().ok_or_else(|| {
                FrontendError::Tokenizer(format!("non-integer token id for '{tok}'"))
            })? as usize;
            max_id = max_id.max(id);
            id_pairs.push((id, tok.clone(), false));
        }

        if let Some(added) = root.get("added_tokens").and_then(Value::as_array) {
            for entry in added {
                if let (Some(id), Some(content)) = (
                    entry.get("id").and_then(Value::as_u64),
                    entry.get("content").and_then(Value::as_str),
                ) {
                    let special = entry
                        .get("special")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    let id_u = id as usize;
                    max_id = max_id.max(id_u);
                    id_pairs.push((id_u, content.to_string(), special));
                }
            }
        }

        let vocab_len = max_id + 1;
        let mut vocab_strings = vec![String::new(); vocab_len];
        let mut is_special = vec![false; vocab_len];
        let mut populated = vec![false; vocab_len];

        for (id, tok, spec) in id_pairs {
            vocab_strings[id] = tok;
            is_special[id] = spec;
            populated[id] = true;
        }
        if populated.iter().any(|&p| !p) {
            return Err(FrontendError::Tokenizer(
                "non-contiguous token IDs in tokenizer.json".into(),
            ));
        }

        let mut merges = Vec::new();
        if let Some(merges_arr) = model.get("merges").and_then(Value::as_array) {
            for item in merges_arr {
                if let Some(s) = item.as_str() {
                    if let Some((l, r)) = s.split_once(' ') {
                        merges.push((l.to_string(), r.to_string()));
                    }
                } else if let Some(pair) = item.as_array() {
                    if pair.len() == 2 {
                        if let (Some(l), Some(r)) = (pair[0].as_str(), pair[1].as_str()) {
                            merges.push((l.to_string(), r.to_string()));
                        }
                    }
                }
            }
        }

        let mut eos_id = (vocab_len - 1) as u32;
        let mut bos_id = None;
        for (i, s) in vocab_strings.iter().enumerate() {
            if s == "<eos>" || s == "<|eot_id|>" || s == "<|im_end|>" || s == "</s>" {
                eos_id = i as u32;
            }
            if s == "<|begin_of_text|>" || s == "<s>" {
                bos_id = Some(i as u32);
            }
        }

        Self::from_parts(vocab_strings, is_special, merges, bos_id, eos_id)
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_strings.len()
    }

    #[inline]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    #[inline]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    #[inline]
    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.is_special
            .get(token_id as usize)
            .copied()
            .unwrap_or(false)
    }

    #[inline]
    pub fn token_str(&self, token_id: u32) -> Option<&str> {
        self.vocab_strings
            .get(token_id as usize)
            .map(String::as_str)
    }

    #[inline]
    pub fn token_id(&self, token_str: &str) -> Option<u32> {
        self.token_to_id.get(token_str).copied()
    }

    #[inline]
    pub fn decode_raw_bytes(&self, token_id: u32) -> &[u8] {
        &self.raw_token_bytes[token_id as usize]
    }

    /// Priority-queue + doubly-linked symbol list $O(N \log N)$ BPE merge encoder over raw bytes.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.byte_fallback_ids[bytes[0] as usize]];
        }

        let mut nodes = Vec::with_capacity(n);
        for (i, &b) in bytes.iter().enumerate() {
            nodes.push(SymbolNode {
                token_id: self.byte_fallback_ids[b as usize],
                prev: if i > 0 { Some(i - 1) } else { None },
                next: if i + 1 < n { Some(i + 1) } else { None },
                version: 0,
                alive: true,
            });
        }

        let mut heap: BinaryHeap<Reverse<MergeCandidate>> = BinaryHeap::new();
        for i in 0..(n - 1) {
            let left_tid = nodes[i].token_id;
            let right_tid = nodes[i + 1].token_id;
            if let Some(&(rank, merged_id)) = self.merge_ranks.get(&(left_tid, right_tid)) {
                heap.push(Reverse(MergeCandidate {
                    rank,
                    left_idx: i,
                    left_ver: 0,
                    right_ver: 0,
                    merged_id,
                }));
            }
        }

        while let Some(Reverse(cand)) = heap.pop() {
            let left = &nodes[cand.left_idx];
            if !left.alive || left.version != cand.left_ver {
                continue;
            }
            let Some(right_idx) = left.next else {
                continue;
            };
            let right = &nodes[right_idx];
            if !right.alive || right.version != cand.right_ver {
                continue;
            }

            // Perform O(1) linked-list merge of `left_idx` and `right_idx`
            let new_next = right.next;
            let prev_idx = left.prev;

            nodes[right_idx].alive = false;
            nodes[cand.left_idx].token_id = cand.merged_id;
            nodes[cand.left_idx].version = nodes[cand.left_idx].version.wrapping_add(1);
            nodes[cand.left_idx].next = new_next;

            if let Some(nn) = new_next {
                nodes[nn].prev = Some(cand.left_idx);
            }

            let cur_ver = nodes[cand.left_idx].version;
            // Enqueue updated `(prev, cur)` candidate
            if let Some(pi) = prev_idx {
                let p_tid = nodes[pi].token_id;
                let p_ver = nodes[pi].version;
                if let Some(&(rank, merged_id)) = self.merge_ranks.get(&(p_tid, cand.merged_id)) {
                    heap.push(Reverse(MergeCandidate {
                        rank,
                        left_idx: pi,
                        left_ver: p_ver,
                        right_ver: cur_ver,
                        merged_id,
                    }));
                }
            }
            // Enqueue updated `(cur, next)` candidate
            if let Some(ni) = new_next {
                let n_tid = nodes[ni].token_id;
                let n_ver = nodes[ni].version;
                if let Some(&(rank, merged_id)) = self.merge_ranks.get(&(cand.merged_id, n_tid)) {
                    heap.push(Reverse(MergeCandidate {
                        rank,
                        left_idx: cand.left_idx,
                        left_ver: cur_ver,
                        right_ver: n_ver,
                        merged_id,
                    }));
                }
            }
        }

        let mut out = Vec::with_capacity(n);
        let mut cur = Some(0usize);
        while let Some(idx) = cur {
            debug_assert!(nodes[idx].alive);
            out.push(nodes[idx].token_id);
            cur = nodes[idx].next;
        }
        out
    }

    /// Encodes UTF-8 text, recognizing registered special control tokens (`<|im_start|>`, `<eos>`, etc.)
    /// when `allow_special` is true.
    pub fn encode_with_special(&self, text: &str, allow_special: bool) -> Vec<u32> {
        if !allow_special || self.special_tokens.is_empty() {
            return self.encode_bytes(text.as_bytes());
        }

        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut cursor = 0usize;
        let mut span_start = 0usize;

        while cursor < bytes.len() {
            let mut matched: Option<(usize, u32)> = None;
            for (pat, tid) in &self.special_tokens {
                let pb = pat.as_bytes();
                if cursor + pb.len() <= bytes.len() && &bytes[cursor..cursor + pb.len()] == pb {
                    matched = Some((pb.len(), *tid));
                    break;
                }
            }

            if let Some((pat_len, tid)) = matched {
                if span_start < cursor {
                    out.extend(self.encode_bytes(&bytes[span_start..cursor]));
                }
                out.push(tid);
                cursor += pat_len;
                span_start = cursor;
            } else {
                cursor += 1;
            }
        }

        if span_start < bytes.len() {
            out.extend(self.encode_bytes(&bytes[span_start..]));
        }
        out
    }

    /// Encodes arbitrary UTF-8 string losslessly (treating literal special-token substrings as text
    /// unless `encode_with_special(text, true)` is requested).
    #[inline]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_bytes(text.as_bytes())
    }

    /// Decodes token IDs back into exact raw bytes (100% lossless even on arbitrary binary).
    pub fn decode_bytes(&self, tokens: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(tokens.len() * 3);
        for &tid in tokens {
            let idx = tid as usize;
            if idx >= self.raw_token_bytes.len() {
                continue;
            }
            if skip_special && self.is_special[idx] {
                continue;
            }
            out.extend_from_slice(&self.raw_token_bytes[idx]);
        }
        out
    }

    /// Decodes token IDs into a validated UTF-8 `String`.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, FrontendError> {
        let bytes = self.decode_bytes(tokens, false);
        Ok(String::from_utf8(bytes)?)
    }
}

/// Incremental UTF-8 token stream decoder that buffers incomplete multi-byte UTF-8 codepoint
/// sequences across tokens so SSE streams never emit broken `U+FFFD` replacement characters mid-emoji.
#[derive(Clone, Debug, Default)]
pub struct TokenStreamDecoder {
    pending: Vec<u8>,
    skip_special: bool,
}

impl TokenStreamDecoder {
    pub fn new(skip_special: bool) -> Self {
        Self {
            pending: Vec::with_capacity(16),
            skip_special,
        }
    }

    pub fn push_token(&mut self, tokenizer: &BpeTokenizer, token_id: u32) -> String {
        if (token_id as usize) >= tokenizer.vocab_size() {
            return String::new();
        }
        if self.skip_special && tokenizer.is_special_token(token_id) {
            return String::new();
        }
        self.pending
            .extend_from_slice(tokenizer.decode_raw_bytes(token_id));
        self.drain_valid_utf8()
    }

    fn drain_valid_utf8(&mut self) -> String {
        let mut emitted = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    emitted.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid_len = err.valid_up_to();
                    if valid_len > 0 {
                        let valid_str = std::str::from_utf8(&self.pending[..valid_len])
                            .expect("valid_up_to prefix guaranteed UTF-8");
                        emitted.push_str(valid_str);
                        self.pending.drain(..valid_len);
                    }
                    match err.error_len() {
                        None => {
                            // Incomplete trailing 2/3/4-byte UTF-8 codepoint prefix: wait for next token!
                            break;
                        }
                        Some(bad_len) => {
                            // Genuine invalid byte sequence (not an incomplete prefix)
                            emitted.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..bad_len);
                        }
                    }
                }
            }
        }
        emitted
    }

    /// Flushes any remaining buffered bytes at end-of-stream.
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Deterministic `VocabularyBuilder` constructing BPE vocabularies (`vocab_size = 512..=4096`)
/// with 256 byte fallbacks, control tokens, and rich JSON / English / code / emoji subword merges.
pub struct VocabularyBuilder;

impl VocabularyBuilder {
    pub const SPECIAL_TOKENS: [&'static str; 5] = [
        "<|begin_of_text|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<eos>",
    ];

    pub fn build_synthetic(target_vocab_size: usize) -> Result<BpeTokenizer, FrontendError> {
        let clamped_size = target_vocab_size.clamp(261, 8192);
        let seed_corpus: &[&str] = &[
            r#"{"status": "ok", "code": 200, "items": [{"id": 1, "name": "alpha", "active": true, "score": -3.14}, {"id": 2, "name": "beta", "active": false, "score": 42.5}], "meta": null}"#,
            r#"{"tool": "search_docs", "arguments": {"query": "PagedAttention RadixCache Speculative", "limit": 10, "strict": true}}"#,
            "The quick brown fox jumps over the lazy dog. High-throughput LLM inference serving engine in Rust with SIMD Q4_0 and Q8_0 kernels! ",
            "fn forward_step(tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, Error> { return Ok(logits); }\n",
            "Streaming UTF-8 emojis 🚀🔥✨🎯🧠 and multilingual text 你好世界 こんにちは 안녕하세요 café naïve résumé!",
            "0123456789 10 20 50 100 256 512 1024 2048 4096 -1 -42 0.5 0.99 1e-5 true false null ",
        ];
        let byte_slices: Vec<&[u8]> = seed_corpus.iter().map(|s| s.as_bytes()).collect();
        Self::train_from_corpus(&byte_slices, clamped_size)
    }

    pub fn train_from_corpus(
        corpus: &[&[u8]],
        target_vocab_size: usize,
    ) -> Result<BpeTokenizer, FrontendError> {
        let min_vocab = 256 + Self::SPECIAL_TOKENS.len();
        if target_vocab_size < min_vocab {
            return Err(FrontendError::Tokenizer(format!(
                "target_vocab_size {target_vocab_size} must be >= {min_vocab}"
            )));
        }

        let (byte_to_char, _) = gpt2_byte_to_unicode();
        let mut vocab_strings = Vec::with_capacity(target_vocab_size);
        let mut is_special = Vec::with_capacity(target_vocab_size);

        // IDs 0..256: exact 256 byte tokens
        for ch in byte_to_char {
            vocab_strings.push(ch.to_string());
            is_special.push(false);
        }

        // IDs 256..261: canonical special tokens (`<eos>` at index 260 unless appended at end)
        let bos_id = Some(vocab_strings.len() as u32);
        for &sp in &Self::SPECIAL_TOKENS {
            vocab_strings.push(sp.to_string());
            is_special.push(true);
        }
        let eos_id = (vocab_strings.len() - 1) as u32;

        // Convert corpus chunks into mutable token ID sequences (with amplified repeats for rich merges)
        let mut words: Vec<Vec<u32>> = Vec::new();
        for chunk in corpus {
            // Split softly after spaces/newlines GPT-style so merges stay within coherent words/punctuation
            let mut cur = Vec::new();
            for &b in *chunk {
                cur.push(b as u32);
                if b == b' ' || b == b'\n' || cur.len() >= 24 {
                    words.push(cur);
                    cur = Vec::new();
                }
            }
            if !cur.is_empty() {
                words.push(cur);
            }
        }

        let mut merges = Vec::new();
        while vocab_strings.len() < target_vocab_size {
            let mut pair_counts: HashMap<(u32, u32), u32> = HashMap::new();
            for w in &words {
                for pair in w.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_insert(0) += 1;
                }
            }

            // Pick highest-count pair with deterministic lowest `(left, right)` tie-break!
            let best = pair_counts
                .into_iter()
                .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                    count_a.cmp(count_b).then_with(|| pair_b.cmp(pair_a))
                });

            let Some(((left_id, right_id), count)) = best else {
                break;
            };
            if count < 1 {
                break;
            }

            let new_id = vocab_strings.len() as u32;
            let left_str = vocab_strings[left_id as usize].clone();
            let right_str = vocab_strings[right_id as usize].clone();
            vocab_strings.push(format!("{left_str}{right_str}"));
            is_special.push(false);
            merges.push((left_str, right_str));

            for w in &mut words {
                if w.len() < 2 {
                    continue;
                }
                let mut i = 0usize;
                let mut next_w = Vec::with_capacity(w.len());
                while i < w.len() {
                    if i + 1 < w.len() && w[i] == left_id && w[i + 1] == right_id {
                        next_w.push(new_id);
                        i += 2;
                    } else {
                        next_w.push(w[i]);
                        i += 1;
                    }
                }
                *w = next_w;
            }
        }

        // If corpus exhausted before `target_vocab_size`, pad deterministically with 2-byte ASCII pairs
        let mut p_left = b'a' as u32;
        let mut p_right = b'a' as u32;
        let mut existing: std::collections::HashSet<String> =
            vocab_strings.iter().cloned().collect();
        while vocab_strings.len() < target_vocab_size {
            let ls = vocab_strings[p_left as usize].clone();
            let rs = vocab_strings[p_right as usize].clone();
            let cand = format!("{ls}{rs}");
            if existing.insert(cand.clone()) {
                vocab_strings.push(cand);
                is_special.push(false);
                merges.push((ls, rs));
            }
            p_right += 1;
            if p_right >= 127 {
                p_right = 32;
                p_left = (p_left + 1) % 256;
            }
        }

        BpeTokenizer::from_parts(vocab_strings, is_special, merges, bos_id, eos_id)
    }
}
