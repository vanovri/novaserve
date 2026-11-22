# Module M02 Design Specification — `crates/novaserve-frontend`

## 1. Architectural Scope
`novaserve-frontend` implements the lexical and syntactic boundary of **NovaServe**:
1. **GPT-4 / Llama-3 Compatible Byte-Level BPE Tokenizer (`bpe.rs`)** with $O(N \log N)$ priority-queue merge encoding, exact special-token splitting, GGUF + `tokenizer.json` loaders, `VocabularyBuilder`, and multi-token UTF-8 buffering (`TokenStreamDecoder`).
2. **Regex & Draft-2020-12 JSON-Schema to Minimized Byte-Level DFA Compiler (`grammar/dfa.rs`, `grammar/json_schema.rs`)**.
3. **Vocabulary-Wide Packed Bitmask Cache (`grammar/mask_cache.rs`)** with deterministic fast-forward (`forced_tokens()`) and `< 5 µs` word-level logit masking.

---

## 2. Byte-Level BPE & Streaming UTF-8 Detokenizer

### 2.1 Reversible 256-Byte Table & Merge Priority Queue
- Implements the canonical 256-entry GPT-2 / Llama-3 byte-to-unicode bijection mapping printable ASCII/Latin-1 bytes directly and remapping control/whitespace bytes (`0x00..=0x20`, `0x7F..=0xA0`, `0xAD`) to `U+0100..U+0143` so every arbitrary byte sequence is representable as valid UTF-8 subword strings.
- **Encoding Algorithm ($O(N \log N)$):**
  1. Split input around registered special/control tokens (`<|begin_of_text|>`, `<|im_start|>`, `<|im_end|>`, `<|eot_id|>`, `<eos>`) using longest-match scanning.
  2. Represent each byte of an ordinary text span as a node in an index-linked doubly-linked list `(sym_id, prev, next, version)` and push adjacent mergeable pairs `(rank, left_idx, left_ver, right_ver)` into a min-heap (`BinaryHeap<Reverse<...>>`).
  3. Pop lowest-rank pair, skip stale entries whose node versions changed, merge in $O(1)$, and push newly formed left/right neighbor pairs.

### 2.2 `TokenStreamDecoder` (Zero Broken `U+FFFD` Mid-Sequence)
- Appends raw decoded token bytes into a small `pending: Vec<u8>` buffer.
- Calls `std::str::from_utf8(&pending)`:
  - Emits the valid prefix `&pending[..valid_up_to]`.
  - When `err.error_len().is_none()`, the remaining `1..=3` tail bytes are a valid prefix of an in-flight multi-byte UTF-8 character (e.g., 4-byte emoji split across 3 BPE tokens) and remain buffered until subsequent continuation tokens arrive.

---

## 3. Regex & JSON-Schema Byte-DFA Compiler

### 3.1 Why Byte-Level (`[StateId; 256]`) DFA?
Byte-level BPE tokens frequently straddle UTF-8 codepoint boundaries or bundle multiple punctuation characters (`{"`, `": "`, `"}]`). Compiling both regular expressions and JSON schemas directly to a **minimized Byte-Level DFA** makes per-token state validation a branch-minimal byte-table lookup with zero UTF-8 re-parsing overhead.

### 3.2 Compilation Pipeline
1. **Regex Parser (`grammar/dfa.rs`):** Recursive-descent AST supporting alternation `|`, concatenation, quantifiers `*`, `+`, `?`, `{m}`, `{m,n}`, `{m,}`, character classes `[a-z0-9_]`, negated ASCII/UTF-8 classes `[^...]`, dot `.`, shorthand `\d`, `\w`, `\s`, and escaped literals.
2. **Thompson NFA Construction:** Explicit byte-range transitions `(lo..=hi -> target)` plus $\varepsilon$-edges.
3. **Subset Construction & Hopcroft Minimization:**
   - Computes $\varepsilon$-closures using bitsets to construct deterministic `[StateId; 256]` transitions.
   - Prunes dead (non-coaccessible) states that cannot reach any accepting state—critical so `AllowedTokenMask` immediately rejects any token entering a dead syntactic alley!
   - Runs Hopcroft partition refinement to minimize active DFA states.

### 3.3 Exact RFC 8259 / RFC 3629 JSON-Schema Lowering (`grammar/json_schema.rs`)
- **Strings:** Exact RFC 3629 UTF-8 safe byte graph excluding raw `0x00..=0x1F`, `"`, `\`, paired with legal JSON escapes (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, and BMP non-surrogate `\u` hex escapes preventing lone-surrogate parse failures in `serde_json`).
- **Integers & Numbers:** Exact RFC 8259 non-leading-zero `-?(0|[1-9][0-9]{0,14})` and bounded exponent floats preventing `f64::INFINITY` overflow.
- **Objects (`properties`, `required`, `additionalProperties: false`):** Linear $2P$-state `(prop_index, has_emitted_field)` spine guaranteeing required fields are always present, optional fields may be omitted, and commas `,` appear strictly between emitted key-value pairs.
- **Arrays (`items`, `minItems`, `maxItems`), `enum`, `boolean`, `null`, and Nested Schemas:** Recursively lowered into exact NFA sub-fragments prior to global determinization and dead-state pruning.

---

## 4. Vocabulary Bitmask Cache (`AllowedTokenMask`)
- Packed `Vec<u64>` of length $\lceil V / 64 \rceil$ per reachable `(dfa_state)` lazily memoized in a `parking_lot::RwLock<HashMap<StateId, Arc<StateMaskEntry>>>`.
- Special/control tokens other than `eos_token_id` are masked `0`; `eos_token_id` is `1` iff `dfa.is_accepting(state)`.
- **`forced_tokens()`:** Pre-identifies states where exactly 1 token is allowed (and `eos` is disallowed), enabling deterministic structural token fast-forwarding.
- **`apply_to_logits(&self, logits: &mut [f32])`:** Word-wise 64-token mask scan setting disallowed entries to `f32::NEG_INFINITY` in `< 1 µs` for `V = 4,096` and `< 5 µs` for `V = 128,256`.
