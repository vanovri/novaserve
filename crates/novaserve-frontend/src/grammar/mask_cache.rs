//! Vocabulary-Wide Token Bitmask Index (`AllowedTokenMask`, `GrammarMaskCache`, and
//! stateful `GrammarConstraintState` with deterministic jump-token fast-forwarding).

use crate::bpe::BpeTokenizer;
use crate::error::FrontendError;
use crate::grammar::dfa::{ByteDfa, DEAD_STATE};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Packed `Vec<u64>` vocabulary bitset ($\lceil V / 64 \rceil$ words) enabling word-level
/// `< 5 µs` logit masking.
#[derive(Clone, Debug)]
pub struct AllowedTokenMask {
    words: Vec<u64>,
    vocab_size: usize,
    allowed_count: u32,
    eos_allowed: bool,
    single_forced_token: Option<u32>,
}

impl AllowedTokenMask {
    pub fn all_allowed(vocab_size: usize, eos_allowed: bool) -> Self {
        let num_words = vocab_size.div_ceil(64);
        let mut words = vec![u64::MAX; num_words];
        let rem = vocab_size % 64;
        if rem != 0 {
            words[num_words - 1] = (1u64 << rem) - 1;
        }
        Self {
            words,
            vocab_size,
            allowed_count: vocab_size as u32,
            eos_allowed,
            single_forced_token: None,
        }
    }

    pub fn from_words(
        words: Vec<u64>,
        vocab_size: usize,
        allowed_count: u32,
        eos_allowed: bool,
        single_forced_token: Option<u32>,
    ) -> Self {
        Self {
            words,
            vocab_size,
            allowed_count,
            eos_allowed,
            single_forced_token,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn allowed_count(&self) -> u32 {
        self.allowed_count
    }

    #[inline]
    pub fn is_eos_allowed(&self) -> bool {
        self.eos_allowed
    }

    #[inline]
    pub fn single_forced_token(&self) -> Option<u32> {
        self.single_forced_token
    }

    #[inline]
    pub fn is_allowed(&self, token_id: u32) -> bool {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return false;
        }
        let w = self.words[idx >> 6];
        ((w >> (idx & 63)) & 1) != 0
    }

    /// Sets disallowed tokens in `logits` to `f32::NEG_INFINITY` using 64-wide word scans.
    #[inline]
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        let limit = logits.len().min(self.vocab_size);
        let full_words = limit >> 6;

        for (wi, &mask_word) in self.words[..full_words].iter().enumerate() {
            let base = wi << 6;
            if mask_word == u64::MAX {
                continue;
            }
            if mask_word == 0 {
                logits[base..base + 64].fill(f32::NEG_INFINITY);
                continue;
            }
            let chunk = &mut logits[base..base + 64];
            for (bit, slot) in chunk.iter_mut().enumerate() {
                if ((mask_word >> bit) & 1) == 0 {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }

        let tail_base = full_words << 6;
        if tail_base < limit {
            let mask_word = self.words[full_words];
            for (bit, slot) in logits[tail_base..limit].iter_mut().enumerate() {
                if ((mask_word >> bit) & 1) == 0 {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }

        if logits.len() > self.vocab_size {
            logits[self.vocab_size..].fill(f32::NEG_INFINITY);
        }
    }

    /// Returns up to `limit` allowed token IDs for Studio / TUI grammar inspection.
    pub fn sample_allowed_ids(&self, limit: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(limit);
        for (wi, &word) in self.words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let tid = (wi * 64 + bit) as u32;
                if (tid as usize) < self.vocab_size {
                    out.push(tid);
                    if out.len() >= limit {
                        return out;
                    }
                }
                w &= w - 1;
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
struct CachedStateEntry {
    mask: Arc<AllowedTokenMask>,
    next_state_for_single: Option<(u32, u32)>,
}

/// Thread-safe vocabulary bitmask cache lazily indexing `ByteDfa` states against a `BpeTokenizer`.
pub struct GrammarMaskCache {
    dfa: Arc<ByteDfa>,
    vocab_size: usize,
    eos_token_id: u32,
    state_masks: RwLock<HashMap<u32, CachedStateEntry>>,
}

impl GrammarMaskCache {
    pub fn new(dfa: Arc<ByteDfa>, tokenizer: &BpeTokenizer) -> Self {
        Self {
            dfa,
            vocab_size: tokenizer.vocab_size(),
            eos_token_id: tokenizer.eos_token_id(),
            state_masks: RwLock::new(HashMap::new()),
        }
    }

    #[inline]
    pub fn dfa(&self) -> &ByteDfa {
        &self.dfa
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.dfa.start_state()
    }

    #[inline]
    pub fn is_accepting(&self, state: u32) -> bool {
        self.dfa.is_accepting(state)
    }

    fn compute_entry(
        &self,
        state: u32,
        tokenizer: &BpeTokenizer,
    ) -> Result<CachedStateEntry, FrontendError> {
        if (state as usize) >= self.dfa.num_states() {
            return Err(FrontendError::InvalidState(state));
        }

        let num_words = self.vocab_size.div_ceil(64);
        let mut words = vec![0u64; num_words];
        let mut allowed_count = 0u32;
        let mut last_non_eos_tok = None;
        let mut last_non_eos_next_state = DEAD_STATE;

        let eos_allowed = self.dfa.is_accepting(state);

        for tid_usize in 0..self.vocab_size {
            let tid = tid_usize as u32;
            if tid == self.eos_token_id {
                if eos_allowed {
                    words[tid_usize >> 6] |= 1u64 << (tid_usize & 63);
                    allowed_count += 1;
                }
                continue;
            }
            if tokenizer.is_special_token(tid) {
                continue;
            }
            let raw = tokenizer.decode_raw_bytes(tid);
            if raw.is_empty() {
                continue;
            }
            if let Some(nxt_state) = self.dfa.advance_bytes(state, raw) {
                words[tid_usize >> 6] |= 1u64 << (tid_usize & 63);
                allowed_count += 1;
                last_non_eos_tok = Some(tid);
                last_non_eos_next_state = nxt_state;
            }
        }

        let (single_forced, next_pair) = if allowed_count == 1 && !eos_allowed {
            let tok = last_non_eos_tok.expect("1 allowed non-eos token");
            (Some(tok), Some((tok, last_non_eos_next_state)))
        } else {
            (None, None)
        };

        let mask = Arc::new(AllowedTokenMask::from_words(
            words,
            self.vocab_size,
            allowed_count,
            eos_allowed,
            single_forced,
        ));
        Ok(CachedStateEntry {
            mask,
            next_state_for_single: next_pair,
        })
    }

    fn get_or_compute(
        &self,
        state: u32,
        tokenizer: &BpeTokenizer,
    ) -> Result<CachedStateEntry, FrontendError> {
        if let Some(hit) = self.state_masks.read().get(&state) {
            return Ok(hit.clone());
        }
        let computed = self.compute_entry(state, tokenizer)?;
        let mut guard = self.state_masks.write();
        let entry = guard.entry(state).or_insert_with(|| computed.clone());
        Ok(entry.clone())
    }

    /// Returns the cached packed `AllowedTokenMask` for `state`.
    pub fn mask_for_state(
        &self,
        state: u32,
        tokenizer: &BpeTokenizer,
    ) -> Result<Arc<AllowedTokenMask>, FrontendError> {
        Ok(self.get_or_compute(state, tokenizer)?.mask)
    }

    /// Returns the chain of deterministic jump tokens starting at `state` whenever `state`
    /// permits strictly 1 non-EOS token (stopping safely before accepting states or cycles).
    pub fn forced_tokens(
        &self,
        mut state: u32,
        tokenizer: &BpeTokenizer,
    ) -> Result<Vec<u32>, FrontendError> {
        let mut chain = Vec::new();
        let mut visited = HashSet::new();

        while visited.insert(state) {
            let entry = self.get_or_compute(state, tokenizer)?;
            if let Some((tok, next_state)) = entry.next_state_for_single {
                chain.push(tok);
                state = next_state;
                if chain.len() >= 64 {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(chain)
    }

    /// Advances DFA `state` by consuming `token_id`. Returns `None` if `token_id` violates the grammar.
    pub fn advance_token(
        &self,
        state: u32,
        tokenizer: &BpeTokenizer,
        token_id: u32,
    ) -> Option<u32> {
        if token_id == self.eos_token_id {
            return if self.dfa.is_accepting(state) {
                Some(state)
            } else {
                None
            };
        }
        if tokenizer.is_special_token(token_id) {
            return None;
        }
        let raw = tokenizer.decode_raw_bytes(token_id);
        if raw.is_empty() {
            return None;
        }
        self.dfa.advance_bytes(state, raw)
    }
}

/// Per-sequence active grammar state tracker used during Continuous Batching & Speculative Decoding.
#[derive(Clone)]
pub struct GrammarMatcher {
    cache: Arc<GrammarMaskCache>,
    state: u32,
    terminated: bool,
}

impl GrammarMatcher {
    pub fn new(cache: Arc<GrammarMaskCache>) -> Self {
        let start = cache.start_state();
        Self {
            cache,
            state: start,
            terminated: false,
        }
    }

    #[inline]
    pub fn state(&self) -> u32 {
        self.state
    }

    #[inline]
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    #[inline]
    pub fn is_accepting(&self) -> bool {
        self.cache.is_accepting(self.state)
    }

    pub fn current_mask(
        &self,
        tokenizer: &BpeTokenizer,
    ) -> Result<Arc<AllowedTokenMask>, FrontendError> {
        self.cache.mask_for_state(self.state, tokenizer)
    }

    pub fn forced_tokens(&self, tokenizer: &BpeTokenizer) -> Result<Vec<u32>, FrontendError> {
        if self.terminated {
            return Ok(Vec::new());
        }
        self.cache.forced_tokens(self.state, tokenizer)
    }

    pub fn accept_token(&mut self, tokenizer: &BpeTokenizer, token_id: u32) -> bool {
        if self.terminated {
            return false;
        }
        if token_id == tokenizer.eos_token_id() {
            if self.cache.is_accepting(self.state) {
                self.terminated = true;
                return true;
            }
            return false;
        }
        if let Some(nxt) = self.cache.advance_token(self.state, tokenizer, token_id) {
            self.state = nxt;
            true
        } else {
            false
        }
    }

    pub fn cache(&self) -> &Arc<GrammarMaskCache> {
        &self.cache
    }
}
