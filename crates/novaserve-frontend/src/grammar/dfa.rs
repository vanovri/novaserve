//! Byte-level Regular Expression -> Thompson NFA -> Subset Construction ->
//! Dead-State Pruned + Hopcroft-Minimized `ByteDfa` (`100%` safe Rust).

use crate::error::FrontendError;
use std::collections::{BTreeMap, HashMap, VecDeque};

pub const DEAD_STATE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct NfaFrag {
    pub start: usize,
    pub out: usize,
}

#[derive(Clone, Debug, Default)]
struct NfaState {
    ranges: Vec<(u8, u8, usize)>,
    eps: Vec<usize>,
}

/// Composable Thompson Byte-NFA builder shared by both the Regex compiler and JSON-Schema compiler.
#[derive(Clone, Debug, Default)]
pub struct ByteNfaBuilder {
    states: Vec<NfaState>,
}

impl ByteNfaBuilder {
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    #[inline]
    pub fn add_state(&mut self) -> usize {
        let id = self.states.len();
        self.states.push(NfaState::default());
        id
    }

    #[inline]
    pub fn add_eps(&mut self, from: usize, to: usize) {
        self.states[from].eps.push(to);
    }

    #[inline]
    pub fn add_range(&mut self, from: usize, lo: u8, hi: u8, to: usize) {
        if lo <= hi {
            self.states[from].ranges.push((lo, hi, to));
        }
    }

    pub fn empty_frag(&mut self) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();
        self.add_eps(s, e);
        NfaFrag { start: s, out: e }
    }

    pub fn lit_byte(&mut self, b: u8) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();
        self.add_range(s, b, b, e);
        NfaFrag { start: s, out: e }
    }

    pub fn lit_bytes(&mut self, bytes: &[u8]) -> NfaFrag {
        if bytes.is_empty() {
            return self.empty_frag();
        }
        let s = self.add_state();
        let mut cur = s;
        for &b in bytes {
            let nxt = self.add_state();
            self.add_range(cur, b, b, nxt);
            cur = nxt;
        }
        NfaFrag { start: s, out: cur }
    }

    pub fn byte_ranges(&mut self, ranges: &[(u8, u8)]) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();
        for &(lo, hi) in ranges {
            self.add_range(s, lo, hi, e);
        }
        NfaFrag { start: s, out: e }
    }

    pub fn concat(&mut self, frags: &[NfaFrag]) -> NfaFrag {
        if frags.is_empty() {
            return self.empty_frag();
        }
        for w in frags.windows(2) {
            self.add_eps(w[0].out, w[1].start);
        }
        NfaFrag {
            start: frags[0].start,
            out: frags[frags.len() - 1].out,
        }
    }

    pub fn alt(&mut self, frags: &[NfaFrag]) -> NfaFrag {
        if frags.is_empty() {
            let s = self.add_state();
            let e = self.add_state();
            return NfaFrag { start: s, out: e };
        }
        if frags.len() == 1 {
            return frags[0];
        }
        let s = self.add_state();
        let e = self.add_state();
        for f in frags {
            self.add_eps(s, f.start);
            self.add_eps(f.out, e);
        }
        NfaFrag { start: s, out: e }
    }

    pub fn optional(&mut self, frag: NfaFrag) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();
        self.add_eps(s, frag.start);
        self.add_eps(s, e);
        self.add_eps(frag.out, e);
        NfaFrag { start: s, out: e }
    }

    pub fn zero_or_more(&mut self, frag: NfaFrag) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();
        self.add_eps(s, frag.start);
        self.add_eps(s, e);
        self.add_eps(frag.out, frag.start);
        self.add_eps(frag.out, e);
        NfaFrag { start: s, out: e }
    }

    pub fn one_or_more(&mut self, frag: NfaFrag) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();
        self.add_eps(s, frag.start);
        self.add_eps(frag.out, frag.start);
        self.add_eps(frag.out, e);
        NfaFrag { start: s, out: e }
    }

    /// Exact RFC 3629 UTF-8 codepoint fragment (for `.` excluding `\n` when `allow_newline == false`).
    pub fn utf8_any_char(&mut self, allow_newline: bool) -> NfaFrag {
        let s = self.add_state();
        let e = self.add_state();

        // 1-byte ASCII
        if allow_newline {
            self.add_range(s, 0x00, 0x7F, e);
        } else {
            self.add_range(s, 0x00, 0x09, e);
            self.add_range(s, 0x0B, 0x7F, e);
        }

        // 2-byte UTF-8: C2..DF -> 80..BF
        let u2 = self.add_state();
        self.add_range(s, 0xC2, 0xDF, u2);
        self.add_range(u2, 0x80, 0xBF, e);

        // 3-byte UTF-8:
        // E0 -> A0..BF -> 80..BF
        let u3_e0 = self.add_state();
        let u3_tail = self.add_state();
        self.add_range(s, 0xE0, 0xE0, u3_e0);
        self.add_range(u3_e0, 0xA0, 0xBF, u3_tail);
        // E1..EC | EE..EF -> 80..BF -> 80..BF
        let u3_norm = self.add_state();
        self.add_range(s, 0xE1, 0xEC, u3_norm);
        self.add_range(s, 0xEE, 0xEF, u3_norm);
        self.add_range(u3_norm, 0x80, 0xBF, u3_tail);
        // ED -> 80..9F -> 80..BF (excludes surrogates D800..DFFF)
        let u3_ed = self.add_state();
        self.add_range(s, 0xED, 0xED, u3_ed);
        self.add_range(u3_ed, 0x80, 0x9F, u3_tail);
        self.add_range(u3_tail, 0x80, 0xBF, e);

        // 4-byte UTF-8:
        let u4_tail2 = self.add_state();
        let u4_tail1 = self.add_state();
        // F0 -> 90..BF -> u4_tail2
        let u4_f0 = self.add_state();
        self.add_range(s, 0xF0, 0xF0, u4_f0);
        self.add_range(u4_f0, 0x90, 0xBF, u4_tail2);
        // F1..F3 -> 80..BF -> u4_tail2
        let u4_mid = self.add_state();
        self.add_range(s, 0xF1, 0xF3, u4_mid);
        self.add_range(u4_mid, 0x80, 0xBF, u4_tail2);
        // F4 -> 80..8F -> u4_tail2
        let u4_f4 = self.add_state();
        self.add_range(s, 0xF4, 0xF4, u4_f4);
        self.add_range(u4_f4, 0x80, 0x8F, u4_tail2);

        self.add_range(u4_tail2, 0x80, 0xBF, u4_tail1);
        self.add_range(u4_tail1, 0x80, 0xBF, e);

        NfaFrag { start: s, out: e }
    }

    fn epsilon_closure(&self, seeds: &[usize], words_len: usize) -> Vec<u64> {
        let mut bits = vec![0u64; words_len];
        let mut stack: Vec<usize> = Vec::with_capacity(seeds.len() * 2);
        for &s in seeds {
            let w = s / 64;
            let mask = 1u64 << (s % 64);
            if (bits[w] & mask) == 0 {
                bits[w] |= mask;
                stack.push(s);
            }
        }
        while let Some(u) = stack.pop() {
            for &v in &self.states[u].eps {
                let w = v / 64;
                let mask = 1u64 << (v % 64);
                if (bits[w] & mask) == 0 {
                    bits[w] |= mask;
                    stack.push(v);
                }
            }
        }
        bits
    }

    /// Determinizes `(root.start -> root.out)` via subset construction, prunes dead non-coaccessible
    /// states, and minimizes via Hopcroft partition refinement.
    pub fn build_minimized_dfa(&self, root: NfaFrag) -> Result<ByteDfa, FrontendError> {
        let n_nfa = self.states.len();
        if n_nfa == 0 {
            return Err(FrontendError::RegexSyntax("empty NFA".into()));
        }
        let words_len = n_nfa.div_ceil(64);
        let accept_w = root.out / 64;
        let accept_mask = 1u64 << (root.out % 64);

        let start_set = self.epsilon_closure(&[root.start], words_len);
        let mut set_to_id: HashMap<Vec<u64>, u32> = HashMap::new();
        let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
        let mut raw_trans: Vec<[u32; 256]> = Vec::new();
        let mut raw_accept: Vec<bool> = Vec::new();

        set_to_id.insert(start_set.clone(), 0);
        queue.push_back(start_set);

        const MAX_DFA_STATES: usize = 65_536;
        let mut active_states_buf = Vec::with_capacity(64);
        let mut next_seeds = Vec::with_capacity(32);

        while let Some(cur_bits) = queue.pop_front() {
            let is_acc = (cur_bits[accept_w] & accept_mask) != 0;
            raw_accept.push(is_acc);
            let cur_dfa_idx = raw_trans.len();
            raw_trans.push([DEAD_STATE; 256]);

            // Collect active NFA states in `cur_bits`
            active_states_buf.clear();
            for (wi, &word) in cur_bits.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    active_states_buf.push(wi * 64 + bit);
                    w &= w - 1;
                }
            }

            // Group bytes that share identical NFA range transitions for fast subset construction!
            // Mark boundary split points in `0..=256`.
            let mut splits = [false; 257];
            splits[0] = true;
            splits[256] = true;
            for &u in &active_states_buf {
                for &(lo, hi, _) in &self.states[u].ranges {
                    splits[lo as usize] = true;
                    splits[(hi as usize) + 1] = true;
                }
            }

            let mut b_start = 0usize;
            while b_start < 256 {
                let mut b_end = b_start + 1;
                while b_end < 256 && !splits[b_end] {
                    b_end += 1;
                }
                let rep_byte = b_start as u8;
                next_seeds.clear();
                for &u in &active_states_buf {
                    for &(lo, hi, target) in &self.states[u].ranges {
                        if lo <= rep_byte && rep_byte <= hi {
                            next_seeds.push(target);
                        }
                    }
                }

                if !next_seeds.is_empty() {
                    let closure = self.epsilon_closure(&next_seeds, words_len);
                    let target_id = if let Some(&existing) = set_to_id.get(&closure) {
                        existing
                    } else {
                        if set_to_id.len() >= MAX_DFA_STATES {
                            return Err(FrontendError::SchemaCompile(format!(
                                "compiled DFA exceeded {MAX_DFA_STATES} states"
                            )));
                        }
                        let new_id = set_to_id.len() as u32;
                        set_to_id.insert(closure.clone(), new_id);
                        queue.push_back(closure);
                        new_id
                    };
                    for slot in &mut raw_trans[cur_dfa_idx][b_start..b_end] {
                        *slot = target_id;
                    }
                }
                b_start = b_end;
            }
        }

        // Step 2: Co-accessibility pruning (prune states that cannot reach ANY accepting state!)
        let num_raw = raw_trans.len();
        let mut rev_edges: Vec<Vec<usize>> = vec![Vec::new(); num_raw];
        for (u, row) in raw_trans.iter().enumerate() {
            let mut prev = DEAD_STATE;
            for &v in row {
                if v != DEAD_STATE && v != prev {
                    rev_edges[v as usize].push(u);
                    prev = v;
                }
            }
        }

        let mut live = vec![false; num_raw];
        let mut rev_q = VecDeque::new();
        for (s, &acc) in raw_accept.iter().enumerate() {
            if acc {
                live[s] = true;
                rev_q.push_back(s);
            }
        }
        while let Some(v) = rev_q.pop_front() {
            for &u in &rev_edges[v] {
                if !live[u] {
                    live[u] = true;
                    rev_q.push_back(u);
                }
            }
        }

        if !live[0] {
            // Language is empty (start state cannot reach any accepting state)
            return Ok(ByteDfa {
                transitions: vec![[DEAD_STATE; 256]],
                accepting: vec![false],
                start_state: DEAD_STATE,
            });
        }

        // Remap live states densely keeping start state at index 0
        let mut live_remap = vec![DEAD_STATE; num_raw];
        let mut pruned_trans = Vec::new();
        let mut pruned_accept = Vec::new();
        for s in 0..num_raw {
            if live[s] {
                live_remap[s] = pruned_trans.len() as u32;
                pruned_trans.push([DEAD_STATE; 256]);
                pruned_accept.push(raw_accept[s]);
            }
        }
        for s in 0..num_raw {
            if live[s] {
                let ns = live_remap[s] as usize;
                for b in 0..256 {
                    let t = raw_trans[s][b];
                    if t != DEAD_STATE {
                        pruned_trans[ns][b] = live_remap[t as usize];
                    }
                }
            }
        }

        // Step 3: Hopcroft / Moore Partition-Refinement Minimization
        Ok(minimize_dfa(&pruned_trans, &pruned_accept))
    }
}

fn minimize_dfa(trans: &[[u32; 256]], accept: &[bool]) -> ByteDfa {
    let n = trans.len();
    if n <= 1 {
        return ByteDfa {
            transitions: trans.to_vec(),
            accepting: accept.to_vec(),
            start_state: 0,
        };
    }

    // Initial partition: 0 = non-accepting (if any), 1 = accepting
    let mut class_of: Vec<u32> = vec![0; n];
    let has_non_acc = accept.iter().any(|&a| !a);
    let mut num_classes = if has_non_acc {
        for (i, &a) in accept.iter().enumerate() {
            class_of[i] = u32::from(a);
        }
        2u32
    } else {
        1u32
    };

    let mut sig_map: BTreeMap<(u32, [u32; 256]), u32> = BTreeMap::new();
    let mut new_class = vec![0u32; n];

    loop {
        sig_map.clear();
        let mut next_id = 0u32;
        for s in 0..n {
            let mut mapped_row = [DEAD_STATE; 256];
            for b in 0..256 {
                let t = trans[s][b];
                if t != DEAD_STATE {
                    mapped_row[b] = class_of[t as usize];
                }
            }
            let key = (class_of[s], mapped_row);
            let cid = *sig_map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            new_class[s] = cid;
        }
        if next_id == num_classes {
            break;
        }
        num_classes = next_id;
        class_of.copy_from_slice(&new_class);
    }

    // Reorder equivalence classes so `class_of[0]` (start state) maps to `0`!
    let mut canonical_id = vec![DEAD_STATE; num_classes as usize];
    let mut next_canon = 0u32;
    let mut min_trans = Vec::with_capacity(num_classes as usize);
    let mut min_accept = Vec::with_capacity(num_classes as usize);
    let mut rep_state_for_canon = Vec::with_capacity(num_classes as usize);

    for s in 0..n {
        let c = class_of[s] as usize;
        if canonical_id[c] == DEAD_STATE {
            canonical_id[c] = next_canon;
            next_canon += 1;
            rep_state_for_canon.push(s);
            min_trans.push([DEAD_STATE; 256]);
            min_accept.push(accept[s]);
        }
    }

    for (canon_idx, &rep_s) in rep_state_for_canon.iter().enumerate() {
        for b in 0..256 {
            let t = trans[rep_s][b];
            if t != DEAD_STATE {
                min_trans[canon_idx][b] = canonical_id[class_of[t as usize] as usize];
            }
        }
    }

    ByteDfa {
        transitions: min_trans,
        accepting: min_accept,
        start_state: 0,
    }
}

/// Minimized deterministic byte-level automaton (`[StateId; 256]` per live state).
#[derive(Clone, Debug)]
pub struct ByteDfa {
    transitions: Vec<[u32; 256]>,
    accepting: Vec<bool>,
    start_state: u32,
}

impl ByteDfa {
    /// Compiles a regular expression string into a minimized `ByteDfa`.
    pub fn from_regex(pattern: &str) -> Result<Self, FrontendError> {
        let ast = RegexParser::new(pattern).parse_expr()?;
        let mut builder = ByteNfaBuilder::new();
        let frag = compile_ast_to_nfa(&ast, &mut builder)?;
        builder.build_minimized_dfa(frag)
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    #[inline]
    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }

    #[inline]
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting.get(state as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub fn next_state(&self, state: u32, byte: u8) -> u32 {
        let Some(row) = self.transitions.get(state as usize) else {
            return DEAD_STATE;
        };
        row[byte as usize]
    }

    /// Advances `state` across all bytes of `slice`, returning `None` immediately on hitting `DEAD_STATE`.
    #[inline]
    pub fn advance_bytes(&self, mut state: u32, slice: &[u8]) -> Option<u32> {
        if state == DEAD_STATE {
            return None;
        }
        for &b in slice {
            let row = self.transitions.get(state as usize)?;
            state = row[b as usize];
            if state == DEAD_STATE {
                return None;
            }
        }
        Some(state)
    }

    /// Full exact match check on `input`.
    pub fn is_match(&self, input: &[u8]) -> bool {
        if let Some(end_state) = self.advance_bytes(self.start_state, input) {
            self.is_accepting(end_state)
        } else {
            false
        }
    }
}

// ---------------- Recursive-Descent Regex AST & Compiler ----------------

#[derive(Clone, Debug)]
enum RegexAst {
    Empty,
    LiteralBytes(Vec<u8>),
    Dot,
    ByteClass(Vec<(u8, u8)>),
    NegatedAsciiOrUtf8(Vec<(u8, u8)>),
    Concat(Vec<RegexAst>),
    Alt(Vec<RegexAst>),
    Repeat {
        inner: Box<RegexAst>,
        min: usize,
        max: Option<usize>,
    },
}

struct RegexParser<'a> {
    chars: Vec<char>,
    pos: usize,
    _src: &'a str,
}

impl<'a> RegexParser<'a> {
    fn new(src: &'a str) -> Self {
        // Strip optional outer `^...$` anchors when compiling full-string grammars
        let trimmed = src.strip_prefix('^').unwrap_or(src);
        let trimmed = if trimmed.ends_with('$') && !trimmed.ends_with(r"\$") {
            &trimmed[..trimmed.len() - 1]
        } else {
            trimmed
        };
        Self {
            chars: trimmed.chars().collect(),
            pos: 0,
            _src: src,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next_ch(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn parse_expr(&mut self) -> Result<RegexAst, FrontendError> {
        let node = self.parse_alternation()?;
        if self.pos < self.chars.len() {
            return Err(FrontendError::RegexSyntax(format!(
                "unexpected trailing character '{}' at pos {}",
                self.chars[self.pos], self.pos
            )));
        }
        Ok(node)
    }

    fn parse_alternation(&mut self) -> Result<RegexAst, FrontendError> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some('|') {
            self.pos += 1;
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(RegexAst::Alt(branches))
        }
    }

    fn parse_concat(&mut self) -> Result<RegexAst, FrontendError> {
        let mut seq = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            seq.push(self.parse_quantified()?);
        }
        match seq.len() {
            0 => Ok(RegexAst::Empty),
            1 => Ok(seq.pop().unwrap()),
            _ => Ok(RegexAst::Concat(seq)),
        }
    }

    fn parse_quantified(&mut self) -> Result<RegexAst, FrontendError> {
        let mut atom = self.parse_atom()?;
        while let Some(c) = self.peek() {
            match c {
                '*' => {
                    self.pos += 1;
                    atom = RegexAst::Repeat {
                        inner: Box::new(atom),
                        min: 0,
                        max: None,
                    };
                }
                '+' => {
                    self.pos += 1;
                    atom = RegexAst::Repeat {
                        inner: Box::new(atom),
                        min: 1,
                        max: None,
                    };
                }
                '?' => {
                    self.pos += 1;
                    atom = RegexAst::Repeat {
                        inner: Box::new(atom),
                        min: 0,
                        max: Some(1),
                    };
                }
                '{' => {
                    let (min, max) = self.parse_braced_quantifier()?;
                    atom = RegexAst::Repeat {
                        inner: Box::new(atom),
                        min,
                        max,
                    };
                }
                _ => break,
            }
        }
        Ok(atom)
    }

    fn parse_braced_quantifier(&mut self) -> Result<(usize, Option<usize>), FrontendError> {
        self.pos += 1; // '{'
        let min = self.parse_uint()?;
        let max = match self.peek() {
            Some('}') => {
                self.pos += 1;
                Some(min)
            }
            Some(',') => {
                self.pos += 1;
                if self.peek() == Some('}') {
                    self.pos += 1;
                    None
                } else {
                    let m = self.parse_uint()?;
                    if self.next_ch() != Some('}') {
                        return Err(FrontendError::RegexSyntax("unclosed '{m,n}'".into()));
                    }
                    if m < min {
                        return Err(FrontendError::RegexSyntax(format!(
                            "invalid quantifier range {{{min},{m}}}"
                        )));
                    }
                    Some(m)
                }
            }
            _ => {
                return Err(FrontendError::RegexSyntax(
                    "malformed '{...}' quantifier".into(),
                ));
            }
        };
        if min > 256 || max.unwrap_or(0) > 256 {
            return Err(FrontendError::RegexSyntax(
                "quantifier bound > 256 not permitted".into(),
            ));
        }
        Ok((min, max))
    }

    fn parse_uint(&mut self) -> Result<usize, FrontendError> {
        let start = self.pos;
        let mut val = 0usize;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                self.pos += 1;
                val = val
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(d as usize))
                    .ok_or_else(|| FrontendError::RegexSyntax("quantifier overflow".into()))?;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(FrontendError::RegexSyntax(
                "expected integer in '{...}' quantifier".into(),
            ));
        }
        Ok(val)
    }

    fn parse_atom(&mut self) -> Result<RegexAst, FrontendError> {
        let c = self
            .next_ch()
            .ok_or_else(|| FrontendError::RegexSyntax("unexpected end of regex".into()))?;
        match c {
            '(' => {
                // Support `(?:...)` non-capturing group syntax transparently
                if self.peek() == Some('?') && self.chars.get(self.pos + 1) == Some(&':') {
                    self.pos += 2;
                }
                let inner = self.parse_alternation()?;
                if self.next_ch() != Some(')') {
                    return Err(FrontendError::RegexSyntax("unclosed '(' group".into()));
                }
                Ok(inner)
            }
            '.' => Ok(RegexAst::Dot),
            '[' => self.parse_char_class(),
            '\\' => self.parse_escape(),
            '*' | '+' | '?' | '{' | ')' => Err(FrontendError::RegexSyntax(format!(
                "dangling metacharacter '{c}'"
            ))),
            ch => {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                Ok(RegexAst::LiteralBytes(s.as_bytes().to_vec()))
            }
        }
    }

    fn parse_escape(&mut self) -> Result<RegexAst, FrontendError> {
        let esc = self
            .next_ch()
            .ok_or_else(|| FrontendError::RegexSyntax("trailing backslash".into()))?;
        match esc {
            'd' => Ok(RegexAst::ByteClass(vec![(b'0', b'9')])),
            'D' => Ok(RegexAst::NegatedAsciiOrUtf8(vec![(b'0', b'9')])),
            'w' => Ok(RegexAst::ByteClass(vec![
                (b'0', b'9'),
                (b'A', b'Z'),
                (b'_', b'_'),
                (b'a', b'z'),
            ])),
            'W' => Ok(RegexAst::NegatedAsciiOrUtf8(vec![
                (b'0', b'9'),
                (b'A', b'Z'),
                (b'_', b'_'),
                (b'a', b'z'),
            ])),
            's' => Ok(RegexAst::ByteClass(vec![(b'\t', b'\r'), (b' ', b' ')])),
            'S' => Ok(RegexAst::NegatedAsciiOrUtf8(vec![
                (b'\t', b'\r'),
                (b' ', b' '),
            ])),
            'n' => Ok(RegexAst::LiteralBytes(vec![b'\n'])),
            'r' => Ok(RegexAst::LiteralBytes(vec![b'\r'])),
            't' => Ok(RegexAst::LiteralBytes(vec![b'\t'])),
            'x' => {
                let h1 = self
                    .next_ch()
                    .and_then(|c| c.to_digit(16))
                    .ok_or_else(|| FrontendError::RegexSyntax("bad \\x escape".into()))?;
                let h2 = self
                    .next_ch()
                    .and_then(|c| c.to_digit(16))
                    .ok_or_else(|| FrontendError::RegexSyntax("bad \\x escape".into()))?;
                Ok(RegexAst::LiteralBytes(vec![((h1 << 4) | h2) as u8]))
            }
            other => {
                let mut buf = [0u8; 4];
                let s = other.encode_utf8(&mut buf);
                Ok(RegexAst::LiteralBytes(s.as_bytes().to_vec()))
            }
        }
    }

    fn parse_class_char(&mut self) -> Result<char, FrontendError> {
        let c = self
            .next_ch()
            .ok_or_else(|| FrontendError::RegexSyntax("unclosed '[' character class".into()))?;
        if c == '\\' {
            let esc = self
                .next_ch()
                .ok_or_else(|| FrontendError::RegexSyntax("unclosed escape in '['".into()))?;
            Ok(match esc {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            })
        } else {
            Ok(c)
        }
    }

    fn parse_char_class(&mut self) -> Result<RegexAst, FrontendError> {
        let negated = if self.peek() == Some('^') {
            self.pos += 1;
            true
        } else {
            false
        };

        let mut ascii_ranges: Vec<(u8, u8)> = Vec::new();
        let mut multi_utf8_alts: Vec<RegexAst> = Vec::new();
        let mut first = true;

        while first || self.peek() != Some(']') {
            first = false;
            if self.peek().is_none() {
                return Err(FrontendError::RegexSyntax("unclosed '[' class".into()));
            }
            // Handle shorthand `\d`, `\w`, `\s` inside `[...]`
            if self.peek() == Some('\\') {
                match self.chars.get(self.pos + 1).copied() {
                    Some('d') => {
                        self.pos += 2;
                        ascii_ranges.push((b'0', b'9'));
                        continue;
                    }
                    Some('w') => {
                        self.pos += 2;
                        ascii_ranges.extend_from_slice(&[
                            (b'0', b'9'),
                            (b'A', b'Z'),
                            (b'_', b'_'),
                            (b'a', b'z'),
                        ]);
                        continue;
                    }
                    Some('s') => {
                        self.pos += 2;
                        ascii_ranges.extend_from_slice(&[(b'\t', b'\r'), (b' ', b' ')]);
                        continue;
                    }
                    _ => {}
                }
            }

            let lo_ch = self.parse_class_char()?;
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.pos += 1; // '-'
                let hi_ch = self.parse_class_char()?;
                if lo_ch > hi_ch {
                    return Err(FrontendError::RegexSyntax(format!(
                        "inverted character range '{lo_ch}-{hi_ch}'"
                    )));
                }
                if lo_ch.is_ascii() && hi_ch.is_ascii() {
                    ascii_ranges.push((lo_ch as u8, hi_ch as u8));
                } else {
                    for cp in lo_ch..=hi_ch {
                        let mut b = [0u8; 4];
                        multi_utf8_alts.push(RegexAst::LiteralBytes(
                            cp.encode_utf8(&mut b).as_bytes().to_vec(),
                        ));
                    }
                }
            } else if lo_ch.is_ascii() {
                ascii_ranges.push((lo_ch as u8, lo_ch as u8));
            } else {
                let mut b = [0u8; 4];
                multi_utf8_alts.push(RegexAst::LiteralBytes(
                    lo_ch.encode_utf8(&mut b).as_bytes().to_vec(),
                ));
            }
        }
        self.pos += 1; // consume ']'

        if negated {
            Ok(RegexAst::NegatedAsciiOrUtf8(ascii_ranges))
        } else if multi_utf8_alts.is_empty() {
            Ok(RegexAst::ByteClass(ascii_ranges))
        } else {
            if !ascii_ranges.is_empty() {
                multi_utf8_alts.push(RegexAst::ByteClass(ascii_ranges));
            }
            Ok(RegexAst::Alt(multi_utf8_alts))
        }
    }
}

fn compile_ast_to_nfa(
    ast: &RegexAst,
    builder: &mut ByteNfaBuilder,
) -> Result<NfaFrag, FrontendError> {
    match ast {
        RegexAst::Empty => Ok(builder.empty_frag()),
        RegexAst::LiteralBytes(bytes) => Ok(builder.lit_bytes(bytes)),
        RegexAst::Dot => Ok(builder.utf8_any_char(false)),
        RegexAst::ByteClass(ranges) => Ok(builder.byte_ranges(ranges)),
        RegexAst::NegatedAsciiOrUtf8(excluded) => {
            let mut forbidden = [false; 128];
            for &(lo, hi) in excluded {
                for b in lo..=hi.min(127) {
                    forbidden[b as usize] = true;
                }
            }
            let mut allowed_ascii = Vec::new();
            let mut b = 0usize;
            while b < 128 {
                if !forbidden[b] {
                    let start = b as u8;
                    while b + 1 < 128 && !forbidden[b + 1] {
                        b += 1;
                    }
                    allowed_ascii.push((start, b as u8));
                }
                b += 1;
            }
            let ascii_frag = builder.byte_ranges(&allowed_ascii);
            // Also permit valid multi-byte UTF-8 non-ASCII characters (`>= U+0080`)
            let s = builder.add_state();
            let e = builder.add_state();
            builder.add_eps(s, ascii_frag.start);
            builder.add_eps(ascii_frag.out, e);

            let u2 = builder.add_state();
            builder.add_range(s, 0xC2, 0xDF, u2);
            builder.add_range(u2, 0x80, 0xBF, e);

            let u3_e0 = builder.add_state();
            let u3_tail = builder.add_state();
            builder.add_range(s, 0xE0, 0xE0, u3_e0);
            builder.add_range(u3_e0, 0xA0, 0xBF, u3_tail);
            let u3_norm = builder.add_state();
            builder.add_range(s, 0xE1, 0xEC, u3_norm);
            builder.add_range(s, 0xEE, 0xEF, u3_norm);
            builder.add_range(u3_norm, 0x80, 0xBF, u3_tail);
            let u3_ed = builder.add_state();
            builder.add_range(s, 0xED, 0xED, u3_ed);
            builder.add_range(u3_ed, 0x80, 0x9F, u3_tail);
            builder.add_range(u3_tail, 0x80, 0xBF, e);

            let u4_tail2 = builder.add_state();
            let u4_tail1 = builder.add_state();
            let u4_f0 = builder.add_state();
            builder.add_range(s, 0xF0, 0xF0, u4_f0);
            builder.add_range(u4_f0, 0x90, 0xBF, u4_tail2);
            let u4_mid = builder.add_state();
            builder.add_range(s, 0xF1, 0xF3, u4_mid);
            builder.add_range(u4_mid, 0x80, 0xBF, u4_tail2);
            let u4_f4 = builder.add_state();
            builder.add_range(s, 0xF4, 0xF4, u4_f4);
            builder.add_range(u4_f4, 0x80, 0x8F, u4_tail2);
            builder.add_range(u4_tail2, 0x80, 0xBF, u4_tail1);
            builder.add_range(u4_tail1, 0x80, 0xBF, e);

            Ok(NfaFrag { start: s, out: e })
        }
        RegexAst::Concat(items) => {
            let mut frags = Vec::with_capacity(items.len());
            for item in items {
                frags.push(compile_ast_to_nfa(item, builder)?);
            }
            Ok(builder.concat(&frags))
        }
        RegexAst::Alt(branches) => {
            let mut frags = Vec::with_capacity(branches.len());
            for br in branches {
                frags.push(compile_ast_to_nfa(br, builder)?);
            }
            Ok(builder.alt(&frags))
        }
        RegexAst::Repeat { inner, min, max } => match (*min, *max) {
            (0, None) => {
                let f = compile_ast_to_nfa(inner, builder)?;
                Ok(builder.zero_or_more(f))
            }
            (1, None) => {
                let f = compile_ast_to_nfa(inner, builder)?;
                Ok(builder.one_or_more(f))
            }
            (0, Some(1)) => {
                let f = compile_ast_to_nfa(inner, builder)?;
                Ok(builder.optional(f))
            }
            (m, None) => {
                let mut parts = Vec::with_capacity(m + 1);
                for _ in 0..m {
                    parts.push(compile_ast_to_nfa(inner, builder)?);
                }
                let tail = compile_ast_to_nfa(inner, builder)?;
                parts.push(builder.zero_or_more(tail));
                Ok(builder.concat(&parts))
            }
            (m, Some(n)) => {
                let mut parts = Vec::with_capacity(n);
                for _ in 0..m {
                    parts.push(compile_ast_to_nfa(inner, builder)?);
                }
                for _ in m..n {
                    let f = compile_ast_to_nfa(inner, builder)?;
                    parts.push(builder.optional(f));
                }
                Ok(builder.concat(&parts))
            }
        },
    }
}
