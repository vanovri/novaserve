#![deny(unsafe_code)]

//! `novaserve-frontend`: GPT-4 / Llama-3 Byte-Level BPE Tokenizer, streaming UTF-8
//! detokenizer, and Regex / Draft-2020-12 JSON-Schema -> Minimized Byte-DFA Token-Bitmask Engine.

pub mod bpe;
pub mod error;
pub mod grammar;

pub use bpe::{gpt2_byte_to_unicode, BpeTokenizer, TokenStreamDecoder, VocabularyBuilder};
pub use error::FrontendError;
pub use grammar::dfa::{ByteDfa, ByteNfaBuilder, NfaFrag, DEAD_STATE};
pub use grammar::json_schema::{compile_json_schema, compile_json_schema_str, validate_json_value};
pub use grammar::mask_cache::{AllowedTokenMask, GrammarMaskCache, GrammarMatcher};
