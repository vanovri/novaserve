#![deny(unsafe_code)]

//! `novaserve-oracle`: Naive `f64` Reference Engine, 5,000-Scenario Continuous-Batching /
//! Preemption Chaos Fuzzer, and 3-Regime Speculative Decoding Statistical Proof Suite.

pub mod batch_fuzzer;
pub mod eager_ref;
pub mod spec_proof;

pub use batch_fuzzer::{run_chaos_batch_equivalence_campaign, ChaosFuzzReport};
pub use eager_ref::EagerReferenceEngine;
pub use spec_proof::{run_3_regime_speculative_proof, RegimeProofStats, SpeculativeProofReport};
