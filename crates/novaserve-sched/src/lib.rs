#![deny(unsafe_code)]

//! `novaserve-sched`: Iteration-Level Continuous Batching Scheduler, Preemption
//! (`Swap` & `Recompute`), and Leviathan Exact Speculative Decoding Engine.

pub mod error;
pub mod scheduler;
pub mod speculative;
pub mod types;

pub use error::SchedError;
pub use scheduler::ServingEngine;
pub use speculative::{
    leviathan_rejection_sample, sample_residual_distribution, AdaptiveLookaheadController,
    RejectionOutcome,
};
pub use types::{
    collect_top_logprobs, sample_categorical, ActiveSequenceSnapshot, BuiltinSampler,
    CompletedRequest, FinishReason, GrammarCandidateLogit, GrammarStepXray, PreemptionMode,
    RequestParams, SampledToken, SamplerStrategy, SamplingConfig, SchedulerConfig, SequenceOutput,
    SequenceStatus, SequenceStepDelta, SpeculativeTrace, StepSummary, TokenLogprob,
};
