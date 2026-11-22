#![deny(unsafe_code)]

//! `novaserve-sampler`: Composable Vectorized 9-Stage Logits Post-Processing & Sampling Pipeline.

pub mod pipeline;

pub use pipeline::{
    sample_from_dist, SamplerError, SamplerScratch, SamplingPipeline, SparseOrDenseProbDist,
};
