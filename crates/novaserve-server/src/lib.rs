#![deny(unsafe_code)]

//! `novaserve-server`: High-Concurrency Async OpenAI-Compatible HTTP & SSE Inference Gateway
//! (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`,
//! `/v1/telemetry/snapshot`) with instantaneous TCP disconnect sequence cancellation.

pub mod api_types;
pub mod engine_actor;
pub mod http;

pub use api_types::*;
pub use engine_actor::{EngineActorGuard, EngineActorHandle};
pub use http::build_router;
