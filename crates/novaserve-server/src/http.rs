use crate::api_types::{ChatCompletionRequest, CompletionRequest, ModelCard, ModelsListResponse};
use crate::engine_actor::{EngineActorHandle, StreamEndpointKind, UnaryResponsePayload};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hyper::body::{Body as HyperBody, Frame};
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc::UnboundedReceiver;

struct SseAbortGuard {
    req_id: u64,
    engine: EngineActorHandle,
    completed_cleanly: bool,
}

impl Drop for SseAbortGuard {
    fn drop(&mut self) {
        if !self.completed_cleanly {
            self.engine.abort_request(self.req_id);
        }
    }
}

/// Zero-dependency SSE HTTP streaming body (`hyper::body::Body`) that instantaneously
/// sends `EngineCommand::Abort` to `ServingEngine` if the client disconnects mid-stream!
struct SseChannelBody {
    rx: UnboundedReceiver<String>,
    abort_guard: SseAbortGuard,
}

impl HyperBody for SseChannelBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(chunk)) => {
                if chunk.starts_with("data: [DONE]") {
                    self.abort_guard.completed_cleanly = true;
                }
                Poll::Ready(Some(Ok(Frame::data(Bytes::from(chunk)))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

fn build_sse_response(
    req_id: u64,
    rx: UnboundedReceiver<String>,
    engine: EngineActorHandle,
) -> Response {
    let sse_body = SseChannelBody {
        rx,
        abort_guard: SseAbortGuard {
            req_id,
            engine,
            completed_cleanly: false,
        },
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("x-novaserve-request-id", req_id.to_string())
        .body(Body::new(sse_body))
        .expect("valid SSE HTTP response headers")
}

fn error_response(status: StatusCode, msg: impl Into<String>) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": msg.into(),
            "type": "invalid_request_error",
        }
    });
    (status, Json(body)).into_response()
}

pub fn build_router(engine: EngineActorHandle) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/completions", post(handle_completions))
        .route("/v1/models", get(handle_list_models))
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_prometheus_metrics))
        .route("/v1/telemetry/snapshot", get(handle_telemetry_snapshot))
        .route("/v1/admin/clear_radix", post(handle_clear_radix))
        .with_state(engine)
}

async fn handle_chat_completions(
    State(engine): State<EngineActorHandle>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let is_stream = req.stream;
    let sub = match engine.build_chat_submission(&req) {
        Ok(s) => s,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, msg),
    };

    if is_stream {
        match engine.execute_stream(sub, StreamEndpointKind::Chat).await {
            Ok((req_id, rx)) => build_sse_response(req_id, rx, engine),
            Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
    } else {
        match engine.execute_unary(sub, StreamEndpointKind::Chat).await {
            Ok(UnaryResponsePayload::Chat(resp)) => (StatusCode::OK, Json(resp)).into_response(),
            Ok(UnaryResponsePayload::Completion(_)) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "mismatched unary payload",
            ),
            Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
    }
}

async fn handle_completions(
    State(engine): State<EngineActorHandle>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let is_stream = req.stream;
    let sub = match engine.build_completion_submission(&req) {
        Ok(s) => s,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, msg),
    };

    if is_stream {
        match engine
            .execute_stream(sub, StreamEndpointKind::Completion)
            .await
        {
            Ok((req_id, rx)) => build_sse_response(req_id, rx, engine),
            Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
    } else {
        match engine
            .execute_unary(sub, StreamEndpointKind::Completion)
            .await
        {
            Ok(UnaryResponsePayload::Completion(resp)) => {
                (StatusCode::OK, Json(resp)).into_response()
            }
            Ok(UnaryResponsePayload::Chat(_)) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "mismatched unary payload",
            ),
            Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
    }
}

async fn handle_list_models(State(engine): State<EngineActorHandle>) -> impl IntoResponse {
    let resp = ModelsListResponse {
        object: "list".into(),
        data: vec![ModelCard {
            id: engine.model_name().to_string(),
            object: "model".into(),
            created: 1_760_000_000,
            owned_by: "novaserve".into(),
        }],
    };
    Json(resp)
}

async fn handle_health(State(engine): State<EngineActorHandle>) -> impl IntoResponse {
    let snap = engine.snapshot_telemetry();
    Json(serde_json::json!({
        "status": "ok",
        "model": snap.model_name,
        "free_kv_blocks": snap.free_kv_blocks,
        "total_kv_blocks": snap.total_kv_blocks,
        "running_groups": snap.running_groups,
    }))
}

async fn handle_telemetry_snapshot(State(engine): State<EngineActorHandle>) -> impl IntoResponse {
    Json(engine.snapshot_telemetry())
}

async fn handle_clear_radix(State(engine): State<EngineActorHandle>) -> Response {
    match engine.clear_radix_cache().await {
        Ok((free, total)) => Json(serde_json::json!({
            "status": "cleared",
            "free_kv_blocks": free,
            "total_kv_blocks": total,
        }))
        .into_response(),
        Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, msg),
    }
}

async fn handle_prometheus_metrics(State(engine): State<EngineActorHandle>) -> Response {
    let s = engine.snapshot_telemetry();
    let out = format!(
        "# HELP novaserve_kv_blocks_total Total PagedAttention physical KV blocks\n\
         # TYPE novaserve_kv_blocks_total gauge\n\
         novaserve_kv_blocks_total {}\n\
         # HELP novaserve_kv_blocks_free Free PagedAttention physical KV blocks\n\
         # TYPE novaserve_kv_blocks_free gauge\n\
         novaserve_kv_blocks_free {}\n\
         # HELP novaserve_radix_cached_blocks Blocks cached in SGLang-style RadixCache\n\
         # TYPE novaserve_radix_cached_blocks gauge\n\
         novaserve_radix_cached_blocks {}\n\
         # HELP novaserve_radix_hit_rate_pct Radix prefix cache block hit rate percentage\n\
         # TYPE novaserve_radix_hit_rate_pct gauge\n\
         novaserve_radix_hit_rate_pct {:.4}\n\
         # HELP novaserve_tokens_generated_total Total generated output tokens\n\
         # TYPE novaserve_tokens_generated_total counter\n\
         novaserve_tokens_generated_total {}\n\
         # HELP novaserve_tokens_per_sec Real-time output throughput in tokens/sec\n\
         # TYPE novaserve_tokens_per_sec gauge\n\
         novaserve_tokens_per_sec {:.4}\n\
         # HELP novaserve_ttft_p50_ms Time-To-First-Token p50 latency in ms\n\
         # TYPE novaserve_ttft_p50_ms gauge\n\
         novaserve_ttft_p50_ms {:.4}\n\
         # HELP novaserve_ttft_p99_ms Time-To-First-Token p99 latency in ms\n\
         # TYPE novaserve_ttft_p99_ms gauge\n\
         novaserve_ttft_p99_ms {:.4}\n\
         # HELP novaserve_itl_p50_ms Inter-Token Latency p50 in ms\n\
         # TYPE novaserve_itl_p50_ms gauge\n\
         novaserve_itl_p50_ms {:.4}\n\
         # HELP novaserve_itl_p99_ms Inter-Token Latency p99 in ms\n\
         # TYPE novaserve_itl_p99_ms gauge\n\
         novaserve_itl_p99_ms {:.4}\n\
         # HELP novaserve_speculative_alpha_pct Leviathan speculative acceptance rate percentage\n\
         # TYPE novaserve_speculative_alpha_pct gauge\n\
         novaserve_speculative_alpha_pct {:.4}\n\
         # HELP novaserve_requests_completed_total Total completed requests\n\
         # TYPE novaserve_requests_completed_total counter\n\
         novaserve_requests_completed_total {}\n\
         # HELP novaserve_requests_aborted_total Total aborted or client-disconnected requests\n\
         # TYPE novaserve_requests_aborted_total counter\n\
         novaserve_requests_aborted_total {}\n",
        s.total_kv_blocks,
        s.free_kv_blocks,
        s.radix_cached_blocks,
        s.radix_hit_rate_pct,
        s.tokens_generated_total,
        s.tokens_per_sec,
        s.ttft_p50_ms,
        s.ttft_p99_ms,
        s.itl_p50_ms,
        s.itl_p99_ms,
        s.speculative_alpha_pct,
        s.completed_requests_total,
        s.aborted_requests_total,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(out))
        .expect("valid metrics response")
}
