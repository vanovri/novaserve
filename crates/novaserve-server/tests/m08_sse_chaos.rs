use novaserve_frontend::validate_json_value;
use novaserve_model::{LlamaConfig, ModelFactory};
use novaserve_sched::SchedulerConfig;
use novaserve_server::{
    build_router, ChatCompletionChunk, ChatCompletionResponse, CompletionResponse,
    EngineActorHandle, ModelsListResponse, TelemetrySnapshot,
};
use novaserve_tensor::DType;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

async fn http_post_json(addr: std::net::SocketAddr, path: &str, payload: &Value) -> (u16, String) {
    let body = serde_json::to_string(payload).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    parse_raw_http_response(&raw)
}

async fn http_get(addr: std::net::SocketAddr, path: &str) -> (u16, String) {
    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    parse_raw_http_response(&raw)
}

fn parse_raw_http_response(raw: &[u8]) -> (u16, String) {
    let text = String::from_utf8_lossy(raw);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .expect("valid HTTP response header/body separator");
    let status_line = head.lines().next().unwrap();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();

    // Decode chunked transfer encoding if present
    let is_chunked = head.lines().any(|l| {
        l.to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
    });
    if !is_chunked {
        return (status, body.to_string());
    }

    let mut out = String::new();
    let mut rest = body;
    while !rest.is_empty() {
        let Some((len_hex, after_len)) = rest.split_once("\r\n") else {
            break;
        };
        let chunk_len = usize::from_str_radix(len_hex.trim(), 16).unwrap_or(0);
        if chunk_len == 0 {
            break;
        }
        out.push_str(&after_len[..chunk_len]);
        rest = &after_len[chunk_len + 2..];
    }
    (status, out)
}

async fn read_sse_chat_to_done(addr: std::net::SocketAddr, payload: &Value) -> String {
    let body = serde_json::to_string(payload).unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    // Skip HTTP headers
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.unwrap();
        assert!(n > 0, "unexpected EOF before SSE headers finished");
        if line == "\r\n" {
            break;
        }
    }

    let mut assembled = String::new();
    let mut saw_done = false;
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if let Some(data_payload) = trimmed.strip_prefix("data: ") {
            if data_payload == "[DONE]" {
                saw_done = true;
                break;
            }
            let chunk: ChatCompletionChunk = serde_json::from_str(data_payload).unwrap();
            for ch in chunk.choices {
                if let Some(c) = ch.delta.content {
                    assembled.push_str(&c);
                }
            }
        }
    }
    assert!(saw_done, "SSE stream must terminate with data: [DONE]");
    assembled
}

async fn sse_connect_and_drop_midstream(addr: std::net::SocketAddr, payload: &Value) {
    let body = serde_json::to_string(payload).unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let mut data_events_seen = 0usize;
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.unwrap();
        if n == 0 {
            break;
        }
        if line.trim_start().starts_with("data: ") {
            data_events_seen += 1;
            if data_events_seen >= 2 {
                // Intentionally sever TCP connection mid-generation!
                drop(reader);
                return;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn m08_64_client_concurrent_sse_schema_cow_and_disconnect_chaos() {
    let vocab_size = 512;
    let tok = Arc::new(novaserve_frontend::VocabularyBuilder::build_synthetic(vocab_size).unwrap());
    let target_cfg = LlamaConfig::micro_2m(vocab_size, DType::Q8_0);
    let draft_cfg = LlamaConfig::draft_500k(vocab_size, DType::F32);
    let (tm, dm) = ModelFactory::build_correlated_pair(target_cfg, draft_cfg, 2026_0864).unwrap();

    let total_blocks = 160usize;
    let sched_cfg = SchedulerConfig {
        max_num_batched_tokens: 64, // exercises Chunked Prefill under 64-client load!
        max_num_seqs: 24,
        num_physical_blocks: total_blocks,
        num_swap_blocks: 256,
        speculative_lookahead: 3,
        adaptive_speculative: true,
        ..SchedulerConfig::default()
    };

    let actor_guard =
        EngineActorHandle::spawn("novaserve-chaos-2m", tm, Some(dm), tok, sched_cfg).unwrap();
    let router = build_router(actor_guard.handle.clone());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Verify metadata endpoints before storm
    let (st_health, body_health) = http_get(addr, "/health").await;
    assert_eq!(st_health, 200);
    assert!(body_health.contains("\"status\":\"ok\""));

    let (st_models, body_models) = http_get(addr, "/v1/models").await;
    assert_eq!(st_models, 200);
    let models_list: ModelsListResponse = serde_json::from_str(&body_models).unwrap();
    assert_eq!(models_list.data[0].id, "novaserve-chaos-2m");

    let tool_schema = json!({
        "type": "object",
        "properties": {
            "tool": { "type": "string", "enum": ["search_docs", "query_kv"] },
            "limit": { "type": "integer" },
            "strict": { "type": "boolean" }
        },
        "required": ["tool", "limit", "strict"],
        "additionalProperties": false
    });

    let shared_sys = "You are NovaServe, a deterministic Rust LLM inference engine with PagedAttention and RadixCache.";

    // Spawn 64 concurrent async clients!
    let mut handles = Vec::with_capacity(64);
    for client_id in 0..64usize {
        let schema_clone = tool_schema.clone();
        handles.push(tokio::spawn(async move {
            match client_id % 4 {
                // Cohort 0 (16 clients): Streaming SSE Chat (`stream: true`)
                0 => {
                    let payload = json!({
                        "model": "novaserve-chaos-2m",
                        "messages": [
                            { "role": "system", "content": shared_sys },
                            { "role": "user", "content": format!("Stream client #{client_id} query") }
                        ],
                        "max_tokens": 18,
                        "temperature": 0.7,
                        "top_k": 16,
                        "seed": 1000 + client_id as u64,
                        "stream": true,
                        "use_speculative": true
                    });
                    let text = read_sse_chat_to_done(addr, &payload).await;
                    assert!(!text.is_empty());
                }
                // Cohort 1 (16 clients): CoW `n = 2` parallel completions + `logprobs`
                1 => {
                    if client_id % 8 == 1 {
                        let payload = json!({
                            "model": "novaserve-chaos-2m",
                            "messages": [
                                { "role": "system", "content": shared_sys },
                                { "role": "user", "content": format!("Parallel CoW #{client_id}") }
                            ],
                            "n": 2,
                            "max_tokens": 14,
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "logprobs": true,
                            "top_logprobs": 5,
                            "seed": 2000 + client_id as u64
                        });
                        let (status, body) =
                            http_post_json(addr, "/v1/chat/completions", &payload).await;
                        assert_eq!(status, 200, "unexpected error: {body}");
                        let resp: ChatCompletionResponse = serde_json::from_str(&body).unwrap();
                        assert_eq!(resp.choices.len(), 2);
                        assert!(resp.choices[0].logprobs.is_some());
                    } else {
                        let payload = json!({
                            "model": "novaserve-chaos-2m",
                            "prompt": format!("{shared_sys}\nCompletion #{client_id}:"),
                            "n": 2,
                            "max_tokens": 12,
                            "temperature": 0.6,
                            "min_p": 0.05,
                            "seed": 2500 + client_id as u64
                        });
                        let (status, body) =
                            http_post_json(addr, "/v1/completions", &payload).await;
                        assert_eq!(status, 200, "unexpected error: {body}");
                        let resp: CompletionResponse = serde_json::from_str(&body).unwrap();
                        assert_eq!(resp.choices.len(), 2);
                    }
                }
                // Cohort 2 (16 clients): Strict `json_schema` structured tool-call generation!
                2 => {
                    let is_sse = client_id % 8 == 2;
                    let payload = json!({
                        "model": "novaserve-chaos-2m",
                        "messages": [
                            { "role": "system", "content": shared_sys },
                            { "role": "user", "content": format!("Invoke tool for task #{client_id}") }
                        ],
                        "max_tokens": 96,
                        "temperature": 0.8,
                        "seed": 3000 + client_id as u64,
                        "stream": is_sse,
                        "use_speculative": false,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "tool_call",
                                "schema": schema_clone
                            }
                        }
                    });
                    let content = if is_sse {
                        read_sse_chat_to_done(addr, &payload).await
                    } else {
                        let (status, body) =
                            http_post_json(addr, "/v1/chat/completions", &payload).await;
                        assert_eq!(status, 200, "schema request failed: {body}");
                        let resp: ChatCompletionResponse = serde_json::from_str(&body).unwrap();
                        resp.choices[0].message.content.clone()
                    };
                    let parsed: Value = serde_json::from_str(&content).unwrap_or_else(|e| {
                        panic!("invalid JSON emitted by client #{client_id}: '{content}' ({e})")
                    });
                    validate_json_value(&parsed, &schema_clone, &schema_clone).unwrap_or_else(|e| {
                        panic!("schema violation on client #{client_id}: '{content}' ({e})")
                    });
                }
                // Cohort 3 (16 clients): Intentional mid-stream TCP socket disconnects!
                _ => {
                    let payload = json!({
                        "model": "novaserve-chaos-2m",
                        "messages": [
                            { "role": "user", "content": format!("Long generation drop #{client_id}") }
                        ],
                        "max_tokens": 256,
                        "temperature": 0.9,
                        "seed": 4000 + client_id as u64,
                        "stream": true,
                        "use_speculative": false
                    });
                    sse_connect_and_drop_midstream(addr, &payload).await;
                }
            }
        }));
    }

    tokio::time::timeout(Duration::from_secs(60), async {
        for h in handles {
            h.await.unwrap();
        }
    })
    .await
    .expect("64 concurrent clients must finish without hanging");

    // Wait briefly for the 16 severed TCP sockets' hyper Drop guards to drain completely
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let (_, snap_body) = http_get(addr, "/v1/telemetry/snapshot").await;
        let snap: TelemetrySnapshot = serde_json::from_str(&snap_body).unwrap();
        if snap.waiting_groups == 0 && snap.running_groups == 0 && snap.swapped_groups == 0 {
            assert!(
                snap.aborted_requests_total >= 1,
                "expected mid-stream disconnects to record aborted requests"
            );
            assert_eq!(
                snap.completed_requests_total, 48,
                "all 48 non-disconnected clients must complete cleanly"
            );
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "engine failed to drain disconnected streams within 5s"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // Check Prometheus `/metrics` endpoint
    let (st_prom, prom_body) = http_get(addr, "/metrics").await;
    assert_eq!(st_prom, 200);
    assert!(prom_body.contains("novaserve_tokens_generated_total"));

    // Evict unpinned RadixCache blocks and verify 100% physical KV & Swap block reclamation!
    let (st_clr, clr_body) = http_post_json(addr, "/v1/admin/clear_radix", &json!({})).await;
    assert_eq!(st_clr, 200, "clear_radix failed: {clr_body}");
    let (_, final_snap_body) = http_get(addr, "/v1/telemetry/snapshot").await;
    let final_snap: TelemetrySnapshot = serde_json::from_str(&final_snap_body).unwrap();

    assert_eq!(
        final_snap.free_kv_blocks, total_blocks,
        "KV block leak detected after 64-client disconnect chaos! free={} != total={}",
        final_snap.free_kv_blocks, total_blocks
    );
    assert_eq!(
        final_snap.used_swap_blocks, 0,
        "Swap block leak detected after 64-client chaos!"
    );

    server_task.abort();
}
