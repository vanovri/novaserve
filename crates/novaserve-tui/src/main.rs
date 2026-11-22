#![deny(unsafe_code)]

use clap::Parser;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use novaserve_server::engine_actor::StreamEndpointKind;
use novaserve_server::{
    ChatCompletionRequest, ChatMessage, EngineActorHandle, ResponseFormatSpec, TelemetrySnapshot,
};
use novaserve_tensor::DType;
use novaserve_tui::{render_cockpit, CockpitState};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use serde_json::json;
use std::collections::BTreeMap;
use std::io;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Parser, Debug)]
#[command(
    name = "novaserve-top",
    about = "NovaServe 4-Quadrant Live ML Systems Terminal Cockpit"
)]
struct TopArgs {
    /// Launch embedded NovaServe engine with continuous synthetic multi-turn, CoW, JSON-schema & speculative traffic.
    #[arg(long)]
    demo: bool,

    /// Remote NovaServe HTTP endpoint URL when not running in `--demo` mode.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,

    /// UI refresh interval in milliseconds.
    #[arg(long, default_value_t = 80)]
    interval_ms: u64,
}

async fn fetch_remote_snapshot(base_url: &str) -> Option<TelemetrySnapshot> {
    let stripped = base_url
        .strip_prefix("http://")
        .unwrap_or(base_url)
        .trim_end_matches('/');
    let (host_port, _) = stripped.split_once('/').unwrap_or((stripped, ""));
    let mut stream = TcpStream::connect(host_port).await.ok()?;
    let req = format!(
        "GET /v1/telemetry/snapshot HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).await.ok()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.ok()?;
    let text = String::from_utf8_lossy(&raw);
    let (_, body) = text.split_once("\r\n\r\n")?;
    serde_json::from_str(body).ok()
}

fn spawn_demo_traffic_generator(handle: EngineActorHandle) {
    tokio::spawn(async move {
        let sys_prompt = "You are NovaServe autonomous SRE assistant monitoring PagedAttention KV slabs, SGLang Radix prefix trees, and Leviathan speculative decoding.";
        let schema = json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["scale_kv", "pin_prefix", "tune_k"] },
                "blocks": { "type": "integer" },
                "healthy": { "type": "boolean" }
            },
            "required": ["action", "blocks", "healthy"],
            "additionalProperties": false
        });

        let mut wave = 0u64;
        loop {
            let snap = handle.snapshot_telemetry();
            if snap.running_groups < 6 && snap.waiting_groups < 4 {
                let is_schema = wave.is_multiple_of(3);
                let req = ChatCompletionRequest {
                    model: handle.model_name().to_string(),
                    messages: vec![
                        ChatMessage {
                            role: "system".into(),
                            content: sys_prompt.into(),
                        },
                        ChatMessage {
                            role: "user".into(),
                            content: format!("Telemetry wave #{wave}: inspect cluster state"),
                        },
                    ],
                    max_tokens: if is_schema { 64 } else { 40 },
                    temperature: Some(0.8),
                    top_p: Some(0.95),
                    top_k: Some(24),
                    min_p: Some(0.02),
                    n: if wave.is_multiple_of(4) && !is_schema {
                        2
                    } else {
                        1
                    },
                    stream: false,
                    seed: Some(10_000 + wave),
                    stop: None,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repetition_penalty: 1.0,
                    logprobs: false,
                    top_logprobs: None,
                    logit_bias: BTreeMap::new(),
                    response_format: if is_schema {
                        Some(ResponseFormatSpec {
                            kind: "json_schema".into(),
                            json_schema: Some(schema.clone()),
                        })
                    } else {
                        None
                    },
                    use_speculative: !is_schema,
                };
                if let Ok(sub) = handle.build_chat_submission(&req) {
                    let h2 = handle.clone();
                    tokio::spawn(async move {
                        let _ = h2.execute_unary(sub, StreamEndpointKind::Chat).await;
                    });
                }
                wave += 1;
            }
            tokio::time::sleep(Duration::from_millis(60)).await;
        }
    });
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = TopArgs::parse();

    let demo_guard = if args.demo {
        let g = EngineActorHandle::spawn_demo(DType::Q8_0, 96, 4)?;
        spawn_demo_traffic_generator(g.handle.clone());
        Some(g)
    } else {
        None
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mode_label = if args.demo {
        "EMBEDDED DEMO ENGINE".to_string()
    } else {
        format!("LIVE {}", args.url)
    };

    let res = run_ui_loop(
        &mut terminal,
        &args,
        demo_guard.as_ref().map(|g| &g.handle),
        &mode_label,
    )
    .await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

async fn run_ui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    args: &TopArgs,
    demo_handle: Option<&EngineActorHandle>,
    mode_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_snap = TelemetrySnapshot::default();
    loop {
        if let Some(h) = demo_handle {
            last_snap = h.snapshot_telemetry();
        } else if let Some(remote) = fetch_remote_snapshot(&args.url).await {
            last_snap = remote;
        }

        let state = CockpitState::new(last_snap.clone(), mode_label);
        terminal.draw(|frame| render_cockpit(frame, &state))?;

        if event::poll(Duration::from_millis(args.interval_ms))? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
            }
        }
    }
    Ok(())
}
