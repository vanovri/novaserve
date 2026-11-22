use novaserve_kv::{PhysicalBlockSnapshot, RadixNodeSnapshot};
use novaserve_server::{ActiveLaneTelemetry, TelemetrySnapshot};
use novaserve_tui::{render_cockpit, CockpitState};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn tui_headless_120x40_renders_all_four_quadrants() {
    let blocks: Vec<PhysicalBlockSnapshot> = (0..64u32)
        .map(|i| PhysicalBlockSnapshot {
            block_id: i,
            ref_count: match i % 4 {
                0 => 0,
                1 => 1,
                2 => 3,
                _ => 1,
            },
            is_radix_cached: i % 4 >= 2,
            content_hash: if i % 4 == 0 {
                None
            } else {
                Some(0xCAFE_0000 + i as u64)
            },
        })
        .collect();

    let snap = TelemetrySnapshot {
        step_index: 128,
        model_name: "novaserve-tiny-16m-q8_0".into(),
        total_kv_blocks: 64,
        free_kv_blocks: 16,
        radix_cached_blocks: 32,
        total_swap_blocks: 64,
        used_swap_blocks: 4,
        external_fragmentation_pct: 0.0,
        waiting_groups: 2,
        running_groups: 4,
        swapped_groups: 1,
        completed_requests_total: 92,
        aborted_requests_total: 3,
        tokens_generated_total: 4820,
        tokens_per_sec: 1485.6,
        ttft_p50_ms: 3.42,
        ttft_p99_ms: 9.81,
        itl_p50_ms: 0.74,
        itl_p99_ms: 2.15,
        radix_hit_rate_pct: 88.5,
        radix_evicted_blocks: 14,
        radix_hit_sparkline: vec![60, 72, 80, 84, 86, 88, 89],
        speculative_alpha_pct: 76.4,
        active_spec_k: 4,
        blocks,
        radix_tree: vec![RadixNodeSnapshot {
            node_id: 1,
            parent_id: Some(0),
            token_preview: vec![12, 44, 91, 103],
            total_edge_tokens: 64,
            block_ids: vec![2, 6, 10, 14],
            min_ref_count: 3,
            last_access_tick: 55,
        }],
        active_lanes: vec![
            ActiveLaneTelemetry {
                req_id: 41,
                child_idx: 0,
                status: "decode".into(),
                prompt_tokens: 64,
                kv_tokens: 82,
                generated_tokens: 19,
                max_tokens: 64,
                text_preview: "{\"tool\":\"search_docs\",\"limit\":10}".into(),
                last_spec_burst: 4,
                has_json_schema: true,
                dfa_state: Some(17),
                total_dfa_states: Some(48),
            },
            ActiveLaneTelemetry {
                req_id: 42,
                child_idx: 0,
                status: "prefill".into(),
                prompt_tokens: 256,
                kv_tokens: 128,
                generated_tokens: 0,
                max_tokens: 32,
                text_preview: String::new(),
                last_spec_burst: 0,
                has_json_schema: false,
                dfa_state: None,
                total_dfa_states: None,
            },
        ],
    };

    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    let state = CockpitState::new(snap, "HEADLESS-TEST");
    terminal.draw(|f| render_cockpit(f, &state)).unwrap();

    let buf = terminal.backend().buffer();
    let mut rendered = String::with_capacity(120 * 41);
    for y in 0..40u16 {
        for x in 0..120u16 {
            rendered.push_str(buf[(x, y)].symbol());
        }
        rendered.push('\n');
    }

    assert!(rendered.contains("NOVASERVE-TOP"));
    assert!(rendered.contains("[Q1] Physical KV-Cache Block Map"));
    assert!(rendered.contains("[Q2] Live Continuous Batch & Chunked-Prefill Lanes"));
    assert!(rendered.contains("[Q3] Radix Prefix Tree & Grammar FSM Inspector"));
    assert!(rendered.contains("[Q4] SLA Latency & Speculative Gauges"));
    assert!(rendered.contains("[+4]"));
    assert!(rendered.contains("[DFA#17]"));
    assert!(rendered.contains("[PREFILL  50%]"));
    assert!(rendered.contains("1485.6 tok/s"));
}
