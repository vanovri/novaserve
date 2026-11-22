#![deny(unsafe_code)]

//! `novaserve-tui`: 4-Quadrant Terminal Serving Cockpit (`novaserve-top`) built on `ratatui`.

use novaserve_server::TelemetrySnapshot;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Wrap};
use ratatui::Frame;

pub struct CockpitState {
    pub snapshot: TelemetrySnapshot,
    pub mode_label: String,
}

impl CockpitState {
    pub fn new(snapshot: TelemetrySnapshot, mode_label: impl Into<String>) -> Self {
        Self {
            snapshot,
            mode_label: mode_label.into(),
        }
    }
}

pub fn render_cockpit(frame: &mut Frame, state: &CockpitState) {
    let area = frame.area();
    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Percentage(52),
            Constraint::Min(10),
        ])
        .split(area);

    render_header(frame, root_chunks[0], state);

    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(root_chunks[1]);

    let bottom_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(root_chunks[2]);

    render_kv_block_map(frame, top_row[0], &state.snapshot);
    render_batch_lanes(frame, top_row[1], &state.snapshot);
    render_radix_and_grammar(frame, bottom_row[0], &state.snapshot);
    render_sla_gauges(frame, bottom_row[1], &state.snapshot);
}

fn render_header(frame: &mut Frame, area: Rect, state: &CockpitState) {
    let s = &state.snapshot;
    let title_line = Line::from(vec![
        Span::styled(
            " NOVASERVE-TOP ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(&state.mode_label, Style::default().fg(Color::Yellow)),
        Span::raw(format!(
            " | model={} | step={} | Q(wait={}, run={}, swap={}) | [q] quit",
            s.model_name, s.step_index, s.waiting_groups, s.running_groups, s.swapped_groups
        )),
    ]);

    let hdr = Paragraph::new(title_line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(hdr, area);
}

fn render_kv_block_map(frame: &mut Frame, area: Rect, s: &TelemetrySnapshot) {
    let outer = Block::default()
        .title(" [Q1] Physical KV-Cache Block Map ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    if inner.height < 3 {
        return;
    }

    let sub = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    let used = s.total_kv_blocks.saturating_sub(s.free_kv_blocks);
    let util_ratio = if s.total_kv_blocks > 0 {
        (used as f64 / s.total_kv_blocks as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .ratio(util_ratio)
        .label(format!(
            "KV Slab {}/{} ({:.1}%) | Swap {}/{} | ExtFrag {:.1}%",
            used,
            s.total_kv_blocks,
            util_ratio * 100.0,
            s.used_swap_blocks,
            s.total_swap_blocks,
            s.external_fragmentation_pct
        ));
    frame.render_widget(gauge, sub[0]);

    let legend = Line::from(vec![
        Span::styled("·Free ", Style::default().fg(Color::DarkGray)),
        Span::styled("█Unique ", Style::default().fg(Color::Cyan)),
        Span::styled("◆CoW(rc>1) ", Style::default().fg(Color::Magenta)),
        Span::styled("◈Radix ", Style::default().fg(Color::Green)),
        Span::styled("⇄Swap ", Style::default().fg(Color::Yellow)),
    ]);
    frame.render_widget(Paragraph::new(legend), sub[1]);

    let cols = (sub[2].width as usize / 2).max(1);
    let mut grid_lines = Vec::new();
    let mut cur_spans = Vec::with_capacity(cols);

    for blk in &s.blocks {
        let (sym, color) = if blk.ref_count == 0 {
            ("· ", Color::DarkGray)
        } else if blk.is_radix_cached && blk.ref_count > 1 {
            ("◆ ", Color::Magenta)
        } else if blk.is_radix_cached {
            ("◈ ", Color::Green)
        } else if blk.ref_count > 1 {
            ("◆ ", Color::Magenta)
        } else {
            ("█ ", Color::Cyan)
        };
        cur_spans.push(Span::styled(sym, Style::default().fg(color)));
        if cur_spans.len() >= cols {
            grid_lines.push(Line::from(cur_spans));
            cur_spans = Vec::with_capacity(cols);
            if grid_lines.len() >= sub[2].height as usize {
                break;
            }
        }
    }
    if !cur_spans.is_empty() && grid_lines.len() < sub[2].height as usize {
        grid_lines.push(Line::from(cur_spans));
    }

    frame.render_widget(Paragraph::new(grid_lines), sub[2]);
}

fn render_batch_lanes(frame: &mut Frame, area: Rect, s: &TelemetrySnapshot) {
    let outer = Block::default()
        .title(" [Q2] Live Continuous Batch & Chunked-Prefill Lanes ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let mut lines = Vec::new();
    if s.active_lanes.is_empty() {
        lines.push(Line::from(Span::styled(
            "Idle — waiting for continuous batching traffic...",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for lane in s.active_lanes.iter().take(inner.height as usize) {
            let status_span = match lane.status.as_str() {
                "prefill" => {
                    let pct = (lane.kv_tokens * 100)
                        .checked_div(lane.prompt_tokens)
                        .unwrap_or(100);
                    Span::styled(
                        format!("[PREFILL {:>3}%] ", pct),
                        Style::default().fg(Color::Yellow),
                    )
                }
                "swapped" => Span::styled("[SWAPPED]      ", Style::default().fg(Color::Red)),
                "waiting" => Span::styled("[WAITING]      ", Style::default().fg(Color::DarkGray)),
                _ => Span::styled("[DECODE]       ", Style::default().fg(Color::Green)),
            };

            let mut row = vec![
                Span::styled(
                    format!("R{:<4}.{:<1} ", lane.req_id, lane.child_idx),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                status_span,
                Span::styled(
                    format!("{:>3}/{:<3}t ", lane.generated_tokens, lane.max_tokens),
                    Style::default().fg(Color::Cyan),
                ),
            ];

            if lane.last_spec_burst > 1 {
                row.push(Span::styled(
                    format!("[+{}] ", lane.last_spec_burst),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if lane.has_json_schema {
                let dfa_lbl = lane
                    .dfa_state
                    .map(|st| format!("[DFA#{st}] "))
                    .unwrap_or_else(|| "[JSON] ".into());
                row.push(Span::styled(dfa_lbl, Style::default().fg(Color::Magenta)));
            }
            row.push(Span::raw(lane.text_preview.clone()));
            lines.push(Line::from(row));
        }
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_radix_and_grammar(frame: &mut Frame, area: Rect, s: &TelemetrySnapshot) {
    let outer = Block::default()
        .title(" [Q3] Radix Prefix Tree & Grammar FSM Inspector ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    if inner.height < 4 {
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(inner);

    let spark_data: Vec<u64> = if s.radix_hit_sparkline.is_empty() {
        vec![0; 16]
    } else {
        s.radix_hit_sparkline.clone()
    };
    let spark = Sparkline::default()
        .block(Block::default().title(format!(
            "Hit Rate: {:.1}% | Cached Blks: {} | LRU Evicted: {}",
            s.radix_hit_rate_pct, s.radix_cached_blocks, s.radix_evicted_blocks
        )))
        .data(&spark_data)
        .max(100)
        .style(Style::default().fg(Color::Green));
    frame.render_widget(spark, chunks[0]);

    let schema_lanes = s.active_lanes.iter().filter(|l| l.has_json_schema).count();
    let mut tree_lines = vec![Line::from(vec![
        Span::styled(
            format!("Active JSON-Schema FSM Streams: {} ", schema_lanes),
            Style::default().fg(Color::Magenta),
        ),
        Span::raw(format!("| Radix Tree Nodes: {}", s.radix_tree.len())),
    ])];

    for node in s
        .radix_tree
        .iter()
        .take((chunks[1].height as usize).saturating_sub(1))
    {
        tree_lines.push(Line::from(format!(
            " └─ node#{} (parent={:?}) toks={} blks={:?} rc={} tick={}",
            node.node_id,
            node.parent_id,
            node.total_edge_tokens,
            node.block_ids,
            node.min_ref_count,
            node.last_access_tick
        )));
    }

    frame.render_widget(
        Paragraph::new(tree_lines).wrap(Wrap { trim: true }),
        chunks[1],
    );
}

fn render_sla_gauges(frame: &mut Frame, area: Rect, s: &TelemetrySnapshot) {
    let outer = Block::default()
        .title(" [Q4] SLA Latency & Speculative Gauges ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    if inner.height < 4 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(2)])
        .split(inner);

    let alpha_ratio = (s.speculative_alpha_pct / 100.0).clamp(0.0, 1.0);
    let spec_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Yellow).bg(Color::Black))
        .ratio(alpha_ratio)
        .label(format!(
            "Speculative Acceptance α: {:.1}% (active K={})",
            s.speculative_alpha_pct, s.active_spec_k
        ));
    frame.render_widget(spec_gauge, rows[0]);

    let metrics_text = vec![
        Line::from(vec![
            Span::styled("Throughput:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1} tok/s", s.tokens_per_sec),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("  (total: {} toks)", s.tokens_generated_total)),
        ]),
        Line::from(vec![
            Span::styled("TTFT p50/p99:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2} ms / {:.2} ms", s.ttft_p50_ms, s.ttft_p99_ms),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("ITL  p50/p99:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2} ms / {:.2} ms", s.itl_p50_ms, s.itl_p99_ms),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("Requests:      ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "completed={}  aborted={}",
                s.completed_requests_total, s.aborted_requests_total
            )),
        ]),
    ];

    frame.render_widget(Paragraph::new(metrics_text), rows[1]);
}
