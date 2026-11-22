import "./styles.css";
import { STUDIO_PRESETS, TOOL_CALL_JSON_SCHEMA } from "./presets";
import type {
  WasmCompletedRequest,
  WasmGrammarXray,
  WasmSpecTrace,
  WasmStepTelemetry,
} from "./types";
import { NovaServeStudioClient } from "./wasm_bridge";

interface UiState {
  activePresetId: string;
  activeTab: "radix" | "speculative" | "grammar";
  playing: boolean;
  speedMs: number;
  hoveredSeqKey: string | null;
  telemetry: WasmStepTelemetry | null;
  specHistory: WasmSpecTrace[];
  latestXray: WasmGrammarXray | null;
  completedLedger: WasmCompletedRequest[];
  errorMsg: string | null;
}

const state: UiState = {
  activePresetId: STUDIO_PRESETS[0].id,
  activeTab: "radix",
  playing: false,
  speedMs: 140,
  hoveredSeqKey: null,
  telemetry: null,
  specHistory: [],
  latestXray: null,
  completedLedger: [],
  errorMsg: null,
};

let client: NovaServeStudioClient | null = null;
let playTimer: number | null = null;

function escapeHtml(raw: string): string {
  return raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function stopTimer(): void {
  state.playing = false;
  if (playTimer !== null) {
    window.clearInterval(playTimer);
    playTimer = null;
  }
}

function startTimer(): void {
  stopTimer();
  state.playing = true;
  playTimer = window.setInterval(() => {
    if (!client || !state.telemetry?.has_unfinished_work) {
      stopTimer();
      renderApp();
      return;
    }
    advanceOneStep();
  }, state.speedMs);
}

function advanceOneStep(): void {
  if (!client) return;
  try {
    const step = client.stepIteration();
    state.telemetry = step;
    if (step.speculative_traces.length > 0) {
      state.specHistory = [
        ...step.speculative_traces,
        ...state.specHistory,
      ].slice(0, 24);
    }
    if (step.grammar_xrays.length > 0) {
      state.latestXray = step.grammar_xrays[step.grammar_xrays.length - 1];
    }
    if (step.completed_this_step.length > 0) {
      state.completedLedger = [
        ...step.completed_this_step,
        ...state.completedLedger,
      ].slice(0, 16);
    }
    renderApp();
  } catch (err) {
    state.errorMsg = err instanceof Error ? err.message : String(err);
    stopTimer();
    renderApp();
  }
}

function loadPreset(presetId: string): void {
  const preset =
    STUDIO_PRESETS.find((p) => p.id === presetId) ?? STUDIO_PRESETS[0];
  stopTimer();
  state.activePresetId = preset.id;
  state.activeTab = preset.defaultTab;
  state.specHistory = [];
  state.latestXray = null;
  state.completedLedger = [];
  state.hoveredSeqKey = null;
  state.errorMsg = null;

  if (!client) return;
  client.reconfigure(preset.config);
  preset.setup(client);
  state.telemetry = client.snapshot();
  renderApp();
}

function renderRadixSvg(t: WasmStepTelemetry): string {
  const nodes = t.radix_tree;
  if (nodes.length === 0) {
    return `<div style="color: var(--text-dim); font-family: var(--font-mono); padding: 14px;">
      RadixCache contains only root (#0). Step requests with shared prefixes to watch compressed edges populate and split!
    </div>`;
  }

  const rowHeight = 62;
  const height = Math.max(180, (nodes.length + 1) * rowHeight + 24);
  let svg = `<svg width="100%" height="${height}" viewBox="0 0 620 ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<rect x="16" y="14" width="130" height="36" rx="6" fill="#131b2a" stroke="#22d3ee" />`;
  svg += `<text x="28" y="36" fill="#22d3ee" font-family="monospace" font-size="12" font-weight="bold">ROOT #0</text>`;

  const posMap = new Map<number, { x: number; y: number }>();
  posMap.set(0, { x: 81, y: 50 });

  nodes.forEach((n, idx) => {
    const depth = n.parent_id === 0 || n.parent_id === null ? 1 : 2;
    const x = 40 + depth * 120;
    const y = 72 + idx * rowHeight;
    posMap.set(n.node_id, { x: x + 90, y: y + 20 });
    const parentPos = posMap.get(n.parent_id ?? 0) ?? { x: 81, y: 50 };

    svg += `<path d="M ${parentPos.x} ${parentPos.y} C ${parentPos.x} ${y + 20}, ${x - 20} ${y + 20}, ${x} ${y + 20}" fill="none" stroke="#10b981" stroke-width="1.8" />`;
    const stroke = n.ref_count > 1 ? "#d946ef" : "#10b981";
    svg += `<rect x="${x}" y="${y}" width="340" height="44" rx="6" fill="#131b2a" stroke="${stroke}" />`;
    svg += `<text x="${x + 10}" y="${y + 17}" fill="#e6edf7" font-family="monospace" font-size="11">Node #${n.node_id} | ${n.total_edge_tokens} toks | blks [${n.block_ids.join(",")}] | rc=${n.ref_count} | age=${n.lru_age}</text>`;
    svg += `<text x="${x + 10}" y="${y + 34}" fill="#8b9bb4" font-family="monospace" font-size="10.5">"${escapeHtml(n.token_preview.slice(0, 38))}"</text>`;
  });

  svg += `</svg>`;
  return svg;
}

function renderSpeculativePanel(): string {
  if (state.specHistory.length === 0) {
    return `<div style="color: var(--text-dim); font-family: var(--font-mono); padding: 14px;">
      Select Preset 3 (Speculative K=4) and click Step Iteration or Play to inspect Leviathan rejection sampling u &lt;= min(1, p(x)/q(x))!
    </div>`;
  }

  return `<div class="spec-tape">
    ${state.specHistory
      .map((tr) => {
        const draftsHtml = tr.draft_items
          .map(
            (d) => `<span class="tok-chip ${
              d.accepted ? "tok-accepted" : "tok-rejected"
            }" title="q=${d.draft_prob.toFixed(3)}, p=${d.target_prob.toFixed(3)}">
              ${escapeHtml(JSON.stringify(d.token_text))} (p/q=${(
              d.target_prob / Math.max(d.draft_prob, 1e-4)
            ).toFixed(2)})
            </span>`
          )
          .join("");

        const finalLabel = tr.is_bonus ? "BONUS" : "RESIDUAL (p-q)+";
        return `<div class="spec-row">
          <strong>R${tr.req_id}.${tr.child_idx} (${tr.accepted_count}/${tr.lookahead_k} acc):</strong>
          ${draftsHtml}
          <span class="tok-chip tok-gold">${finalLabel}: ${escapeHtml(
          JSON.stringify(tr.residual_or_bonus_text)
        )}</span>
        </div>`;
      })
      .join("")}
  </div>`;
}

function renderGrammarXrayPanel(): string {
  const x = state.latestXray;
  const schemaDisplay = x?.schema_json || TOOL_CALL_JSON_SCHEMA;
  if (!x) {
    return `<div>
      <div class="schema-box"><pre>${escapeHtml(schemaDisplay)}</pre></div>
      <div style="color: var(--text-dim); font-family: var(--font-mono); padding: 12px 4px;">
        Select Preset 4 (JSON-Schema DFA X-Ray) and click Step Iteration to watch real Transformer logits get masked to -∞!
      </div>
    </div>`;
  }

  const maxMag = Math.max(
    1,
    ...x.top_allowed.map((i) => Math.abs(i.raw_logit)),
    ...x.top_masked.map((i) => Math.abs(i.raw_logit))
  );

  const renderBars = (
    items: WasmGrammarXray["top_allowed"],
    kind: "allowed" | "masked"
  ) =>
    items
      .map((item) => {
        const widthPct = Math.min(
          100,
          Math.max(12, (Math.abs(item.raw_logit) / maxMag) * 100)
        );
        const isChosen =
          kind === "allowed" && item.token_id === x.chosen_token_id;
        return `<div class="logit-row">
          <span style="width: 96px; overflow: hidden; text-overflow: ellipsis; ${
            kind === "masked"
              ? "text-decoration: line-through; color: #f87171;"
              : isChosen
              ? "color: #fbbf24; font-weight: bold;"
              : ""
          }">${escapeHtml(JSON.stringify(item.token_text))}</span>
          <div class="logit-bar-track">
            <div class="logit-bar-fill ${kind}" style="width: ${widthPct.toFixed(
          1
        )}%"></div>
          </div>
          <span style="width: 78px; text-align: right; color: ${
            kind === "masked" ? "#ef4444" : "#10b981"
          };">
            ${
              kind === "masked"
                ? "-∞ MASKED"
                : item.raw_logit.toFixed(2)
            }
          </span>
        </div>`;
      })
      .join("");

  return `<div style="display: flex; flex-direction: column; gap: 10px;">
    <div style="display: flex; flex-wrap: wrap; gap: 8px; font-family: var(--font-mono); font-size: 11.5px;">
      <span class="kpi-chip">DFA State: <strong>#${x.dfa_state} / ${
    x.total_dfa_states
  }</strong></span>
      <span class="kpi-chip">Accepting: <strong>${
        x.is_accepting ? "YES" : "NO"
      }</strong></span>
      <span class="kpi-chip">Allowed Tokens: <strong>${x.allowed_count} / ${
    x.vocab_size
  }</strong></span>
      <span class="kpi-chip">Chosen: <strong>${escapeHtml(
        JSON.stringify(x.chosen_token_text)
      )}</strong></span>
      ${
        x.forced_jump_tokens.length > 0
          ? `<span class="kpi-chip">Forced Jump: <strong>${escapeHtml(
              x.forced_jump_tokens.join("")
            )}</strong></span>`
          : ""
      }
    </div>
    <div class="schema-box"><pre>${escapeHtml(schemaDisplay)}</pre></div>
    <div class="xray-cols">
      <div>
        <div style="font-family: var(--font-mono); font-size: 11px; color: var(--green); margin-bottom: 6px;">
          LEGAL NEXT TOKENS (Top-10 Allowed Logits)
        </div>
        ${renderBars(x.top_allowed, "allowed")}
      </div>
      <div>
        <div style="font-family: var(--font-mono); font-size: 11px; color: var(--red); margin-bottom: 6px;">
          SYNTACTICALLY ILLEGAL TOKENS (Crossed Out → -∞)
        </div>
        ${renderBars(x.top_masked, "masked")}
      </div>
    </div>
  </div>`;
}

function renderApp(): void {
  const root = document.getElementById("app");
  if (!root) return;

  const t = state.telemetry;
  const hoveredBlocks = new Set<number>();
  if (t && state.hoveredSeqKey) {
    const seq = t.sequences.find((s) => s.seq_key === state.hoveredSeqKey);
    seq?.block_ids.forEach((b) => hoveredBlocks.add(b));
  }

  const usedBlocks = t ? t.total_kv_blocks - t.free_kv_blocks : 0;

  root.innerHTML = `
    <div class="studio-shell">
      <header class="topbar">
        <div class="topbar-row">
          <div class="brand">
            <span class="brand-pill">NOVASERVE WASM STUDIO</span>
            <span class="brand-sub">In-Browser Rust PagedAttention + RadixCache + Leviathan Speculative + JSON-DFA Engine</span>
          </div>
          <div class="kpi-strip">
            <span class="kpi-chip">Step: <strong>${t?.step_index ?? 0}</strong></span>
            <span class="kpi-chip">KV Slab: <strong>${usedBlocks}/${t?.total_kv_blocks ?? 0}</strong></span>
            <span class="kpi-chip">Radix Hit: <strong>${(t?.radix_hit_rate_pct ?? 0).toFixed(1)}%</strong></span>
            <span class="kpi-chip">Swap Pool: <strong>${t?.used_swap_blocks ?? 0}/${t?.total_swap_blocks ?? 0}</strong></span>
            <span class="kpi-chip">Spec α: <strong>${(t?.rolling_spec_alpha_pct ?? 0).toFixed(1)}% (K=${t?.active_spec_k ?? 0})</strong></span>
          </div>
          <div class="controls-group">
            <button id="btn-play" class="primary">${state.playing ? "⏸ Pause" : "▶ Play"}</button>
            <button id="btn-step">⏭ Step +1 Iteration</button>
            <button id="btn-clear-radix">🧹 Clear Unpinned Radix</button>
            <label style="font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);">
              Speed:
              <input id="rng-speed" type="range" min="30" max="500" step="10" value="${state.speedMs}" style="vertical-align: middle;" />
              ${state.speedMs}ms
            </label>
          </div>
        </div>
        <div class="preset-bar">
          ${STUDIO_PRESETS.map(
            (p) => `
            <button class="preset-btn ${
              p.id === state.activePresetId ? "active" : ""
            }" data-preset="${p.id}">
              <span class="preset-badge">${escapeHtml(p.badge)}</span>
              <strong>${escapeHtml(p.title)}</strong>
              <span style="color: var(--text-dim); font-size: 10.5px;">${escapeHtml(
                p.description
              )}</span>
            </button>
          `
          ).join("")}
        </div>
        ${
          state.errorMsg
            ? `<div style="color: var(--red); font-family: var(--font-mono);">${escapeHtml(
                state.errorMsg
              )}</div>`
            : ""
        }
      </header>

      <main class="workbench-grid">
        <section class="panel">
          <div class="panel-header">
            <span class="panel-title">PHYSICAL KV-BLOCK SLAB MATRIX (${
              t?.block_size ?? 8
            } tok/blk, 0% Ext Fragmentation)</span>
            <span style="color: var(--text-dim); font-size: 11px;">Hover a sequence below to highlight its non-contiguous blocks</span>
          </div>
          <div class="slab-matrix">
            ${
              t
                ? t.kv_slab
                    .map((blk) => {
                      const dots = Array.from(
                        { length: t.block_size },
                        (_, idx) =>
                          `<span class="slot-dot ${
                            idx < blk.filled_slots ? "filled" : ""
                          }"></span>`
                      ).join("");
                      return `
                      <div class="kv-tile status-${blk.status} ${
                        hoveredBlocks.has(blk.block_id) ? "hovered-seq" : ""
                      }" title="Owners: ${blk.owner_seq_ids.join(", ") || "none"}">
                        <div class="kv-tile-top">
                          <strong>B${String(blk.block_id).padStart(2, "0")}</strong>
                          <span>
                            ${
                              blk.is_radix_cached
                                ? `<span class="rdx-badge">RDX</span>`
                                : ""
                            }
                            ${
                              blk.ref_count > 1
                                ? `<span class="rc-badge">x${blk.ref_count}</span>`
                                : ""
                            }
                          </span>
                        </div>
                        <div class="slot-dots">${dots}</div>
                        <div style="color: var(--text-dim); font-size: 9.5px;">
                          ${blk.filled_slots}/${t.block_size} slots
                        </div>
                      </div>`;
                    })
                    .join("")
                : ""
            }
          </div>

          <div class="panel-header" style="margin-top: 4px;">
            <span class="panel-title">LIVE CONTINUOUS BATCHING &amp; COMPLETED STREAMS</span>
            <span style="color: var(--text-dim); font-family: var(--font-mono); font-size: 11px;">
              prefill=${t?.scheduled_prefill_tokens ?? 0}t | decode=${
    t?.scheduled_decode_tokens ?? 0
  }t | preempt=${t?.preempted_this_step ?? 0}
            </span>
          </div>
          <div class="streams-list">
            ${
              t && t.sequences.length > 0
                ? t.sequences
                    .map(
                      (s) => `
                    <div class="seq-card" data-seq-key="${s.seq_key}">
                      <div class="seq-card-hdr">
                        <span>
                          <strong style="color: var(--cyan);">${s.seq_key}</strong>
                          [${s.status.toUpperCase()}]
                          kv=${s.kv_tokens}/${s.prompt_len + s.generated_count}
                          gen=${s.generated_count}/${s.max_new_tokens}
                          ${
                            s.cached_prefix_tokens > 0
                              ? `<span style="color: var(--green);">radix_hit=${s.cached_prefix_tokens}t</span>`
                              : ""
                          }
                          ${
                            s.preemptions > 0
                              ? `<span style="color: var(--amber);">swaps=${s.preemptions}</span>`
                              : ""
                          }
                        </span>
                        <span>blks=[${s.block_ids.join(",")}]</span>
                      </div>
                      <div class="seq-text">${escapeHtml(
                        s.decoded_text || "(prefilling prompt chunks...)"
                      )}</div>
                    </div>`
                    )
                    .join("")
                : ""
            }
            ${state.completedLedger
              .map((comp) =>
                comp.choices
                  .map(
                    (ch) => `
                  <div class="seq-card" style="opacity: 0.82;">
                    <div class="seq-card-hdr">
                      <span><strong style="color: var(--green);">R${comp.req_id}.${ch.child_idx} [DONE: ${ch.finish_reason}]</strong> (${ch.token_count} toks, radix_hit=${comp.cached_prefix_tokens}t, swaps=${comp.preemptions})</span>
                    </div>
                    <div class="seq-text">${escapeHtml(ch.decoded_text)}</div>
                  </div>`
                  )
                  .join("")
              )
              .join("")}
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <div class="tabs-header">
              <button class="tab-btn ${
                state.activeTab === "radix" ? "active" : ""
              }" data-tab="radix">Radix Prefix Tree</button>
              <button class="tab-btn ${
                state.activeTab === "speculative" ? "active" : ""
              }" data-tab="speculative">Speculative Tape</button>
              <button class="tab-btn ${
                state.activeTab === "grammar" ? "active" : ""
              }" data-tab="grammar">Grammar DFA X-Ray</button>
            </div>
          </div>
          <div>
            ${
              state.activeTab === "radix" && t
                ? renderRadixSvg(t)
                : state.activeTab === "speculative"
                ? renderSpeculativePanel()
                : renderGrammarXrayPanel()
            }
          </div>
        </section>
      </main>
    </div>
  `;

  document.getElementById("btn-play")?.addEventListener("click", () => {
    if (state.playing) {
      stopTimer();
      renderApp();
    } else {
      startTimer();
      renderApp();
    }
  });
  document.getElementById("btn-step")?.addEventListener("click", () => {
    stopTimer();
    advanceOneStep();
  });
  document.getElementById("btn-clear-radix")?.addEventListener("click", () => {
    if (!client) return;
    state.telemetry = client.clearRadixCache();
    renderApp();
  });
  document.getElementById("rng-speed")?.addEventListener("input", (ev) => {
    const val = Number((ev.target as HTMLInputElement).value);
    state.speedMs = val;
    if (state.playing) startTimer();
    else renderApp();
  });

  document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pid = btn.dataset.preset;
      if (pid) loadPreset(pid);
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab as UiState["activeTab"] | undefined;
      if (tab) {
        state.activeTab = tab;
        renderApp();
      }
    });
  });

  document.querySelectorAll<HTMLElement>("[data-seq-key]").forEach((el) => {
    el.addEventListener("mouseenter", () => {
      state.hoveredSeqKey = el.dataset.seqKey ?? null;
      renderApp();
    });
    el.addEventListener("mouseleave", () => {
      state.hoveredSeqKey = null;
      renderApp();
    });
  });
}

async function bootstrap(): Promise<void> {
  renderApp();
  try {
    client = await NovaServeStudioClient.createInBrowser(
      STUDIO_PRESETS[0].config
    );
    loadPreset(STUDIO_PRESETS[0].id);
  } catch (err) {
    state.errorMsg = err instanceof Error ? err.message : String(err);
    renderApp();
  }
}

void bootstrap();
