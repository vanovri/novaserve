import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { STUDIO_PRESETS, TOOL_CALL_JSON_SCHEMA } from "../src/presets";
import { NovaServeStudioClient } from "../src/wasm_bridge";

const WASM_PATH = resolve(
  __dirname,
  "../../target/wasm32-unknown-unknown/release/novaserve_wasm.wasm"
);

describe("NovaServe WASM Engine & Studio Preset E2E Suite", () => {
  it("runs Preset 1 (Radix Hit + CoW n=4 Forking) with shared refcounts and 100% block reclamation", async () => {
    const wasmBytes = readFileSync(WASM_PATH);
    const preset = STUDIO_PRESETS[0];
    const client = await NovaServeStudioClient.createFromBytes(
      wasmBytes,
      preset.config
    );
    preset.setup(client);

    let maxRefCountSeen = 0;
    let steps = 0;
    while (steps < 80) {
      const snap = client.stepIteration();
      steps++;
      for (const blk of snap.kv_slab) {
        if (blk.ref_count > maxRefCountSeen) {
          maxRefCountSeen = blk.ref_count;
        }
      }
      if (!snap.has_unfinished_work) break;
    }

    expect(maxRefCountSeen).toBeGreaterThanOrEqual(4);
    const beforeClear = client.snapshot();
    expect(beforeClear.radix_hit_rate_pct).toBeGreaterThan(50);

    const afterClear = client.clearRadixCache();
    expect(afterClear.free_kv_blocks).toBe(afterClear.total_kv_blocks);
    client.dispose();
  });

  it("runs Preset 2 (Chunked Prefill + Memory Squeeze Swap Preemption) with zero block leaks", async () => {
    const wasmBytes = readFileSync(WASM_PATH);
    const preset = STUDIO_PRESETS[1];
    const client = await NovaServeStudioClient.createFromBytes(
      wasmBytes,
      preset.config
    );
    preset.setup(client);

    let sawPreemptionOrSwap = false;
    let completedCount = 0;
    for (let i = 0; i < 200; i++) {
      const snap = client.stepIteration();
      if (snap.preempted_this_step > 0 || snap.used_swap_blocks > 0) {
        sawPreemptionOrSwap = true;
      }
      completedCount += snap.completed_this_step.length;
      if (!snap.has_unfinished_work) break;
    }

    expect(sawPreemptionOrSwap).toBe(true);
    expect(completedCount).toBe(4);
    const finalSnap = client.clearRadixCache();
    expect(finalSnap.free_kv_blocks).toBe(finalSnap.total_kv_blocks);
    expect(finalSnap.used_swap_blocks).toBe(0);
    client.dispose();
  });

  it("runs Preset 3 (Leviathan Speculative Decoding K=4) emitting verified draft & residual/bonus traces", async () => {
    const wasmBytes = readFileSync(WASM_PATH);
    const preset = STUDIO_PRESETS[2];
    const client = await NovaServeStudioClient.createFromBytes(
      wasmBytes,
      preset.config
    );
    preset.setup(client);

    let totalSpecTraces = 0;
    for (let i = 0; i < 80; i++) {
      const snap = client.stepIteration();
      for (const tr of snap.speculative_traces) {
        totalSpecTraces++;
        expect(tr.draft_items.length).toBeGreaterThanOrEqual(1);
        expect(tr.draft_items.length).toBeLessThanOrEqual(4);
      }
      if (!snap.has_unfinished_work) break;
    }

    expect(totalSpecTraces).toBeGreaterThan(0);
    const finalSnap = client.clearRadixCache();
    expect(finalSnap.free_kv_blocks).toBe(finalSnap.total_kv_blocks);
    client.dispose();
  });

  it("runs Preset 4 (JSON-Schema Constrained Decoding DFA X-Ray) producing 100% valid schema-conforming JSON", async () => {
    const wasmBytes = readFileSync(WASM_PATH);
    const preset = STUDIO_PRESETS[3];
    const client = await NovaServeStudioClient.createFromBytes(
      wasmBytes,
      preset.config
    );
    preset.setup(client);

    let sawGrammarXray = false;
    let generatedJson = "";
    for (let i = 0; i < 120; i++) {
      const snap = client.stepIteration();
      if (snap.grammar_xrays.length > 0) {
        sawGrammarXray = true;
        expect(snap.grammar_xrays[0].top_allowed.length).toBeGreaterThan(0);
        expect(snap.grammar_xrays[0].top_masked.length).toBeGreaterThan(0);
      }
      for (const comp of snap.completed_this_step) {
        generatedJson = comp.choices[0].decoded_text;
      }
      if (!snap.has_unfinished_work) break;
    }

    expect(sawGrammarXray).toBe(true);
    const parsed = JSON.parse(generatedJson) as {
      tool: string;
      limit: number;
      strict: boolean;
    };
    const schema = JSON.parse(TOOL_CALL_JSON_SCHEMA);
    expect(schema.properties.tool.enum).toContain(parsed.tool);
    expect(typeof parsed.limit).toBe("number");
    expect(Number.isInteger(parsed.limit)).toBe(true);
    expect(typeof parsed.strict).toBe("boolean");
    expect(Object.keys(parsed).sort()).toEqual(["limit", "strict", "tool"]);

    const finalSnap = client.clearRadixCache();
    expect(finalSnap.free_kv_blocks).toBe(finalSnap.total_kv_blocks);
    client.dispose();
  });
});
