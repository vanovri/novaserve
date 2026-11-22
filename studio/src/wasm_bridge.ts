import type {
  EngineConfigPayload,
  SubmitRequestPayload,
  WasmStepTelemetry,
} from "./types";

interface NovaServeRawExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  ns_wasm_engine_new_configured: () => number;
  ns_wasm_engine_free: (handle: number) => void;
  ns_wasm_prepare_input: (byteLen: number) => number;
  ns_wasm_submit_json: (handle: number) => number;
  ns_wasm_step_json: (handle: number) => number;
  ns_wasm_snapshot_json: (handle: number) => number;
  ns_wasm_clear_radix: (handle: number) => number;
  ns_wasm_output_ptr: () => number;
  ns_wasm_output_len: () => number;
}

export class NovaServeStudioClient {
  private readonly exports: NovaServeRawExports;
  private handle: number;
  private readonly encoder = new TextEncoder();
  private readonly decoder = new TextDecoder();

  private constructor(exports: NovaServeRawExports, cfg: EngineConfigPayload) {
    this.exports = exports;
    this.writeInputJson(cfg);
    this.handle = this.exports.ns_wasm_engine_new_configured();
    if (this.handle === 0) {
      throw new Error(`Failed to initialize NovaServeWasmEngine: ${this.readOutputString()}`);
    }
  }

  static async createFromBytes(
    wasmBytes: BufferSource,
    cfg: EngineConfigPayload
  ): Promise<NovaServeStudioClient> {
    const mod = await WebAssembly.compile(wasmBytes);
    const imports: WebAssembly.Imports = {};

    // Populate inert stubs for any generated wasm-bindgen JS glue symbols when driving via linear-memory JSON FFI
    for (const imp of WebAssembly.Module.imports(mod)) {
      if (!imports[imp.module]) {
        imports[imp.module] = {};
      }
      const modObj = imports[imp.module] as Record<string, WebAssembly.ImportValue>;
      if (imp.kind === "function") {
        modObj[imp.name] = () => 0;
      }
    }

    const instance = await WebAssembly.instantiate(mod, imports);
    return new NovaServeStudioClient(
      instance.exports as NovaServeRawExports,
      cfg
    );
  }

  static async createInBrowser(
    cfg: EngineConfigPayload,
    url = "/novaserve_wasm.wasm"
  ): Promise<NovaServeStudioClient> {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${url}: HTTP ${resp.status}`);
    }
    const buf = await resp.arrayBuffer();
    return NovaServeStudioClient.createFromBytes(buf, cfg);
  }

  reconfigure(cfg: EngineConfigPayload): WasmStepTelemetry {
    if (this.handle !== 0) {
      this.exports.ns_wasm_engine_free(this.handle);
      this.handle = 0;
    }
    this.writeInputJson(cfg);
    this.handle = this.exports.ns_wasm_engine_new_configured();
    if (this.handle === 0) {
      throw new Error(`Engine reconfigure failed: ${this.readOutputString()}`);
    }
    return this.snapshot();
  }

  submitRequest(req: SubmitRequestPayload): number {
    this.writeInputJson(req);
    const reqId = this.exports.ns_wasm_submit_json(this.handle);
    if (reqId === 0) {
      throw new Error(`submit_request failed: ${this.readOutputString()}`);
    }
    return reqId;
  }

  stepIteration(): WasmStepTelemetry {
    const ok = this.exports.ns_wasm_step_json(this.handle);
    const text = this.readOutputString();
    if (ok === 0) {
      throw new Error(`step_iteration failed: ${text}`);
    }
    return JSON.parse(text) as WasmStepTelemetry;
  }

  snapshot(): WasmStepTelemetry {
    const ok = this.exports.ns_wasm_snapshot_json(this.handle);
    const text = this.readOutputString();
    if (ok === 0) {
      throw new Error(`snapshot failed: ${text}`);
    }
    return JSON.parse(text) as WasmStepTelemetry;
  }

  clearRadixCache(): WasmStepTelemetry {
    this.exports.ns_wasm_clear_radix(this.handle);
    return this.snapshot();
  }

  dispose(): void {
    if (this.handle !== 0) {
      this.exports.ns_wasm_engine_free(this.handle);
      this.handle = 0;
    }
  }

  private writeInputJson(payload: unknown): void {
    const bytes = this.encoder.encode(JSON.stringify(payload));
    const ptr = this.exports.ns_wasm_prepare_input(bytes.length);
    const view = new Uint8Array(this.exports.memory.buffer, ptr, bytes.length);
    view.set(bytes);
  }

  private readOutputString(): string {
    const ptr = this.exports.ns_wasm_output_ptr();
    const len = this.exports.ns_wasm_output_len();
    const view = new Uint8Array(this.exports.memory.buffer, ptr, len);
    return this.decoder.decode(view);
  }
}
