import { defineConfig } from "vite";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

function copyNovaServeWasmPlugin() {
  const srcWasm = resolve(
    __dirname,
    "../target/wasm32-unknown-unknown/release/novaserve_wasm.wasm"
  );
  const pubDir = resolve(__dirname, "public");
  const dstWasm = resolve(pubDir, "novaserve_wasm.wasm");

  const syncWasm = () => {
    if (existsSync(srcWasm)) {
      mkdirSync(pubDir, { recursive: true });
      copyFileSync(srcWasm, dstWasm);
    }
  };

  return {
    name: "copy-novaserve-wasm",
    buildStart() {
      syncWasm();
    },
    configureServer() {
      syncWasm();
    },
  };
}

export default defineConfig({
  plugins: [copyNovaServeWasmPlugin()],
  build: {
    target: "es2022",
    outDir: "dist",
  },
});
