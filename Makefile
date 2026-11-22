NIGHTLY ?= nightly-x86_64-pc-windows-gnu

.PHONY: check test miri fuzz-oracle grammar-fuzz bench wasm-studio telemetry-summary verify-all verify-live

check:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace --release

miri:
	cargo +$(NIGHTLY) miri test -p novaserve-tensor

fuzz-oracle:
	cargo test -p novaserve-oracle --release -- --nocapture

grammar-fuzz:
	cargo test -p novaserve-frontend --release --test grammar_fuzz -- --nocapture

bench:
	cargo run --release -p novaserve-server --bin novaserve-bench

wasm-studio:
	cargo build --release -p novaserve-wasm --target wasm32-unknown-unknown
	cd studio && npm run build && npm test

telemetry-summary:
	node scripts/emit_telemetry_summary.mjs --report docs/benchmarks/serving_sla.md --progress PROGRESS.md --out studio/public/telemetry/summary.json --strict

verify-all: check test fuzz-oracle grammar-fuzz bench wasm-studio

verify-live: verify-all telemetry-summary
