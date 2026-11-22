NIGHTLY ?= nightly-x86_64-pc-windows-gnu

.PHONY: check test miri fuzz-oracle grammar-fuzz bench wasm-studio verify-all

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

verify-all: check test fuzz-oracle grammar-fuzz bench wasm-studio
