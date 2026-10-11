#!/usr/bin/env bash
# Build the browser module (crates/sky-web: game cores + System One engine) with SIMD128 into
# site/public/pkg, where the VitePress site loads it at runtime.
# Usage: scripts/build-wasm.sh [--dev]
set -euo pipefail
cd "$(dirname "$0")/.."

profile=--release
[[ "${1:-}" == "--dev" ]] && profile=--dev

command -v wasm-pack >/dev/null || {
  echo "wasm-pack not found: cargo install wasm-pack (or brew install wasm-pack)" >&2
  exit 1
}
rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown ||
  rustup target add wasm32-unknown-unknown

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-web-wasm}
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"

wasm-pack build crates/sky-web --target web "$profile" --out-dir ../../site/public/pkg --no-typescript
# wasm-pack writes a .gitignore and a package.json the site does not need.
rm -f site/public/pkg/.gitignore site/public/pkg/package.json site/public/pkg/README.md

ls -l site/public/pkg
