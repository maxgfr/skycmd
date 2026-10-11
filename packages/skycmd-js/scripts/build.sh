#!/usr/bin/env sh
# Build the npm package `skycmd` into packages/skycmd-js/dist:
#   sky_infer.js + sky_infer_bg.wasm  wasm-pack (--target web) of crates/sky-infer, feature `wasm`, SIMD128
#   skycmd.js + skycmd.d.ts           the bundler-free wrapper (src/)
# Usage: sh scripts/build.sh [--dev]   (from packages/skycmd-js, or `npm run build`)
set -eu
pkg=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$pkg/../.." && pwd)

profile=--release
[ "${1:-}" = "--dev" ] && profile=--dev

command -v wasm-pack >/dev/null || {
  echo "wasm-pack not found: cargo install wasm-pack (or brew install wasm-pack)" >&2
  exit 1
}
if command -v rustup >/dev/null; then
  rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown ||
    rustup target add wasm32-unknown-unknown
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target/skycmd-js-wasm}"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"

rm -rf "$pkg/dist"
# wasm-pack options go before the crate path, cargo options after it.
wasm-pack build --target web "$profile" --out-dir "$pkg/dist" --out-name sky_infer \
  --no-pack "$root/crates/sky-infer" --features wasm
rm -f "$pkg/dist/.gitignore" "$pkg/dist/package.json" "$pkg/dist/README.md"
cp "$pkg/src/skycmd.js" "$pkg/src/skycmd.d.ts" "$pkg/dist/"
[ -f "$root/LICENSE" ] && cp "$root/LICENSE" "$pkg/LICENSE"

ls -l "$pkg/dist"
