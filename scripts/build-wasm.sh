#!/usr/bin/env bash
# Build the browser module (crates/sky-web: game cores + System One engine) into site/public/pkg,
# where the VitePress site loads it at runtime. Two builds of the same code:
#   sky_web_bg.wasm          SIMD128 (every browser with WebAssembly SIMD)
#   sky_web_relaxed_bg.wasm  SIMD128 + relaxed SIMD (`f32x4.relaxed_madd`, fused on arm64)
# The loader (site/.vitepress/theme/lib/wasm.ts) takes the relaxed one when the browser validates
# a relaxed instruction, else the baseline. Both share one JS glue (sky_web.js): the script fails
# if the two builds produce different glue.
# Usage: scripts/build-wasm.sh [--dev]   (--dev builds the baseline only)
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
rustflags=${RUSTFLAGS:-}
pkg=$PWD/site/public/pkg

RUSTFLAGS="$rustflags -C target-feature=+simd128" \
  wasm-pack build crates/sky-web --target web "$profile" --out-dir "$pkg" --no-typescript
# wasm-pack writes a .gitignore and a package.json the site does not need.
rm -f "$pkg/.gitignore" "$pkg/package.json" "$pkg/README.md"

if [[ "$profile" == --dev ]]; then
  # A relaxed module left by an earlier release build would not match this glue: drop it, the
  # loader then falls back to the baseline.
  rm -f "$pkg/sky_web_relaxed_bg.wasm"
else
  # Same sources, relaxed SIMD on top. The dependencies keep one build per set of flags in the
  # same target dir, so switching back and forth only recompiles sky-infer and sky-web.
  relaxed=$(mktemp -d)
  trap 'rm -rf "$relaxed"' EXIT
  RUSTFLAGS="$rustflags -C target-feature=+simd128,+relaxed-simd" \
    wasm-pack build crates/sky-web --target web "$profile" --out-dir "$relaxed" --no-typescript
  cmp "$pkg/sky_web.js" "$relaxed/sky_web.js" || {
    echo "the relaxed build wrote a different sky_web.js: the loader needs one glue for both modules" >&2
    exit 1
  }
  cp "$relaxed/sky_web_bg.wasm" "$pkg/sky_web_relaxed_bg.wasm"
fi

ls -l site/public/pkg
