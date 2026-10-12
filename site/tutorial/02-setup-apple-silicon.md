# 02 Setup on Apple Silicon

This chapter sets up a Mac with an M-series chip to build every skycmd crate, train with mlx-rs on the GPU, compile the browser module to WebAssembly and run this site locally. Every command below comes from the repository's own scripts and manifests.

## What you need

| tool | why | checked by |
|---|---|---|
| macOS on Apple Silicon (M1 to M5) | mlx-rs trains on the GPU through Metal | `sky-train` |
| Xcode (full app) and its Metal toolchain | MLX compiles Metal kernels when it builds | `scripts/train-ladder.sh` |
| CMake | `mlx-sys` builds MLX through the `cmake` crate | `mlx-sys` build script |
| Rust stable through rustup | the whole workspace, edition 2021 | `Cargo.toml` |
| `wasm32-unknown-unknown` target and `wasm-pack` | the browser module | `scripts/build-wasm.sh` |
| Node 20 or newer | the VitePress site | `site/package.json` |

The reference training machine is an Apple M5 (`docs/SPEC.md`). Smaller chips work too, just more slowly. On the M5, `sky-train bench` measures 598,104 tokens/s for the s1-pico decision phase in bf16 and 243,002 for s1-nano (`BENCH.md`). Your own numbers come from the same command (see the end of this page).

## Xcode and the Metal toolchain

mlx-rs 0.32.0 builds MLX from source, and MLX needs the Metal compiler that ships with Xcode. The Command Line Tools alone are not enough. Install Xcode from the App Store, open it once to accept the license, then fetch the Metal toolchain (`docs/research.md`):

```bash
xcodebuild -downloadComponent MetalToolchain
brew install cmake
```

The training scripts point `DEVELOPER_DIR` at the full Xcode, even if `xcode-select` points somewhere else:

`scripts/train-ladder.sh`

```bash
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-train}
cargo build -q --release -p sky-train
T=$CARGO_TARGET_DIR/release/sky-train
```

Do the same in your shell before any manual `cargo` command on `sky-train` or `sky-arena`:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun --find metal   # must print a path inside Xcode.app
```

## Rust, the wasm target and wasm-pack

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup target add wasm32-unknown-unknown
cargo install wasm-pack        # or: brew install wasm-pack
```

`scripts/build-wasm.sh` checks for both and adds the target if it is missing:

`scripts/build-wasm.sh`

```bash
command -v wasm-pack >/dev/null || {
  echo "wasm-pack not found: cargo install wasm-pack (or brew install wasm-pack)" >&2
  exit 1
}
rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown ||
  rustup target add wasm32-unknown-unknown
```

## Node for the site

The site declares its engine requirement and scripts:

`site/package.json`

```json
  "scripts": {
    "docs:dev": "vitepress dev",
    "docs:build": "vitepress build",
    "docs:preview": "vitepress preview"
  },
  "devDependencies": {
    "vitepress": "^1.6.4",
    "vue": "^3.5.22"
  },
  "engines": {
    "node": ">=20"
  }
```

Any Node 20+ install works (Homebrew, fnm, nvm). Check it with `node --version`.

## One target directory per crate

The workspace uses `members = ["crates/*"]`. The project rule (`docs/SPEC.md`) is to give each crate its own Cargo target directory. Parallel builds then never wait on each other's lock, and a heavy mlx-rs build never invalidates a light one:

```bash
export CARGO_TARGET_DIR=$PWD/target/<crate-name>
```

All commands below run from the repository root.

## Build and test each crate

### sky-schema: types and validators

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-schema
cargo test -p sky-schema
```

### sky-games: game cores and dataset generator

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-games
cargo test -p sky-games
cargo test -p sky-games --release          # also runs flappy_oracle_survives_500_pipes
cargo run --release -p sky-games --example eval
cargo check -p sky-games --target wasm32-unknown-unknown
```

The slow Flappy test is skipped in debug builds. The attribute in `crates/sky-games/tests/games.rs` is `#[cfg_attr(debug_assertions, ignore = "slow in debug builds, runs with --release")]`.

### sky-gguf: GGUF reader and writer

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-gguf
cargo test -p sky-gguf
cargo run --release -p sky-gguf --bin gguf-inspect -- site/public/models/s1-pico.gguf
```

The usage string is `gguf-inspect <file.gguf> [--max-array N]`.

### sky-infer: the inference engine

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo test -p sky-infer
cargo run --release -p sky-infer --bin sky-bench -- --random s1-pico
cargo run --release -p sky-infer --bin sky-bench -- site/public/models/s3-pico.gguf
```

The s1 golden tests compare against the reference model in `models-src/oscar`, which is gitignored. Without it they print `skipping golden test: ... is absent` and pass. The s2 and s3 golden files are committed under `crates/sky-infer/tests/golden/`. `sky-bench` takes a GGUF file, a Laya directory, or `--random NAME` with `s1-pico|s1-nano|s1-micro` or `s2-pico|s2-nano|s2-micro|s3-pico|s3-nano`.

### sky-data: Hugging Face pipeline, tokenizer, teacher

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-data
cargo test -p sky-data
cargo run --release -p sky-data -- --help
```

The unit tests make no network calls. Downloads only happen through the subcommands in [chapter 05](./05-hugging-face-data).

### sky-train: training on the GPU

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-train
cargo build --release -p sky-train
$CARGO_TARGET_DIR/release/sky-train --help
```

The first build compiles MLX and its Metal kernels, which takes a while. The s1 subcommands are `bench`, `pretrain`, `decide`, `calibrate`, `export` and `eval`; `scripts/train-ladder.sh` chains them for each model size. The s2 and s3 models use `s2-bench`, `s2-pretrain`, `s2-decide`, `s2-calibrate`, `s2-export`, `s2-eval`, `s2-golden`, `s2-play`, `s2-seqgen` and `s2-numeric` ([chapter 19](./19-s2-architecture), [chapter 21](./21-s3-recursive-tiny)).

### sky-web and sky-arena

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-web
cargo test -p sky-web

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-arena
cargo build --release -p sky-arena
```

`sky-arena` plays every game with a model and prints game metrics and latency. Its usage line is `sky-arena <model.gguf | laya_dir> [--episodes N] [--seed S] [--game G|all|a,b] [--json out.json] [--max-pipes N] [--exit-threshold T]`, where `G` is `flappy`, `rescue`, `dodge` or `doom`. It also takes `--dagger N --out DIR` to collect DAgger data ([chapter 20](./20-dagger-lessons)).

## Build the browser module

The site loads a single wasm module, built from `crates/sky-web`, which bundles the game cores and the inference engine:

```bash
scripts/build-wasm.sh          # release build
scripts/build-wasm.sh --dev    # faster, unoptimized
```

The script turns on WebAssembly SIMD and writes into `site/public/pkg`:

`scripts/build-wasm.sh`

```bash
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-web-wasm}
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"

wasm-pack build crates/sky-web --target web "$profile" --out-dir ../../site/public/pkg --no-typescript
# wasm-pack writes a .gitignore and a package.json the site does not need.
rm -f site/public/pkg/.gitignore site/public/pkg/package.json site/public/pkg/README.md
```

The output is `sky_web.js` plus `sky_web_bg.wasm`. `site/public/pkg` is gitignored, so run the script after every fresh clone.

## Models for the site

The published models are already in the repository: `site/public/models/` holds `s3-pico.gguf` (the default), `s1-pico.gguf`, `s2-nano.gguf` and `s2-nano-doom.gguf`, listed in `manifest.json`.

`scripts/export-models.sh` converts each trained s1 model in `models/<name>/` (the Laya layout written by `sky-train export`) to `site/public/models/<name>.gguf`, then rewrites `manifest.json` and `bench.json`. Encoder matrices are stored in f16 by default. `--type f16|q8_0|f32` overrides this. The script only knows s1 directories: the s2 and s3 files are GGUFs written by `sky-train s2-export` and copied to `site/public/models/`, and running the script replaces the published manifest with the s1 models it finds.

```bash
scripts/export-models.sh
```

When no model is listed, the site falls back to the oracle policy. You can still play the games.

## Run the site

```bash
cd site && npm install && npm run docs:dev
```

VitePress prints a local URL. `npm run docs:build` produces the static build.

## Troubleshooting

- **`xcrun: error: unable to find utility "metal"`**: `DEVELOPER_DIR` points at the Command Line Tools. Export the Xcode path shown above, or run `xcodebuild -downloadComponent MetalToolchain` again.
- **CMake not found while building `mlx-sys`**: run `brew install cmake`.
- **Builds block each other ("Blocking waiting for file lock")**: two shells share a target directory. Give each crate its own `CARGO_TARGET_DIR`.
- **The site shows no model**: run `scripts/build-wasm.sh`, check that `site/public/models/manifest.json` lists the models, then reload.
- **Measuring your own machine**: `sky-train bench` appends throughput rows to `BENCH.md`. The rows already there were measured on the reference machine with mlx-rs 0.32 and MLX 0.32.2.

## Next

[03 Typed decisions: schema, unknown and the veto rule](./03-typed-decisions)
