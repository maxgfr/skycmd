# 17 Scaling up and troubleshooting

The s1 ladder stops at s1-micro. The recipe still trains there, but a micro decision no longer fits in a browser frame (about 32.6 ms in WASM against a 16.7 ms frame) and micro is no better than pico on the games. This chapter shows which levers to pull for a bigger model (config, data, schedule) and which constraints of the GGUF and WASM paths to respect. It ends with the failures you are most likely to meet, each tied to the line of code that produces it.

## The ladder today

Parameter counts and training throughput come from `BENCH.md` (`sky-train bench`, MLX on Metal, bf16, decision phase at batch 64 × 192 tokens). Accuracy is the `games` row of each `runs/s1-*/report.md`. WASM latency is the `bench()` median of one Flappy decision (2 questions + physics) from `BENCH.md`, except s1-micro, measured on a busy machine for [chapter 12](./12-size-ladder).

| model | d | L | head layers | vocab | decision params | train tok/s (bf16) | held-out accuracy, games | wasm decision (Flappy) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1-pico | 64 | 2 | 1 | 1024 | 202,305 | 598,104 | 0.8806 | 694 µs |
| s1-nano | 128 | 4 | 1 | 2048 | 1,134,209 | 243,002 | 0.7837 | 3,541 µs |
| s1-micro | 256 | 6 | 2 | 4096 | 6,630,913 | 70,881 | 0.8215 | about 32,600 µs |

Each step multiplies the parameter count by about 5.5 to 6 and divides training throughput by about 2.5 to 3.5. WASM latency grows faster than the parameter count from pico to nano (5.1× for 5.6× the parameters) and from nano to micro (about 9× for 5.8×), because the embedding table, a lookup that is almost free at inference, is a smaller share of the bigger models. Before scaling s1, read [chapter 20](./20-dagger-lessons): bigger was not better in closed loop either. The architecture change of [chapter 19](./19-s2-architecture) bought more than size did.

## Scaling the model

### Write a config

A ladder config is a ModernBERT `config.json` plus an `rl_agent` block. Start from `configs/s1-micro.json` and change the sizes. These fields matter:

```json
{
  "name": "s1-mini",
  "vocab_size": 8192,
  "hidden_size": 384,
  "num_hidden_layers": 8,
  "num_attention_heads": 6,
  "intermediate_size": 768,
  "global_attn_every_n_layers": 3,
  "local_attention": 128,
  "max_position_embeddings": 256,
  "rl_agent": { "head_layers": 2, "max_len": 256, "head_max_len": 96,
                "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {} }
}
```

(`s1-mini` is an example name, not a shipped config; keep the other keys of `s1-micro.json` as they are.)

Three constraints come from the code, not from taste.

**Heads of 64.** The decision head always uses `d / 64` heads (`head_heads: (d / 64).max(1)` in `ModelConfig::from_hf_json`). llama.cpp reads one head count for the encoder and the head, so `sky-convert` refuses any other encoder head count.

From `crates/sky-infer/src/convert.rs`:

```rust
        // llama.cpp reads one head count and one norm eps for the encoder and the decision head
        if cfg.head_layers > 0
            && (cfg.head_heads != cfg.num_heads || cfg.head_norm_eps != cfg.norm_eps)
        {
            return Err(Error::Unsupported(format!(
                "decision head with {} heads / eps {} while the encoder has {} / {}: not representable in GGUF",
                cfg.head_heads, cfg.head_norm_eps, cfg.num_heads, cfg.norm_eps
            )));
        }
```

Set `num_attention_heads = hidden_size / 64`. That is 6 for d = 384 above.

**Multiples of 32 for int8.** Q8_0 needs every matrix row to split into 32-value blocks. `d`, `intermediate_size` and `4d` should all be multiples of 32; otherwise the converter falls back to F16 for that tensor.

**`ModelConfig::validate`.** The hidden size must split into heads of even size, and `max_len` and `head_max_len` must be non-zero.

`sky-convert --random` and `sky-bench --random` only know the shipped shapes: `s1-pico`, `s1-nano` and `s1-micro` for s1 (`ladder_config` in `convert.rs`), plus `s2-pico`, `s2-nano`, `s2-micro`, `s3-pico` and `s3-nano` (see `sky-convert --help`). For a new size, measure inference speed on a trained (or briefly trained) checkpoint instead.

### Match the tokenizer to the vocabulary

`vocab_size` in the config must equal the size of the tokenizer you train with. Train one per size with `sky-data`:

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-data
cargo run --release -p sky-data -- train-tokenizer --vocab 8192
# writes data/tok/tok-8192.json
```

`--int-prior` (default `vocab / 32`) makes the small integers single tokens, which matters because game states are space-separated small integers (SPEC, "Game state text"). A larger vocabulary covers a wider integer range for free.

### Keep `max_len` and `head_max_len`

Our games, and the truncation rules shared with llama.cpp, were designed around `max_len = 256` and `head_max_len = 96`. Raising them makes every buffer in `Workspace` larger (they are sized for `max_len` tokens) and makes attention cost grow with the square of the sequence length on global layers. Raise them only if your questions really need it, and retrain if you do: the model only knows the lengths it saw.

## Scaling the data and the schedule

`scripts/train-ladder.sh` is the reference recipe. Each size gets a fixed time budget for MLM pretraining and for decision training, plus a sampling weight for game records:

From `scripts/train-ladder.sh`:

```sh
# name  vocab  pretrain-min  decide-min  game-weight
ladder() {
  case "$1" in
    pico)  echo "1024 8 12 0.65" ;;
    nano)  echo "2048 12 20 0.65" ;;
    micro) echo "4096 15 25 0.5" ;;
  esac
}
```

The levers, from cheapest to most expensive:

- **Longer schedules.** `sky-train pretrain` and `decide` take `--minutes`. The WSD schedule starts its decay at 80% of the budget, so a longer run keeps the same shape. `--resume` continues from `<out>/last`. Bigger models need more steps to reach the same loss, and BENCH.md shows they also take fewer tokens per second.
- **More game data.** `sky-games-gen --game {flappy,rescue,dodge,all} --n N --seed S [--out PATH]` writes as many oracle-labelled records as you want, at no cost.
- **On-policy data.** `scripts/specialize.sh <pico|nano|micro> <rounds> <minutes-per-round>` runs DAgger rounds. `sky-arena --dagger` lets the model play, the oracle labels the states the model reaches, and `decide` fine-tunes on games + DAgger + a little general data.
- **More general data.** `sky-data fetch-decisions --cap N` (default 60,000 records per source) and `sky-data fetch-text --max-mb M` (default 150 MB) raise the size caps of the Hugging Face sources. Licenses are in `DATA_LICENSES.md`.
- **Mixture weights.** `decide --weights "games=0.5,decisions=0.5"` sets per-source sampling weights. Bigger models have more room for general decisions; micro already uses 0.5 instead of 0.65 for games.

Every run should end with the same three steps as the ladder script, so the temperatures, the report and the export stay consistent:

```sh
T=target/sky-train/release/sky-train
$T calibrate --run runs/s1-mini/decide/last --records data/decisions,data/games
$T export --run runs/s1-mini --out models/s1-mini
$T eval --run runs/s1-mini --records data/decisions,data/games
scripts/export-models.sh
```

`sky-train` uses the GPU through MLX. Do not start it while another training run is using the GPU.

## Scaling inference in the browser

- **Watch the frame budget.** About 16.7 ms at 60 Hz, shared with rendering. Read `GameSession.timing()` and raise `set_decide_every(n)` when the median decision time gets close (chapter 15).
- **Ship F16, not Q8_0.** The browser computes in f32 anyway. Q8_0 shrinks only the encoder matrices (embeddings stay F16, the head stays F32) and costs up to 3e-2 in probability on oscar (chapter 13).
- **Measure both ways.** Run `sky-bench` for native latency per question and `bench()` from the WASM module for a full decision (`node crates/sky-web/bench/bench-node.mjs <file.gguf>` runs it under Node). The ladder values are in the table at the top of this page; on a shared machine, interleave the models and keep the best median of several rounds, as `BENCH.md` does.

## Troubleshooting

### `invalid request: options do not fit in head_max_len=96 tokens`

From `crates/sky-infer/src/engine.rs`:

```rust
    fn run(&mut self, qtype: QType, n_options: usize) -> Result<()> {
        if self.markers.len() != n_options {
            return Err(Error::InvalidRequest(format!(
                "options do not fit in head_max_len={} tokens",
                self.model.cfg.head_max_len
            )));
        }
```

The error fires when some option markers did not survive truncation. `build_prefix` shrinks options evenly, but never below 4 tokens each (the marker plus 3). With many options the prefix can therefore exceed `head_max_len`, and `finish_sequence` then drops every marker past `max_len`. Fixes: fewer options, shorter keys (`key` without a description is rendered as just `key`), or a model trained with a larger `head_max_len`. Long descriptions are not the problem: they are cut to 48 tokens and then shrunk.

### Tokenizer mismatch

Symptoms and causes:

- `invalid request: token id N out of vocabulary` (from `Model::forward`): the tokenizer has more tokens than the embedding table. You paired a model with a tokenizer of another size, for example `tok-2048.json` with an s1-pico config.
- `invalid format: tensor token_embd.weight: shape [...], expected [...]` (from `Loader::req`): the config's `vocab_size` does not match the trained tensor.
- `token ids differ` in `tests/golden.rs`, or `tests/tokenizer.rs` failures: the BPE no longer matches HF `tokenizers`. `Tokenizer` rejects what it cannot reproduce: `unsupported: BPE byte_fallback`, `unsupported: BPE dropout`, `unsupported: tokenizer model type …`.
- `missing: mask token in the tokenizer`: the decision prompt needs `[CLS]`, `[SEP]` and `[MASK]`. Our tokenizers fix them at ids 2, 3 and 4 (SPEC, "Tokenizer").
- Different answers under llama.cpp only: llama.cpp ignores the `skycmd.tokenizer.*` keys (NFC normalizer, lstrip/rstrip). Text that is not already NFC can tokenize differently there. Normalize states to NFC before sending them.

### `` can't find crate for `core` `` when building for wasm32

The `wasm32-unknown-unknown` standard library is not installed. `scripts/build-wasm.sh` adds it with `rustup target add wasm32-unknown-unknown`, which only works with a rustup-managed toolchain. If `rustc` comes from a system package manager, install rustup or that package manager's wasm32 standard library.

### `wasm-pack not found`

The script stops with `wasm-pack not found: cargo install wasm-pack (or brew install wasm-pack)`. Install it as suggested. wasm-pack downloads its own `wasm-opt` on first use, so a separate binaryen install is not needed.

### `wasm-opt` rejects the module, or SIMD is silently off

Two separate problems:

- `wasm-opt` validates features. A module with SIMD128 needs `--enable-simd` (and our build also uses bulk memory, non-trapping float-to-int, sign extension and mutable globals). Both `crates/sky-infer/Cargo.toml` and `crates/sky-web/Cargo.toml` set these flags in `[package.metadata.wasm-pack.profile.release]`. A new crate built with wasm-pack needs the same block.
- The kernels pick their SIMD backend with `#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]`. Without `+simd128` they compile the portable fallback and still work, only slower. Cargo also lets a `RUSTFLAGS` environment variable **replace** the `rustflags` of `crates/sky-web/.cargo/config.toml`, so a stray `RUSTFLAGS` drops the flag. `build-wasm.sh` appends `-C target-feature=+simd128` to `RUSTFLAGS` for this reason; prefer the script over a bare `wasm-pack build`. If `bench()` is suddenly several times slower, check this first.

### Metal toolchain and `DEVELOPER_DIR` (training)

`sky-train` builds MLX through mlx-rs, which compiles Metal shaders. If `xcode-select` points to the Command Line Tools rather than Xcode, the Metal compiler is not found. Both training scripts default to Xcode:

```sh
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
```

Export the same variable when you run `cargo build -p sky-train` by hand. With a recent Xcode, the Metal toolchain is a separate download (`xcodebuild -downloadComponent MetalToolchain`, docs/research.md).

### The site works locally but not on GitHub Pages

The site is served from `https://<user>.github.io/skycmd/`, not from the domain root. VitePress therefore needs `base: '/skycmd/'` in the site config. Everything under `site/public/` (`pkg/sky_web.js`, `pkg/sky_web_bg.wasm`, `models/*.gguf`, `models/manifest.json`, `bench.json`) must be fetched with that base: use VitePress's `withBase()` or `import.meta.env.BASE_URL`, never a hard-coded `/models/...`. The usual symptom is a 404 on the `.gguf` or `.wasm` request in the browser's network panel. Also run `scripts/build-wasm.sh` before building the site (`npm --prefix site run docs:build`): `site/public/pkg` is gitignored, so a fresh clone has no module.

### Other messages you may see

| message | where | meaning |
|---|---|---|
| `skipping golden test: …/models-src/oscar is absent` | `tests/golden.rs` | the oscar reference model is not downloaded; the golden tests pass without checking anything |
| `unknown ladder configuration X` | `sky-convert --random`, `sky-bench --random` | only the shipped shapes exist there (`s1-pico`, `s1-nano`, `s1-micro`, the s2 ladder, `s3-pico`, `s3-nano`) |
| `skip models/X/ (no model.safetensors)` | `scripts/export-models.sh` | the directory is not an exported checkpoint |
| `no trained model under models/: the site falls back to the oracle policy` | `scripts/export-models.sh` | nothing to convert; the site plays with the oracle |
| `unsupported GGUF feature: GGUF version N` | `sky-gguf` reader | only GGUF v2 and v3 are read |
| `decision model has no valid max_head_tokens` | llama.cpp server | the GGUF lacks `modern-bert.decision.max_head_tokens`; `sky-convert` always writes it, so the file came from elsewhere |
| (no message) the game stutters | `GameSession.timing()` | a decision is slower than the frame; raise `set_decide_every` |

## Where to go from here

You now have every piece: data, tokenizer, training, calibration, GGUF export, a SIMD128 engine and the games wired around it. Go back to the [tutorial index](./) for any chapter, or [play the games](/play/) with the model this tutorial built.
