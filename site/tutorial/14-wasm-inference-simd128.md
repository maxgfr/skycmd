# 14 A WASM inference engine with SIMD128

`sky-infer` is a small, dependency-light inference engine for Laya-format decision models (ModernBERT encoder plus decision head) and, since chapter 19, for the s2 and s3 models (`skycmd-s2` architecture, `src/s2/`). It compiles to native code for tools and to `wasm32` with SIMD128 for the browser. This chapter covers the s1 path: how it loads a GGUF file, runs one forward pass per question without allocating, and why the browser build computes in f32 even when the file is int8.

## Why a dedicated engine

A decision model never generates text. One question is one forward pass over at most `max_len = 256` tokens, and the answer is a softmax over a handful of marker positions. That makes the runtime simple, and it allows a few optimizations a general LLM runtime cannot make:

- **Only the question's type runs.** The head is evaluated for one type embedding only. llama.cpp's graph evaluates it for all three types, because the type is not a graph input (`build_decision_head` in `models-src/modern-bert.cpp`).
- **The last head layer runs only at the markers.** Keys and values are needed for every position, but queries, the MLP and the scorer are only needed at the `[MASK]` tokens.
- **The prompt prefix is cached.** In a game loop the question and its options never change; only the state does.
- **No allocation in the forward pass.** Every buffer is sized once from the config.

Besides `sky-gguf`, the default build depends only on `unicode-normalization` and `serde_json` (for the JSON API). The `wasm` feature adds `wasm-bindgen` and `js-sys`.

## Module map

| file | role |
|---|---|
| `tokenizer/` | byte-level BPE that reproduces HF `tokenizers` exactly, with a word cache |
| `prompt.rs` | option rendering, `build_prefix` / `finish_sequence` (truncation rules), temperature buckets |
| `model.rs` | `ModelConfig`, weights (`Weight::F32` or `Weight::Q8`), the forward pass, `Workspace` |
| `kernels.rs` | matrix products (f32 and Q8_0), attention, LayerNorm, RoPE, `fast_exp`, `erf` |
| `quant.rs` | calibration and outlier channels for Q8_0 (chapter 13) |
| `engine.rs` | `Engine`: tokenizer + model + prompt + readout, with caches |
| `api.rs` | System One request/response types (JSON behind the `json` feature) |
| `wasm.rs` | `SkyEngine`, the wasm-bindgen API (feature `wasm`) |
| `convert.rs` | Laya directory to GGUF (chapter 13) |
| `ternary.rs`, `lut.rs`, `num.rs` | 2-bit ternary weights, lookup-table kernels and number value features for s2/s3 (chapters 19 and 21) |
| `s2/` | the s2/s3 model, prompt and engine (static cache, early exits, loops) |

## Loading a GGUF file

Each matrix is held as either f32 or Q8_0 blocks; there is no f16 compute path. F16 and BF16 tensors are widened to f32 when the file is read, while Q8_0 tensors stay quantized.

From `crates/sky-infer/src/model.rs`:

```rust
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(g)?;
        let n_enc = cfg.num_layers;
        let get = move |name: &str| -> Result<Option<Loaded>> {
            let Some(info) = g.tensor(name) else {
                return Ok(None);
            };
            let shape = info.shape();
            let bytes = g.tensor_bytes(info);
            let w = match info.dtype {
                GgmlType::Q8_0 => Weight::Q8(kernels::blocks_from_ggml(bytes)),
                ty => Weight::F32(sky_gguf::dequantize(ty, bytes)?),
            };
            Ok(Some(Loaded { shape, w }))
        };
```

`ModelConfig::from_gguf` reads the `modern-bert.*` keys back. The encoder depth is `block_count − decision.block_count`. The decision temperatures are every key under `modern-bert.decision.temperature.`. `max_len` comes from `decision.max_tokens`, or else from `context_length` capped at 512. `ModelConfig::validate` rejects configs the engine cannot run, for example a hidden size that does not split into heads of even size.

## The forward pass

`Model::forward(ws, ids, markers, qtype, logits)` follows the SPEC pseudo-code: embeddings and their LayerNorm, then `L` encoder layers. Every third layer is global (θ = 160000); the others attend inside `|i − j| ≤ local_attention / 2` with θ = 10000. Then come `final_norm`, the type embedding, the head layers and the scorer.

### The last head layer, at the markers only

Below is the branch that runs the final head layer. Keys and values are computed for the `t` positions; queries, attention output, MLP and residual only for the `k` markers (k is 2 to 7 in our games, against up to 256 tokens).

From `crates/sky-infer/src/model.rs`:

```rust
                // last head layer: keys/values for every position, everything else only at the markers
                let kv = &mut qkv[..t * 2 * d];
                linear(hl.in_proj.rows(d, 3 * d), hn, t, kv, false, xq);
                for (m, &p) in markers.iter().enumerate() {
                    mh[m * d..(m + 1) * d].copy_from_slice(&hn[p * d..(p + 1) * d]);
                    mx[m * d..(m + 1) * d].copy_from_slice(&x[p * d..(p + 1) * d]);
                }
                let mq = &mut mq[..k * d];
                linear(hl.in_proj.rows(0, d), mh, k, mq, false, xq);
                let a = Attention {
                    k: &kv[..],
                    v: &kv[d..],
                    kv_stride: 2 * d,
                    n_kv: t,
                    n_heads: hh,
                    head_dim: hhd,
                    window: None,
                };
                let matt = &mut matt[..k * d];
                a.run(mq, d, markers, matt, d, attn);
```

`Linear::rows(start, end)` is a view over a range of output rows, so the packed `in_proj` matrix `[3d, d]` is split into its Q and KV parts without copying. With one head layer (s1-pico, s1-nano), this branch is the whole head.

### No allocation per decision

`Workspace::new(cfg)` sizes every scratch buffer for `max_len` tokens once: hidden states, QKV, MLP, quantized activations, attention scores, marker rows. `forward` only slices them. The engine keeps its own `ids`, `markers` and `logits` vectors with the same capacity. A frame in the browser therefore makes no allocation beyond the probability `Vec` returned to the caller.

## Kernels

All matrix products use one layout: activations `x[rows][k]` and weights `w[n][k]`, which is the PyTorch `nn.Linear` layout and also the GGUF byte order. Both operands are contiguous along `k`, so every output is a dot product. `matmul_f32` computes tiles of 4 rows × 4 outputs with SIMD accumulators, cache-blocked so that a panel of weights (`L1_BLOCK_BYTES = 32 KiB`) stays hot while the rows stream through. A bias and a residual connection (`accumulate`) are fused into the store.

### One SIMD layer, four back ends

The kernels are written once against a tiny vector API (`zero`, `splat`, `load`, `store`, `fma`, `hsum`, `hsum4`, plus `prep`, `pdot`, `i2f`, `shl7_add` for int8). Four `mod simd` implementations are selected with `cfg`: NEON on aarch64, SIMD128 on wasm32, SSE2 on x86_64, and a portable scalar fallback. Here is the SIMD128 version.

From `crates/sky-infer/src/kernels.rs`:

```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use core::arch::wasm32::*;

    pub type V = v128;
    pub type VI = v128;

    #[inline(always)]
    pub fn zero() -> V {
        f32x4_splat(0.0)
    }
    #[inline(always)]
    pub fn splat(v: f32) -> V {
        f32x4_splat(v)
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        v128_load(p as *const v128)
    }
```

Baseline SIMD128 has no fused multiply-add, so `fma(acc, a, b)` is `f32x4_add(acc, f32x4_mul(a, b))`; a build with `+relaxed-simd` uses `f32x4.relaxed_madd` instead (Chrome, Edge and Node run it, Safari does not; `BENCH.md`). The `cfg` also means a wasm build without `+simd128` still compiles and silently uses the scalar fallback. Chapter 17 shows how to make sure the flag is set.

### int8 products on SIMD128

Baseline SIMD128 has no 8-bit dot product either. The closest instruction is `i32x4.dot_i16x8_s`, so each 32-byte Q8_0 block is first widened to four `i16x8` vectors.

From `crates/sky-infer/src/kernels.rs`:

```rust
    #[inline(always)]
    pub fn prep(a: &[i8; 32]) -> P {
        unsafe {
            let a0 = v128_load(a.as_ptr() as *const v128);
            let a1 = v128_load(a.as_ptr().add(16) as *const v128);
            [
                i16x8_extend_low_i8x16(a0),
                i16x8_extend_high_i8x16(a0),
                i16x8_extend_low_i8x16(a1),
                i16x8_extend_high_i8x16(a1),
            ]
        }
    }

    #[inline(always)]
    pub fn pdot(a: &P, b: &P) -> VI {
        let d0 = i32x4_dot_i16x8(a[0], b[0]);
        let d1 = i32x4_dot_i16x8(a[1], b[1]);
        let d2 = i32x4_dot_i16x8(a[2], b[2]);
        let d3 = i32x4_dot_i16x8(a[3], b[3]);
        i32x4_add(i32x4_add(d0, d1), i32x4_add(d2, d3))
    }
```

That widening, plus the activation quantization, costs as much as it saves. Native aarch64 is different: with `dotprod`, `sdot` does 16 int8 products per instruction. So the browser engine dequantizes Q8_0 files at load and computes in f32. Q8_0 keeps one advantage there, a smaller download.

From `crates/sky-infer/src/wasm.rs`:

```rust
    /// Load a GGUF model (s1 or s2; f32, f16, Q8_0 or ternary file) and compute in f32: with
    /// baseline SIMD128, int8 dot products are not faster than f32 in browsers, so Q8_0 files are
    /// dequantized at load (smaller download, f32 speed and accuracy).
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<SkyEngine, JsError> {
        Self::with_options(gguf_bytes, false, false)
    }
```

`SkyEngine.withOptions(bytes, q8, fast)` keeps the Q8_0 path reachable from JavaScript for experiments.

### Attention, exp and erf

`Attention::run` handles a block of 16 queries at a time. Per head, the scores are one strided matrix product restricted to the keys inside the block's window. Positions outside a query's window become `-inf` before the softmax, then a second product multiplies by the transposed values. Local layers therefore cost O(t · window) rather than O(t²).

`fast_exp` (Cody-Waite range reduction plus a degree-6 polynomial, relative error below 2e-7) and an Abramowitz-Stegun `erf` (about 1.5e-7) avoid libm calls. Those calls are slow in wasm, and avoiding them lets the loops vectorize. `gelu` is the exact erf form, as in PyTorch.

## The engine: caches for the game loop

`Engine::prepare` looks up the `(type, instructions, options)` prefix in a small most-recently-used list (`prefix_cache = 16` entries by default). On a miss it tokenizes and truncates the prefix once. Every call then appends only the state.

From `crates/sky-infer/src/engine.rs`:

```rust
        let idx = match hit {
            Some(i) => i,
            None => {
                let opts: Vec<&str> = options.iter().map(|o| o.as_ref()).collect();
                let prefix = prompt::build_prefix(
                    &self.tok,
                    &pt,
                    qtype,
                    instructions,
                    &opts,
                    self.model.cfg.head_max_len,
                    Some(&mut self.bpe),
                );
                if self.prefixes.len() >= self.prefix_cap.max(1) {
                    self.prefixes.pop();
                }
```

A second cache, `BpeCache` (4096 words), remembers the BPE split of words already seen. Game states such as `vy 0 dx 36 up 9 dn 7 nxt 10 alt 32` reuse a tiny set of words, so after a few frames tokenization is mostly lookups.

The engine exposes three levels of API:

- `decide_raw(qtype, instructions, options, state)` returns probabilities over already rendered option texts. The games use this fast path.
- `decide(&DecisionRequest)` / `decide_json(&str)` answer a full System One request: argmax, `confidence`, expected score, `noul`.
- `scores_raw` and `build_sequence` return raw logits and token ids, for debugging.

`decide_json` accepts the same request shape as llama.cpp's `/v1/systemone` (chapter 16 has a full `request.json`). The `decide` example runs it from the command line:

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo run --release -p sky-infer --example decide -- site/public/models/s1-pico.gguf request.json
```

## Building for the browser with SIMD128

The site does not load `sky-infer` on its own; it loads `crates/sky-web`, which links `sky-infer` (feature `wasm`) and the game cores into one module (chapter 15). The build script is `scripts/build-wasm.sh`.

From `scripts/build-wasm.sh`:

```sh
rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown ||
  rustup target add wasm32-unknown-unknown

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-web-wasm}
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"

wasm-pack build crates/sky-web --target web "$profile" --out-dir ../../site/public/pkg --no-typescript
```

Three things work together:

1. `RUSTFLAGS` with `-C target-feature=+simd128` makes rustc emit SIMD128 and selects the `mod simd` above. `crates/sky-web/.cargo/config.toml` sets the same flag for wasm builds started from that directory.
2. `[package.metadata.wasm-pack.profile.release]` in both `Cargo.toml` files passes `--enable-simd` (plus bulk memory, non-trapping float-to-int, sign extension and mutable globals) to `wasm-opt -O3`. Without them, `wasm-opt` rejects the SIMD instructions.
3. `--target web` produces an ES module with `init()` / `initSync()` that the VitePress site imports at runtime.

```sh
scripts/build-wasm.sh          # release, writes site/public/pkg/sky_web.js + sky_web_bg.wasm
scripts/build-wasm.sh --dev    # unoptimized, faster to build
```

The release module currently on disk (`site/public/pkg/sky_web_bg.wasm`) is about 0.9 MB and contains all four games and the s1, s2 and s3 engine.

## Measuring

The module exports `bench(engine, n)`. It plays Flappy Drone and times `n` AI decisions in batches of 10, because browsers coarsen `performance.now()`. Each decision is two forward passes (the `action` choice and the `danger` yes/no question) plus one physics step. The same call works under Node:

```sh
node --input-type=module -e '
import { readFileSync } from "node:fs";
const pkg = await import(process.cwd() + "/site/public/pkg/sky_web.js");
pkg.initSync({ module: readFileSync("site/public/pkg/sky_web_bg.wasm") });
const engine = new pkg.Engine(readFileSync("site/public/models/s1-pico.gguf"));
console.log(pkg.bench(engine, 500));
'
```

The native counterpart is `sky-bench` (one choice question per iteration, state changing every frame):

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo run --release -p sky-infer --bin sky-bench -- site/public/models/s1-pico.gguf --iters 300
```

Measured on an Apple M5 (Node 24 / V8 for wasm, F16 GGUF files, medians). The s1-pico row is current: the WASM number is `wasm_us` in `site/public/bench.json`, the native ones are `sky-bench --questions 2|1` in `BENCH.md` (s3 section). The s1-nano and s1-micro rows were measured with the engine as it was before the s3 kernel work of [chapter 21](./21-s3-recursive-tiny) and have not been re-measured:

| model | params | wasm `bench()` per decision (2 forwards + physics) | native `sky-bench`, 2 questions on one state | native `sky-bench`, 1 question |
|---|---:|---:|---:|---:|
| s1-pico | 202,305 | 588.6 µs | 300.6 µs | 137.6 µs |
| s1-nano (earlier engine) | 1,134,209 | 3,541 µs | 1,980 µs | 974 µs |
| s1-micro (earlier engine) | 6,630,913 | about 32,600 µs | about 9,700 µs | about 4,600 µs |

The s1-micro row was measured for [chapter 12](./12-size-ladder) on the same machine while other jobs ran, so read it as an order of magnitude. The comparable columns are the first two: the same two questions on the same state, WASM against native. WASM costs about 2× native on pico today (1.8× on nano with the earlier engine). Expect a few percent of noise between runs, more if the machine is busy (with the earlier engine, the s1-pico WASM median moved between 631 and 697 µs across three runs on a loaded machine).

The s2 and s3 architectures of [chapter 19](./19-s2-architecture) and [chapter 21](./21-s3-recursive-tiny) use the same SIMD layer and the same SIMD128 build, plus panel kernels for products with few rows (`kernels::PanelMatrix`). They do less work per frame: the same `bench()` costs 409.1 µs for the 1.5M-parameter s2-nano and 49.6 µs for s3-pico, the site's default model (`site/public/bench.json`). `BENCH.md` ("Inference") breaks these down per question, with and without relaxed SIMD.

## Next

[15 Wiring the games](./15-wiring-the-games)
