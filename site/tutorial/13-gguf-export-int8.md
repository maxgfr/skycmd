# 13 GGUF export and int8 quantization

A trained skycmd model leaves `sky-train` as a Laya-layout directory of safetensors and JSON configs. This chapter turns it into one self-contained GGUF file that both our WASM engine and llama.cpp can read, and shows how far you can push the weights down to int8 (Q8_0) before the probabilities move.

Everything here is about s1 files (`modern-bert`). The s2 and s3 models, including the site's default s3-pico, are written directly as GGUF by `sky-train s2-export` with the `skycmd-s2` architecture and ternary weights ([chapter 19](./19-s2-architecture), [chapter 21](./21-s3-recursive-tiny)); those files run only in the skycmd engine (CLI, server, npm, Python, WASM), not in llama.cpp.

## Where we are in the pipeline

```text
runs/s1-pico/decide/last   --sky-train export-->   models/s1-pico/          (Laya layout, f32 safetensors)
models/s1-pico/            --sky-convert------->   site/public/models/s1-pico.gguf
```

`sky-train export` writes `model.safetensors`, `encoder/config.json`, `rl_agent_config.json`, `config.json` and a `tokenizer/` directory, the same layout as the reference model `oscar-1-17m`. Training is not covered here; we only read what it produced. The `scripts/train-ladder.sh` loop ends with exactly this call:

```sh
$T export --run "$run" --out "models/s1-$name"
```

## The GGUF container (`crates/sky-gguf`)

GGUF is a flat binary file: a header, a list of typed metadata keys, a table of tensor descriptors, then aligned tensor data. `sky-gguf` reads and writes version 3 (it also parses version 2) with no dependencies. The reader borrows the bytes it was given and never copies tensor data, which matters in the browser where the model arrives as one `Uint8Array`.

Two details are easy to get wrong.

- **Dimension order.** ggml lists dimensions innermost first. A PyTorch `[rows, cols]` matrix is stored with `dims = [cols, rows]` and the same bytes. `gguf-inspect` prints the ggml order; `TensorInfo::shape` returns the PyTorch order.
- **Alignment.** Every tensor starts on a multiple of `general.alignment` (32 by default). The writer always emits that key and pads each tensor.

The writer computes offsets and padding in one pass.

From `crates/sky-gguf/src/writer.rs`:

```rust
        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            put_string(&mut head, &t.name);
            head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&t.dtype.to_u32().to_le_bytes());
            head.extend_from_slice(&offset.to_le_bytes());
            offsets.push(offset);
            offset = (offset + t.data.len() as u64).div_ceil(self.alignment) * self.alignment;
        }
        pad_to(&mut head, self.alignment);
        w.write_all(&head)?;
```

### Element types

Only four ggml types are converted to and from f32: F32 (id 0), F16 (1), Q8_0 (8) and BF16 (30). Any other id is kept as `Other(n)`, so a file that uses other types can still be inspected.

The f16 conversion rounds to nearest even, like numpy and torch. A unit test checks the round trip of all 65,536 bit patterns.

### The Q8_0 block

Q8_0 groups 32 consecutive values of a row into a block: one f16 scale `d = max|x| / 127`, then 32 signed bytes `q = round(x / d)`. That is 34 bytes per 32 values, against 64 for f16.

From `crates/sky-gguf/src/dtype.rs`:

```rust
pub fn quantize_q8_0_into(x: &[f32], out: &mut [u8]) {
    assert!(x.len().is_multiple_of(QK8_0));
    assert_eq!(out.len(), x.len() / QK8_0 * Q8_0_BLOCK_BYTES);
    for (blk, dst) in x
        .chunks_exact(QK8_0)
        .zip(out.chunks_exact_mut(Q8_0_BLOCK_BYTES))
    {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        dst[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (q, &v) in dst[2..].iter_mut().zip(blk) {
            *q = ((v * id).round() as i32).clamp(-128, 127) as i8 as u8;
        }
    }
}
```

This is ggml's `quantize_row_q8_0_ref`, so our Q8_0 bytes are interchangeable with llama.cpp's. A row whose length is not a multiple of 32 cannot be split into blocks, and the writer refuses it.

### Inspecting a file

`gguf-inspect` prints every key and every tensor. Its usage string is `gguf-inspect <file.gguf> [--max-array N]`.

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-gguf
cargo run -q --release -p sky-gguf --bin gguf-inspect -- site/public/models/s1-pico.gguf --max-array 4
```

Abridged output for the shipped s1-pico file:

```text
site/public/models/s1-pico.gguf: GGUF v3, 545280 bytes, alignment 32

51 metadata keys:
  general.architecture (string) = "modern-bert"
  modern-bert.block_count (u32) = 3
  modern-bert.decision.type (string) = "laya"
  modern-bert.decision.block_count (u32) = 1
  modern-bert.decision.max_head_tokens (u32) = 96
  modern-bert.decision.temperature.choice.2 (f32) = 1.0013216
  tokenizer.ggml.model (string) = "gpt2"
  tokenizer.ggml.mask_token_id (u32) = 4
  ...
33 tensors (ggml dims, ne[0] = innermost):
  token_embd.weight                        [64, 1024]             F16        131072 bytes
  blk.0.attn_qkv.weight                    [64, 192]              F16         24576 bytes
  blk.2.attn_qkv.weight                    [64, 192]              F32         49152 bytes
  ...
total: 202305 parameters, 514308 bytes of tensor data
```

Note `block_count = 3`: two encoder layers plus one decision-head layer. The head blocks continue the `blk.N` numbering, as llama.cpp expects.

## Converting a checkpoint (`sky-convert`)

`sky-convert` lives in `crates/sky-infer/src/bin/sky-convert.rs`. Its usage string is:

```text
usage: sky-convert <laya_dir | --random s1-pico|s1-nano|s1-micro> <out.gguf> [--type f32|f16|bf16|q8_0] [--head-type f32|f16|bf16|q8_0] [--pre NAME] [--name NAME]
       sky-convert --random s2-pico|s2-nano|s2-micro|s3-pico|s3-nano <out.gguf> [--format f32|f16|ternary] [--tokenizer tok-s2.json] [--name NAME]
```

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo build --release -p sky-infer --bin sky-convert --bin sky-bench
$CARGO_TARGET_DIR/release/sky-convert models/s1-pico /tmp/s1-pico-f16.gguf --name skycmd-s1-pico
$CARGO_TARGET_DIR/release/sky-convert models/s1-pico /tmp/s1-pico-q8.gguf --type q8_0 --name skycmd-s1-pico
```

`--random s1-nano` writes a model with random weights and a small test tokenizer instead. Use it to exercise the runtime before a real model exists. The second form does the same for the s2 and s3 shapes (`--tokenizer` takes a real s2 tokenizer such as `data/tok/tok-s2-1024.json`).

### Which tensor gets which type

The type policy is in `LayaSources::to_gguf`.

From `crates/sky-infer/src/convert.rs`:

```rust
            let mut ty = if t.shape.len() < 2 {
                GgmlType::F32
            } else if is_head {
                opts.head_type
            } else if gg == "token_embd.weight" && opts.ftype == GgmlType::Q8_0 {
                GgmlType::F16
            } else {
                opts.ftype
            };
            let inner = *t.shape.last().unwrap_or(&1);
            if ty == GgmlType::Q8_0 && inner % sky_gguf::QK8_0 != 0 {
                ty = GgmlType::F16; // as llama.cpp: fall back when rows do not split into blocks
            }
```

The rules, in plain words:

- Norm weights and biases (1-D tensors) are always F32.
- The decision head (`blk.N` for N at or above the encoder depth, `token_types`, `cls.*`) uses `--head-type`, F32 by default. It is small, and its errors go straight into the scores.
- Encoder matrices use `--type`, F16 by default.
- With `--type q8_0`, the token embeddings stay F16. A lookup gains no speed from int8, and on oscar the rounding of the embeddings alone moves the probabilities by about 0.02.

### Metadata: the `modern-bert` and `decision` keys

Names and keys follow llama.cpp's `bert.py` converter (`ModernBertDecisionModel`), so the file loads in llama.cpp without a second conversion. The decision-specific part is short.

From `crates/sky-infer/src/convert.rs`:

```rust
    // decision head
    w.add_kv(k("decision.type"), "laya");
    w.add_kv(k("decision.block_count"), u(cfg.head_layers));
    w.add_kv(k("decision.max_head_tokens"), u(cfg.head_max_len));
    // not read by llama.cpp: the Laya max_len (the server only limits to the context size)
    w.add_kv(k("decision.max_tokens"), u(cfg.max_len));
    for (name, t) in &cfg.temperatures {
        w.add_kv(
            k(&format!("decision.temperature.{name}")),
            OwnedValue::F32(*t),
        );
    }
```

The temperatures come from `rl_agent_config.json`. Laya bucket names such as `choice:3-5` become `choice.3_5` (see `laya_bucket_to_key` in `model.rs`), the same rewrite `bert.py` does.

### Tokenizer and the `systemone` template

The tokenizer goes in too: `tokenizer.ggml.tokens`, `merges`, the special ids (`[CLS]=2`, `[SEP]=3`, `[MASK]=4`), and `tokenizer.ggml.token_type_count = 3` (one token type per question type). The converter also writes the Jinja `systemone` chat template that llama.cpp's server renders. `systemone_template()` builds it from the special token names, so it renders the same `[CLS] {type} question: … [SEP] ([MASK] option)* [SEP] state [SEP]` sequence as the training code.

A few keys under `skycmd.tokenizer.*` (NFC normalizer, lstrip/rstrip ids, `ignore_merges`) let `sky-infer` reproduce Hugging Face `tokenizers` exactly. llama.cpp ignores them; chapter 17 explains when that matters.

## int8 that keeps its accuracy

Plain round-to-nearest Q8_0 does not work well on these small encoders, for a reason explained at the top of the quantization module.

From `crates/sky-infer/src/quant.rs`:

```rust
//! The residual stream of these small encoders has a few "massive" channels (~20x the others).
//! Inside a Q8_0 block such a channel sets the scale of the 31 others, and its weight rounding
//! error is amplified 20x. As in LLM.int8() (Dettmers et al., 2022), the input channels whose
//! calibrated magnitude stands far above the rest are computed in f32 with exact weight columns,
//! while everything else stays an int8 x int8 product. [`Calibration`] collects the per-channel
//! input magnitudes of every matrix product with a forward pass over [`CALIBRATION_PROMPTS`].
```

The fix has three steps.

1. **Calibrate.** Run the f32 model on twelve built-in prompts (`CALIBRATION_PROMPTS`: game states, English and French prose, JSON). Record `max |x|` per input channel at every matrix product ("site").
2. **Pick the outliers.** A channel is an outlier when its magnitude is above `OUTLIER_RATIO = 3.0` times the median channel. There are at most `k / 16` outliers per matrix.
3. **Choose the scales.** `quantize_q8_exact_cols` tries, in each block that contains an outlier column, the scales that make the outlier weight an exact multiple without clipping the block. It keeps the scale with the lowest weighted error (the outlier columns weigh 10⁴ times more). The output is still standard Q8_0 bytes.

At load time, the engine moves those columns out of the int8 product and adds them back in f32 (`Outliers::split_q8` and the `axpy` loop in `matmul_q8`).

### Activations: one or two int8 passes

Weights are only half of an int8 product; the activations are quantized on the fly too. `ActQuant` has two modes.

From `crates/sky-infer/src/kernels.rs`:

```rust
pub enum ActQuant {
    /// One Q8_0 block per 32 values (ggml `q8_0` activations): fastest.
    Q8,
    /// Q8_0 plus a second Q8_0 pass on the rounding residual at 1/128 of the scale
    /// (`x ~ d * (q + r / 128)`), summed in int32 as `128 * dot(q, w) + dot(r, w)`: ~15-bit
    /// activations, still i8 x i8 -> i32 dot products.
    #[default]
    Q8x2,
}
```

`Q8x2` is the default. It costs a second integer dot product per block but keeps about 15 bits of each activation.

## How much accuracy does each format cost?

`crates/sky-infer/tests/golden.rs` runs the real oscar-1-17m model against reference outputs from the Python Laya runtime. It checks that the token ids and markers match exactly, then bounds the largest absolute probability error over all cases:

| path | test | tolerance (max abs prob error) |
|---|---|---|
| f32 from safetensors | `golden_f32_safetensors` | 1e-3 |
| F32 or F16 GGUF | `golden_gguf_conversion` | 1e-3 |
| Q8_0 weights quantized at load, `Q8x2` activations | `golden_q8_safetensors` | 2e-2 |
| F16 GGUF quantized at load | `golden_gguf_f16_quantized_at_load` | 2e-2 |
| Q8_0 GGUF file (plain llama.cpp Q8_0) | `golden_gguf_conversion` | 3e-2 (measures about 0.022) |
| Q8_0 weights, single-pass `Q8` activations | `golden_q8_fast_activations` | 4e-2 (measures about 0.025) |

A Q8_0 file is a little worse than quantizing at load. The file holds plain Q8_0 blocks in which the massive channels share their scale with the rest; only quantizing at load can move the outlier columns to f32. The test file therefore calls "ship F16, quantize at load" the recommended Q8_0 deployment.

The golden tests need `models-src/oscar/`, which is gitignored. Without it they print a skip message and pass.

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo test --release -p sky-infer --test golden -- --nocapture
cargo test --release -p sky-gguf
```

## What the site ships, and why

`scripts/export-models.sh` converts every `models/*/` directory that holds an s1 export (`model.safetensors`) with the default types: F16 encoder, F32 head. The script's header explains the choice. The golden tests measure 1e-3 for F16 against 3e-2 for Q8_0, and the browser engine computes in f32 anyway (chapter 14). On models this small, Q8_0 would save little download for a visible accuracy cost.

```sh
scripts/export-models.sh              # f16 (default)
scripts/export-models.sh --type q8_0  # Q8_0 encoder matrices, for comparison
```

Both commands overwrite the files in `site/public/models/`. To compare formats without touching what the site serves, call `sky-convert` directly with an output path under `/tmp`, as shown above.

The script also runs `sky-bench` on each file and writes `site/public/models/manifest.json` and `site/public/bench.json`. A value that is not available yet is written as `null`, and the Bench page shows `n/a` instead of a number. The script only knows s1 directories: the s2 and s3 files on the site (`s3-pico.gguf`, `s2-nano.gguf`, `s2-nano-doom.gguf`) come from `sky-train s2-export`, and the current `manifest.json` and `bench.json` also carry closed-loop and WASM columns that the script does not write.

| model | params | F16 GGUF bytes | Q8_0 GGUF bytes | native latency per question |
|---|---:|---:|---:|---:|
| s1-pico | 202,305 | 545,280 | 468,512 | 137.6 µs (`BENCH.md`) |
| s1-nano | 1,134,209 | 2,765,792 | 2,151,456 | 974 µs (earlier engine) |
| s1-micro | 6,630,913 | 16,694,464 | 13,008,128 | about 4,600 µs (earlier engine, [chapter 12](./12-size-ladder)) |

Of the s1 sizes, only s1-pico is published. "Earlier engine" marks numbers measured before the s3 kernel work of [chapter 21](./21-s3-recursive-tiny) and not re-measured since. The other sizes come from running `sky-convert models/s1-<name> /tmp/s1-<name>.gguf [--type q8_0]` on the trained checkpoints. Q8_0 saves 14% on pico and 22% on micro: the embedding table stays F16 and the head F32, and on pico those are most of the file.

To time the int8 path natively, `sky-bench` takes `--q8` (two-pass activations), `--q8-fast` (single pass) and `--no-calib`:

```sh
$CARGO_TARGET_DIR/release/sky-bench site/public/models/s1-pico.gguf --iters 300
$CARGO_TARGET_DIR/release/sky-bench site/public/models/s1-pico.gguf --q8 --iters 300
$CARGO_TARGET_DIR/release/sky-bench --random s1-micro --q8-fast --iters 300
```

On these models the int8 path is not a speed-up by default. Best median of 3 rounds per mode, one question per frame, measured on the M5 for this chapter while other jobs ran, with the engine as it was before the s3 kernel work (`sky-bench <file> [--q8 | --q8-fast] --questions 1 --iters 1000`; not re-measured since):

| model | F16 file, f32 compute | `--q8` | `--q8-fast` | Q8_0 file |
|---|---:|---:|---:|---:|
| s1-pico | 145 µs | 199 µs | 153 µs | 202 µs |
| s1-nano | 804 µs | 878 µs | 594 µs | 927 µs |
| s1-micro | 4,896 µs | 4,991 µs | 3,293 µs | 5,376 µs |

The two-pass `--q8` activations cost more than they save at every size. The single-pass `--q8-fast` mode is about 26% faster than f32 on nano and 33% on micro, but it is also the least accurate path in the golden tests (about 0.025 probability error, against 1e-3 for f32). The Q8_0 file timings come from one round each. The site keeps F16 and f32 compute.

## Next

[14 A WASM inference engine with SIMD128](./14-wasm-inference-simd128)
