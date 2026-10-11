# 09 Muon, WSD and the mlx-rs training loop

`sky-train` trains on Apple Silicon with mlx-rs. Muon updates the encoder's hidden matrices and AdamW updates everything else. A warmup-stable-decay schedule runs on wall-clock time, and one compiled function computes the loss and the gradients. This chapter goes through `crates/sky-train/src/optim.rs` and `session.rs` and ends with the throughput measured in `BENCH.md`.

## Why write an optimizer at all

`docs/research.md` lists what mlx-rs 0.32 provides: `fast::scaled_dot_product_attention`, `fast::rope`, `fast::layer_norm`, `value_and_grad` and graph compilation. It has no Muon and no transformer training example. The optimizer is about 200 lines of Rust over plain `Array`s, ported from `mlx.optimizers.Muon` (Python) and the usual AdamW.

## Two optimizers, one parameter map

The parameters are a flat map keyed by SPEC names (chapter 08). The optimizer group of a tensor is decided from its name alone:

`crates/sky-train/src/params.rs`
```rust
pub fn group_of(name: &str, ndim: usize) -> Group {
    let hidden = name.starts_with("encoder.layers.")
        && (name.ends_with(".attn.Wqkv.weight")
            || name.ends_with(".attn.Wo.weight")
            || name.ends_with(".mlp.Wi.weight")
            || name.ends_with(".mlp.Wo.weight"));
    if hidden {
        Group::Muon
    } else if ndim == 2 && !name.contains("embeddings") && name != "type_emb.weight" {
        Group::AdamDecay
    } else {
        Group::AdamNoDecay
    }
}
```

Three groups:

- **Muon**: the four matrices of every encoder layer.
- **AdamW with weight decay**: the other matrices, meaning the decision head, the scorer and the pretraining-only `mlm.dense`.
- **AdamW without decay**: embeddings, `type_emb`, every norm and every bias.

The research notes explain the split. Muon can collapse at the interface between representation and readout (paper 2608.07436), so embeddings, the head and the scorer stay on AdamW.

## Muon: orthogonalised momentum

Muon keeps a momentum buffer per matrix and replaces the raw update with its nearest semi-orthogonal matrix. All singular values are pushed towards 1, so the step is the same size in every direction, whatever the conditioning of the gradient. The orthogonalisation uses five Newton–Schulz iterations with the quintic coefficients of the reference implementation:

`crates/sky-train/src/optim.rs`
```rust
/// Newton–Schulz quintic coefficients of Muon.
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);
```

`crates/sky-train/src/optim.rs`
```rust
pub fn newton_schulz(g: &Array, steps: usize) -> R<Array> {
    let (a, b, c) = NS_COEFFS;
    let transpose = g.dim(0) > g.dim(1);
    let mut x = if transpose { g.t() } else { g.clone() };
    let norm = x.square()?.sum(None)?.sqrt()?;
    x = x.divide(norm.add(Array::from_f32(1e-7))?)?;
    for _ in 0..steps {
        let gram = ops::matmul(&x, x.t())?;
        let poly = gram
            .multiply(Array::from_f32(b))?
            .add(ops::matmul(&gram, &gram)?.multiply(Array::from_f32(c))?)?;
        x = x.multiply(Array::from_f32(a))?.add(ops::matmul(&poly, &x)?)?;
    }
    Ok(if transpose { x.t() } else { x })
}
```

Tall matrices are transposed first, so the Gram matrix is always the small side. For `Wqkv` (`[3d, d]`) that is `d × d`. The iteration is tuned for speed, not exact convergence. The `muon_newton_schulz_orthogonalizes` test only asks that every singular value lands in `[0.6, 1.3]`, with a mean within 0.2 of 1.

The update itself uses Nesterov momentum, a small L2 term added to the gradient (`muon_weight_decay = 0.01`, as in the MLX version), and a scale of `sqrt(max(1, rows / cols))` for tall matrices:

`crates/sky-train/src/optim.rs`
```rust
                Group::Muon => {
                    let g = if c.muon_weight_decay != 0.0 {
                        g.add(p.multiply(Array::from_f32(c.muon_weight_decay))?)?
                    } else {
                        g
                    };
                    let buf = match self.mom.get(name) {
                        Some(b) => b.multiply(Array::from_f32(mu))?,
                        None => ops::zeros_like(&g)?,
                    }
                    .add(g.multiply(Array::from_f32(1.0 - mu))?)?;
                    // Nesterov.
                    let upd = g
                        .multiply(Array::from_f32(1.0 - mu))?
                        .add(buf.multiply(Array::from_f32(mu))?)?;
                    let o = newton_schulz(&upd, c.ns_steps)?;
                    let ratio = (p.dim(0) as f32 / p.dim(1) as f32).max(1.0).sqrt();
                    *p = p.subtract(o.multiply(Array::from_f32(muon_lr * ratio))?)?;
                    self.mom.insert(name.clone(), buf);
                }
```

The momentum `mu` warms up linearly from 0.85 to 0.95 over the first 300 updates (`muon_momentum()`), which keeps the first steps from overshooting while the buffer is still mostly zeros.

## AdamW and clipping

The AdamW branch is standard: β1 = 0.9, β2 = 0.95, ε = 1e-8, bias correction, and decoupled weight decay (`--weight-decay`, default 0.01) applied only to the `AdamDecay` group. Before either branch, all gradients are scaled by one global factor `min(1, clip_norm / ‖g‖)` (`--clip-norm`, default 1.0, `0` disables). `update` returns the norm from before clipping, and the training log records it as `grad_norm`.

The defaults live in `OptConfig::default()`. Each subcommand then picks its own peak learning rates, which `--adam-lr` and `--muon-lr` override:

`crates/sky-train/src/main.rs`
```rust
impl OptFlags {
    fn build(&self, adam_lr: f32, muon_lr: f32) -> OptConfig {
        OptConfig {
            adam_lr: self.adam_lr.unwrap_or(adam_lr),
            muon_lr: self.muon_lr.unwrap_or(muon_lr),
            weight_decay: self.weight_decay,
            clip_norm: self.clip_norm,
            ..OptConfig::default()
        }
    }
}
```

`pretrain` calls `opt.build(3e-3, 0.02)` and `decide` calls `opt.build(1e-3, 0.01)`. Decision training starts from a pretrained encoder, so it uses smaller steps.

## WSD on a time budget

Both training phases are **time-budgeted**: you pass `--minutes`, not a step count. The learning-rate multiplier is a warmup-stable-decay curve over the *fraction of the time budget used*:

`crates/sky-train/src/optim.rs`
```rust
/// Warmup-stable-decay multiplier at training progress `frac` in `[0, 1]`:
/// linear warmup over the first 2%, constant, then linear decay over the last
/// 20% down to 0.1x.
pub fn wsd(frac: f64) -> f64 {
    let f = frac.clamp(0.0, 1.0);
    if f < 0.02 {
        (f / 0.02).max(0.01)
    } else if f < 0.8 {
        1.0
    } else {
        1.0 - 0.9 * (f - 0.8) / 0.2
    }
}
```

| progress | 0% | 1% | 2–80% | 90% | 100% |
|---|---:|---:|---:|---:|---:|
| multiplier | 0.01 | 0.5 | 1.0 | 0.55 | 0.1 |

The training loop computes `frac = elapsed_secs / budget_secs` before every step. Both the Muon and the AdamW rates are multiplied by `wsd(frac)`. This has two consequences:

- A model gets the full schedule (warmup, plateau, cooldown) whatever its speed. pico, nano and micro can share one recipe with different minute budgets, and nobody has to guess step counts in advance.
- The cooldown starts at 80% of the time, which is when WSD gives most of its gain. Training at a fixed budget always ends on a decayed checkpoint.

The pico pretraining log shows it. The last logged step of `runs/s1-pico/pretrain/log.jsonl` (step 21,180 of an 8-minute run) has `lr` 0.000311, about 0.104 × the 3e-3 peak, just before the multiplier reaches 0.1. Runs may be retrained, so your numbers will differ slightly.

## One compiled step

A training step is "loss and gradient of every parameter, then one optimizer update". `Session` compiles the first half with `mlx_rs::transforms::compile`. The parameters go in as a sorted list of arrays, followed by the batch arrays, and the compiled function returns the loss followed by one gradient per parameter:

`crates/sky-train/src/session.rs`
```rust
impl CompiledGrad {
    /// Compiles `loss(params, batch_arrays)`; parameters are passed in sorted name order.
    fn new(names: Vec<Rc<str>>, loss: impl Fn(&ParamMap, &[Array]) -> R<Array> + 'static) -> Self {
        let keys = names.clone();
        let step = move |args: &[Array]| -> R<Vec<Array>> {
            let (pv, bv) = args.split_at(keys.len());
            let params: ParamMap = keys.iter().cloned().zip(pv.iter().cloned()).collect();
            let f = |p: HashMap<Rc<str>, Array>, b: &[Array]| -> R<Vec<Array>> { Ok(vec![loss(&p, b)?]) };
            let (mut values, grads) = keyed_value_and_grad(f)(params, bv)?;
            for k in &keys {
                values.push(grads[k].clone());
            }
            Ok(values)
        };
        Self { names, f: Box::new(mlx_rs::transforms::compile::compile(step, false)) }
    }
```

A few details keep the compiled graph stable from step to step:

- The step always passes a `key_valid` array, even when the batch has no padding (`all_valid` fills it with `true`). The graph then has the same inputs every step.
- MLM prediction lists are padded to a multiple of 256 entries with zero weight (`mask_batch`), and decision batches are padded to a multiple of 16 tokens (`make_batch`). The number of distinct shapes stays small.
- The optimizer update runs eagerly. After it, `apply_update` evaluates the parameters, the optimizer state, the loss and the gradient norm together, so each step ends with one synchronisation.

### bf16 compute, f32 master weights

The default `--precision bf16` makes bfloat16 copies of the weights inside the loss function and runs matmuls and activations in bf16. The master weights, the optimizer state and the losses stay in f32. Logits are cast back with `to_f32` before any softmax.

`crates/sky-train/src/model.rs`
```rust
fn compute_params<'a>(p: &'a ParamMap, cfg: &ModelConfig) -> R<Cow<'a, ParamMap>> {
    if !cfg.bf16 {
        return Ok(Cow::Borrowed(p));
    }
    let mut out = ParamMap::with_capacity(p.len());
    for (k, v) in p {
        let v = if v.dtype() == Dtype::Float32 { v.as_dtype(Dtype::Bfloat16)? } else { v.clone() };
        out.insert(k.clone(), v);
    }
    Ok(Cow::Owned(out))
}
```

## Checkpoints and resume

`Session::save` writes `model.safetensors`, `optimizer.safetensors`, `state.json` (step, elapsed seconds, budget, RNG state, data cursor, model and optimizer configs) and `config.json`, plus phase-specific files such as `tokenizer.json` and `rl_agent_config.json`. It writes into `<dir>.tmp` and renames, so a crash never leaves a half-written `last/`. Checkpoints are written every `--checkpoint-minutes` (default 10) and at the end.

`--resume` reloads `<out>/last`. The time budget counts across resumes, because `elapsed_secs` is saved, so the WSD curve continues where it stopped. The `checkpoint_resume_is_reproducible` test checks that save, load and three more steps give the same weights as three uninterrupted steps. It runs on the CPU device: on the GPU, the scatter-add of the embedding gradient is not bit-deterministic.

## Throughput

`sky-train bench` times forward, backward and update on random tokens and appends rows to `BENCH.md`:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-train
cargo build --release -p sky-train
$CARGO_TARGET_DIR/release/sky-train bench --config configs/s1-pico.json
$CARGO_TARGET_DIR/release/sky-train bench --config configs/s1-pico.json --precision f32
```

Defaults: `--mlm-batch 128 --mlm-seq 128 --dec-batch 64 --dec-seq 192 --dec-options 4 --warmup 10 --steps 50`. These are the bf16 rows currently in `BENCH.md` (mlx-rs 0.32, MLX 0.32.2, Metal):

| model | phase | step (ms) | tok/s | peak mem (MB) |
|---|---|---:|---:|---:|
| s1-pico | mlm | 24.3 | 673,236 | 367 |
| s1-pico | decision | 20.5 | 598,104 | 308 |
| s1-nano | mlm | 55.8 | 293,518 | 1035 |
| s1-nano | decision | 50.6 | 243,002 | 926 |
| s1-micro | mlm | 172.7 | 94,883 | 1939 |
| s1-micro | decision | 173.4 | 70,881 | 1702 |

bf16 is faster than f32 on every rung: about 1.2× for pico MLM, about 2× for nano, and about 1.7× for micro (compare the f32 rows in `BENCH.md`). The gain is smallest for pico, where matmuls are a small part of each step. Real runs match the benchmark: the end of the pico pretraining log shows about 710k tokens/s.

To check the optimizer and the gradients without training anything:

```bash
cargo test -p sky-train --release optim::
cargo test -p sky-train --release session::
```

`session::tests::numeric_gradient_check` compares autodiff gradients of the full decision forward against central finite differences for tensors in the encoder, the head and the scorer.

## Next

[10 Pretraining and decision training](./10-pretraining-and-decision-training)
