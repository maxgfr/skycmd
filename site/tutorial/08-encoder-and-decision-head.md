# 08 The encoder and the decision head

A skycmd-s1 model is a small ModernBERT encoder followed by the Laya decision head. The head reads one hidden state per `[MASK]` marker and turns it into one logit per option. This chapter walks through the layout in `docs/SPEC.md`, the configs, the mlx-rs forward pass in `sky-train`, and the matching f32 path in `sky-infer`. The s2 and s3 models, including the default s3-pico of the site, use a different two-stream graph, described in [chapter 19](./19-s2-architecture) and [chapter 21](./21-s3-recursive-tiny).

## The model in one page

The SPEC gives the whole forward pass as pseudo-code. Every crate implements exactly this graph:

`docs/SPEC.md`
```text
x = tok_embeddings[ids]                      # [T, d]
x = LayerNorm(embeddings.norm)(x)
for l in 0..L:
    h = x if l == 0 else LayerNorm(layers.l.attn_norm)(x)     # layer 0 has NO attn_norm (Identity)
    qkv = h @ layers.l.attn.Wqkv^T            # [T, 3d], split q,k,v; heads of head_dim 64 (n_heads = d/64)
    RoPE (NeoX/rotate_half style, as HF ModernBERT) on q,k: theta = global_rope_theta if l % global_attn_every_n_layers == 0 else local_rope_theta
    attention: bidirectional; local layers only attend |i-j| <= local_attention/2 ; scale 1/sqrt(64)
    x = x + attn_out @ layers.l.attn.Wo^T
    h = LayerNorm(layers.l.mlp_norm)(x)
    a, g = split(h @ layers.l.mlp.Wi^T, 2)    # Wi: [2*intermediate, d]; first half = input, second = gate
    x = x + (gelu_erf(a) * g) @ layers.l.mlp.Wo^T
enc = LayerNorm(final_norm)(x)
```

A few choices to note:

- **No biases and no learned positions in the encoder.** All encoder linears and LayerNorms are bias-free, with eps `1e-5`. Positions enter only through RoPE.
- **Pre-norm, with layer 0 as the exception.** Layer 0 reads the embedding norm output directly. Later layers normalise before attention.
- **GeGLU MLP.** `Wi` produces `2 × intermediate` features. The first half goes through exact (erf) GELU and gates the second half.
- **Alternating attention.** Every `global_attn_every_n_layers`-th layer (3 in all our configs) attends over the whole sequence with `global_rope_theta = 160000`. The others attend a window of `|i − j| ≤ 64` (half of `local_attention = 128`) with `local_rope_theta = 10000`.

The decision head comes next. It runs once, for the question's type `t ∈ {0 choice, 1 score, 2 noul}`:

`docs/SPEC.md`
```text
h = enc + type_emb[t]
for each head layer (PyTorch nn.TransformerEncoderLayer, norm_first=True, activation=relu, d_ff = 4d, nhead = d/64, WITH biases, NO positional encoding, full attention):
    h = h + out_proj(MHA(LayerNorm(norm1)(h)))     # in_proj_weight [3d,d], in_proj_bias [3d]
    h = h + linear2(relu(linear1(LayerNorm(norm2)(h))))
s = scorer.3( gelu_erf( scorer.1( LayerNorm(scorer.0)(h) ) ) )   # [T,1]; scorer LayerNorm/Linears HAVE bias
logits = s[marker positions]   ->   probs = softmax(logits / T_bucket)
```

The head is a plain PyTorch transformer layer stacked on top of the encoder. Its role is to let every option marker look at the whole question, the other options and the state once more, after the type embedding has told it which kind of answer is expected.

## Configuration files

The three ladder configs are HF ModernBERT `config.json` files plus an `rl_agent` block that mirrors Laya's `rl_agent_config.json`. This is the end of `configs/s1-pico.json`:

`configs/s1-pico.json`
```json
  "initializer_range": 0.02,
  "initializer_cutoff_factor": 2.0,
  "rl_agent": {
    "head_layers": 1,
    "max_len": 256,
    "head_max_len": 96,
    "temperature": [
      1.0,
      1.0,
      1.0
    ],
    "temperature_by_options": {}
  }
}
```

The temperatures start at 1.0. They are fitted after training (chapter 11). `sky-train` deserialises the file into `ModelConfig` and enforces one rule throughout: every attention head has 64 dimensions.

`crates/sky-train/src/config.rs`
```rust
    pub fn validate(&self) -> Result<()> {
        if !self.hidden_size.is_multiple_of(HEAD_DIM as usize) {
            bail!("hidden_size {} is not a multiple of {HEAD_DIM}", self.hidden_size);
        }
        if self.num_attention_heads != self.n_heads() {
            bail!(
                "num_attention_heads {} must be hidden_size / {HEAD_DIM} = {}",
                self.num_attention_heads,
                self.n_heads()
            );
        }
        if self.global_attn_every_n_layers == 0 {
            bail!("global_attn_every_n_layers must be >= 1");
        }
```

So `d = 64` gives one head, `d = 128` two and `d = 256` four, in the encoder and in the head alike.

## Parameters as a flat map of SPEC names

`sky-train` does not use mlx-rs modules. The parameters live in a `HashMap<Rc<str>, Array>` keyed by the exact safetensors names of the SPEC. Each forward function looks its weights up by name. Export then becomes a plain dump, and `keyed_value_and_grad` can differentiate with respect to the whole map. `specs` lists every tensor with its shape and initialiser:

`crates/sky-train/src/params.rs`
```rust
    let mut out = vec![
        ("encoder.embeddings.tok_embeddings.weight".to_string(), vec![v, d], Init::TruncNormal(std)),
        ("encoder.embeddings.norm.weight".to_string(), vec![d], Init::Ones),
    ];
    for l in 0..cfg.num_hidden_layers {
        let p = format!("encoder.layers.{l}");
        if l > 0 {
            out.push((format!("{p}.attn_norm.weight"), vec![d], Init::Ones));
        }
        out.push((format!("{p}.attn.Wqkv.weight"), vec![3 * d, d], Init::TruncNormal(std)));
        out.push((format!("{p}.attn.Wo.weight"), vec![d, d], Init::TruncNormal(std_out)));
        out.push((format!("{p}.mlp_norm.weight"), vec![d], Init::Ones));
        out.push((format!("{p}.mlp.Wi.weight"), vec![2 * i, d], Init::TruncNormal(std)));
        out.push((format!("{p}.mlp.Wo.weight"), vec![d, i], Init::TruncNormal(std_out)));
    }
    out.push(("encoder.final_norm.weight".to_string(), vec![d], Init::Ones));
```

Encoder matrices follow HF ModernBERT: a normal truncated at ±2σ with σ = 0.02, and output projections scaled down by `sqrt(2L)`. Head layers use the PyTorch defaults that Laya inherits: xavier-uniform for `in_proj_weight` and `U(±1/sqrt(fan_in))` for the other linears. `type_emb` starts as a unit normal, like `nn.Embedding`.

## The encoder in mlx-rs

The encoder forward reads almost line by line like the SPEC. `fast::rope` with `traditional = false` is the NeoX "rotate half" variant (the `rope_is_neox_rotate_half` test compares it with an explicit implementation), and `fast::scaled_dot_product_attention` does the attention:

`crates/sky-train/src/model.rs`
```rust
    for l in 0..cfg.num_hidden_layers {
        let pre = format!("encoder.layers.{l}");
        let h = if l == 0 {
            x.clone()
        } else {
            layer_norm(&x, w(p, &format!("{pre}.attn_norm.weight"))?, None, eps)?
        };
        let qkv = linear(&h, w(p, &format!("{pre}.attn.Wqkv.weight"))?, None)?;
        let (q, k, v) = split_heads(&qkv, h_count)?;
        let theta = cfg.rope_theta(l);
        let q = fast::rope(&q, HEAD_DIM, false, theta, 1.0, 0, None)?;
        let k = fast::rope(&k, HEAD_DIM, false, theta, 1.0, 0, None)?;
        let mask = if cfg.is_global(l) { global.as_ref() } else { local.as_ref() };
        let a = merge_heads(&attend(&q, &k, &v, mask)?)?;
        x = x.add(linear(&a, w(p, &format!("{pre}.attn.Wo.weight"))?, None)?)?;
        let h = layer_norm(&x, w(p, &format!("{pre}.mlp_norm.weight"))?, None, eps)?;
        let u = linear(&h, w(p, &format!("{pre}.mlp.Wi.weight"))?, None)?;
        let half = cfg.intermediate_size as i32;
        let parts = u.split_at_indices(&[half], -1)?;
        let m = gelu_erf(&parts[0])?.multiply(&parts[1])?;
        x = x.add(linear(&m, w(p, &format!("{pre}.mlp.Wo.weight"))?, None)?)?;
    }
    layer_norm(&x, w(p, "encoder.final_norm.weight")?, None, eps)
```

### Masks: padding and the local window

Inference always runs one unpadded sequence. Training batches are padded, so `attention_masks` builds two boolean masks: a padding mask for global layers and the padding mask combined with the `|i − j| ≤ local_attention / 2` window for local layers. A padded query always sees itself, so no row of the softmax is ever fully masked:

`crates/sky-train/src/model.rs`
```rust
    let half = (cfg.local_attention / 2) as i32;
    let pos = Array::from_iter(0..t, &[t]);
    let qi = pos.reshape(&[t, 1])?;
    let kj = pos.reshape(&[1, t])?;
    let pad = match key_valid {
        Some(kv) => {
            let b = kv.dim(0);
            let eye = qi.eq(&kj)?;
            Some(kv.reshape(&[b, 1, 1, t])?.logical_or(&eye)?)
        }
        None => None,
    };
    let window = if t - 1 > half {
        Some(ops::abs(qi.subtract(&kj)?)?.le(Array::from_int(half))?)
    } else {
        None
    };
```

The test `padding_does_not_change_valid_outputs` checks the property that matters: padding a sequence leaves its marker logits unchanged (to 1e-4).

## The decision head in mlx-rs

`decision_logits` adds the type embedding to every position and runs the head layers with full attention, using only the padding mask:

`crates/sky-train/src/model.rs`
```rust
    let type_row = w(p, "type_emb.weight")?.take_axis(qtype, 0)?.reshape(&[b, 1, d])?;
    let mut h = enc.add(&type_row)?;
    // Full attention with a key padding mask only.
    let (pad, _) = attention_masks(cfg, t, key_valid)?;
    for i in 0..cfg.rl_agent.head_layers {
        let pre = format!("head.layers.{i}");
        let y = layer_norm(&h, w(p, &format!("{pre}.norm1.weight"))?, Some(w(p, &format!("{pre}.norm1.bias"))?), eps)?;
```

After the head layers, it gathers the marker rows and scores each one with the same small MLP:

`crates/sky-train/src/model.rs`
```rust
    // Gather the marker rows, then score them.
    let k = marker_pos.dim(1);
    let idx = ops::broadcast_to(marker_pos.reshape(&[b, k, 1])?, &[b, k, d])?;
    let m = h.take_along_axis(&idx, 1)?;
    let s = layer_norm(&m, w(p, "scorer.0.weight")?, Some(w(p, "scorer.0.bias")?), eps)?;
    let s = gelu_erf(&linear(&s, w(p, "scorer.1.weight")?, Some(w(p, "scorer.1.bias")?))?)?;
    let s = linear(&s, w(p, "scorer.3.weight")?, Some(w(p, "scorer.3.bias")?))?;
    to_f32(s.reshape(&[b, k])?)
```

Every option is scored by the same weights from its own marker position, so the head supports any number of options. The softmax over the K logits is the only place where options compete. Padding options in a batch get `−1e4` from `masked_logits` and receive no probability mass.

## Where the markers come from

The marker positions are produced by the prompt builder, a port of Laya's `rl_common.build_sequence`:

```text
[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")* [SEP] {state} [SEP]
```

The order of the budget rules matters, because they decide what gets cut when a question is long. Options are capped at 48 tokens each. If the options leave less than 16 tokens of the 96-token head budget, every option is shrunk evenly. The question then gets what remains, but never less than 8 tokens, and the state fills the room left up to `max_len = 256`:

`crates/sky-train/src/sequence.rs`
```rust
    let total: usize = opt_ids.iter().map(Vec::len).sum();
    let mut opt_budget = head_max_len as i64 - total as i64;
    if opt_budget < 16 {
        // Too many / too long options: shrink every option text evenly.
        let per = 4.max((head_max_len as i64 - 16).div_euclid(opt_ids.len().max(1) as i64)) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len as i64 - opt_ids.iter().map(Vec::len).sum::<usize>() as i64;
    }
    let head_keep = opt_budget.max(8) as usize;
    let mut ids = Vec::with_capacity(max_len);
    ids.push(CLS_ID);
    ids.extend_from_slice(&p.head[..p.head.len().min(head_keep)]);
    ids.push(SEP_ID);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
```

The state is truncated on the right, and options always come before it. The answer space survives truncation even when a long passage does not. `sky-infer/src/prompt.rs` implements the same rules for inference, split into a reusable `Prefix` (question + options) and `finish_sequence` (state). A game asks the same question every frame, so only the state is tokenized again.

## The same graph in `sky-infer`

The browser engine runs the same computation on f32 slices with pre-allocated scratch buffers (`Workspace`), so `Model::forward` does not allocate. It adds one optimisation that the training code does not need. Only the marker rows are read in the end, so the **last** head layer computes keys and values for all positions but queries, the MLP and the scorer for the K markers only:

`crates/sky-infer/src/model.rs`
```rust
            } else {
                // last head layer: keys/values for every position, everything else only at the markers
                let kv = &mut qkv[..t * 2 * d];
                linear(hl.in_proj.rows(d, 3 * d), hn, t, kv, false, xq);
                for (m, &p) in markers.iter().enumerate() {
                    mh[m * d..(m + 1) * d].copy_from_slice(&hn[p * d..(p + 1) * d]);
                    mx[m * d..(m + 1) * d].copy_from_slice(&x[p * d..(p + 1) * d]);
                }
                let mq = &mut mq[..k * d];
                linear(hl.in_proj.rows(0, d), mh, k, mq, false, xq);
```

For a game question with 4 options in a 60-token sequence, that last layer's feed-forward work drops by a factor of 15. With a single head layer (pico and nano), this covers the whole head.

## Where the parameters go

The `exact_parameter_counts` test in `crates/sky-train/src/params.rs` pins the totals:

`crates/sky-train/src/params.rs`
```rust
        // (config, encoder, decision model total, MLM head extra)
        let expected = [
            ("s1-pico", 147_776, 202_305, 5_184),
            ("s1-nano", 918_656, 1_134_209, 18_560),
            ("s1-micro", 4_984_064, 6_630_913, 69_888),
        ];
```

Here is the pico split, worked out from the shapes above:

| part | params | share |
|---|---:|---:|
| token embeddings (1024 × 64) + norm | 65,600 | 32% |
| 2 encoder layers | 82,112 | 41% |
| final norm | 64 | 0% |
| 1 head layer (d_ff = 256) | 49,984 | 25% |
| `type_emb` + scorer | 4,545 | 2% |
| **total** | **202,305** | |

The head is a quarter of the smallest model. Its feed-forward layer is `4d` wide, while the encoder's GeGLU MLP has `intermediate = 2d` features. Chapter 12 compares the three rungs.

To run the shape, gradient and padding tests of the model code (they need the Metal toolchain, like all of `sky-train`):

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-train
cargo test -p sky-train --release model::
cargo test -p sky-train --release params::
```

## Next

[09 Muon, WSD and the mlx-rs training loop](./09-muon-wsd-training-loop)
