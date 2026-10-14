# skycmd — shared engineering spec

This is the contract that every crate follows. Read it fully before you write code.

## Goal
skycmd builds tiny "System One" decision models: a bidirectional encoder with a typed decision head that scores options in **one forward pass**, with no generation. They are trained from scratch in Rust (mlx-rs) on an Apple M5 and run in the browser through a Rust → WASM engine, where they play small games live. Everything is Rust, except the web page glue.
All docs, comments and UI text are in **English**. The project is civil only: there are no targeting or weapon features.

## Reference material (local, gitignored, read-only)
`models-src/`:
- `rl_common.py`, `rl_agent_api.py`: the Laya reference runtime (prompt building, model, losses, temperature buckets).
- `modern-bert.cpp`: the llama.cpp graph for ModernBERT + decision head (PR #29818).
- `server-decision.cpp/.h`: the llama.cpp `/v1/systemone` server (prompt, truncation, readout, temperatures).
- `bert.py`: the llama.cpp HF → GGUF converter for ModernBERT/decision models. It gives the GGUF keys and tensor names.
- `oscar/`: `mgoeckel/oscar-1-17m` (Apache-2.0). This is a real Laya-format decision model with encoder `ettin-encoder-17m`: `model.safetensors` (bf16), `encoder/config.json`, `rl_agent_config.json`, `tokenizer/tokenizer.json`. **It is the golden reference model for our inference engine.**

## Crates (`crates/`, one owner each — never edit another crate)
| crate | purpose |
|---|---|
| `sky-schema` | System One request/response types, decision record type, command-plan JSON schema + safety validator |
| `sky-games` | pure-Rust game cores (flappy, rescue, dodge), state → text, questions, oracles, rollout harness. `no_std`-friendly (uses `alloc`), compiles to wasm32 |
| `sky-gguf` | GGUF v3 reader + writer, F32/F16/BF16/Q8_0, quantizer, `gguf-inspect` bin |
| `sky-infer` | inference engine: byte-level BPE tokenizer, ModernBERT encoder, Laya decision head, prompt builder, f32 + int8 paths, wasm-bindgen API (feature `wasm`) |
| `sky-data` | Hugging Face download (hf-hub + parquet) → decision/MLM records; tokenizer training (HF `tokenizers`); GLM teacher client (Anthropic Messages protocol) |
| `sky-train` | mlx-rs training: model, MLM pretrain, decision training, Muon + AdamW + WSD, temperature calibration, export to HF-layout safetensors + GGUF |
| `site/` | VitePress site + games UI (TS) using the wasm pkg of `sky-infer` + `sky-games` |

Add dependencies **in your own crate's Cargo.toml** with explicit versions. Do not edit the root `Cargo.toml`, except that the root workspace uses `members = ["crates/*"]`, so create your crate with `cargo new --lib crates/<name>` before anything else.
To avoid lock contention with parallel workers, use your own target dir: `export CARGO_TARGET_DIR=$PWD/target/<crate-name>`.
**Never** run `git commit` / `git push`. Never add any AI/assistant attribution anywhere.

## Model family `skycmd-s1` (exactly the Laya / ModernBERT layout)
Encoder = ModernBERT. All linear layers have no bias, and all LayerNorms have no bias (`norm_bias=false`), eps 1e-5.
```
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
Decision head (Laya `DecisionModel`). It runs once for the question's type t ∈ {0 choice, 1 score, 2 noul}:
```
h = enc + type_emb[t]
for each head layer (PyTorch nn.TransformerEncoderLayer, norm_first=True, activation=relu, d_ff = 4d, nhead = d/64, WITH biases, NO positional encoding, full attention):
    h = h + out_proj(MHA(LayerNorm(norm1)(h)))     # in_proj_weight [3d,d], in_proj_bias [3d]
    h = h + linear2(relu(linear1(LayerNorm(norm2)(h))))
s = scorer.3( gelu_erf( scorer.1( LayerNorm(scorer.0)(h) ) ) )   # [T,1]; scorer LayerNorm/Linears HAVE bias
logits = s[marker positions]   ->   probs = softmax(logits / T_bucket)
```
Padding / attention mask: single sequence, no padding at inference.

Safetensors tensor names (HF layout, as in oscar `model.safetensors`, prefixed `encoder.` for the encoder):
`encoder.embeddings.tok_embeddings.weight`, `encoder.embeddings.norm.weight`, `encoder.layers.{l}.attn_norm.weight` (l≥1), `encoder.layers.{l}.attn.Wqkv.weight`, `encoder.layers.{l}.attn.Wo.weight`, `encoder.layers.{l}.mlp_norm.weight`, `encoder.layers.{l}.mlp.Wi.weight`, `encoder.layers.{l}.mlp.Wo.weight`, `encoder.final_norm.weight`, `head.layers.{i}.{norm1,norm2}.{weight,bias}`, `head.layers.{i}.self_attn.in_proj_{weight,bias}`, `head.layers.{i}.self_attn.out_proj.{weight,bias}`, `head.layers.{i}.{linear1,linear2}.{weight,bias}`, `type_emb.weight` [3,d], `scorer.0.{weight,bias}`, `scorer.1.{weight,bias}`, `scorer.3.{weight,bias}`.
(Verify the exact names against `models-src/oscar/model.safetensors`. If oscar differs, oscar wins and you update this spec.)
The GGUF names and keys follow `models-src/bert.py` + `modern-bert.cpp` (`general.architecture = "modern-bert"`, `modern-bert.decision.*`, `blk.N.*`, `token_types.weight`, `cls.*`).

### Ladder configs
| name | d | L | heads | intermediate | head_layers | vocab | global_every | local_attention | rope θ global/local |
|---|---|---|---|---|---|---|---|---|---|
| s1-pico  | 64  | 2 | 1 | 128 | 1 | 1024 | 3 | 128 | 160000 / 10000 |
| s1-nano  | 128 | 4 | 2 | 256 | 1 | 2048 | 3 | 128 | 160000 / 10000 |
| s1-micro | 256 | 6 | 4 | 512 | 2 | 4096 | 3 | 128 | 160000 / 10000 |
`max_len = 256`, `head_max_len = 96` for our models. Config JSON files live in `configs/s1-{pico,nano,micro}.json` with the ModernBERT `encoder/config.json` keys plus a `rl_agent_config.json`-style block (`head_layers`, `max_len`, `head_max_len`, `temperature` [3], `temperature_by_options` {bucket: T}).

## Prompt / sequence (identical to `rl_common.build_sequence` and llama.cpp `server-decision.cpp`)
`[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")* [SEP] {state} [SEP]`
- `t` ∈ `choice|score|noul`.
- Option texts follow `render_options`: choice `key` or `key: desc`, score `level i: desc`, noul `["false: …", "true: …"]` with defaults `no, the statement does not hold` / `yes, the statement holds`.
- Each option is tokenized as `" " + text`, truncated to 48 tokens, with the shrinking rule; the head is truncated to `max(8, opt_budget)`; the state is truncated to the room left.
- The temperature bucket is `type.bucket` where bucket = `2` (n≤2), `3_5` (≤5), `6_10` (≤10), `11`; it falls back to `type`, else 1.0 (see `temp_bucket` in rl_common.py and the server).
- Response: choice → argmax, probs, `confidence = (pmax − 1/n)/(1 − 1/n)`; score → Σ i·p_i plus probs; noul → p(true).

## Tokenizer
Byte-level BPE (GPT-2 style: ByteLevel pre-tokenizer, `add_prefix_space=false`, GPT-2 split regex), trained with HF `tokenizers` and saved as `tokenizer.json`. Special tokens and fixed ids: `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`. The `sky-infer` tokenizer must reproduce HF `tokenizers` encodings exactly (oscar's tokenizer.json is a ByteLevel BPE too, so use it as a test).

## Decision record (JSONL, produced by sky-data / sky-games, consumed by sky-train)
```json
{"src":"boolq","t":"noul","ins":"Is the passage saying that ...?","opts":["false: no, the statement does not hold","true: yes, the statement holds"],"state":"<free text or compact game state>","target":[0.0,1.0],"split":"train"}
```
`opts` are already rendered option texts (see above). `target` is a probability vector over `opts` (one-hot or soft), and `split` ∈ `train|val|test`.

## MLM record (JSONL)
`{"src":"simplestories","text":"..."}`.

## Game state text (compact, produced by sky-games)
Space-separated `key value` pairs with small integers, e.g. `alt 12 vy -3 dx 40 gt 20 gb 35`. Values are integers that are binned/clamped so the BPE learns them as single tokens.
