# skycmd-s2 — the innovative decision architecture

This spec is binding for `sky-train` (training and export) and `sky-infer` (native and WASM inference). Read `docs/SPEC.md` first: everything not overridden here (records, tokenizer conventions, prompt texts, temperature buckets, response formats) still applies.

**s1** is the Laya/ModernBERT clone, kept as the llama.cpp-compatible baseline. **s2** changes the architecture to be smaller and faster per decision. It runs in our Rust/WASM engine and in the `skycmd` server, which speaks the same `/v1/systemone` API.

## 1. Ideas
1. **Two streams with a static cache.** A decision is `static part` (question + options, identical every game frame) + `state` (changes every frame). s2 encodes them separately:
   - The **static encoder** reads question and options **once** and caches the resulting option-marker vectors.
   - The **state encoder** reads only the state, and is **question-agnostic**: one state encoding serves every question asked about that state in the same frame.
   - A small **fusion head** lets each option marker cross-attend to the state tokens and scores it.
   - Per-frame cost is therefore O(L · S²) for the state encoder (S ≈ 12–24 tokens) plus O(n_opts · S) for the fusion head. It does not depend on the question and option lengths.
2. **Permutation equivariance by construction.** Options never attend to each other, neither in the static encoder nor in fusion. Shuffling the options shuffles the probabilities identically, with no augmentation needed (cf. 2610.06744, 2610.02586).
3. **Native numbers.** Every numeric literal becomes one `[NUM]` token whose embedding carries its value through continuous features. Ordering and magnitude come built in, and the vocabulary no longer spends entries on integers.
4. **Adaptive depth (early exit).** Exits are attached after chosen state-encoder layers, all trained jointly (deep supervision) and calibrated separately. At inference the engine stops at the first exit whose calibrated top probability is ≥ τ.
5. **Ternary weights.** In a final QAT phase, every linear layer of the encoders and of the fusion head uses ternary weights {−1, 0, +1} × a per-output-row scale (absmean, straight-through estimator). Embeddings are stored int8 with per-row scales, and norms stay f32.

## 2. Tokenization
- Numeric extraction runs before BPE on every text (instructions, option texts, state):
  - regex `-?\d+(?:\.\d+)?` on ASCII digits;
  - a match glued to a preceding **ASCII** letter is not extracted, e.g. `b2`, `glm-5` (see the fixtures in `crates/sky-data/tests/fixtures/num_extract.json`, which every implementation must pass);
  - each extracted match becomes `[NUM]`, and its f32 value goes in a parallel `values` array (0 for non-number tokens).
- s2 uses dedicated tokenizers `data/tok/tok-s2-<vocab>.json`:
  - trained by `sky-data train-tokenizer --num` on text where numbers are already replaced by `[NUM]`;
  - special ids `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4 [NUM]=5`;
  - `[NUM]` must also be matched as a special token, so the replaced text tokenizes to id 5. It has `lstrip = true` (like `[MASK]` in oscar's tokenizer), so the space before a number is absorbed and `vy [NUM]` costs 2 tokens.
- **Value features**: with `u = sign(v)·ln(1+|v|)`, `φ(v) = [u, sin(u·ω_k), cos(u·ω_k) for k=0..F-1]` where `ω_k = 2^k · π / 8` and `F = 8`, giving 17 dims.
- **Embedding**: `e = tok_embd[id] + is_num · (num_proj.weight · φ(v) + num_proj.bias)`, where `num_proj.weight` is [d, 17].

## 3. Model
Notation: `d` hidden, `h = d/64` heads (head_dim 64), `ff` intermediate.
- **Block** (used by both encoders):
  - pre-LayerNorm without bias, eps 1e-5;
  - fused QKV without bias, RoPE (rotate_half, θ = 10000) on q and k, bidirectional attention under a mask, output projection without bias;
  - pre-LayerNorm, then GeGLU: `Wi` is [2·ff, d], first half is the input and second half the gate, `gelu_erf(a)·g`; then `Wo` [d, ff];
  - layer 0 has no attention norm, as in ModernBERT;
  - after the last block of each encoder, a final LayerNorm.
- **Static stream** for question type `t`, instructions `ins` and rendered options `opt_i`:
  - tokens `Q = [CLS] "{t} question: {ins}" [SEP]`, then for each option `O_i = [MASK] " {opt_i}"`;
  - attention mask: Q attends to Q, and O_i attends to Q ∪ O_i;
  - RoPE positions: Q gets 0..|Q|−1, and every O_i restarts at |Q| (same positions for every option);
  - truncation: as in SPEC.md, each option is capped at 48 tokens and the question at max(8, budget);
  - the static encoder has `static_layers` blocks plus `static_norm`. Its output at each O_i's `[MASK]` position is the marker vector `m_i`, and the outputs at Q positions are `q_tokens`.
- **State stream**:
  - tokens `[CLS] {state} [SEP]` (numbers become `[NUM]` with values), truncated to `max_state` = 64 tokens, positions 0..S−1, full bidirectional attention;
  - `state_layers` blocks;
  - exit `e` sits after block `exits[e]` (1-based) and normalizes the hidden states with `exit.{e}.norm` (LayerNorm without bias) to give `H_e` [S, d]. The last exit equals `state_layers`.
- **Fusion head**, shared by all exits, `fusion_layers` blocks. For one question and one exit `e`:
  ```
  x_i = m_i + type_embd[t]                                    # [n, d]
  kv  = concat(H_e, q_tokens)                                  # state tokens + cached question tokens
  for each fusion block:
      x = x + attn_out( MHA( q = attn_q(LN(x)), k,v = attn_kv(LN_kv(kv)) ) )   # cross-attention only; markers do NOT attend to each other
      x = x + ffn_down( relu( ffn_up( LN(x) ) ) )               # ff_fusion = 2·d, linear layers WITH bias
  s_i = cls.output( gelu_erf( cls( LN(cls.norm)(x_i) ) ) )      # scalar logit per option
  p   = softmax(s / T[e, t, bucket])
  ```
  `attn_kv` is a single [2d, d] projection producing k and v, and there is no RoPE in fusion.
  The q_tokens part of `kv` is cached with the static stream: its LN_kv + attn_kv projection is computed once per question.
- **Early exit at inference**: for e in exits order, compute fusion on H_e and stop when `max(p) ≥ τ_t`. τ is stored per type, default 0.9, and τ = 1.0 disables early exit. The response also reports `exit` (index) and `layers_used`.
- **Multi-question frames**: the state is encoded once and reused for every question on the same state (e.g. flappy `action` + `danger`).

### Ladder
| name | d | state_layers | static_layers | fusion_layers | ff | vocab | exits |
|---|---|---|---|---|---|---|---|
| s2-pico | 64 | 3 | 2 | 1 | 128 | 1024 | [1, 3] |
| s2-nano | 128 | 4 | 2 | 2 | 256 | 2048 | [1, 2, 4] |
| s2-micro | 256 | 6 | 3 | 2 | 512 | 4096 | [2, 4, 6] |
Configs live in `configs/s2-{pico,nano,micro}.json`.

## 4. Training (sky-train, mlx-rs; new subcommands `s2-pretrain`, `s2-decide`, `s2-calibrate`, `s2-export`, `s2-eval`)
- **Pretrain:**
  - MLM over one encoder stack of depth `state_layers`. Inputs use numeric extraction, so `[NUM]` tokens carry values.
  - 30 % masking. A masked `[NUM]` keeps the `[NUM]` id but its value is replaced by 0 and an `is_masked_num` flag. Loss = CE on masked non-number tokens + 0.5 · MSE on the u = signlog value of masked numbers, predicted by a linear `mlm_num` head (xVal-style).
  - At the end, the state encoder is initialized from this stack and the static encoder from a copy of its first `static_layers` blocks.
- **Decision training:**
  - Same records as s1. Static = (t, ins, opts), state = state.
  - Loss = Σ_e w_e · soft-target CE(exit e), with `w = linspace(0.5, 1.0)` over the exits, normalized.
  - Muon on the encoder hidden matrices; AdamW on embeddings, num_proj, type_embd, fusion head and scorer. WSD schedule.
  - Batching: group by question so the static stream is shared across a batch.
- **QAT:** during the last 30 % of decision training (`--qat-frac 0.3`), every Linear in the encoders and fusion uses fake-quant ternary weights: `γ_r = mean(|W_r|)` per output row, `Wq = clip(round(W/γ_r), −1, 1)·γ_r`, with a straight-through gradient. `cls.output` and the embeddings are excluded. Report the accuracy before and after ternarization.
- **Calibration:** temperature per (exit, type, bucket) on val, then τ per type: the smallest value whose early-exit accuracy is within 0.5 points of full depth on val.
- **Eval:**
  - accuracy, NLL, Brier, ECE per source and per exit;
  - mean layers used and the early-exit rate;
  - permutation invariance check (max probability difference under option shuffle; should be ≤ 1e-5);
  - coherence p(X) + p(¬X) on paired noul questions where available.

## 5. GGUF `skycmd-s2`
- `general.architecture = "skycmd-s2"`, `general.name`.
- Keys, all prefixed `skycmd-s2.`:
  - `embedding_length`, `feed_forward_length`, `head_count`
  - `state_block_count`, `static_block_count`, `fusion_block_count`
  - `exits` (u32 array), `max_state_tokens`, `max_head_tokens`
  - `num_features` (= 17), `num_token_id` (= 5)
  - `rope_freq_base`, `layer_norm_epsilon`
  - `exit_threshold.{choice,score,noul}`
  - `temperature.e{E}.{type}[.{bucket}]`
  - `weight_format` (`f16`, `f32` or `ternary`)
- The tokenizer uses the same `tokenizer.ggml.*` keys as s1: gpt2 model, tokens, merges, token types and special ids, plus `skycmd.tokenizer.*` for NFC and the extra rules already used by sky-infer.
- Tensors (PyTorch shape order; F32/F16, or ternary as below):
  - `token_embd.weight` [V, d]
  - `num_proj.weight` [d, 17], `num_proj.bias` [d]
  - `type_embd.weight` [3, d]
  - per static block `static.blk.N.*` and per state block `blk.N.*`: `attn_norm.weight` (absent for N = 0), `attn_qkv.weight` [3d, d], `attn_output.weight` [d, d], `ffn_norm.weight`, `ffn_up.weight` [2ff, d], `ffn_down.weight` [d, ff]
  - `static_norm.weight`, `exit.E.norm.weight`
  - per fusion block `fusion.N.*`: `q_norm.weight/bias`, `kv_norm.weight/bias`, `attn_q.weight/bias` [d, d], `attn_kv.weight/bias` [2d, d], `attn_output.weight/bias`, `ffn_norm.weight/bias`, `ffn_up.weight/bias` [2d, d], `ffn_down.weight/bias` [d, 2d]
  - `cls.norm.weight/bias`, `cls.weight/bias` [d, d], `cls.output.weight` [1, d], `cls.output.bias` [1]
- **Ternary storage**, when `weight_format = ternary`. For each ternary matrix `X`:
  - `X.tq`: GGUF type I8, packed with 4 weights per byte (2 bits each: 00 = 0, 01 = +1, 10 = −1), row-major and row-padded to a multiple of 4;
  - `X.scale`: F16 [rows].
  - `token_embd.weight` is stored as `token_embd.q8`: I8 [V, d] plus `token_embd.scale` F16 [V].
  - Everything else is F32.

## 6. Engine API (sky-infer)
- `Engine::from_gguf(bytes)` dispatches on `general.architecture`: `modern-bert` gives s1 and `skycmd-s2` gives s2. `decide_json` / `decide_raw` / `decide` work for both, and s2 responses add `exit` and `layers_used`.
- The s2 fast path caches the static stream keyed by `(qtype, instructions, options)` (LRU of 16 entries) and the state encoding keyed by the state string (for multi-question frames).
- `EngineOptions { exit_threshold: Option<f32> }` overrides τ; 1.0 means full depth.
- The ternary kernels unpack 2-bit weights to sign masks and accumulate with add/sub (SIMD128 / NEON), then multiply once by the row scale.
- Golden test: `sky-train s2-golden` writes `crates/sky-infer/tests/golden/s2.json` (random-init tiny s2: GGUF bytes path, request, expected probabilities per exit, computed by mlx-rs). sky-infer matches it within 1e-4 (f32) and within 1e-3 (ternary, against mlx-rs fake-quant).
