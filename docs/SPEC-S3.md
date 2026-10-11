# skycmd-s3: recursive, value-fused, ternary decision model (an extension of s2)

s3 is **s2 with two new options** stored in the same `skycmd-s2` GGUF architecture. Everything in `docs/SPEC-S2.md` still holds unless this file says otherwise. A GGUF without the new keys must behave exactly as before, which keeps every existing s2 file and test valid.

Each option is motivated by recent work, summarized in `docs/research-night.md`:
- **Value-fused tokens**, like FT-Transformer's feature tokenizer and FoNE single-token numbers: a `key value` pair costs **one** token.
- **A weight-shared recursive state encoder**, as in TRM, Ouro and Universal Transformers: depth comes from looping a tiny block, and every loop ends at an exit, which gives adaptive computation without adding parameters.
- **Closed-loop exit calibration**, as in risk-controlled early exit: τ is chosen on the states the model visits while playing, not on i.i.d. validation questions.

## 1. `fuse_numbers` (GGUF key `skycmd-s2.fuse_numbers`, bool, default false)
Tokenization runs exactly as in SPEC-S2 §2 (numeric extraction, then BPE with `[NUM]`). Afterwards, when this flag is true:
- **Merge rule**: each `[NUM]` token at position i > 0 is removed, and its value moves onto token i−1.
  - Token i−1 gets `has_value = 1` and `value = v`.
  - If token i−1 already carries a value (two numbers in a row, e.g. `10-20` → `[NUM] [NUM]`), the second `[NUM]` is kept as a standalone value token, which is the s2 behaviour.
  - A `[NUM]` at position 0, or right after `[CLS]`, `[SEP]` or `[MASK]`, is also kept standalone.
- **Embedding**: `e_i = tok_embd[id_i] + has_value_i · (num_proj.weight · φ(v_i) + num_proj.bias)`, with the same φ as in s2. A standalone `[NUM]` keeps id 5, exactly as in s2.
- **Streams and positions**: the rule applies to the static stream and the state stream alike. RoPE positions are the positions after merging.
- **Effect**: a game state like `vy -1 dx 35 up 7 dn 9 nxt 1 alt 32` becomes 6 tokens plus `[CLS]` and `[SEP]`, instead of about 14 with s2.

## 2. `state_loops` (GGUF key `skycmd-s2.state_loops`, u32, default 1)
- **Loop structure**: the state encoder's `state_block_count` distinct blocks (`blk.0..blk.{B-1}`) are applied in order, and the whole sequence is repeated `state_loops` times. The effective depth is `B × state_loops`.
- **First block**: block 0 has no attention norm only on the very first application. From the second loop on, the input is normalized with `blk.0.attn_norm_loop.weight`, a LayerNorm without bias that exists only when `state_loops > 1`.
- **Loop embedding**: before each loop `l ≥ 1`, add `loop_embd.weight[l]` ([state_loops, d]) to the hidden states. It tells the shared block which iteration it is in, in the spirit of TRM and Universal Transformers.
- **Exits**: `skycmd-s2.exits` keeps counting **block applications** (1-based effective depth), so `[1, 2, 3, 4]` with B = 1 and 4 loops means an exit after every loop. Exit norms `exit.E.norm` and temperatures `temperature.e{E}.*` are unchanged.
- **Inference**: the engine stops at the first exit whose calibrated top probability is ≥ τ, as in s2.

## 3. Training additions (sky-train)
- **Configs** use the new fields `fuse_numbers` and `state_loops`.
- **Deep supervision** covers every exit: the loss is a weighted sum of the soft-target CE at every exit, with weights rising linearly from 0.5 to 1.0.
- **Stochastic depth** (`--loop-dropout 0.25`): during training, each batch skips the extra loops beyond a random exit with that probability, which acts as stochastic stopping and keeps intermediate states useful.
- **Numeric curriculum** (programmatic families with exact gold): `sky-train s2-numeric --n N --out data/numeric/numeric.jsonl` generates DecisionRecords whose states hold `key value` pairs, for example `a 12 b -3 c 40`. Questions:
  - which key is largest or smallest (choice);
  - is a > b (noul);
  - which value is closest to t (choice);
  - the sign of a − b (choice);
  - bucket a value into 5 ordered levels (score);
  - does a lie between b and c (noul);
  - EN and FR instructions.

  These records are mixed into s2-pretrain/s2-decide (weight 0.1) so the value channel learns order and magnitude.
- **Ternary QAT** runs for the whole decision phase of the small models (`--qat-frac 1.0` after a short float warm-up of about 10% of steps).
- **GGUF export** writes the new keys and tensors (`loop_embd.weight`, `blk.0.attn_norm_loop.weight`) and leaves everything else unchanged.

## 4. Ladder (configs/s3-*.json)
| name | d | heads | state blocks × loops | static layers | fusion layers | ff | vocab | exits |
|---|---|---|---|---|---|---|---|---|
| s3-pico | 96 | 1 (head_dim 96) | 1 × 4 | 1 | 2 | 192 | 1024 | [1,2,3,4] |
| s3-nano | 128 | 2 | 2 × 3 | 2 | 2 | 256 | 2048 | [2,4,6] |

The head_dim of 96 for s3-pico is allowed: head_dim = d / heads. The SPEC-S2 rule `h = d/64` is relaxed through a new optional key `skycmd-s2.head_count`, which is already written.

## 5. Closed-loop exit calibration (sky-arena)
`sky-arena <model> --calibrate-exit --episodes N` plays each game at full depth to get the reference metrics. It then sweeps τ ∈ {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98} and picks the smallest τ (per question type) whose metrics stay within 2 % of full depth with no extra crashes. It prints the `gguf-set` command that writes those thresholds.

## 6. Golden
`sky-train s2-golden --config configs/s3-pico.json` writes `crates/sky-infer/tests/golden/s3.json` and `s3-ternary.gguf`, with tiny random weights and the new options on. sky-infer must match it within 1e-4 (f32) and 1e-3 (ternary).

## 7. Temporal recursion: carry the latent across frames (`skycmd-s2.temporal_carry`, bool, default false)
Inspiration: TRM carries its latent, detached, across supervision steps. RIN warm-starts latents from the previous step through a zero-initialised term. Delta networks skip computation on unchanged inputs. Here the recursion of §2 is **amortized over time**: the brain does not start from scratch every frame.
- **New tensor** `carry_gate.weight` [d], initialised to 0 in training.
- **Starting state:** for a frame t of the same stream (same question set, same state token count S and the same state token ids, i.e. the same keys), the state encoder starts from
  `h⁰_t = embed(state_t) + carry_gate ⊙ LN₀(h^final_{t−1})`,
  where `h^final_{t−1}` is the output of the last block applied at frame t−1, before the exit norm. `LN₀` is a LayerNorm without bias whose weight is `carry_norm.weight` [d]. If the token count or the ids differ, or there is no previous frame, the carry term is 0 (a cold start).
- **Training:**
  - Sequences of K = 4 consecutive decision points from the same episode, with an oracle target for each frame. The generator emits a `seq` id and a `t` index per record.
  - Frame 1 starts cold. Frames 2..K use the detached final latent of the previous frame.
  - Deep supervision applies at every exit of every frame.
  - With probability 0.3 the carry is dropped (cold start), so the model also works without history.
- **Inference API:** `Engine::decide_raw_stream(stream_id, qtype, ins, opts, state)` keeps one carry buffer per `stream_id` and clears it on `reset_stream(stream_id)`. JSON `/v1/systemone` stays stateless (cold). GameSession uses one stream per game session and resets it on episode reset.
- **Expected effect:** frames after the first converge in about 1 loop, so early exit fires at exit 0 and the per-frame cost drops toward one block application.

## 8. Ternary lookup-table kernels (inference only)
These are T-MAC and bitnet.cpp TL style kernels:
- Quantize each activation row to int8 with a per-row scale.
- For each group of 4 input columns, precompute the 16 partial sums of ±x over the sign patterns (zeros are handled by splitting each ternary weight into a "nonzero" mask and a sign).
- Each output row then does one table lookup per group, with `vqtbl1q_s8` on NEON and `i8x16.swizzle` on WASM SIMD128, accumulated in int16/int32, then multiplied once by the row scale times the activation scale.
- Keep the f32-expanded path, and pick the faster path per platform at load time, as today.

## 9. Ablation (optional, only if time allows): expand-and-sparsify FFN
Fly mushroom body: about 50 projection neurons feed about 2,500 Kenyon cells, and k-WTA keeps about 5% of them active. The ablation is:
- the FFN up-projection becomes a fixed **seeded** sparse random ±1 projection to 8·d units, stored as a u64 seed (0 bytes of weights);
- k-WTA keeps the top 5%;
- the learned down-projection only reads the k active units.

Compare accuracy and µs against the GeGLU FFN at equal file size.
