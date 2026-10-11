# 19 The s2 architecture

s1 (chapters 08 to 17) is a clone of the Laya / ModernBERT decision model, so its files run in llama.cpp and wllama. That compatibility has a cost in a game: every frame, s1 re-reads the question and the options to answer a question about a 20-token state. s2 is a different architecture, specified in `docs/SPEC-S2.md`, that keeps the same request and response format and drops the parts a game does not need.

The result is s2-nano: 1,532,033 parameters in a 701,568-byte ternary GGUF (`site/public/models/manifest.json`) that plays the three drone games at oracle level. It got there in two steps, and the first one failed in closed loop. This chapter covers the five ideas of the architecture, then that failure.

s2 files run only in the skycmd engine: the browser module, the `skycmd` CLI and server, the Python wheel and the npm package (`docs/compat.md`). No other runtime knows `general.architecture = "skycmd-s2"`.

## The cost s2 removes

An s1 Flappy question is one sequence of about 66 tokens (`data/STATS.md`, `tok-1024`): the question, both options with their descriptions, and the state. A Flappy frame asks two questions (`action` and `danger`), so s1 encodes two sequences of that size, and the question and options are identical in every frame. Only the state changes, and it is 20 tokens.

s2 splits the work along that line.

## 1. Two streams and a static cache

```text
static stream (once per question)        state stream (once per frame)
[CLS] question [SEP] [MASK] opt_1 ...    [CLS] vy [NUM] dx [NUM] ... [SEP]
        │ static encoder                          │ state encoder (blocks 1..4)
        ▼                                          ▼  exits after blocks 1, 2, 4
   option markers m_i  ──── fusion head ────►  state tokens H_e
   question tokens        (markers cross-attend      │
   (cached, LRU of 16)     to H_e + question)        ▼
                                              one logit per option
```

- **Static encoder.** It reads `[CLS] "{type} question: {instructions}" [SEP]` followed by one `[MASK] " {option}"` group per option. The output at each option's `[MASK]` is that option's marker vector. The engine caches the markers and the question tokens, keyed by `(type, instructions, options)`, in an LRU of 16 entries (`docs/SPEC-S2.md` §6).
- **State encoder.** It reads only `[CLS] {state} [SEP]`, at most 64 tokens. It does not see the question, so one state encoding serves every question asked about that state in the same frame.
- **Fusion head.** Each option marker cross-attends to the state tokens and the cached question tokens, then a small scorer turns it into one logit. The fusion blocks have no RoPE and no self-attention between markers.

The per-frame cost no longer depends on the length of the question or the options: it is the state encoder (O(L·S²) for S state tokens) plus one fusion pass per question (O(n_opts·S)).

`BENCH.md` ("Inference") measures what the cache buys, on random-init models of the ladder shapes (`sky-bench`, native, µs per frame, median):

| s2-nano, full depth | 1 question | 2 questions on one state |
|---|---:|---:|
| static cache on | 306 | 338 |
| static cache off | 599 | 1,015 |

With the cache, the second question costs 32 µs instead of a full second pass, because it only runs its fusion head. Without the cache, s2 loses most of its advantage. The cache is what makes the architecture fast.

## 2. Permutation equivariance by construction

In s1, the options sit in one sequence and attend to each other, so the answer can depend on the order in which they are listed. s1 training fights this with data: options are renamed and shuffled in every record ([chapter 04](./04-game-cores-and-oracles)).

s2 removes the cause instead:

- in the static encoder, option `i` attends to the question and to itself only (mask `Q ∪ O_i`);
- every option starts at the same RoPE position (`|Q|`), so its position does not reveal its rank;
- in the fusion head, markers attend to the state and the question, never to each other.

Shuffling the options therefore shuffles the probabilities in exactly the same way. `s2-eval` checks it on 256 shuffled questions, on the CPU: the maximum probability difference over all exits is 1.8e-7 for the shipped `s2-nano.gguf` (`runs/s2-nano-g2/report.md`), and at most 3e-6 on every model and exit in `BENCH.md`. That is float rounding. On the Metal GPU, batch-dependent kernel rounding alone reaches about 1e-3, which is why the check runs on the CPU.

## 3. Numbers as values: `[NUM]` and its features

Before BPE, every number that matches `-?\d+(?:\.\d+)?` becomes one `[NUM]` token (id 5), and its value goes into a parallel array. A number glued to a preceding ASCII letter, such as `b2` or `glm-5`, stays text (`crates/sky-data/tests/fixtures/num_extract.json`). The embedding of a `[NUM]` token adds a projection of 17 value features:

```text
u    = sign(v) · ln(1 + |v|)
φ(v) = [u, sin(u·ω_k), cos(u·ω_k)]  for k = 0..7,  ω_k = 2^k · π / 8
e    = tok_embd[id] + is_num · (num_proj.weight · φ(v) + num_proj.bias)     # num_proj.weight is [d, 17]
```

Order and magnitude are built into the embedding, and the vocabulary no longer spends entries on integers. It does not make the game states shorter, though. Our s1 tokenizers already make small integers single tokens ([chapter 07](./07-tokenizer)), With the 2k tokenizers, a state costs 20.0 tokens in s2 against 19.4 in s1 for Flappy, 23.0 against 17.3 for Dodge and 34.8 against 26.0 for Rescue (`data/STATS.md`, "Game state tokens per frame").

Pretraining is a masked LM with a numeric term: a masked `[NUM]` keeps its id, loses its value, and a linear head predicts `u`. `docs/SPEC-S2.md` sets the weight of that MSE to 0.5. In practice it swamped the token loss: after 1.5 minutes the token cross-entropy was still on the unigram plateau at 4.81, against 3.07 without the MSE and 3.19 with a weight of 0.05 (`BENCH.md`, notes). The runs use `--num-weight 0.05` and clip `u` to ±12 (`scripts/train-s2.sh`).

## 4. Early exits with deep supervision

Exits sit after chosen blocks of the state encoder: after blocks 1, 2 and 4 for s2-nano, after blocks 1 and 3 for s2-pico. Each exit normalizes the hidden states with its own LayerNorm and feeds the shared fusion head.

- **Training.** All exits are trained together. The loss is a weighted sum of the soft-target cross-entropy of each exit, with weights from 0.5 to 1.0 (`linspace`, normalized), so the deep exits matter most.
- **Calibration.** A temperature per (exit, question type, option-count bucket), fitted on val. Then a threshold τ per question type: the smallest value whose early-exit accuracy stays within 0.5 points of full depth on val.
- **Inference.** The engine runs the exits in order and stops at the first one whose top probability reaches τ. The response reports `exit` and `layers_used`.

On held-out questions this works as designed. For the DAgger-trained s2-nano (`runs/s2-nano-g2/report.md`, ternary), the calibrated τ is 0.62 / 0.62 / 0.69 for choice / score / noul. Early exit costs half a point of accuracy (0.5402 against 0.5454 at the last exit) and uses 2.88 of 4 layers on average. On Dodge questions it uses 1.28 layers. Natively, stopping at the first exit takes a question from 306 µs to 118 µs (`BENCH.md`).

Section 6 shows why that τ was still wrong.

## 5. Ternary weights with QAT

During the last 30% of decision training (`--qat-frac 0.3`), every linear layer of the encoders and the fusion head runs with fake-quantized ternary weights: per output row, `γ = mean(|W|)` and `W_q = clip(round(W/γ), −1, 1) · γ`, with a straight-through gradient. The final scorer and the embeddings are excluded. The GGUF stores each ternary matrix as 2 bits per weight (4 weights per byte) plus an F16 scale per row, and the token embeddings as int8 with a per-row scale.

The numbers from `BENCH.md` and the run logs:

- **QAT is necessary.** Ternarizing a model trained without QAT costs 6 to 11 points on val (s2-nano 0.542 → 0.432). With 30% of the budget in QAT, s2-nano reaches 0.556 on val.
- **The latent weights adapt to the quantizer.** Evaluated in plain f32, the final QAT master weights of s2-nano score 0.2969 on the test split, against 0.5454 for the same weights ternarized (`runs/s2-nano-g2/report.md`). The f32 export is therefore taken from the snapshot just before QAT starts.
- **Files are 7 to 9 times smaller.** The s2-nano export went from 6,050.1 KiB in f32 to 685.1 KiB ternary (`runs/s2-nano.log`). The published file is 701,568 bytes.
- **Ternary compute is not faster here.** The packed add/sub kernels need three vector operations per 4 weights against one FMA, and these matrices fit in L1. On s2-nano, a question takes 667 µs with packed kernels and 298 µs with the ternary weights expanded to f32 at load. The default (`TernaryCompute::Auto`) therefore expands them at load: the download stays small and the compute is f32.

The DAgger rounds of s2 run QAT on every step (`--qat-frac 1.0`, `scripts/specialize-s2.sh`), so the shipped weights never leave the ternary grid.

## 6. The τ that failed in closed loop

After two DAgger rounds ([chapter 20](./20-dagger-lessons)), s2-nano was played by `sky-arena` with its calibrated thresholds. Same weights, three ways to stop, 30 episodes per game, default seed:

| τ (choice / score / noul) | Flappy pipes (crashes) | Rescue rate | Dodge completion | Dodge vetoes | layers used, Flappy / Rescue / Dodge |
|---|---:|---:|---:|---:|---|
| 0.62 / 0.62 / 0.69 (calibrated on val) | 16.6 (30) | 1.00 | 0.50 | 2,163 | 1.33 / 1.95 / 1.08 |
| 0.95 for every type (published) | 200.0 (0) | 1.00 | 0.87 | 25 | 3.43 / 3.16 / 3.28 |
| 1.0 (full depth) | 200.0 (0) | 1.00 | 0.87 | 25 | 4.00 / 4.00 / 4.00 |
| oracle | 200.0 (0) | 1.00 | 0.87 | 0 | |

Sources: `runs/s2-nano-g2/arena.json` (calibrated τ), `runs/ship-s2-nano.arena.json` (τ = 0.95). The full-depth row was measured for this chapter with `sky-arena site/public/models/s2-nano.gguf --episodes 30 --game flappy,rescue,dodge --exit-threshold 1.0`. The published file holds the same weights as `s2-nano-g2`: replaying it at the calibrated thresholds reproduces 16.6 pipes and 30 crashes.

At the calibrated τ, the model crashes in every Flappy episode, and 48,405 of its 57,319 Flappy decisions stopped at the first exit, after one layer. At full depth the same weights fly 200 pipes every time and match the oracle in Dodge with only 25 reflex vetoes. At τ = 0.95 the results are identical to full depth while using 3.16 to 3.43 layers of 4 depending on the game (3.29 averaged over the three games, `site/public/bench.json`).

Why the val calibration misled: it measures average accuracy on independent questions, and most game states are easy. On those, the first exit is nearly as good as the last one. A game punishes the rare frame that goes wrong near a pipe, and one such frame ends a Flappy episode. Accuracy within 0.5 points on val says nothing about where the remaining errors fall.

The fix is to calibrate the exit on the states the model visits while playing. `docs/SPEC-S3.md` §5 and `scripts/calibrate-exit.sh` describe the procedure: play every game at full depth to get reference metrics, sweep τ over {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98} on the same episodes, and keep the smallest τ whose metrics stay within 2% of full depth with no extra crash or veto. The published file does not come from that script yet: τ = 0.95 was written by hand with `gguf-set`, and there is no `runs/calibrate-exit/` directory. The Doom model of [chapter 18](./18-doom) uses the same τ = 0.95 for the same reason.

```bash
gguf-set models/s2-nano-g2/skycmd-s2-nano-g2-ternary.gguf s2-nano.gguf \
  skycmd-s2.exit_threshold.choice=0.95 skycmd-s2.exit_threshold.score=0.95 skycmd-s2.exit_threshold.noul=0.95
scripts/calibrate-exit.sh s2-nano.gguf     # the closed-loop sweep, EPISODES=20 by default
```

## 7. s2-pico is too small

s2-pico has 310,337 parameters and a 177.3 KiB ternary file, smaller than s1-pico on disk. It does not play. After two DAgger rounds, at its calibrated τ (0.5 for every type): 1.9 Flappy pipes with 30 crashes, a rescue rate of 0.00 and 0.30 Dodge completion (`runs/s2-pico-g2/arena.json`). At full depth, measured for this chapter, the results are identical: 1.9 pipes, 30 crashes, rescue 0.00, Dodge 0.30 with 4,129 vetoes. Here the exit is not the problem. Its two exits give the same validation accuracy (0.527 at both, `runs/specialize-s2.log`), and `BENCH.md` points at the single 64-wide head in a single fusion block as the bottleneck. Rescue, with 7 options to match against sensor fields, stayed at 0.58 held-out accuracy before DAgger.

## Sizes and latencies

From `BENCH.md` ("Inference"), Apple M5, medians, best of 3 interleaved rounds. **The s2 rows use random-init models of the ladder shapes**, so only their speed is meaningful. Random weights never reach the default τ, so they are given at full depth (worst case) and at the first exit (best case).

| model | params | file | native, 1 question | native, 2 questions | WASM, one Flappy decision |
|---|---:|---:|---:|---:|---:|
| s1-pico (trained) | 202,305 | 533 KB (f16) | 188 µs | 407 µs | 694 µs |
| s1-nano (trained) | 1,134,209 | 2.7 MB (f16) | 974 µs | 1,980 µs | 3,541 µs |
| s2-pico ternary, full depth | 310,337 | 176 KB | 67 µs | 73 µs | 105 µs |
| s2-nano ternary, full depth | 1,532,033 | 684 KB | 298 µs | 329 µs | 501 µs |
| s2-nano, first exit | 1,532,033 | 5.9 MB (f32) | 118 µs | 147 µs | 196 µs |

Ternary rows use the default compute (weights expanded to f32 at load). A WASM decision is both Flappy questions on one state plus the physics step (`node crates/sky-web/bench/bench-node.mjs`).

- Per frame, s2-pico is 5.6× faster than s1-pico natively (73 against 407 µs) and 6.7× faster in WASM (104 against 694 µs, f32 file).
- s2-nano has 7.6× the parameters of s1-pico, a file 1.3× larger, and a WASM decision that is still faster (501 against 694 µs).

The trained file was also timed for this chapter, on a machine running other jobs (load average 9 to 14). Natively, `sky-bench site/public/models/s2-nano.gguf --questions 1` gives 280 µs per question. In WASM, three runs of `bench-node.mjs` gave medians between 627 and 1,036 µs per Flappy decision, with 3.6 layers used on average, while s1-pico measured 631 to 697 µs in the same runs. Treat the 501 µs above as the figure for a quiet machine.

## Training it yourself

```bash
cargo run --release -p sky-data -- train-tokenizer --vocab 2048 --num   # data/tok/tok-s2-2048.json
scripts/train-s2.sh nano                 # numeric MLM 12 min, decision training 20 min (QAT last 30%), calibrate, export, eval
scripts/specialize-s2.sh nano 2 8        # 2 DAgger rounds of 8 minutes, fully ternary
```

`train-s2.sh` writes `runs/s2-nano/report.md` and `models/s2-nano/skycmd-s2-nano-{f32,ternary}.gguf`. Each DAgger round writes `runs/s2-nano-gN/arena.json`, played at the calibrated τ. Play the result again at full depth or at τ = 0.95 before you trust it.

## Next

[20 What DAgger taught us](./20-dagger-lessons)
