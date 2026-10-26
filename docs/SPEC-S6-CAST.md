# SPEC-S6-CAST: probabilistic forecasting with the s6 encoder

s6-cast forecasts a time series: given up to 2,048 past points, it returns nine quantiles (the deciles) for each of the next 64 steps, and longer horizons by rolling out. It reuses the s6 state encoder unchanged:
- weight-shared blocks applied in loops;
- Parallax attention;
- an exit after each loop;
- ternary QAT and Muon v2.

It runs in sky-infer on native targets and in WASM. Code only so far: no model has been trained.

Why this design (`docs/research.md`, forecasting):
- At this size, the best zero-shot forecasters are small and long-context. TinyCast has 146K parameters; Reverso-Nano has 200K and Reverso 2.6M.
- They pretrain mostly or only on synthetic series, which Simulation Alone and TempoPFN show is enough for a decent zero-shot model.
- s6-cast keeps that recipe and adds what skycmd already has: ternary weights, early exits and the browser runtime.

## 1. Architecture (`configs/s6-cast-*.json`)

| name | d | heads | blocks × loops | exits | patch × context | horizon | decoder | params | ternary GGUF |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| s6-cast-pico | 96 | 2 | 2 × 2 | 1, 2, 3, 4 | 32 × 64 (2,048) | 64 | 2 | 388,320 | 144 KB |
| s6-cast-femto | 32 | 2 | 1 × 2 | 1, 2 | 8 × 40 (320) | 16 | 1 | 23,128 | golden only |

1. **Input.** The context is normalized by min-max over its observed values (`x̃ = (x − lo) / (hi − lo)`; a constant window maps to 0). TinyCast and Reverso both found min-max better than z-score. It is cut into patches of `patch` points, right-aligned on the last observation. A missing value is 0 with mask 0. `patch_proj` (Linear `2·patch → d`, with bias) reads each patch's values followed by its mask.
2. **Encoder.** The s2/s3/s6 state encoder:
   - blocks `blk.{l}` with pre-LN, fused QKV, RoPE and Parallax (`attn_qkv` is `[4d, d]`, `attn_px_norm`), optional XSA, and a GeGLU feed-forward;
   - `loop_embd` before each loop after the first, and `blk.0.attn_norm_loop`;
   - one exit per loop: `H_e = LN(x; exit.{e}.norm)`.

   The left padding of a short context (the patches before the first observed one) is masked out of every attention. Patches after it that hold no observation remain keys.
3. **Decoder.** `horizon_patches` learned queries (`cast_query`, `[H/patch, d]`) read `H_e` through `decoder_layers` cross-attention blocks `dec.{n}`. The layout is the s2 fusion block: LayerNorms with bias, `attn_q`, `attn_kv`, `attn_output`, a ReLU feed-forward of width `2d`, and residuals. There is no self-attention between queries. This is the decoupled horizon of Timestral, and Reverso's attention decoder.
4. **Head.** `out_norm` then `quant_head` (Linear `d → patch·Q`, with bias) gives `Q` raw outputs per step. The middle output is the median. Each level above adds `softplus` of its output to the level below it, and each level below subtracts, so the quantiles cannot cross. The output is mapped back with the window's normalization.
5. **Initialization.** `quant_head.bias` is 0.5 on the median outputs (the middle of the normalized range) and −3 on the others: neighbouring levels start `softplus(−3) ≈ 0.05` apart.

## 2. Inference (`sky_infer::cast`, `sky-cast`)

- **`CastModel::forecast(history, horizon, opts)`.** It returns `[horizon][Q]` in the units of the history.
  - Past one pass (64 steps), each pass's median is appended to the context of the next, as in TinyCast and Timestral.
  - `flip` averages with the sign-flipped context, level by level (`q_k = (q_k(x) − q_{Q−1−k}(−x)) / 2`, the Reverso / TinyCast trick, two passes).
- **Early exit.** The exits run in order, and a pass stops when an exit's median differs from the previous exit's by less than `exit_tolerance` (normalized units, max over the horizon). `exit_tolerance` 0 (the default) runs every exit. The calibration that picks it comes with the evaluation harness.
- **Left padding.** It is dropped, not masked: RoPE scores depend only on relative positions, so the kept patches keep their absolute positions and match the masked training graph.
- **Compute.** The kernels are those of s2: full attention through `self_attention_probe` up to 32 patches and the tiled `run_probe` above, and every `TernaryCompute` mode.
- **`sky-cast <model.gguf> --values … | --file … | --synth kind [--horizon H] [--flip] [--bench N]`.** It prints the JSON forecast, or the latency with `--bench`.
- **Latency.** On the 60-step probe checkpoint, ternary computed in f32 on one Apple M5 core, full depth, 64-step horizon: 869 µs with 2,048 points of history and 131 µs with 256.

## 3. Training (`sky-train cast-train`)

- **Data** (`cast::data`). Windows come from a pool of synthetic series (4,096 series of `context + horizon + 1,024` points). Every 50 steps a block of 256 is regenerated in place. The pool at a given step depends only on that step, so a resumed run rebuilds it exactly.
  - Real series can be mixed in (`--series` directories of JSONL `{"id", "values"}`; one in ten held out by id; `--real-frac 0.3`).
  - Context length: the full context half of the time, a whole number of patches from 4 up 30 % of the time, and 8–256 points (log-uniform, a partly filled first patch) the other 20 %: GIFT-Eval's short configurations give 13 to 300 points.
  - Normalization and patching read only the last `context()` points, at training and at inference alike (`prep::prepare`).
  - Normalized targets (and teacher quantiles) are clamped to [−5, 6]: a nearly flat context followed by a jump would otherwise dominate its batch's gradient.
- **Generators** (`sky_infer::cast::synth`, deterministic per seed, mixed 0.3 / 0.15 / 0.3 / 0.25):
  - **`gp`**: KernelSynth with Reverso's kernel bank (constant, linear, RBF, rational quadratic, Matérn ν ∈ {0.5, 1.5, 2.5}, and periodic with a period from Reverso's set of 19), length scales 0.1, 1 and 10 of the series.
    - Stationary kernels are sampled with 48 random Fourier features instead of a Cholesky factor (O(n·M), not O(n³)).
    - One to five draws are combined by sum or product of the functions. A product of draws is not a draw of the product kernel, but it gives the same kinds of shapes.
  - **`pulse`**: trapezoid trains (spikes or dips) over a baseline, with noise.
  - **`tsi`**: a trend (linear, exponential, polynomial or piecewise), up to three seasonal waves (sine, sawtooth or square), noise (Gaussian, Laplace or Student-t), outliers and level shifts.
  - **`sarima`**: seasonal ARIMA, with stationary AR parts drawn through their partial autocorrelations (Durbin–Levinson), seasonal AR and MA terms, optional differencing and seasonal differencing, and Gaussian or Student-t noise. This is Simulation Alone's simulator family.
- **Augmentations** (Reverso), each with probability 0.3, missing values left missing:
  - stride-2 or stride-4 downsampling, cut from a segment 2 or 4 times as long when the series has it;
  - amplitude modulation;
  - sign flip and time reversal;
  - censoring at a quantile;
  - mixup with a second series (probability 0.1);
  - 1–5 % missing values in the context (probability 0.1).
- **Loss.** Pinball over the deciles and the observed target steps, at every exit (deep supervision with s2's `linspace(0.5, 1)` weights).
- **Distillation** (`--teacher tinycast.gguf`). A share of the windows (`--teacher-frac 0.5`) get the teacher's deciles, computed on the CPU by the batch producer thread. The loss adds `w·L1(student, teacher)` per exit, with `w` decaying linearly from `--teacher-weight 0.5` to 0 over the budget, so the end of training follows the ground truth only. Past 90 % of the budget the producer stops computing teacher targets (the weight is then under a tenth of its start). Students that fit both the labels and a teacher can surpass it (arXiv 1805.04770, 2206.08491). The batches depend only on `(seed, step)`, so prefetching and resuming draw the same data.
- **Optimizer.** Muon v2 on `blk.*`, and AdamW elsewhere. `patch_proj`, `cast_query`, `loop_embd` and the norms get no weight decay. WSD schedule.
- **QAT and checkpoints.** Ternary QAT covers the last 20 % (`--qat-frac`), with the encoder blocks, the decoder matrices and `quant_head` ternary, and optional StableQAT. The weight EMA is optional (`--ema`). Checkpoints go to `last/`, `pre-qat/` and `last-ema/`.
- **Speed.** About 2,000 windows per second for s6-cast-pico at batch 256 on an M5.
- **`cast-synth`** writes generator samples as JSONL.

## 4. GGUF (`general.architecture = skycmd-cast`)

- **Keys** (`skycmd-cast.*`):
  - `embedding_length`, `feed_forward_length`, `head_count`;
  - `state_block_count`, `state_loops`, `exits`;
  - `parallax` and `xsa` (written only when true);
  - `patch_length`, `context_patches`, `horizon_patches`, `quantiles` (f32 array);
  - `decoder_block_count`, `rope_freq_base`, `layer_norm_epsilon`, `exit_tolerance`, `weight_format`.
- **Tensors.**
  - `patch_proj.{weight,bias}`;
  - `blk.{l}.*`, `blk.0.attn_norm_loop.weight` and `loop_embd.weight`;
  - `exit.{e}.norm.weight`, `cast_query.weight`;
  - `dec.{n}.{q_norm,kv_norm,ffn_norm}.{weight,bias}`;
  - `dec.{n}.{attn_q,attn_kv,attn_output,ffn_up,ffn_down}.{weight,bias}`;
  - `out_norm.{weight,bias}` and `quant_head.{weight,bias}`.
- **Encodings** (as s2). Ternary matrices are stored as `{base}.tq` (I8, 4 weights per byte) plus `{base}.scale` (F16, per row). Other tensors are F32, or F16 for matrices in the f16 format. `sky-train cast-export --run <run>/last --format ternary --out …` writes the file.

## 5. Golden

`sky-train cast-golden` uses `configs/s6-cast-femto.json`, which has Parallax and XSA, 40 patches, and both attention kernels. It writes three files to `crates/sky-infer/tests/golden/`:
- `cast-f32.gguf`;
- `cast-ternary.gguf`;
- `cast.json`, the normalized quantiles of every exit computed by the mlx-rs graph on the weights exactly as stored, for six contexts:
  - one per generator family;
  - a 29-point context (left padding);
  - a series with missing values and an empty interior patch;
  - a constant series.

`crates/sky-infer/tests/golden_cast.rs` recomputes them from the raw contexts in every `TernaryCompute` mode. Tolerances are 1e-4 for f32 and 1e-3 for ternary; int8 and LUT get 2e-3, since their activations are quantized.

## 6. Evaluation (`scripts/gift-eval.py`, `sky-cast-eval`)
GIFT-Eval with its official protocol, so the numbers are comparable to the leaderboard and to TinyCast's:

1. `scripts/gift-eval.py download` / `export` (uv, Python 3.12): the 97 configurations' test windows, exported locally as f32 contexts (last 16,384 points) with their prediction length and season (371,330 windows, 7 GB; GIFT-Eval is for research use and is never redistributed).
2. `sky-cast-eval run --model m.gguf --windows data/gift-eval --out <dir> [--flip]`: the nine deciles of every window, on every core. A window the model cannot read (no observation in its last `context()` points) gets the seasonal naive on every level, and is counted. `sky-cast-eval naive` writes the seasonal-naive forecasts.
3. `scripts/gift-eval.py score`: the forecasts replayed through gluonts' `evaluate_model` exactly as TinyCast's driver does (`axis=None`, masked invalid labels, the dataset's seasonality), the leaderboard-format CSV, then nGMASE, nWQL and nMSIS: geometric means of each configuration's ratio to the seasonal-naive reference (`eval/gift-eval/seasonal_naive.csv`).

The harness is checked before any s6-cast number is read (`docs/s6-cast-prereg.md` §5): our seasonal naive scores nGMASE 0.99986 over the 97 configurations (the same windows and scale as the leaderboard; its nWQL is 0.888, not 1, because it repeats the point forecast on every decile while the reference is probabilistic), and our TinyCast port must reproduce its published 0.774 / 0.545.

**Exit calibration** (`sky-cast-eval calibrate --model m.gguf --out m-cal.gguf`). On 2,000 synthetic windows unseen in training, it picks the largest `exit_tolerance` whose early-exit median stays within δ = 0.02 (normalized units) of the full-depth median, with the conformal bound `(n·R̂ + 1)/(n + 1) ≤ α` (α = 0.05) on the share of windows that do not. Tolerances are tested from the smallest and the search stops at the first that fails (fixed-sequence testing: the risk is not monotone in the tolerance). The tolerance is written into the GGUF.

## 7. Training plan
`scripts/train-cast.sh [data] [pilot] [train] [export] [eval]` follows `docs/s6-cast-prereg.md`:
- **data**: `sky-data fetch-series`, `cast-synth --games` and the GIFT-Eval export;
- **pilot**: optional learning-rate pilot;
- **train**: 480 minutes, QAT over the last 20 %, EMA;
- **export**: the EMA or the last weights, by validation pinball, then exit calibration;
- **eval**: GIFT-Eval for s6-cast (with and without flip), the seasonal naive and TinyCast.
