# skycmd-s6: Parallax attention and the Muon v2 recipe (an extension of s3)

s6 is **s3 with one new option** in the same `skycmd-s2` GGUF architecture, plus a new optimizer recipe for training. Everything in `docs/SPEC-S2.md` and `docs/SPEC-S3.md` still holds unless this file says otherwise. A GGUF without the new key behaves exactly as before, so every s2 and s3 file and test stays valid. No s6 model is trained yet (issue #34): this file fixes the format, the recipe and the ablation that will decide whether s6 ships.

What motivates each part:
- **Parallax** (arXiv 2605.29157, Zuo et al., Northwestern / Tilde Research). Softmax attention is a local *constant* estimator; local linear attention (LLA) fits a local *linear* one and has strictly smaller integrated error, but needs a solver per query. Parallax learns the probe instead of solving for it. Its gain over softmax attention shows up under Muon and mostly vanishes under AdamW (§4.3 of the paper).
- **Muon v2**: two fixes from "Muon is Scalable for LLM Training" (Kimi, arXiv 2502.16982), decoupled weight decay and an update scale that matches AdamW's RMS; and DeepSeek V4's two-stage Newton–Schulz iteration.
- **Representation diagnostics**: the logit margin and the effective rank of the hidden states, the two mechanisms behind the robustness and transfer of Muon-trained features.

## 1. `parallax` (GGUF key `skycmd-s2.parallax`, bool, default false)
In every encoder block (state and static), when this flag is true:
- **Probe**: `attn_qkv.weight` is `[4d, d]` instead of `[3d, d]`. Its rows are q, k, v, then the probe `ρ` (`W_R` in the paper). It is one fused projection, so the first-block lookup table (`QkvTable`) and the ternary storage need no new tensor.
- **Probe norm**: `attn_px_norm.weight` `[head_dim]` (f32) RMS-normalizes each head of `ρ`: `ρ_h ← ρ_h · w / sqrt(mean(ρ_h²) + norm_eps)`. The paper applies an RMSNorm to `ρ` as well.
- **RoPE**: `ρ` is rotated like `q`, at the query's position. The paper finds RoPE on `ρ` helps under Muon.
- **Attention** (paper eq. 14), per head, with `p_ij = softmax_j(q_i · k_j / √hd)` under the usual mask and the unscaled probe scores `s_ij = ρ_i · k_j`:

  `out_i = Σ_j p_ij v_j · (1 + Σ_j p_ij s_ij) − Σ_j p_ij s_ij v_j`

  This equals softmax attention minus the softmax-weighted covariance of the values with the probe scores: `o_SA − Σ_KV ρ`. With `ρ = 0` it is softmax attention exactly, so the static stream's option isolation (permutation equivariance) still holds.
- **Kernels**: the correction is a reweighting of the softmax weights, `p_ij ← p_ij (1 + d2_i − s_ij)` with `d2_i = Σ_j p_ij s_ij`. That costs one extra `ρ Kᵀ` product, and the product with V stays single (`kernels::parallax_weights`, used by `self_attention_probe` and `Attention::run_probe`).
- **Not in fusion**: the fusion cross-attention (option markers over state and question tokens) stays softmax attention.
- **Initialization**: `ρ`'s rows are drawn like the other projections and `attn_px_norm` starts at 1. As in the paper, the correction is active from the first step.

## 2. Muon v2 (`OPT_RECIPE = 2`, sky-train)
There is one recipe in the code. A checkpoint written by recipe 1 has no `recipe` field and reads as 1. `Optimizer::load_state` then refuses to resume it, and it has to be resumed with the commit that wrote it.
- **Decoupled weight decay** (Kimi): `W ← W · (1 − lr · weight_decay) − lr · 0.2 · √max(A, B) · NS(M)`. It uses the same `weight_decay` as AdamW (0.01). Recipe 1 added an L2 term to the gradient before Newton–Schulz, which the orthogonalization mostly washed out.
- **Update scale** (Kimi): `0.2 · √max(A, B)` gives the orthogonalized update the RMS of an AdamW update, whatever the matrix shape, so `muon_lr` defaults to the AdamW learning rate.
  - Recipe 1 used `√max(1, A/B)` with `muon_lr = 0.02`. At the s3-pico shapes (96 × 96, 288 × 96), recipe 2 makes the per-entry step about 3.4× smaller at the default rates.
  - Retune with `--muon-lr` if the ablation shows underfitting.
- **Two-stage Newton–Schulz** (DeepSeek V4): `ns_steps = 5` fast iterations with Keller Jordan's quintic `(3.4445, −4.7750, 2.0315)`, then `ns_polish = 2` iterations of `(15/8, −10/8, 3/8)`.
  - That polynomial has `f(1) = 1` and `f'(1) = 0`, so it pulls the singular values the fast stage leaves in ~[0.7, 1.2] to within 0.5% of 1. The test checks [0.98, 1.02].
  - The exact DeepSeek V4 coefficients (8 + 2 iterations) are not public in a form we could check. These are the classical convergent ones.
- **CLI**: `--no-muon` puts the encoder matrices in AdamW (the AdamW arm of the ablation), `--ns-polish N` sets the polish iterations (also on `s2-bench`), and `--muon-lr` defaults to `--adam-lr`.
- **SODA weight decay** (`--wd-mode soda`, arXiv 2605.11172; the default stays `decoupled`).
  - Each decayed matrix, AdamW or Muon, is pulled toward its initialization `W₀` before the update of step k (0-based): `W ← W + (W₀ − W) / (k + 2)`.
  - The pull is not multiplied by the learning rate and has no hyper-parameter, so `--weight-decay` is ignored. It fades over a long run instead of being tuned for it.
  - `W₀` is optimizer state (`soda.z0.*`), saved and resumed with the checkpoint.
  - "Spectral Scaling Laws of Muon" (arXiv 2606.04058) finds that 5 Newton–Schulz steps orthonormalize the layers that matter at small scale. The `parallax-muon-nopolish` arm tests whether the polish iterations are worth keeping here.

## 3. Diagnostics (sky-train)
- **Logit margin** (`Metrics.margin`, in every metrics JSON): the gold option's raw logit minus the best other logit, averaged over questions with two options or more. A larger margin leaves more room before noise flips the answer.
- **Effective rank** (`erank` in the s2 `log.jsonl` at every validation): `exp(H)`, with H the entropy of the normalized singular values of the raw state latents of each exit. It is computed over the valid state tokens of the first 128 validation items, from the `d × d` Gram matrix on the CPU. A low value means the representation collapses onto a few directions.

## 4. Ablation (`cargo xtask s6ab [--arms a,b]`, not run yet)
A 2 × 2 grid on the s3-pico shape, then each change of §8–10 alone against it:

| arm | config | optimizer | decide |
|---|---|---|---|
| softmax-adamw | `configs/s3-pico.json` | AdamW (`--no-muon`) | |
| softmax-muon | `configs/s3-pico.json` | Muon v2 | |
| parallax-adamw | `configs/s6-pico.json` | AdamW (`--no-muon`) | |
| parallax-muon | `configs/s6-pico.json` | Muon v2 | |
| xsa-muon | `configs/s3-pico-xsa.json` | Muon v2 | |
| parallax-muon-soda | `configs/s6-pico.json` | Muon v2, `--wd-mode soda` | |
| parallax-muon-nopolish | `configs/s6-pico.json` | Muon v2, `--ns-polish 0` | |
| parallax-muon-kd | `configs/s6-pico.json` | Muon v2 | `--exit-kd 0.5` |
| parallax-muon-ema | `configs/s6-pico.json` | Muon v2 | `--ema 0.999` (the EMA weights are calibrated, exported and reported) |
| parallax-muon-stableqat | `configs/s6-pico.json` | Muon v2 | `--qat-grad-a 0.21` |
| parallax-muon-curriculum | `configs/s6-pico.json` | Muon v2, `--curriculum 0.1` in pretraining | |

Eleven arms × 3 seeds is 33 runs of about 27 minutes on an M5, so about 15 hours. `--arms` runs a subset.

- **Each arm and seed** (7, 11, 13):
  - `s2-pretrain` for 4,800 steps and `s2-decide` for 6,400 steps (the step counts of the 8 + 15 minute s3-pico runs, with the same data weights, QAT after 10%, loop dropout 0.25);
  - calibration and a ternary export;
  - `sky-arena` at full depth (τ = 1) on the four games, 30 episodes each.
- **Learning rates**: `cargo xtask s6lr` runs first. It trains `softmax-adamw` and `parallax-muon` at ×0.5, ×1 and ×2 of the default peak rate (seed 7, same step budgets) and writes its choice to `runs/s6lr/summary.json`. `s6ab --adamw-lr-scale a --muon-lr-scale m` then applies it: `--adam-lr` for the AdamW arms, `--muon-lr` for the Muon arms.
- **Output**: `runs/s6ab/summary.json` gives the mean and sample std over seeds of the last exit's ternary validation (accuracy, ECE, margin, effective rank), of each game's arena score and of the wall time of both training phases (`train_minutes`).
- **Question**: does Parallax beat softmax attention under Muon v2, and does that gain disappear under AdamW, as in the paper?
- **Rules**: fixed in `docs/s6-prereg.md` before any run: thresholds for each arm, the combination of the adopted changes (`scripts/train-s6.sh`, with `configs/s6-pico-xsa.json` if Parallax and XSA both win) and the final gate against s3-pico-g3.

## 5. Ladder (configs/s6-*.json)
| name | as | params (decision / pretraining stack) |
|---|---|---:|
| s6-pico | s3-pico + `parallax` | 463,969 / 212,705 (s3-pico: 445,345 / 203,393) |
| s6-femto | s3-femto + `parallax` | golden only |
| s6-femto-xsa | s6-femto + `xsa` | golden only |
| s3-pico-xsa | s3-pico + `xsa` | 445,345 / 203,393 (no new tensor) |
| s6-secrets-pico | s6-pico with a 320-token state and a 192-token head (§14) | 463,969 / 212,705 (RoPE: no new tensor) |
| s6-nano | general-purpose, vocabulary 2,048, d 144, 256 / 128 windows (§15) | 1,569,457 / 778,065 |
| s6-micro | s4-micro + `parallax` (§15) | 5,849,601 / 2,960,769 (s4-micro: 5,587,201) |

Each encoder block gains `d²` probe weights and `head_dim` norm weights: 9,312 per block for s6-pico (one state block, one static block).

## 6. Cost (Apple M5, measured on random data)
| | s3-pico | s6-pico |
|---|---:|---:|
| `s2-bench` numeric MLM step (batch 128 × 128, f32) | 55.7–55.9 ms | 62.3–63.7 ms (+13%) |
| `s2-bench` decision step (batch 64, f32) | 17.5–17.8 ms | 17.9–18.1 ms (+2%) |
| `sky-bench --random`, ternary, full depth, per decision | 68.2 µs | 74.1 µs (+9%) |

- **MLM step**: the explicit attention path (no fused SDPA kernel) costs most during pretraining, where the state stream is 128 tokens long.
- **Newton–Schulz polish** (`s2-bench --ns-polish 0` vs `2`, s3-pico): no measurable change on the MLM step (57.8–58.5 ms both ways); 17.3 ms → 17.5–18.8 ms on the decision step, within the machine's noise.

## 7. Golden
`sky-train s2-golden --config configs/s6-femto.json --prefix s6` writes `crates/sky-infer/tests/golden/s6{.json,-f32.gguf,-ternary.gguf}`, checked by `tests/golden_s6.rs`:
- **Tolerances**: 1e-4 for f32, 1e-3 for ternary.
- **Measured max |Δp|**: 4.6e-6 (f32), 1.8e-6 / 2.0e-6 (ternary, packed / f32 compute), 2.5e-4 / 1.5e-4 (int8 / LUT compute).

`kernels::tests::parallax_matches_naive_in_both_kernels` checks the tiled kernel (`Attention::run_probe`) and the small one (`self_attention_probe`) against eq. 14 directly, on 1 to 40 tokens and 1 to 3 heads.

`sky-train s2-golden --config configs/s6-femto-xsa.json --prefix s6x` writes the XSA golden (Parallax and XSA together), checked by `tests/golden_s6x.rs`. Max |Δp|: 5.7e-6 (f32), 1.6e-6 / 2.1e-6 (ternary packed / f32), 2.3e-4 / 2.1e-4 (int8 / LUT).

## 8. `xsa` (GGUF key `skycmd-s2.xsa`, bool, default false)
Exclusive self-attention: in every encoder block, each token's attention output `o` loses its component along the token's own value `v`, per head, before `attn_output`:

`o ← o − (o · v / (‖v‖² + 1e-6)) · v`

- **Effect**: the block carries what the other tokens say. Under Muon in nanochat it is stable and slightly better in bits-per-byte (0.7193 → 0.7171 at d24; DOI 10.21203/rs.3.rs-10673227/v1).
- **Cost**: no tensor, a few multiply-adds per token and head. It applies after Parallax when both are on, and is per token, so the static stream's option isolation holds.
- **Code**: training is `s2::model::exclusive`, inference `kernels::exclusive_inplace`.

## 9. Exit thresholds by conformal risk control (`sky-arena --calibrate-crc ALPHA`)
After SAFE-KD (arXiv 2602.03043) and CalexNet: thresholds with a stated guarantee, calibrated on the decisions the model actually meets while playing.
- **Data**: the games are played at full depth, and every decision records each exit's top calibrated probability and answer.
- **Loss**: at threshold τ, the engine answers at the first exit whose top probability reaches τ. The loss is 1 when that answer differs from the full-depth one.
- **Choice**: per question type, the smallest τ of {1.0, 0.99, …, 0.50} such that the conformal bound `(n·R̂(τ) + 1)/(n + 1)` is ≤ α, for it and for every larger τ (the scan stops at the first violation).
- **Output**: the command prints the `gguf-set` line that writes the thresholds, and `--json` writes the report.
- **Caveat**: the guarantee assumes exchangeability. Playing with early exits changes the states visited, so check the play at the chosen thresholds, as `scripts/calibrate-exit.sh` does for its sweep.
- **First run**: published s3-pico, 10 episodes of flappy, rescue and dodge, α = 0.05. The result is τ = 0.5 for choice and noul, a disagreement of 0.53% with full depth, and 1 layer of 4 on average. No score question was asked.

## 10. Exit self-distillation and DAgger mixing
- **`s2-decide --exit-kd W [--exit-kd-temp T]`** (T defaults to 2). Each exit before the last also learns `W · T² · KL(p_T(last) ‖ p_T(e))`, the last exit detached, so an early exit learns where the full-depth answer is uncertain (CalexNet).
  - It combines with `--dis` and loop dropout.
  - The test checks the term against its definition, and that no gradient reaches the last exit.
- **`sky-arena --dagger N --dagger-beta B`**: the oracle takes a share B of the decisions during DAgger collection, the student the rest; this is the mixing policy of DAgger, applied per turn as in arXiv 2605.12913.
  - B was a fixed 0.2, and stays the default.
  - Lowering it round after round moves the data toward the student's own states.

## 11. Weight EMA (`s2-decide --ema DECAY`)
An exponential moving average of the float weights follows every training step, the ternary QAT phase included. In low-bit QAT under the straight-through estimator, the weights oscillate around the quantization thresholds, and an EMA of the model damps that (arXiv 2311.05109; BEMA, arXiv 2508.00180, and Pathwise EMA, ICML 2026, refine the averaging itself).
- **Checkpoint**: the average is saved as a complete run in `<out>/last-ema`, next to `last`, so `s2-calibrate` and `s2-export` take it like any run. A resumed run resumes its EMA too.
- **Validation**: at the end of the run, the EMA is validated as `ema-f32` and `ema-ternary` in `log.jsonl`, after the raw weights.
- **Verified**: a 40-step run on the new Hugging Face sources (issue #38) trained, saved, calibrated and exported `last-ema` to a ternary GGUF.

## 12. StableQAT gradient (`s2-decide --qat-grad-a A`)
In the ternary QAT, the straight-through estimator passes a gradient of 1 through the rounding. StableQAT (arXiv 2601.19320, eq. 5, first order) multiplies it instead by

`g = (1 − c·cos π(x + x_q)) / (1 + c·cos π(x + x_q))`, with `c = A·√2·π`,

on the normalized weight `x = w / γ`, where `γ` is the per-row absmean and carries no gradient, and `x_q = round(x)`.
- **Effect**: the forward pass is unchanged. For ternary weights `g` lies in `[(1 − c)/(1 + c), 1]`, which is small on a grid point and 1 half-way between two points, so the updates that flip a weight back and forth across a threshold are damped.
- **Range**: `A = 0` is the straight-through estimator. `A` must stay below `1/(√2·π) ≈ 0.225` (Theorem 4.2), and the paper uses 0.21.
- **Code**: `s2::model::ternarize_stable`. The test checks the gradient against the formula and that `A = 0` gives 1.

## 13. Pretraining curriculum (`s2-pretrain --curriculum W`)
Pacing-based sampling (arXiv 2506.11300).
- **Difficulty**: each text row's lexical diversity, distinct tokens over tokens, one of the paper's most effective difficulty signals.
- **Warm-up**: over the first fraction `W` of training, a row is drawn uniformly from the easiest share of the rows. That share grows linearly from 20% of the corpus to all of it.
- **After the warm-up**: the usual epoch-shuffled order resumes.
- **Numeric rows**: the numeric curriculum rows (`--numeric`) are unaffected.
- **Reproducibility**: the draws use the session's RNG, so a run is reproducible and resumes exactly.

## 14. s6-secrets: API-key and secret detection (`configs/s6-secrets-pico.json`, not trained yet)

A model of its own, with the s6-pico architecture, specialized in one task: telling whether a short text (an `.env` file, code, a config, a `curl` command, a log, a diff) exposes a usable secret. The protocol and the publication gate are in `docs/s6-secrets-prereg.md`.

- **The jev** (`sky-data secret-jev`, `crates/sky-data/src/secrets.rs`). It has four questions per text, each its own source in `s2-eval` (`secret_bench:secret`, `:service`, `:kind`, `:line`):
  - **secret** (noul): is a usable secret exposed?
  - **service** (choice): which of 14 services issued it, or `none`?
  - **kind** (choice): a secret, a test key, a publishable key, a placeholder (or an example, a redacted value, a reference), or no credential at all?
  - **line** (choice): which line exposes the secret?
- **Generated, never real.** Keys follow each provider's public format with random bodies. Half the texts expose a secret; the rest are the negatives a scanner has to tell apart, often under a secret-sounding name.
- **A bench of unseen contexts.** `secret_synth` (50,000 texts) draws from 9 contexts; `secret_bench` (2,000 texts) from 4 others (docker-compose, toml, go, ini).
- **Wider windows, same weights.** With the s6 tokenizer a random key costs about one token per character: states run to 93 tokens (median) and 312 (max), and the service question with its options to 179. The config widens the state window to 320 tokens and the head to 192. RoPE has no position table, so the parameter count is s6-pico's. The jev stays out of `data/decisions`: the general s6, with 64 state tokens, would read it cut off before the key.
- **The bar: a rule scanner** (`sky-data secret-baseline`). It combines gitleaks-style prefixes for 20 formats, connection strings with an inline password, and an entropy rule on secret-sounding names, with allowlists for documentation, test and publishable keys. On `secret_bench` it scores 0.977 on the secret question (precision 0.966, recall 0.989) and 0.787 on the service question. All 35 of its false positives are random non-credentials under names like `CACHE_KEY`.
- **Training** (`scripts/train-secrets.sh data train eval`, prepared and not run): numeric MLM pretraining in 256-token sequences, decision training with QAT on `data/secrets` alone, then calibration, ternary export and `s2-eval`.

## 15. General-purpose sizes: s6-nano and s6-micro (not trained yet)

The s6 that ships is general-purpose: the four games and multiple-choice questions in free text. It comes in two sizes and replaces the s6-pico final model. The pico pilot and ablation (§4) still run, but only to choose the recipe the two sizes inherit. The protocol and the publication gates are in `docs/s6-gen-prereg.md`.

| name | vocabulary | d / ff | blocks × loops | heads | state / head window | params (decision / pretraining stack) |
|---|---:|---|---|---:|---|---:|
| s6-nano | 2,048 | 144 / 288 | 2 × 2 (exits 1, 2, 3, 4) | 2 | 256 / 128 | 1,569,457 / 778,065 |
| s6-micro | 4,096 | 256 / 768 | 2 × 3 (exits 2, 4, 6) | 4 | 384 / 128 | 5,849,601 / 2,960,769 |

- **Why not a pico.** With 1,024 tokens and windows of 64/96, 96.8% of the decision sequences are truncated, and s3-pico (same size as s6-pico) scores 0.382 on `typed_decisions_bench`, below the majority class (0.398). s4-micro (5.6M, same shape as s6-micro without Parallax) failed at 0.343 after only 30 minutes of MLM pretraining on 150 MB.
- **Inherited recipe.** Both configs have `parallax` and `fuse_numbers`. Before the first run, `parallax` and `xsa` are set from the ablation's adoption rules (`docs/s6-prereg.md` §4), and so are the training variables `OPT`, `LR_SCALE`, `PRETRAIN` and `DECIDE`.
- **GLM teacher** (`scripts/distill-glm.sh`; `docs/s6-gen-prereg.md`, amendment 1). `sky-data teacher label-decisions` asks GLM-5.3-flash, through the z.ai API, for one probability per option on up to 1,000 hard-labelled train records per source family of `data/decisions` (20,876 records, 1,044 calls of 20). The benchmark, held-out and GLM-generated sources are never sent. Workers call the endpoint in parallel, and the file of labels resumes after an interruption. `sky-data teacher mix-labels` keeps the labels whose argmax is the gold answer and writes `0.5 · gold + 0.5 · teacher` to `data/distill/glm.jsonl`, with the agreement per source. The teacher runs remotely, so labelling puts no load on the machine.
- **Tokenizers.** `data/tok/tok-s6-2048.json` and `tok-s6-4096.json` (`sky-data train-tokenizer --num` on `data/text`, `data/text-open`, `data/decisions`, `data/games`), written by `scripts/prepare-data.sh`.
- **Training** (`SIZE=nano|micro scripts/train-s6.sh [lr] [train] [dagger] [exit] [eval]`). The steps:
  - `lr`: a per-size learning-rate pilot at ×0.5, ×1 and ×2 of the adopted rate (20 + 20 min each);
  - `train`: MLM pretraining on `data/text-open,data/text` in 256-token sequences (nano 90 min, micro 180 min), then decision training with QAT on `decisions=0.40,distill=0.15,games=0.35,numeric=0.10`, batches grouped by length (nano 90 min, micro 180 min);
  - `dagger`: three rounds that keep 30% decisions (`games=0.4,rN=0.25,decisions=0.2,distill=0.1,numeric=0.05`; nano 15 min, micro 20 min each);
  - `exit`: exit thresholds by conformal risk control (α = 0.02);
  - `eval`: the evaluation and the gate (`scripts/s6-gate.py`).
- **Learning rates.** `LR_SCALE` × √(96/d) × the size's pilot scale × each phase's default (3e-3 pretraining, 1e-3 decision), on `--muon-lr`, or on `--adam-lr` with `--no-muon`. `OPT` carries no rate. √(96/d) moves the pico rate to width d as the spectral μP condition predicts for AdamW-matched Muon (×0.816 nano, ×0.612 micro); the pilot then tries ×0.5, ×1 and ×2 of that centre.
- **Test sets:**
  - `typed_decisions_bench`;
  - MASSIVE en/fr intents (`data/s4/heldout`, written by `sky-data s4-corpus --only-heldout`): zero-shot, since MASSIVE is not in `data/decisions`;
  - notam, test only;
  - the When2Call test split.

  Banking77 is now in `data/decisions`, so it is trained on: it is reported as in-distribution and never counted as zero-shot.
- **Truncation report** (`sky-train s2-truncation --config C --tokenizer T --records R [--per-source 500] [--json out]`): the share of records whose state passes `max_state_tokens`, and whose head or an option passes `max_head_tokens` or the 48-token option cap. It reads the first records of each source, applies the config's number fusion, and reports per source and overall.
