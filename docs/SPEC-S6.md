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

Eight arms × 3 seeds is 24 runs of about 27 minutes on an M5, so about 11 hours. `--arms` runs a subset.

- **Each arm and seed** (7, 11, 13):
  - `s2-pretrain` for 4,800 steps and `s2-decide` for 6,400 steps (the step counts of the 8 + 15 minute s3-pico runs, with the same data weights, QAT after 10%, loop dropout 0.25);
  - calibration and a ternary export;
  - `sky-arena` at full depth (τ = 1) on the four games, 30 episodes each.
- **Output**: `runs/s6ab/summary.json` gives the mean and sample std over seeds of the last exit's ternary validation (accuracy, ECE, margin, effective rank) and of each game's arena score.
- **Question**: does Parallax beat softmax attention under Muon v2, and does that gain disappear under AdamW, as in the paper?
- **Proposed rule**: s6 ships only if parallax-muon beats softmax-muon on validation accuracy without losing arena score. Fix the rule before reading the results.

## 5. Ladder (configs/s6-*.json)
| name | as | params (decision / pretraining stack) |
|---|---|---:|
| s6-pico | s3-pico + `parallax` | 463,969 / 212,705 (s3-pico: 445,345 / 203,393) |
| s6-femto | s3-femto + `parallax` | golden only |
| s6-femto-xsa | s6-femto + `xsa` | golden only |
| s3-pico-xsa | s3-pico + `xsa` | 445,345 / 203,393 (no new tensor) |

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
