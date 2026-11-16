# Training a fast-weight associative memory in a tiny ternary recurrent network: what the literature prescribes

Research date: 16 November 2026. A dossier of 52 fetched sources (43 cited), written for the diagnosis of s6-sleep (`docs/s6-sleep3-diagnosis.md`). Every statement cites the source it was read in as `[S#]` (list at the end); `[M]` and the notes marked "not from a source" are the author's own knowledge or reading, not something a fetched page says. Nothing here is a result of this project.

## Abstract / TL;DR

- A purely additive (Hebbian, outer-product) fast-weight write cannot edit an association: re-writing a key adds a second value on top of the first, and the published fix is the error-correcting delta rule, `S_t = S_{t-1} - β_t (S_{t-1} k_t - v_t) k_t^T`, which first reads the value currently stored under the key and replaces it [S3][S1][S26].
- The delta rule only behaves when keys are normalised: with `0 ≤ β_t ≤ 1` the update is stable only if `‖k_t‖² ≤ 2`, and with unit-L2 keys and `β_t = 1` the factor `I - k_t k_t^T` is an exact projector that erases one direction and preserves the others [S53][S1].
- Retrieval from a sum of outer products returns the wanted value plus a crosstalk term weighted by `k_i^T k_j`, so keys of different items must be (near-)orthogonal, and at most `d` orthogonal keys fit in `d` dimensions [S54][S3].
- Every fast-weight system in this dossier that trained reliably normalises something around the memory: layer normalisation inside the fast-weight loop [S10], layer normalisation of the retrieved vector [S26], L2-normalised keys and queries plus a normalisation of the read before the output projection [S1][S53][S34], or a unit-norm memory [S44].
- Published fast-weight memories do not start with a learned write gate that can close: the original used fixed scalars (fast learning rate 0.5, decay 0.95) [S10], and the learned-update variant initialised its learned coefficients at those same working values [S32].
- A controller that can answer through a direct path tends to ignore the memory; the documented counter-measures are dropout on the bypass path, removing the recurrent shortcut to the output, and an auxiliary read-back loss on the memory itself [S15][S45][S46].
- For quantised recurrent networks, the two papers that study it find that quantisation distorts gate pre-activations and inflates weight norms, and that normalising the pre-activations is what restores trainability [S19][S30].

> **Note (not from a source).** The problem this review was written for (not from the sources): a 0.3-3M-parameter ternary QAT recurrent model with a per-sequence fast-weight matrix S, write gate β = σ(wβ·x), decay gate α, shared key/query projection, read r = S·q through a zero-initialised read-out Wo added to the hidden state; delayed recall over 64 frames with 4 colours and 52% of asks on re-shown doors. Measured: (1) β collapses to ~0 within 35 steps, reading an un-normalised state of norm ~96 at 10× the base learning rate; (2) gates frozen open keep the gradient alive but loss stays at chance; (3) keys nearly collinear (mean squared cosine 0.73), each frame writes the mean of ~9 tokens at 6% strength; (4) re-shown keys superpose; (5) an auxiliary linear colour decoder on the read rises from 0.25 to 0.41 in 84 steps while the answer head ignores the read; (6) with the colour placed in the ask frame the model learns in ~30 steps.

## Background & motivation

The fast-weight idea is a third kind of variable, faster than slow weights and slower than activities, that stores a temporary memory of the recent past in a matrix whose capacity is O(H²) rather than the O(H) of a hidden vector [S10]. Linearised self-attention is formally the same object: a slow network emits keys and values and programs a fast weight matrix by additive outer products [S3]. Fast weight programmers are recurrent networks with a two-dimensional, matrix-form hidden state that serves as short-term memory and is programmed by another network trained by gradient descent [S4].

Two limits of the additive rule are documented. First, associations interfere unless keys are orthogonal, and no more than `d_dot` orthogonal keys exist, so storing more associations than that produces retrieval error [S3]. Second, the additive rule cannot update an association: an ideal memory should "dynamically interact with the memory contents and selectively determine which associations to remember or to forget", which a purely additive update does not do [S3].

> **Note (not from a source).** This project's symptoms (3) and (4) are these two limits exactly: collinear keys give crosstalk on every read, and re-shown doors need an edit, not a sum. Symptom (6) says the answer head and optimiser are fine and the failure is in what the memory stores and how its read reaches the head.

## Key papers (chronological)

The year column is a bibliographic label taken from arXiv identifiers and venue lines, not a claim checked against the extracts. [M]

| Year | Paper | What it contributes to this problem |
|---|---|---|
| 2014 | Neural Turing Machines | Each write is decomposed into an erase followed by an add, with an erase vector in (0,1) [S25] |
| 2015 | Jozefowicz et al. | Default init leaves the forget gate at 0.5; initialise its bias to 1 or 2 so the gate starts near 1 and gradient flows [S52] |
| 2015 | End-To-End Memory Networks | "Linear start" training, 10% random empty memories, and restarts to escape bad local minima [S41] |
| 2016 | Ba et al., Fast Weights | `A(t) = λA(t-1) + ηh(t)h(t)^T`, layer normalisation inside the inner loop, fixed η and λ [S10] |
| 2017 | Recurrent Entity Networks | Key-gated additive memory update followed by renormalisation, which is the forgetting mechanism [S44] |
| 2017 | Zhang & Zhou | Replaces the fixed λ, η by learned matrices initialised at the fixed values; solves length-50 recall where layer-normalised Fast Weights fail [S32] |
| 2018 | Tallec & Ollivier | Chrono initialisation of gate biases from the expected dependency range [S36] |
| 2018 | Trinh et al. | Unsupervised auxiliary loss at anchor points, two-phase training [S27] |
| 2018 | Miconi et al.; Backpropamine | Plastic coefficient α and plasticity rate η as trained parameters; decay, Oja or hard clip to bound the trace [S17][S11] |
| 2018 | rsDNC; Collier & Beel | Normalisation plus Bypass Dropout to force early memory use; memory-content initialisation decides NTM convergence [S15][S40] |
| 2019 | Ardakani et al.; Hou et al. | Normalisation is what makes binary/ternary LSTMs trainable [S19][S30] |
| 2019 | Metalearned Neural Memory | A read-back reconstruction "meta objective" added to the task loss [S46] |
| 2020 | Gu et al. | Saturated gates stop learning because the gate gradient is f(1-f); refine gate and uniform gate initialisation [S43] |
| 2021 | Schlag et al. (FWM, LT-FWP); Irie et al. | Delta-rule write with learned write strength, sum normalisation, capacity analysis, recurrent variants [S26][S3][S38] |
| 2024-2026 | DeltaNet, Gated DeltaNet, Gated DeltaNet-2, OSDN | L2-normalised keys, gated delta rule with decay α, decoupled erase/write gates, preconditioned write [S1][S8][S5][S2] |

## Methods & approaches compared

### 1. Fast Weights (Ba et al. 2016): what made it train

The update is `A(t) = λ A(t-1) + η h(t) h(t)^T`, with A equal to zero at the start of each sequence, and the next hidden state is obtained by an inner loop `h_{s+1}(t+1) = f([W h(t) + C x(t)] + A(t) h_s(t+1))` in which the bracketed term is held as a sustained boundary condition [S10].

The stated failure mode is that "the scalar product of two hidden vectors could vanish or explode depending on the norm of the hidden vectors"; the remedy is layer normalisation over the whole pre-activation at every inner-loop iteration, `h_{s+1}(t+1) = f(LN[W h(t) + C x(t) + A(t) h_s(t+1)])`, which the authors report makes the memory "more robust to the choice of learning rate and decay hyper-parameters" [S10].

Exact settings: fast learning rate η = 0.5 and decay λ = 0.95 by default, η = 0.5 and λ = 0.9 on associative retrieval; fast weights updated once per time step (more inner iterations performed similarly); slow recurrent weights initialised to an identity matrix scaled by 0.05; ReLU units; Adam with mini-batches of 128 [S10].

Evidence: with 20 recurrent units the fast-weight model reaches 1.81% error on associative retrieval where the LSTM is at 60.81% [S10].

Limits reported by later work: layer-normalised Fast Weights failed to converge on length-50 associative recall, even with 100 units and more inner loops [S32]; a fast-weight RNN and a layer-normalised LSTM both failed on a harder associative task where an LSTM with fast weights succeeded [S29]; and Schlag et al. could not find hyper-parameters that made Ba's fast weights converge on their long-sequence benchmark [S26].

### 2. Delta-rule writes: overwriting instead of superposing

**Schlag, Irie & Schmidhuber 2021.** With `k, v, q = W_k x, W_v x, W_q x`: read the old value `v̄ = W^(i-1) φ(k)`, compute `β = σ(W_β x)` with `W_β` a 1×d row, form `v_new = β v + (1-β) v̄`, and write `W^(i) = W^(i-1) + β (v - v̄) ⊗ φ(k)`, with `W^(0) = 0` [S3]. β is called the write strength because it defines how far the new value replaces the old one [S3].

Normalisation in that paper: divide the key and the query by the sum of their components ("sum normalisation"), `φ'(q) = φ(q) / Σ_j φ(q)_j`; the accumulator-style attention normalisation is rejected because it grows with the number of steps and does not balance the write and remove terms [S3].

Why not a plain gate on the whole matrix: the gated rule `W ← (1-β) W + β v ⊗ φ(k)` also scales every other stored association by (1-β), whereas the delta rule updates the addressed key to `(1-β) v_old + β v_new` and leaves an orthogonal key's value unchanged [S3].

Evidence on this project's exact situation (keys re-assigned within a sequence): with sequence length 40 and 20 unique keys sampled with replacement, the delta rule with sum normalisation outperforms all variants and "the baseline sum update rule fails" [S3]. On a code-execution task that requires maintaining variable values, linear Transformers with the additive rule "completely fail" while delta-rule models learn it [S38].

**Fast Weight Memory (Schlag, Munkhdalai & Schmidhuber 2021).** `[k1, k2, v] = tanh(W_write h_t)`, `β = σ(W_β h_t)`, `F' = F_{t-1} + β vec(k1 ⊗ k2) ⊗ (v - v_old)`, then `F_t = F' / max(1, ‖F'‖_2)` so the fast weights cannot grow without bound [S26]. The read is `n = LN(F_t vec(n_prev ⊗ e))` with a layer norm that has no learned affine map, the result goes through `W_o`, and the output is `softmax(W(h_t + FWM(h_t, F_t)))` [S26]. The tensor product of two keys is used because it "guarantees unique orthogonal vector representations for novel key pairs" [S26]. Sizes and optimiser: LSTM of 256, memory dimension 32, 3 reads, Adam, learning rates searched from 0.0001 to 0.002 [S26].

**DeltaNet (Yang et al. 2024).** `S_t = S_{t-1} - β_t (S_{t-1} k_t - v_t) k_t^T`, read as one SGD step on the online regression loss `½‖S k_t - v_t‖²`, whereas the additive rule corresponds to a linear loss whose gradient does not scale with the error [S1]. Keys and queries are `k = SiLU(W_K x) / ‖SiLU(W_K x)‖_2` and likewise for q with a separate `W_Q`; replacing the L1 norm by the L2 norm "greatly increases performance" [S1]. A normalisation is applied before the output projection for stable training [S1].

The stability argument: the transition `I - β_t k_t k_t^T` has eigenvalue `1 - β_t‖k_t‖²` along the key and 1 elsewhere, so with `0 ≤ β_t ≤ 1` stability requires `‖k_t‖² ≤ 2` [S53]. The full pipeline is: query/key Linear → short convolution → SiLU → L2 norm; value Linear → short convolution → SiLU; beta Linear → sigmoid; output delta rule → RMSNorm → Linear [S53].

The reference implementation computes beta as `b_proj(hidden_states).sigmoid()` where `b_proj` is a bias-free linear layer, defaults to `qk_norm='l2'` and `qk_activation='silu'`, sets beta to ones when `use_beta` is false, and multiplies beta by 2 when negative eigenvalues are allowed [S34].

> **Note (not from a source).** A bias-free projection with a standard zero-mean initialisation puts β near 0.5 at the start; that is my reading of the code, the file does not state the initial value.

**Gated DeltaNet (Yang, Kautz & Hatamizadeh 2025).** `S_t = S_{t-1} (α_t (I - β_t k_t k_t^T)) + β_t v_t k_t^T` with a data-dependent decay `α_t ∈ (0,1)`; `α_t → 1` recovers the pure delta rule and `α_t → 0` clears the memory [S8]. The paper's case study finds that decay hurts retention (the decay-only model degrades because it forgets too quickly), gating helps filtering, and the delta rule helps memorisation [S8]. In the reference layer the decay gate has its own bias-free projection `a_proj` plus two per-head parameters excluded from weight decay: `A_log` initialised as the log of a uniform draw in (0, 16) and `dt_bias` initialised as the inverse softplus of a log-uniform draw between 0.001 and 0.1 [S42].

> **Note (not from a source).** As I recall the kernel, the decay is α = exp(-exp(A_log) · softplus(a_proj(x) + dt_bias)); the layer file fetched here passes these tensors to the kernel but does not show the formula. Under that formula the initial per-step log-decay is small, i.e. α starts close to 1.

**Follow-ups.** Gated DeltaNet-2 argues that one scalar β carries two decisions that need not agree, and replaces it by an erase gate `b_t ∈ [0,1]^{d_k}` on key coordinates and a write gate `w_t ∈ [0,1]^{d_v}` on value coordinates [S5]. OSDN makes the same point from the optimisation side: a scalar write gate is vanilla SGD on the memory, must compromise across key directions, and can be replaced by an online diagonal preconditioner [S2].

### 3. Key geometry and capacity when keys are not orthogonal

For unit-length keys, `S k_j = v_j + Σ_{i≠j} (k_i^T k_j) v_i`; the second term is the retrieval error and vanishes only when keys are mutually orthogonal [S54]. Experimentally, linear attention with `d_key = 64` starts accumulating errors at 60 or more associations [S3].

Ways the sources obtain separated keys: a feature map that raises the effective key dimension (DPFP) [S3]; a tensor product of two learned keys [S26]; ReLU keys and values feeding a Hebbian matrix [S14]; and, for binary or bipolar key memories, training the encoder so unrelated items map to quasi-orthogonal high-dimensional vectors, with a "softabs" sharpening function chosen because a softmax pushes dissimilar keys towards anti-correlation rather than uncorrelation [S55].

The delta rule itself reduces interference: with unit keys each update removes the state component parallel to the current key, which "helps maintain cleaner separations between different key vectors over time" [S53].

### 4. Hebbian plastic networks: how plasticity is initialised and bounded

Differentiable plasticity uses an effective weight `w_ij + α_ij Hebb_ij(t)` with `Hebb(t+1) = η x_i x_j + (1-η) Hebb(t)`; the trace is reset to zero at each episode, α and η are trained by gradient descent, and η is a single learned scalar [S17]. The published listing initialises both w and α as `0.01 * randn`, uses the fixed trace update `hebb = .99 * hebb + .01 * outer(y, yout)`, and trains with Adam at 3e-4 [S17][S37]. Oja's rule is offered as the alternative that keeps traces stable without decay [S17].

Backpropamine replaces the decay by `Hebb(t+1) = Clip(Hebb(t) + η x_i x_j)` with a hard clip to [-1, 1], and distinguishes η (how fast new information enters) from α (the maximum magnitude of the plastic component) [S11]. Its plastic-LSTM code initialises α at `0.0001 * rand`, η at `0.01 * ones(1)`, and clamps the trace at ±2.0 [S24].

A Hebbian rule that overwrites without a delta term is H-Mem's: `ΔW_kl = γ+ (w_max - W_kl) v_k k_l - γ- W_kl k_l²`, where the last term weakens connections from the currently active key components so the current association is emphasised and old ones de-emphasised; keys and values are ReLU projections and the matrix is reset to zeros for each sequence [S14].

### 5. Gates: initial values, saturation, and what they read

A gate whose bias is left at the default sits at 0.5; for a gate that must stay open the recommendation is a bias of 1 or 2, and a badly initialised gate can make one "erroneously conclude" the model cannot learn long dependencies [S52]. Chrono initialisation refines this: for dependencies up to `T_max`, `b_f ~ log(U([1, T_max - 1]))` and `b_i = -b_f`, used as initial values only; it converged 7 times faster than a bias of 1 on the adding task at T = 750 [S36]. A forget-gate-only LSTM with chrono-initialised biases outperforms the standard LSTM on several benchmarks [S9].

Why a collapsed gate does not reopen: the gradient of a sigmoid gate is `f(1-f)`, which decays rapidly as f approaches 0 or 1, so a saturated gate "is unlikely to change" [S43]. The refine gate is the published repair: `r = σ(P_r(x, h))`, `g = r·(1-(1-f)²) + (1-r)·f²`, used in place of f, combined with a uniform initialisation of gate biases that also allows negative biases [S43].

What the gate reads matters. Chrono initialisation assumes inputs and hidden layers are centred, so that the gate takes values centred around `σ(b_g)` [S36]. In a binarised LSTM the measured gate pre-activations became all positive and the gates lost the ability to control information flow; batch-normalising each quantised matrix-vector product restored it [S19]. In the DNC, layer normalisation is applied to the controller output after its weighting and before it is split into gates, vectors and keys [S15].

Learned updates initialised at known-good constants: Zhang & Zhou replace Ba's scalars by learned matrices, `A_t = W_A ∘ A_{t-1} + W_h ∘ (h_t ⊗ h_t)`, initialised from N(0.9, 0.1) and N(0.5, 0.1) "to use the corresponding scalar parameters to initialize", with A_0 = 0, Adam at 1e-4, batch 128 and gradient clipping at (-5, 5) [S32].

### 6. Making the network use the read path

**Diagnosis in the literature.** In the DNC, runs that fail to converge are those where the memory unit has nearly no influence on the answer; the authors attribute this to the direct training signal over the controller→output bypass and to the memory output being noisy at the start [S15]. Taguchi & Tsuruoka state the same for NTM-style models: a recurrent controller lets the model solve the task directly with its own small memory [S45].

**Bypass Dropout.** Multiply the controller's contribution to the output by a Bernoulli mask, `y_t = W_h (h_t ∘ r_t) + W_μ μ_t`, during training only; all models used a dropout probability of 10%, and normalisation plus bypass dropout gave the best performance, with normalisation alone cutting time to convergence by more than half [S15].

**Structural removal of the shortcut.** Give the output a vector computed without the controller's recurrent memory, `h'_t = φ(W_xh x_t + W_rh r_{t-1} + b)`, while the recurrent state still drives the read and write heads [S45].

**Auxiliary losses.** Metalearned Neural Memory adds a meta objective that is the mean squared error between what the memory returns for a written key after the update and the value written there, and trains the controller on the plain sum of task and meta gradients [S46]. Trinh et al. train in two phases, first the auxiliary loss alone, then the sum of supervised and auxiliary losses, with one anchor per sequence in most experiments and a learning rate of 0.001; a truncated LSTM with the reconstruction loss reaches 65.9 on pixel CIFAR10 where a fully back-propagated LSTM reaches 58.8 [S27].

**Normalising the read.** The retrieved vector is layer-normalised without affine parameters before the read-out matrix in the Fast Weight Memory [S26], and passed through an RMSNorm before the output projection in DeltaNet [S53][S34].

**Zero-init versus open-init of an added path.** ReZero adds a branch as `x_{i+1} = x_i + α_i F(x_i)` with the scalar α initialised to zero, and notes that a small constant learning rate for these residual weights was important because the model otherwise diverges easily (0.1 in their fast-schedule experiment) [S21]. Fixup zero-initialises the last layer of each residual branch [S21]. SkipInit sets the branch scalar to 0 or a small constant and cannot train deep networks with it at 1 [S28]. Against this, a study of tanh-gated adapters reports that zero-initialised gates "change very slowly during training, requiring many thousands of iterations", that initialising them at tanh(1) is markedly better when the surrounding network is not pretrained, and that scalar gates diverged at learning rates above 10⁻³ where per-dimension gates tolerated 7·10⁻³ [S48].

**Curricula and schedules.** Linear start removes the softmax in the memory layers until validation loss stops improving and "seems to help avoid local minima" (task 16: 53.6% error without, 1.6% with) [S41]. Memory networks also showed large run-to-run variance, handled by repeating each training 10 times [S41]. Zaremba & Sutskever found LSTMs hard to train on execution and memorisation tasks without a curriculum, and their curriculum beat both no curriculum and the naive one [S31].

**Memory initialisation.** For NTMs, initialising memory contents to a small constant (10⁻⁶) converged about 3.5 times faster on Copy than a learned initialisation, and random initialisation failed in the allotted time [S40].

### 7. Quantised, binary and ternary networks

Naively binarised LSTMs fail on sequential tasks; the proposed cure is batch normalisation of every product between a binarised/ternarised weight matrix and its input, with full-precision weights kept for the parameter update [S19]. Hou et al. explain it: quantisation increases the weight norms that bound the back-propagated gradient, making exploding gradients more severe, and weight, layer or batch normalisation make the gradient invariant to weight scaling; a scaling factor below one on the quantised weights has the same effect [S30].

For ternary weights at scale, BitNet b1.58 uses absmean ternarisation in the forward pass, an added normalisation for stability in quantised training, a two-stage schedule with a high initial learning rate followed by an abrupt drop, and weight decay of 0.1 in the first stage and zero in the second [S56]. A separate study argues that binary networks benefit from random rather than zero bias initialisation and suggests saturating activations in recurrent models may benefit too [S23].

On delta-rule memories specifically, the only quantisation results found are post-training studies of large models: 4-bit weights and activations on all Gated DeltaNet projections, gates included, matched the 16-bit baseline [S49], while quantising the recurrent state itself degrades accuracy because errors propagate through successive updates [S50].

## Findings & consensus

- Delta rule over Hebbian sum for any task where keys recur with new values; the evidence is direct and repeated [S3][S38][S1].
- Normalise keys and queries (L2 preferred to L1/sum in the later work) and normalise the read before it is projected [S1][S53][S26].
- Uniform decay and targeted overwrite are complementary, not substitutes: decay alone forgets too fast, the delta rule alone cannot clear [S8].
- Gates need a deliberate initial value and normalised inputs; a saturated gate has almost no gradient [S52][S36][S43][S19].
- A memory that the output can bypass will be bypassed unless the bypass is weakened or the memory is trained by its own loss [S15][S45][S46].
- Quantised recurrent networks need normalisation after quantised products [S19][S30].

## Ranked mechanisms for this problem

The ranking is my judgement of fit to the six measurements of the diagnosis; each recipe is the one printed in the cited source. [M]

1. **Replace the gated Hebbian write by a delta rule on unit-norm keys.** Recipe: `k = φ(W_K x)/‖φ(W_K x)‖_2`, `q = φ(W_Q x)/‖φ(W_Q x)‖_2` with separate projections, `v_old = S_{t-1} k`, `S_t = S_{t-1} + β (v - v_old) k^T`, `β = σ(W_β x)` from a bias-free projection, `S_0 = 0`; keep `‖k‖² ≤ 2` [S1][S3][S53][S34]. Support: the sum rule fails on re-assigned keys where this rule succeeds [S3]. This addresses measurement (4) directly. [M]
2. **Take the write gate out of the failure loop before making it learnable.** Recipe: start with fixed coefficients, as in the original (η = 0.5, λ = 0.95) [S10] or β fixed at one [S34], then make them learnable initialised at those values [S32]; feed any gate a normalised vector [S15][S19]; give gate parameters no special learning-rate boost, since scalar gates diverged at high learning rates [S48][S21]; if a gate still saturates, use the refine gate `g = r(1-(1-f)²) + (1-r)f²` [S43]. This addresses measurement (1): a gate reading an un-normalised state at ten times the base rate is the configuration the sources warn against. [M]
3. **Initialise the decay gate for the horizon.** Recipe: `b ~ log(U([1, T_max - 1]))` on the retain gate with `T_max` of the order of the sequence length [S36], or a fixed bias of 1 to 2 [S52]; treat α → 1 as the default because decay hurts retention [S8]. [M]
4. **Decorrelate the keys at the source.** Recipe options from the sources: separate key and query projections with L2 normalisation [S1]; a tensor product of two keys [S26]; ReLU keys [S14]; a feature map that raises key dimension [S3]; a sharpening function that drives unrelated keys to uncorrelated rather than anti-correlated [S55]. Crosstalk is `Σ (k_i^T k_j) v_i`, so the measured key collinearity bounds what any read can return [S54]. Writing one association per item at full strength rather than a diluted average is what the reference systems do (write strength 0.5 or a learned β per token) [S10][S3]. [M]
5. **Normalise the read and open its path.** Recipe: `r = LN_no_affine(S q)` then `W_o` [S26], or RMSNorm then Linear [S53]. Because the rest of the network is trained from scratch, prefer a small non-zero initialisation of the read-out or an open gate over a strict zero start, which was found to move very slowly in that setting [S48]; if a scalar gate is kept, give it a small constant learning rate [S21]. [M]
6. **Keep the auxiliary loss and weaken the bypass.** Recipe: add the read-back loss to the task loss with unit weight [S46]; optionally train the auxiliary loss alone first, then the sum [S27]; apply Bernoulli dropout with probability 10% to the hidden-state path into the answer head at ask frames [S15], or compute the answer from the read plus a non-recurrent projection of the current input [S45]. This targets measurement (5), where the read carries colour but the head ignores it. [M]
7. **Quantisation hygiene.** Recipe: normalise after each ternary matrix-vector product, especially those feeding gates [S19][S30]; expect to need a higher initial learning rate than full precision [S56]. [M]
8. **Curriculum and restarts.** Recipe: start with a simpler regime and widen it, as the combined curriculum does [S31]; use several seeds, since memory models show large run-to-run variance [S41][S15]. [M]

## Open questions / contradictions

- Zero-init versus open-init of the read path is contested: ReZero, Fixup and SkipInit report zero or near-zero starts as what makes deep residual stacks trainable [S21][S28], while the adapter study finds zero-started gates too slow when the surrounding network is not pretrained [S48]. None of these experiments involves a memory read path. [M]
- Decay is described both as necessary for filtering and as harmful for retention in the same paper [S8]; the right initial α for a 64-frame recall task is not given by any source. [M]
- Ba's layer-normalised fast weights are reported as working by their authors [S10] and as failing to converge by three later groups on longer or harder tasks [S32][S29][S26].

> **Note (not from a source).** What the sources do not cover:
> - No fetched source trains a fast-weight or delta-rule memory under ternary or binary quantisation-aware training, or at the 0.3-3M-parameter scale with hidden size 128-256; the quantisation results on delta-rule models are post-training and on multi-billion-parameter models.
> - No source gives a learning-rate multiplier for write or decay gates, nor reports a write gate collapsing to zero and the cure; the gate-saturation literature fetched here concerns LSTM forget gates.
> - No source specifies a weight for an auxiliary retrieval loss other than a plain sum, nor a schedule for annealing it.
> - No source reports bypass dropout, linear start or a delay curriculum applied to a fast-weight memory; these come from DNC, memory-network and LSTM work.
> - The modern Hopfield capacity and pattern-separation theorems were not in the passages retrieved from that paper, so no capacity bound as a function of key similarity is given here beyond the crosstalk formula and the d-orthogonal-keys limit.
> - The exact decay formula of the Gated DeltaNet kernel and the initial value of β are not stated in the fetched files; the Gers et al. (2000) forget-gate paper and the DNC curriculum were not fetched.

## Sources

- [S1] Parallelizing Linear Transformers with the Delta Rule over Sequence Length — https://arxiv.org/html/2406.06484v6
- [S2] OSDN — https://arxiv.org/pdf/2605.13473
- [S3] Linear Transformers Are Secretly Fast Weight Programmers — https://arxiv.org/html/2102.11174v3
- [S4] Fast weight programming and linear transformers — https://arxiv.org/html/2508.08435v5
- [S5] Gated DeltaNet-2 — https://arxiv.org/html/2605.22791v1
- [S8] Gated Delta Networks — https://arxiv.org/html/2412.06464v3
- [S9] The unreasonable effectiveness of the forget gate — https://arxiv.org/html/1804.04849v3
- [S10] Using Fast Weights to Attend to the Recent Past — https://arxiv.org/pdf/1610.06258
- [S11] Backpropamine — https://arxiv.org/pdf/2002.10585
- [S14] H-Mem — https://proceedings.neurips.cc/paper/2020/file/f6876a9f998f6472cc26708e27444456-Paper.pdf
- [S15] Robust and Scalable Differentiable Neural Computer — https://arxiv.org/html/1807.02658
- [S17] Differentiable plasticity — https://arxiv.org/html/1804.02464v3
- [S19] Learning Recurrent Binary/Ternary Weights — https://arxiv.org/pdf/1809.11086
- [S21] ReZero — https://arxiv.org/pdf/2003.04887
- [S23] Random Bias Initialization Improves Quantized Training — https://arxiv.org/pdf/1909.13446
- [S24] backpropamine mylstm.py — https://github.com/uber-research/backpropamine/blob/master/awd-lstm-lm/mylstm.py
- [S25] Neural Turing Machines — https://arxiv.org/pdf/1410.5401
- [S26] Learning Associative Inference Using Fast Weight Memory — https://arxiv.org/pdf/2011.07831
- [S27] Learning Longer-term Dependencies in RNNs with Auxiliary Losses — https://arxiv.org/pdf/1803.00144
- [S28] Batch Normalization Biases Residual Blocks Towards the Identity Function — https://arxiv.org/pdf/2002.10444v3
- [S29] Fast Weight Long Short-Term Memory — https://arxiv.org/pdf/1804.06511
- [S30] Normalization Helps Training of Quantized LSTM — https://home.cse.ust.hk/~jamesk/papers/nips19b.pdf
- [S31] Learning to Execute — https://arxiv.org/pdf/1410.4615
- [S32] Learning to update Auto-associative Memory in RNNs — https://arxiv.org/pdf/1709.06493
- [S34] flash-linear-attention delta_net.py — https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py
- [S36] Can recurrent neural networks warp time? — https://arxiv.org/pdf/1804.11188
- [S37] differentiable-plasticity simplest.py — https://github.com/uber-research/differentiable-plasticity/blob/master/simple/simplest.py
- [S38] Going Beyond Linear Transformers with Recurrent Fast Weight Programmers — https://arxiv.org/html/2106.06295v2
- [S40] Implementing Neural Turing Machines — https://arxiv.org/html/1807.08518v3
- [S41] End-To-End Memory Networks — https://arxiv.org/pdf/1503.08895
- [S42] flash-linear-attention gated_deltanet.py — https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
- [S43] Improving the Gating Mechanism of Recurrent Neural Networks — https://arxiv.org/pdf/1910.09890
- [S44] Tracking the World State with Recurrent Entity Networks — https://arxiv.org/html/1612.03969v3
- [S45] Partially Non-Recurrent Controllers for Memory-Augmented Neural Networks — https://arxiv.org/pdf/1812.11485
- [S46] Metalearned Neural Memory — https://arxiv.org/pdf/1907.09720
- [S48] Scalable and Accurate Self-supervised Multimodal Representation Learning — https://arxiv.org/pdf/2304.02080v1
- [S49] Why Gated DeltaNet Survives 4-Bit Quantization — https://arxiv.org/pdf/2609.04098
- [S50] STEPQuant — https://arxiv.org/html/2609.38169
- [S52] An Empirical Exploration of Recurrent Network Architectures — https://proceedings.mlr.press/v37/jozefowicz15.pdf
- [S53] DeltaNet Explained (Part III) — https://sustcsonglin.github.io/blog/2024/deltanet-3/
- [S54] DeltaNet Explained (Part I) — https://sustcsonglin.github.io/blog/2024/deltanet-1/
- [S55] Robust High-dimensional Memory-augmented Neural Networks — https://arxiv.org/pdf/2010.01939
- [S56] BitNet b1.58 2B4T Technical Report — https://arxiv.org/html/2504.12285v2
