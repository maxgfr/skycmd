# 27 s6-nano: one small file for the games and for free-text questions

*Trained in November 2026 and published on 14 November 2026, at full depth. It passed a gate fixed before training; its larger sibling s6-micro, trained the same way, failed its own and is [reported below](#s6-micro-the-same-recipe-four-times-larger-not-published).*

Every earlier small model had to choose. s3-pico plays the four games at the oracle's level but scores 0.382 on `typed_decisions_bench`, below the 0.398 of always giving the most frequent answer ([chapter 21](./21-s3-recursive-tiny)). s4-micro was meant to answer any question and fell lower ([chapter 22](./22-s4-generic-decisions)). s5-lite answers well, but it is a 20.7 MB model built on a pretrained encoder ([chapter 23](./23-s5-lite)).

s6-nano asks whether one ternary model trained from scratch can do both: play the games, and answer typed questions written in free text. It has 1,486,225 parameters in a 715,136-byte file, and the answer is yes, above the majority class and well below s5-lite.

## The question it answers

The same as every skycmd decision model: a state, a typed question (`choice`, `score` or `noul`), and its options, each with a one-line definition. One forward pass returns a calibrated probability per option.

What is new is the range of states:

- **game states**, such as `vy 11 dx 35 up -8 dn 20 nxt 4 alt 34` for Flappy Drone, which are a few tokens long;
- **free text and JSON**: a support conversation, an agent trace, a document. The states of `typed_decisions_bench` are long JSON objects.

## The model

`configs/s6-nano.json` is the s3 architecture at language scale, as s4-micro was ([`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §15):

| | s3-pico | s6-nano |
|---|---:|---:|
| vocabulary | 1,024 | 2,048 |
| hidden / FFN | 96 / 192 | 144 / 288 |
| heads | 1 | 2 |
| state stack | 1 block × 4 loops | 2 blocks × 2 loops |
| exits (block passes) | 1, 2, 3, 4 | 1, 2, 3, 4 |
| state / head tokens | 64 / 96 | 512 / 160 |
| parameters | 445,345 | 1,486,225 |

**Why the windows are wide.** With 1,024 tokens and windows of 64 / 96, 96.8 % of the decision sequences were cut ([chapter 12](./12-size-ladder)). With 256 / 128, s6-nano's tokenizer would have cut the state of 71.6 % of the bench questions; at 512 / 160, 1.8 % ([amendment 2](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md), measured before training). Measured again on the trained model: 1.4 % of the bench states.

**Why 1,486,225 and not 1,569,457.** The pre-registration counted the probe rows of Parallax attention. The ablation did not adopt Parallax, so the trained model does not have them.

## The recipe: chosen by an ablation, not by hand

s6 came with eight additions in the code: Muon v2, Parallax attention, exclusive self-attention (XSA), SODA weight decay, exit self-distillation, a weight EMA, the StableQAT gradient and a pretraining curriculum. Before any s6 size was trained, a pre-registered ablation on the pico shape tested each of them over 3 seeds ([`docs/s6-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-prereg.md)). An addition was adopted only if it brought a validation gain of 0.01 (0.005 for the training options) without losing a game.

None was adopted (`runs/recipe/recipe.json`):

| addition | gain on validation | needed | also |
|---|---:|---:|---|
| Muon v2 | 0.0051 | 0.01 | lost Flappy score |
| Parallax | 0.0054 | 0.01 | |
| XSA | 0.0081 | 0.01 | |
| SODA, exit distillation, EMA, StableQAT, curriculum | all negative | 0.005 | four of them lost a game |

So s6-nano trains with AdamW at a learning-rate scale of 2.0, and the additions stay in the code, off. A per-size rate pilot then tried ×0.5, ×1 and ×2 around the pico rate moved to width 144 (× √(96/144)); the rule keeps ×1 unless another scale beats it by 0.01, and it kept ×1.

## The data and the teacher

- **Pretraining text.** `data/text-open` (the non-web parts of the Dolma 3 and TxT360 corpora, [Sources](/sources/)) then `data/text`, plus 10 % of the numeric curriculum, in 256-token sequences.
- **Decisions.** The Hugging Face decision sets of [chapter 05](./05-hugging-face-data), the four game oracles and the numeric curriculum, mixed as `decisions=0.25,distill=0.30,games=0.35,numeric=0.10`.
- **The teacher.** GLM-5.3-flash, an open-weight model under MIT ([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)), called through the z.ai API, so the laptop does not run it. It gave one probability per option to every training question of the decision sets: 747,077 of the 747,198 selected. A label is kept only when its argmax is the gold answer, and the target becomes `0.5 · gold + 0.5 · teacher` (the `distill` share above).
- **Never sent to the teacher:** `typed_decisions_bench`, NOTAM, Banking77, the records GLM had generated, and every validation and test split.

How the labelling survives quotas, network cuts and the provider's moderation is in [`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §15 and amendment 3 of the pre-registration.

## The run

One run, in a throttled mode (a duty cycle and cooling pauses), so the budgets count working minutes only. Steps actually done (`BENCH.md`):

| phase | budget | steps |
|---|---:|---:|
| MLM pretraining | 90 min | 53,360: 0.87B tokens, 588 per parameter, 8 epochs, MLM loss 2.356 |
| decision training, QAT after 10 % in float, loop dropout 0.25 | 90 min | 19,893 batches of 128 |
| DAgger rounds 1, 2, 3 (the model plays, the oracle labels) | 15 min each | 4,645, 4,660, 4,644 |

The decision training was stopped once by the memory cap, at step 16,020, and resumed from its checkpoint at step 15,573: [amendment 8](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md) explains the defect and the fix, which changes no value.

```bash
SIZE=nano scripts/train-s6.sh lr train dagger exit eval
```

## The gate, and how to read it

Fixed before the run (§6 of the pre-registration), read on the ternary file of the last DAgger round at full depth:

1. `typed_decisions_bench` above the majority class, **> 0.398**;
2. no game whose arena score falls more than 5 % below s3-pico;
3. **ECE ≤ 0.08** over all the test sources together.

| model | params | bench | MASSIVE en | MASSIVE fr | When2Call | NOTAM subject | NOTAM multi | all | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **s6-nano** | 1,486,225 | **0.480** | 0.425 | 0.239 | 0.404 | 0.055 | 0.493 | 0.360 | **0.0479** |
| s3-pico | 445,345 | 0.382 | 0.187 | 0.117 | 0.368 | 0.163 | 0.319 | 0.268 | 0.0629 |
| s4-micro | 5,587,201 | 0.352 | 0.335 | 0.240 | 0.399 | 0.065 | 0.298 | 0.285 | 0.0666 |
| s5-lite | 17,060,097 | 0.696 | 0.726 | 0.444 | 0.376 | 0.177 | 0.554 | 0.500 | 0.0328 |

The games, 30 episodes each, on the same seeds as s3-pico: 200 of 200 Flappy pipes with no crash, every survivor rescued, the oracle's 87 % Dodge completion with 23 reflex vetoes, every Doom-like level cleared with the oracle's kill rate (0.772). **The gate passes.**

What the result does and does not say:

- **It clears a floor, not a ceiling.** 0.480 is above the majority class, where s3-pico and s4-micro stayed below it on the same records. It is far from s5-lite (0.696) and oscar-1-17m (0.6775).
- **No single cause is claimed.** From s4-micro to s6-nano, several things changed at once: the pretraining (0.87B tokens against 30 minutes on 150 MB), the GLM teacher, the tokenizer and the windows. No ablation separates them.
- **Early exits were tried and rejected.** Conformal risk control (α = 0.02, [`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §9) cut the depth to 1.01 to 1.83 of the 4 passes per game, but Rescue fell to 93 % and the Dodge vetoes rose from 23 to 1,016. The rule allows 2 %, so the thresholds were dropped. The published file stores a threshold of 1 for each question type, so every answer uses the 4 passes ([amendment 13](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)).

## Limits

- **NOTAM and When2Call.** 0.055 on the NOTAM subject question, below s3-pico (0.163); 0.404 on When2Call. NOTAM is never trained on.
- **English MASSIVE is not zero-shot.** Found by the project after the evaluation ([amendment 14](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)): the training data hold 3,113 questions of a source derived from MASSIVE, asked with 3 options instead of 8, and 206 of the 2,000 English test utterances are also states of training questions. The seen ones are not answered better (0.393 on the 206, 0.429 on the 1,794 others), and without them the ECE is 0.0486, so the gate does not change. French MASSIVE stays zero-shot.
- **Banking77 is in distribution.** It is trained on; its 0.631 is reported apart and counts in no gate.
- **The teacher's own training data are not published.** It never saw a test record from us, but it may have seen the public benchmarks.
- **One run, one seed.** No interval comes with these numbers.

## In the browser

The WASM module answers like the native engine: on the first 200 bench questions, the largest probability gap is 9.5e-7 and the answer is the same on all 200 (`parity-node.mjs`). Measured under Node on the build machine, SIMD128 build:

- a long JSON state of the bench: a median of 14.6 to 15.2 ms per decision, in two runs;
- a Flappy frame (two questions and the physics step, full depth): 436 µs.

The site loads the relaxed-SIMD build where the browser supports it ([chapter 14](./14-wasm-inference-simd128)). Three alternating runs of each build in one session, with a GPU training running: 15.8 ms (SIMD128) against 12.5 ms (relaxed) per bench question, 445 µs against 358 µs per Flappy frame, and the same answers on the 200 questions (largest gap 9.5e-7 and 8.9e-7).

## s6-micro: the same recipe, four times larger, not published

s6-micro has the s4-micro shape (5,587,201 parameters) and longer budgets: 180 minutes of pretraining and 180 of decision training, then three DAgger rounds of 20 minutes. It reaches 0.547 on the bench, above its gate (0.45) and its announced goal (0.50), and plays the games at s3-pico's score. It fails the calibration guard: ECE 0.0821 at full depth and 0.0804 at its conformal thresholds, against 0.08. It is not published.

Where the ECE comes from was analysed after the gate, as exploration ([amendment 12](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)). In distribution the model is calibrated (ECE 0.037 on validation). The largest part of the pooled error is one zero-shot NOTAM statement: 412 of its 11,000 items, false for 99.5 % of them, which s6-micro answers "true" almost every time at a confidence near 0.5, while s6-nano answers "false". Those 412 items alone weigh up to about 0.02 in the pooled ECE. The gate was not changed for that. A second version with twice the decision training is pre-registered in the same amendment, with its rules written before it runs.

## Try it

- **On [Live](/live/#s6-nano-the-questions-it-was-trained-for):** the game and numeric questions it was trained for, as editable typed questions, with the time of each decision measured in your browser. s6-nano can also be chosen in every other demo of that page.
- **On [Play](/play/):** choose s6-nano in the model list and watch it fly, rescue, dodge and fight.
- **In your own page**, with the module of the [Install](/install/) page:

```js
import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";

const model = await SkyCmd.load("https://maxgfr.github.io/skycmd/models/s6-nano.gguf");
const [pClimb, pDrift] = model.decideRaw(
  "choice",
  "Pick the control input for this frame so the drone clears the pipes.",
  ["climb: gain altitude now", "drift: let gravity pull the drone down"],
  "vy 11 dx 35 up -8 dn 20 nxt 4 alt 34",
);
```

The numbers of this chapter are in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md) (section "skycmd-s6 (s6-nano, s6-micro)") and on the [results page](/results/#s6-nano-passed-published-at-full-depth).
