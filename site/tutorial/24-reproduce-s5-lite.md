# 24 Reproducing s5-lite step by step

[Chapter 23](/tutorial/23-s5-lite) explains what s5-lite is and what its bets showed. This chapter runs the whole pipeline again, command by command, in the order of the night it was built. Everything is Rust:

- `sky-data` builds the data;
- `sky-train` (mlx-rs) scores, trains and exports;
- `sky-eval` and `sky-s5-report` evaluate and judge;
- `cargo xtask s5` chains the stages.

Timings are those of an Apple M5 with 16 GB.

## 0. The store and the runner

Every artefact lives in a **store**: the main checkout of the repository, given by `SKYCMD_STORE`. It holds `data/s5/`, `models-src/`, `models/s5-*`, `runs/s5-*`, all gitignored. The runner refuses any other path, so work done in a git worktree survives its removal.

```sh
export SKYCMD_STORE=$PWD            # the main checkout, not a worktree
cargo xtask s5 list                 # every stage, [GPU] when it trains or scores
```

What the runner does for you:

- **Builds** the binaries it needs in release mode, with the full Xcode toolchain for mlx (`DEVELOPER_DIR`).
- **Runs** each step in the store and tees the output to `runs/s5-logs/<stage>-<time>.log`.
- **Resumes.** A finished step leaves `runs/s5-state/<stage>/<step>.done`, and a rerun skips steps whose command line is unchanged (`--force` reruns them).
- **Locks the GPU.** GPU stages hold an exclusive lock on `runs/s5-gpu.lock`: two GPU stages queue instead of fighting.
- **Runs independent jobs together.** `--jobs N` runs up to N independent runs at once under one lock. A single 17M run already saturates an M5 GPU (about 42k tokens/s), so three at once are not faster in total. They help only when CPU work interleaves.
- **Caps MLX memory.** Training steps set `SKY_MLX_CACHE_GB=1`. Without that cap, MLX's allocator cache grew to 12 GB per run, and three runs swapped the machine to a crawl.

## 1. Downloads (Phase 0)

```sh
cargo xtask s5 fetch-models   # Ettin-17M/32M, ettin-reranker-17m, mmBERT-small, the teachers, oscar-1-17m
cargo xtask s5 fetch-evals    # btzsc, SIB-200, HWU64, the 2609.33209 battery, typed-decisions
```

The downloads are reproducible:

- **Pinned.** Each repository is pinned to a commit.
- **Verified.** Every file is checked against the sha256 (LFS) or the git blob sha1 that the host publishes.
- **Recorded.** The commit and the hashes are written to `.skycmd-fetch.json` next to the files.
- **Authenticated where needed.** `HF_TOKEN` is used only for gated repositories, and it is sent to the Hub only.

## 2. Data (Phase 3)

```sh
cargo xtask s5 data           # typed-decisions train split, sky-data s5-build, sky-data s5-battery
```

`sky-data s5-build` writes `data/s5/train/{corpus,gen,pairs,coherence}.jsonl` and `data/s5/eval/*.jsonl`, with a `MANIFEST.json` that records every hash. Its steps:

- **Splits.** Val and cal take 5 % each, assigned by a hash of the state, so all the questions about one state fall in the same split.
- **Generators**, written in code:
  - rules with 50 % violations, and minimal pairs;
  - tone pairs;
  - buried facts, with a uniform target when the evidence is missing;
  - "none of the above" on a quarter of the choice questions;
  - paraphrase and distractor pairs.
- **Coherence families.** For each typed-decisions question: the choice, "the correct answer is: x" and "… is not: x". The "x or y" questions exist only in the test families.
- **Decontamination.** A training record is dropped when it covers half of an evaluation or demo state in 13-grams, or when it is a MinHash near duplicate (Jaccard ≥ 0.8).
- **Class gate.** No class above 60 %, with a deterministic rebalance.

`sky-data s5-battery` converts the original 160-item coherence battery of arXiv 2609.33209 for the comparison with Jev. It is used for evaluation only, never redistributed.

## 3. Encoders and teacher (Phase 1)

```sh
cargo xtask s5 import         # HF checkpoints -> sky-train checkpoints, with an MLM probe in the log
cargo xtask s5 teachers       # laya-typed-decisions soft labels of the English corpus and pairs
cargo xtask s5 mix            # data/s5/mixed: gold + calibrated teacher, train split only
```

Scoring is the long step: 155k corpus records, two option orders each, about 1 h 50 on the M5. The rows are sorted by length, so the last ones are the slowest.

The mix proceeds as follows:

- **Calibration.** The teacher gets a temperature and a per-option bias for each group (source × type × number of options).
- **Filter.** A soft label is kept only where the teacher agrees with the gold argmax: a single teacher never "agrees with itself".
- **Target.** It mixes 0.5 gold and 0.5 teacher, on the train split only. Val and cal keep the gold, so model selection and calibration are judged against the gold.

## 4. Pre-register, then train (Phase 4)

**Commit `docs/s5-prereg.md` before the first test pass.** It must list the sha256 of every frozen evaluation file. `sky-eval` refuses the test split of a frozen set if the file is not committed, has local changes or misses the set's hash. Each pass is appended to `runs/s5-test-log.jsonl`. When a rule must change, add a dated amendment that says what had been seen; the night of s5 needed three.

```sh
cargo xtask s5 explore --jobs 3   # A' and B-i × lr {5e-5, 1e-4, 2e-4} × init {encoder, reranker}, 600 steps
cargo xtask s5 explore-b-ii       # B-ii at the best B-i configuration
cargo xtask s5 final --jobs 3     # A' and B* × 3 seeds, A × 1 seed
```

Every selection uses validation files only (`runs/s5-<run>/val.json`, lowest NLL over every val record). It is frozen in `runs/s5-select/*.json` the first time a stage is built:

| file | what it freezes |
|---|---|
| `explore.json` | the best configuration of each variant |
| `final.json` | the choice of B-i or B-ii |
| `steps.json` | the step count of the finalists |
| `lambda.json` | λ\* of bet 2 |

The step count is not chosen by hand. It is fixed mechanically from the exploration throughput alone.

The exploration took about 30 minutes and the training of the 7 finalists about 1 hour.

A lesson worth repeating: both variants picked the highest learning rate on 600 steps. Two of the three A′ seeds then went unstable at 2,000 steps. Short runs favour aggressive rates.

Under the runner, every training run is one `sky-train s5-train` call:

```sh
sky-train s5-train --init models/s5-base/ettin-reranker-17m --records data/s5/mixed \
  --out runs/s5-f-b-s0 --variant b-i --steps 2000 --lr 2e-4 --seed 0 --kl 0.5 --coh 0 \
  --eval-every 1000 --bf16 --resume
```

`--variant` takes `a`, `a-prime`, `b-i` or `b-ii`. `--coh λ` adds the bet-2 losses (L_neg + L_part + L_xf). `--resume` continues from `<out>/last`.

## 5. Evaluate (CPU, while the GPU trains)

```sh
cargo xtask s5 eval --runs f-b-s0     # one run; without --runs, every finished finalist and bet-2 run
```

For one run, `eval` does the following:

1. **Exports** f32 and q8_0 GGUF files:
   - `sky-train s5-export` for iso;
   - `sky-train export` then `sky-convert` for the Laya format.
2. **Calibrates.** It fits temperatures per question type and option count on the val split and writes them into the GGUF (`sky-s5-report calibrate`).
3. **Reads the cal split** with the calibrated model, for the escalation thresholds.
4. **Evaluates** every frozen test set in f32 and the fitted bench in q8_0, plus the 2609.33209 battery and the demo cases. Each pass goes through `sky-eval --split test --prereg docs/s5-prereg.md`.

Iso runs take about 25 minutes, Laya runs 30 to 40, on one CPU core each. `eval-q8 --runs <run>` adds the other q8_0 sets for the shipped model.

## 6. Judge

```sh
cargo xtask s5 report --force         # every analysis into runs/s5-report/, then the site summary
```

`sky-s5-report` reads the `--dump-probs` files:

- **`pari1`**: bet 1, non-inferiority on bench accuracy and ECE, two-level bootstrap over items and seeds, Holm for the secondary comparisons;
- **`pari2`**: bet 2;
- **`coherence`**: the 2609.33209 identities, `--set battery_2609_33209` for the original battery;
- **`robust`**: paraphrases, distractors, label reassignment, PolicyBench, coverage;
- **`selective`**: Learn-then-Test escalation thresholds;
- **`gates`**: the s5-lite gates;
- **`summary`**: one JSON for the results page (`site/public/results/s5.json`).

The summary also applies the pre-registered rule for the shipped model.

## 7. Bet 2, the sensitivity analysis and the long run

```sh
cargo xtask s5 lambda --jobs 2        # λ ∈ {0.1, 0.3} on the support chosen by bet 1
cargo xtask s5 lambda-final --jobs 2  # λ* on two more seeds
cargo xtask s5 sens --jobs 3          # amendment 2: A' at the next lower learning rate
cargo xtask s5 long                   # amendment 3: B* for 8,000 steps, the /live model
cargo xtask s5 oscar-ref              # oscar-1-17m on the reassignment set and the bench
```

## 8. Ship to the site

The site serves the calibrated q8_0 file byte for byte as it was evaluated. Its sha256 matches the test log.

```sh
cp $SKYCMD_STORE/models/s5-long-b-s0/q8_0.gguf site/public/models/s5-lite.gguf
# add the manifest entry: name, arch skycmd-iso, weight_format q8_0, task generic, params, bytes
```

/live picks the `skycmd-iso` entry. /results reads `results/s5.json`. To check both pages in a headless Chrome before merging, build the site and open them:

```sh
cd site && npm run docs:build && npx vitepress preview
```

Pushing to `main` deploys GitHub Pages; [chapter 25](/tutorial/25-hugging-face) publishes the same files on Hugging Face.
