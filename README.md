# skycmd

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24418316.svg)](https://doi.org/10.5281/zenodo.24418316) [![Site](https://img.shields.io/badge/site-maxgfr.github.io%2Fskycmd-blue)](https://maxgfr.github.io/skycmd/) [![Hugging Face collection](https://img.shields.io/badge/%F0%9F%A4%97%20collection-skycmd-yellow)](https://huggingface.co/collections/maxgfr/skycmd-6b001dd30a7cb5724995b603) [![Every demo](https://img.shields.io/badge/%F0%9F%A4%97%20Space-every%20demo-yellow)](https://huggingface.co/spaces/maxgfr/skycmd) [![License](https://img.shields.io/badge/code-Apache--2.0-green)](LICENSE)

skycmd trains small decision models and runs them in the browser. A decision model does not generate text: it reads a state (a game frame, a text, a request with a list of tools, a few seconds of audio) and typed questions, and returns a calibrated probability for every allowed option in one forward pass. The models were trained from scratch in Rust on one Apple M5 MacBook, and every demo runs them in WebAssembly from the files their numbers were measured on.

## The papers

[*Pre-registered gates for small models trained on one laptop: what passed, what failed, and what the readings show*](https://maxgfr.github.io/skycmd/paper/), Maxime Golfier, technical report, 21 November 2026 (not peer-reviewed).

- Read it on the site: [maxgfr.github.io/skycmd/paper](https://maxgfr.github.io/skycmd/paper/), or as a [PDF](https://maxgfr.github.io/skycmd/paper/skycmd-technical-report.pdf).
- Archived on Zenodo: [DOI 10.5281/zenodo.24418316](https://doi.org/10.5281/zenodo.24418316) (all versions; this version is [10.5281/zenodo.24418317](https://doi.org/10.5281/zenodo.24418317)), with its LaTeX source.
- The LaTeX source is in [`paper/`](paper/); `scripts/build-paper.sh` builds the PDF.

It reports every gated experiment of the project: from s5 on, each one had a pre-registration committed before the computation it governs, its test was read once, and a model that missed its gate was published with its file. Of the 20 published model files, 15 were judged by a gate fixed before training: 5 passed and 10 failed. The report also states what the project takes from earlier work and what it adds.

```bibtex
@misc{golfier2026skycmd,
  author    = {Golfier, Maxime},
  title     = {Pre-registered gates for small models trained on one laptop:
               what passed, what failed, and what the readings show},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24418316},
  url       = {https://maxgfr.github.io/skycmd/paper/},
  note      = {Technical report, not peer-reviewed}
}
```

[*Looped ternary encoders for typed decisions: a recipe from laptop to browser*](https://maxgfr.github.io/skycmd/paper/methods/), Maxime Golfier, technical report, 21 November 2026 (not peer-reviewed), the companion on method.

- Read it on the site: [maxgfr.github.io/skycmd/paper/methods](https://maxgfr.github.io/skycmd/paper/methods/), or as a [PDF](https://maxgfr.github.io/skycmd/paper/skycmd-methods.pdf).
- The LaTeX source is in [`paper/methods/`](paper/methods/); `scripts/build-paper.sh --methods` builds the PDF.

It describes how the models, their training, their calibration and the engine work, precisely enough to reimplement them: the ternary decision graph with its two streams, cache and looped exits, the training recipe of seven published files, exit thresholds chosen by playing the games, the task-specific systems, and the Rust engine that runs every file natively and in WebAssembly. It cites the first report for the protocol and the verdicts.

```bibtex
@misc{golfier2026skycmdmethods,
  author       = {Golfier, Maxime},
  title        = {Looped ternary encoders for typed decisions:
                  a recipe from laptop to browser},
  year         = {2026},
  publisher    = {Zenodo},
  howpublished = {\url{https://maxgfr.github.io/skycmd/paper/methods/}},
  note         = {Technical report, not peer-reviewed}
}
```

## The state of the project

The experiments are finished, as of 20 November 2026. Every line is closed or published: 20 model files (5 passed their pre-registered gate, 10 failed it and are published as results, 5 had no gate), 9 versions closed without a model, 16 data sets and 24 pre-registrations. The repository stays as the record: the models, the data sets, the pre-registrations, the result files and the technical report.

## Links

- **The site**: <https://maxgfr.github.io/skycmd/>, with [Use it for real](https://maxgfr.github.io/skycmd/use/) (which files fit a real project, and the code), [Models](https://maxgfr.github.io/skycmd/models/), [Results](https://maxgfr.github.io/skycmd/results/) (every gate and its numbers, failures included) and [What we bring](https://maxgfr.github.io/skycmd/novelty/) (every claim, its measurement, its evidence and the prior work it builds on).
- **Hugging Face**:
  - every demo in one Space: [maxgfr/skycmd](https://huggingface.co/spaces/maxgfr/skycmd);
  - the collection [skycmd](https://huggingface.co/collections/maxgfr/skycmd-6b001dd30a7cb5724995b603): the hub Space, the models by verdict, their Spaces and the 16 data sets;
  - every model, Space and data set: [huggingface.co/maxgfr](https://huggingface.co/maxgfr). Each model card opens with its verdict; each data set gives the licence of each source ([`DATA_LICENSES.md`](DATA_LICENSES.md)).
- **The papers**: the technical report ([site](https://maxgfr.github.io/skycmd/paper/), [PDF](https://maxgfr.github.io/skycmd/paper/skycmd-technical-report.pdf), [Zenodo DOI 10.5281/zenodo.24418316](https://doi.org/10.5281/zenodo.24418316)) and the methods paper ([site](https://maxgfr.github.io/skycmd/paper/methods/), [PDF](https://maxgfr.github.io/skycmd/paper/skycmd-methods.pdf)).
- **The raw archive**: the release [`archive-2026-11-20`](https://github.com/maxgfr/skycmd/releases/tag/archive-2026-11-20): raw traces of two gates, run logs and intermediate model files ([`docs/data-inventory.md`](docs/data-inventory.md), section 13).

## The 20 published models

Sizes are the files of [`site/public/models/manifest.json`](site/public/models/manifest.json). A gate is a bar written down before the model was trained and tested; a model that missed it is published as a result, not for use.

| model | what it does | size | verdict | Hugging Face |
|---|---|---:|---|---|
| [s6-ear2-nano](https://maxgfr.github.io/skycmd/results/#s6-ear2-nano-passed) | detects the end of a speaker's turn from the last 4 s of audio | 632 KB | passed | [model](https://huggingface.co/maxgfr/s6-ear2-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-ear2-nano) |
| [s6-route](https://maxgfr.github.io/skycmd/results/#s6-route-passed) | picks a tool from a list given at run time, or says that none fits | 3.7 MB, two files | passed, on fresh slates; system defined after v2's test was read | [model](https://huggingface.co/maxgfr/s6-route) · [demo](https://huggingface.co/spaces/maxgfr/s6-route) |
| [s6-nano](https://maxgfr.github.io/skycmd/results/#s6-nano-passed-published-at-full-depth) | plays four browser games and answers typed questions in free text | 715 KB | passed | [model](https://huggingface.co/maxgfr/s6-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-nano) |
| [s6-micro](https://maxgfr.github.io/skycmd/results/#s6-micro-v2-passed-on-a-second-look) | the same, four times larger | 2.4 MB | passed, on a second look | [model](https://huggingface.co/maxgfr/s6-micro) · [demo](https://huggingface.co/spaces/maxgfr/s6-micro) |
| [s6-secrets-pico](https://maxgfr.github.io/skycmd/results/#s6-secrets-pico-passed-published-at-full-depth) | says whether a short text exposes a secret, and which service issued it | 250 KB | passed on its synthetic bench; fails on real files | [model](https://huggingface.co/maxgfr/s6-secrets-pico) · [demo](https://huggingface.co/spaces/maxgfr/s6-secrets-pico) |
| [s6-micro-v1](https://maxgfr.github.io/skycmd/results/#s6-micro-v1-failed-on-calibration-only) | the first version of s6-micro | 2.4 MB | failed its gate (calibration) | [model](https://huggingface.co/maxgfr/s6-micro-v1) · [demo](https://huggingface.co/spaces/maxgfr/s6-micro-v1) |
| [s6-ear-nano](https://maxgfr.github.io/skycmd/results/#s6-ear-nano-failed) | answers typed questions about the last 10 s of audio | 899 KB | failed its gate | [model](https://huggingface.co/maxgfr/s6-ear-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-ear-nano) |
| [s6-ear3-nano](https://maxgfr.github.io/skycmd/results/#s6-ear3-nano-failed) | s6-ear2-nano trained again on real two-speaker conversations | 632 KB | failed its gate | [model](https://huggingface.co/maxgfr/s6-ear3-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-ear3-nano) |
| [s6-cast-pico](https://maxgfr.github.io/skycmd/results/#s6-cast-pico-failed) | forecasts the next 64 steps of a time series as nine deciles | 143 KB | failed its gate | [model](https://huggingface.co/maxgfr/s6-cast-pico) · [demo](https://huggingface.co/spaces/maxgfr/s6-cast-pico) |
| [s6-cast2-pico](https://maxgfr.github.io/skycmd/results/#s6-cast2-pico-failed) | the same, with the periods it detects as an extra input | 149 KB | failed its gate | [model](https://huggingface.co/maxgfr/s6-cast2-pico) · [demo](https://huggingface.co/spaces/maxgfr/s6-cast2-pico) |
| [s6-secrets-verify-pico](https://maxgfr.github.io/skycmd/results/#s6-secrets-verify-pico-failed) | judges whether a value found by a secret-scanning rule is a real credential | 250 KB | failed its gate | [model](https://huggingface.co/maxgfr/s6-secrets-verify-pico) · [demo](https://huggingface.co/spaces/maxgfr/s6-secrets-verify-pico) |
| [s6-route-nano](https://maxgfr.github.io/skycmd/results/#s6-route-nano-failed) | picks a tool from a run-time list, with each tool's BM25 score in its input | 2.4 MB | failed its gate | [model](https://huggingface.co/maxgfr/s6-route-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-route-nano) |
| [s6-route7-nano](https://maxgfr.github.io/skycmd/results/#s6-route7-nano-failed) | s6-route with its model part retrained, read on tools of a new family | 3.7 MB, two files | failed its gate | [model](https://huggingface.co/maxgfr/s6-route7-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-route7-nano) |
| [s6-sleep3-nano](https://maxgfr.github.io/skycmd/results/#s6-sleep3-nano-failed) | s6-nano with a fast-weight memory that carries a fact across frames | 752 KB | failed its gate | [model](https://huggingface.co/maxgfr/s6-sleep3-nano) · [demo](https://huggingface.co/spaces/maxgfr/s6-sleep3-nano) |
| [s4-micro](BENCH.md#skycmd-s4-s4-micro-generic-decisions-distilled-from-qwen35-experimental) | generic decisions with options in free text, distilled from Qwen3.5-4B | 2.4 MB | failed its gate (fixed before training, no pre-registration document) | [model](https://huggingface.co/maxgfr/s4-micro) · [demo](https://huggingface.co/spaces/maxgfr/s4-micro) |
| [s5-lite](hf/cards/s5-lite.md) | answers questions you write, each option defined in plain words | 20.7 MB | no gate (the pre-registered s5 bets failed; this longer run is outside them) | [model](https://huggingface.co/maxgfr/s5-lite) · [demo](https://huggingface.co/spaces/maxgfr/s5-lite) |
| [s3-pico](hf/cards/s3-pico.md) | plays four browser games from the game state | 251 KB | no gate | [model](https://huggingface.co/maxgfr/s3-pico) · [demo](https://huggingface.co/spaces/maxgfr/s3-pico) |
| [s1-pico](hf/cards/s1-pico.md) | plays three browser games; the same file runs in llama.cpp and wllama | 545 KB | no gate | [model](https://huggingface.co/maxgfr/s1-pico) · [demo](https://huggingface.co/spaces/maxgfr/s1-pico) |
| [s2-nano](hf/cards/s2-nano.md) | plays the three drone games | 702 KB | no gate | [model](https://huggingface.co/maxgfr/s2-nano) · [demo](https://huggingface.co/spaces/maxgfr/s2-nano) |
| [s2-nano-doom](hf/cards/s2-nano-doom.md) | s2-nano fine-tuned for the Doom-like game | 702 KB | no gate | [model](https://huggingface.co/maxgfr/s2-nano-doom) · [demo](https://huggingface.co/spaces/maxgfr/s2-nano-doom) |

Every number behind these verdicts is on [Results](https://maxgfr.github.io/skycmd/results/) and in [`BENCH.md`](BENCH.md); what each model is for and where to try it is on [Models](https://maxgfr.github.io/skycmd/models/).

### Which model for a real project

Two files can go into a project outside the games: **s6-ear2-nano**, the end of a speaker's turn for a voice agent, used with a fallback timer, and **s6-route**, a tool router for tool lists with distinctive names (on tools of a new family it does not beat BM25, and all-MiniLM-L6-v2 does better). **s6-secrets-verify-pico** failed its gate and runs only as an opt-in experiment inside [secretgate](https://github.com/maxgfr/secretgate). The other files are for the games, the demos and the playground, or are published as results. [Use it for real](https://maxgfr.github.io/skycmd/use/) gives the code and the limits of each.

## Try it

Nothing to install for the demos: [Play](https://maxgfr.github.io/skycmd/play/) four games, ask your own questions on [Live](https://maxgfr.github.io/skycmd/live/), pick a tool on [Router](https://maxgfr.github.io/skycmd/router/), forecast a series on [Forecast](https://maxgfr.github.io/skycmd/forecast/), test end-of-turn detection on [Listen](https://maxgfr.github.io/skycmd/ear/). The two failed models with a page of their own are on [Secrets verifier](https://maxgfr.github.io/skycmd/secrets-verify/) and [Memory](https://maxgfr.github.io/skycmd/memory/). To call the end-of-turn detector or the tool router from your own code, see [Use it for real](https://maxgfr.github.io/skycmd/use/). In the games the model only advises, and coded rules (geofence, ceiling, battery, collision reflex) keep a veto: a demo and a tutorial, not flight software.

To run a model yourself, a `.gguf` file and the `skycmd` command line are enough:

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve    # installs the `skycmd` binary
curl -LO https://maxgfr.github.io/skycmd/models/s3-pico.gguf
skycmd info s3-pico.gguf                                         # sizes, temperatures, exit thresholds
skycmd decide --model s3-pico.gguf --request req.json --pretty   # one request from a file
```

The request format, the browser module, Node, Deno, Bun, Python, the HTTP server and llama.cpp are on [Install](https://maxgfr.github.io/skycmd/install/). To train the models yourself, the [tutorial](https://maxgfr.github.io/skycmd/tutorial/) rebuilds everything step by step and gives every training command in one block.

## Reproduce a result

Each s6 model was trained by one script under its pre-registration; the small result files it read are under `runs/`, and its card opens with its verdict. The scripts read their data from `../skycmd-data/`, a folder next to the repository (or the environment variable each script names), and from `data/`; where every data set lives and how it is rebuilt is in [`docs/data-inventory.md`](docs/data-inventory.md).

| model | pre-registration | script | result files | card |
|---|---|---|---|---|
| s6-nano | [`docs/s6-gen-prereg.md`](docs/s6-gen-prereg.md) | `scripts/train-all.sh nano` (`scripts/train-s6.sh`) | [`runs/s6-nano/eval/`](runs/s6-nano/eval/results.md) | [`hf/cards/s6-nano.md`](hf/cards/s6-nano.md) |
| s6-micro-v1 | [`docs/s6-gen-prereg.md`](docs/s6-gen-prereg.md) | `scripts/train-all.sh micro` (`scripts/train-s6.sh`) | [`runs/s6-micro/eval/`](runs/s6-micro/eval/results.md) | [`hf/cards/s6-micro-v1.md`](hf/cards/s6-micro-v1.md) |
| s6-micro (v2) | [`docs/s6-gen-prereg.md`](docs/s6-gen-prereg.md), amendments 12, 16, 17 | `SIZE=micro scripts/train-s6.sh`, environment in `runs/s6-micro2/env.sh` | [`runs/s6-micro2/`](runs/s6-micro2/runs/s6-micro/eval/results.md) | [`hf/cards/s6-micro.md`](hf/cards/s6-micro.md) |
| s6-secrets-pico | [`docs/s6-secrets-prereg.md`](docs/s6-secrets-prereg.md) | `scripts/train-all.sh secrets` (`scripts/train-secrets.sh`) | [`runs/s6-secrets-pico/`](runs/s6-secrets-pico/report.md) | [`hf/cards/s6-secrets-pico.md`](hf/cards/s6-secrets-pico.md) |
| s6-secrets-verify-pico | [`docs/s6-secrets2-prereg.md`](docs/s6-secrets2-prereg.md) | `scripts/train-secrets2.sh` | [`runs/s6-secrets2/`](runs/s6-secrets2/), v3: [`runs/s6-secrets3/`](runs/s6-secrets3/) | [`hf/cards/s6-secrets-verify-pico.md`](hf/cards/s6-secrets-verify-pico.md) |
| s6-ear-nano | [`docs/s6-ear-prereg.md`](docs/s6-ear-prereg.md) | `scripts/train-all.sh ear` (`scripts/train-ear.sh`) | [`runs/s6-ear-nano/eval/`](runs/s6-ear-nano/eval/gate.json) | [`hf/cards/s6-ear-nano.md`](hf/cards/s6-ear-nano.md) |
| s6-ear2-nano | [`docs/s6-ear2-prereg.md`](docs/s6-ear2-prereg.md) | `scripts/train-turn.sh` | [`runs/s6-ear2-nano/`](runs/s6-ear2-nano/) | [`hf/cards/s6-ear2-nano.md`](hf/cards/s6-ear2-nano.md) |
| s6-ear3-nano | [`docs/s6-ear3-prereg.md`](docs/s6-ear3-prereg.md) | `scripts/train-turn3.sh` (otoSpeech is gated: its records are rebuilt, never downloaded from here) | [`runs/s6-ear3-nano/`](runs/s6-ear3-nano/GATE.md), with `runs/s6-ear3-pilot/` | [`hf/cards/s6-ear3-nano.md`](hf/cards/s6-ear3-nano.md) |
| s6-cast-pico | [`docs/s6-cast-prereg.md`](docs/s6-cast-prereg.md) | `scripts/train-all.sh cast` (`scripts/train-cast.sh`) | [`runs/s6-cast-pico/`](runs/s6-cast-pico/gift-eval/summary.json) | [`hf/cards/s6-cast-pico.md`](hf/cards/s6-cast-pico.md) |
| s6-cast2-pico | [`docs/s6-cast2-prereg.md`](docs/s6-cast2-prereg.md) | `scripts/train-cast2.sh` | [`runs/s6-cast2-pico/`](runs/s6-cast2-pico/GATE.md) | [`hf/cards/s6-cast2-pico.md`](hf/cards/s6-cast2-pico.md) |
| s6-route-nano | [`docs/s6-nano2-prereg.md`](docs/s6-nano2-prereg.md) | `scripts/build-route-data.sh`, `scripts/train-route.sh`, `scripts/route-eval.sh` | [`runs/s6-nano2/`](runs/s6-nano2/) | [`hf/cards/s6-route-nano.md`](hf/cards/s6-route-nano.md) |
| s6-route | [`docs/s6-nano3-prereg.md`](docs/s6-nano3-prereg.md) | `scripts/route3-read.sh` (no training) | [`runs/s6-nano3/`](runs/s6-nano3/) | [`hf/cards/s6-route.md`](hf/cards/s6-route.md) |
| s6-route7-nano | [`docs/s6-route7-prereg.md`](docs/s6-route7-prereg.md) | `scripts/build-route7-data.sh`, `scripts/train-route7.sh`, `scripts/route7-gate.sh` | [`runs/s6-route7/`](runs/s6-route7/) | [`hf/cards/s6-route7-nano.md`](hf/cards/s6-route7-nano.md) |
| s6-sleep3-nano | [`docs/s6-sleep3-prereg.md`](docs/s6-sleep3-prereg.md) | `scripts/train-sleep3.sh` | [`runs/s6-sleep3/`](runs/s6-sleep3/gate.json) | [`hf/cards/s6-sleep3-nano.md`](hf/cards/s6-sleep3-nano.md) |

## How the experiments were run

- From s5 on, every experiment had a pre-registration in [`docs/`](docs/README.md), committed before any computation it governs, with its gate.
- Each test set was read once, by the step its pre-registration names; there was no rerun "to see".
- Every change was a dated amendment, written and committed before the computation it concerns.
- A gate was read as written, and a model that missed it is published as a result, with a card that says so first.
- [`docs/README.md`](docs/README.md) indexes every line with its status, pre-registration and result folder; [`AGENTS.md`](AGENTS.md) holds the rules in full.

## Repository layout

`docs/` holds the pre-registrations, studies, diagnoses and specifications ([index](docs/README.md)); `runs/` the small result files of each gated run; `scripts/` the training, evaluation and publishing scripts; `site/` the VitePress site, its demos and the technical report; `hf/` the Hugging Face model cards and Space definitions. The code is one Rust workspace:

| crate | what it does |
|---|---|
| `sky-schema` | System One request/response types, decision records, drone command plan + safety validator |
| `sky-games` | game cores, oracles, DAgger-ready data generator |
| `sky-data` | Hugging Face download and conversion, BPE tokenizer training, GLM teacher client |
| `sky-train` | mlx-rs training: MLM, decision training, Muon/AdamW/WSD, calibration, export |
| `sky-gguf` | GGUF v3 reader/writer, Q8_0 |
| `sky-infer` | inference engine (native + WASM), tokenizer, converter to GGUF |
| `sky-arena` | plays the games natively with a model, DAgger collection |
| `sky-web` | WASM bindings used by the site (games + model) |
| `sky-serve` | the `skycmd` CLI: `serve`, `decide`, `info`, `bench` |
| `sky-py` | Python bindings (PyO3), the `skycmd` wheel |
| `sky-metrics` | evaluation metrics: proper scores, calibration (ECE), selective prediction, paired bootstrap, non-inferiority, the Holm correction |
| `packages/skycmd-js` | the `skycmd` npm package (WASM) |

## Licence and citation

The code is under the Apache-2.0 licence ([`LICENSE`](LICENSE)). The data sets keep their own licences, listed source by source in [`DATA_LICENSES.md`](DATA_LICENSES.md). To cite the project, use [`CITATION.cff`](CITATION.cff), which points to the [technical report](https://maxgfr.github.io/skycmd/paper/) and its DOI, [10.5281/zenodo.24418316](https://doi.org/10.5281/zenodo.24418316).
