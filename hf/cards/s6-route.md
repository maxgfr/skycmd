---
license: apache-2.0
language:
  - en
  - fr
library_name: gguf
pipeline_tag: text-classification
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - tool-routing
  - static-embedding
  - rust
  - wasm
  - skycmd
---

# s6-route

> **Passed its pre-registered gate on fresh slates; the system was defined after v2's test was
> read.** v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps.
> Its parameters come from validation. Its gate is read on requests no system had scored; on
> T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30
> times v2's and was set after v2 was measured. The system needs distinctive tool names.
>
> On T2-tools, 945 fresh requests about tools never seen in training (273 clusters), it picks the
> right option on **0.8868**, against **0.8169 for BM25** (+0.0698 [0.0455, 0.0952]) and **0.9048
> for all-MiniLM-L6-v2** (91.6 MB; the bar was MiniLM − 0.03). Every gate held. Two readings
> reported with it say what the pass does not: **without the three largest clusters the advantage
> over BM25 is +0.0202 [−0.0062, +0.0481]**, and the bootstrap that resamples clusters gives
> +0.0698 [−0.0079, +0.1404]: both intervals hold zero. **With every tool renamed `tool_<k>` it
> falls to 0.2497**, where BM25 keeps 0.8138. Numbers:
> [BENCH.md, section s6-route](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-route-s6-nano-v3-the-v2-router-fused-with-a-static-embedding-pre-registered-gate-passed)
> and [`runs/s6-nano3/eval-fresh/results.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-nano3/eval-fresh/results.md).

**A 3.7 MB tool router in two files.** Given a request and a list of up to 7 tools, each a name and
one sentence, plus the option "none of these tools fits the request", it gives a probability to
each option, and says "ask the user" when the top probability is below τ_ask3 = 0.671047. It is
the file of [s6-route-nano](https://huggingface.co/{{USER}}/s6-route-nano) (v2, which failed its
own gate), unchanged, fused option by option with a 4-bit copy of the static embedding
[minishlab/potion-base-2M](https://huggingface.co/minishlab/potion-base-2M):

`p(i) ∝ (p_model(i) · p_embedding(i))^0.81`, renormalised over the options.

Nothing was trained for this version and no GPU was used: the partner, β and τ_ask3 were chosen on
validation, and the gate was read once on fresh files.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-route](https://huggingface.co/spaces/{{USER}}/s6-route)
  (your own tool list and request, in the browser), and the
  [Router page](https://maxgfr.github.io/skycmd/router/) of the project site
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust; the router
  is `crates/sky-web/src/router.rs` and `crates/sky-infer/src/embed.rs`)
- **Pre-registration:** [docs/s6-nano3-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md)
  (the system in §2, the gate in §5, the disclosure in §7, amendments 1 and 2 before the reading,
  the result block)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## The gate, as read

| | condition | measured | verdict |
|---|---|---|---|
| G1 | T2-tools: above BM25, with the lower bound of the 95% paired bootstrap above 0 | 0.8868 against 0.8169: +0.0698 [0.0455, 0.0952] | holds |
| G1 | ≥ MiniLM − 0.03 | 0.8868 against 0.9048 − 0.03 = 0.8748 | holds |
| G1 | above the model alone and the embedding alone | 0.8868 against 0.7926 and 0.7386 | holds |
| G2 | MetaTool, CLINC150, MASSIVE: ≥ the better part − 0.02 | 0.5175 against 0.5140; 0.8040 against 0.7655; 0.5340 against 0.4555 | holds |
| G3 | ECE ≤ 0.05 on T2-tools | 0.0162 | holds |
| G4 | WASM against the Python reference on 200 slates: max \|Δp\| ≤ 1e-3, same answer on ≥ 199 | 1.6e-6, 200 of 200 | holds |
| G4 | WASM SIMD128 median ≤ 60 ms per request, tool list cached | 30.9 ms | holds |
| G4 | the two files ≤ 4 MB | 3,692,224 bytes | holds |

On MetaTool the system is below its embedding part alone (0.5175 against 0.5340), within the
0.02 that G2 allows. The 60 ms bar replaces v2's 2 ms, which v2 missed at 38.9 ms; it was set
after that measurement. The generated `results.md` prints 3,422,864 bytes in its header line (the
embedding table alone, as §2 counted it); the two files as shipped are 3,692,224 bytes, which is
what G4 read.

Announced goals (not gates): T2-tools ≥ 0.87, met (0.8868); ≥ 0.95 among answered requests at
τ_ask3 with coverage ≥ 0.80, **not met** (0.9433 at coverage 0.8593); the fast mode within 0.015,
met (0.8836).

## Results

Accuracy on the fresh files (T2-tools: 945 slates, 20% with *none* right; the three others: 2,000
each):

| system | size | T2-tools | names masked | MetaTool | CLINC150 | MASSIVE (held-out scenarios, en and fr) |
|---|---:|---:|---:|---:|---:|---:|
| **s6-route** | 3.7 MB | **0.8868** | 0.2497 | 0.5175 | 0.8040 | 0.5340 |
| s6-route, fast mode | 3.7 MB | 0.8836 | 0.2899 | 0.5130 | 0.8020 | 0.5300 |
| s6-route-nano alone (the model part) | 2.4 MB | 0.7926 | 0.0720 | 0.3190 | 0.4240 | 0.3680 |
| potion-base-2M, cosine, 4 bits (the embedding part) | 1.3 MB | 0.7386 | 0.7280 | 0.5340 | 0.7855 | 0.4755 |
| BM25 (the slate's own tools) | no parameters | 0.8169 | 0.8138 | 0.4330 | 0.4990 | 0.3205 |
| BM25 fused with the same embedding | 1.3 MB | 0.8624 | 0.8423 | 0.5420 | 0.7750 | 0.4515 |
| all-MiniLM-L6-v2, cosine | 91.6 MB | 0.9048 | 0.9005 | 0.6570 | 0.8790 | 0.5730 |
| toolrouter (stanley-nv) | 7.66 MB | 0.7735 | 0.7513 | 0.5910 | 0.7470 | 0.5900 |

On T2-tools: ECE 0.0162 (MiniLM 0.0254, BM25 0.0779), *none* F1 0.7732 (MiniLM 0.8010, BM25
0.6368); accuracy among the most confident 50% / 70% / 90% of the slates 0.996 / 0.977 / 0.929.
Paired bootstrap (slates): +0.0942 [0.0720, 0.1185] over the model alone, +0.1481 [0.1228,
0.1714] over the embedding alone, +0.0243 [0.0085, 0.0413] over BM25 fused with the embedding,
−0.0180 [−0.0381, +0.0032] against MiniLM.

## What it does well, and badly

- **Well:** on fresh requests about tools it never saw, above BM25 and above each of its parts,
  within 0.02 of a sentence encoder about 25 times its size, and well calibrated (ECE 0.0162). It
  abstains usefully: at τ_ask3 it answers 86% of the requests and is right on 94% of those. It runs
  in the browser in about 31 ms per request.
- **The advantage over BM25 is fragile.** The three largest clusters hold 300 of the 945 slates.
  Without them the system is at 0.8806 against BM25 0.8605: +0.0202 [−0.0062, +0.0481]. Resampling
  clusters instead of slates gives +0.0698 [−0.0079, +0.1404]. Both intervals hold zero: the
  gated advantage over BM25 leans on the three largest clusters. The advantages over the two parts
  hold under both readings.
- **It needs distinctive tool names.** With every tool renamed `tool_<k>` it falls to 0.2497; the
  model part is then at 0.0720, and the fusion drags the embedding part (0.7280 alone) down with
  it. BM25 keeps 0.8138 and MiniLM 0.9005 on the same slates. Do not use it on tool lists with
  generic or numbered names.
- **MiniLM is more accurate on every file.** If 91.6 MB and about 12 ms of CPU per request are
  affordable, all-MiniLM-L6-v2 with cosine is the better router on these tests.
- **Outside the tool lists of its training families it stays near its embedding part, below it on MetaTool and above it on CLINC150 and MASSIVE:** MetaTool
  0.5175 (the embedding alone 0.5340, toolrouter 0.5910), CLINC150 0.8040 (0.7855), MASSIVE
  0.5340 (0.4755; toolrouter 0.5900).
- **The tools of T2-tools are v2's held-out test tools**, whose results this project had already
  read; the requests are fresh. A confirmation on tools of other families has not been run.

## How to use it

Both files are needed: `s6-route-nano.gguf` (the model part, `skycmd-s2`) and
`s6-route-embed.gguf` (the embedding part: the 4-bit table, the tokenizer of potion-base-2M, θ,
the temperature, β and τ_ask3).

**The input.** A request (one user turn) and up to 7 tools, each rendered `name: description`, the
description cut to its first sentence (up to the first `.`, `!` or `?` followed by a space or the
end) and to 30 words, as the training data were (`sky_data::route::cut_description`); with no
description, the name alone. Do not add *none*: the router appends `none of these tools fits the
request` as the last option. The model part reads at most 256 head tokens and caps each option at
48 tokens, so a long description loses its tail. The BM25 line that the model part reads in its
state is computed by the router (`sky_infer::route`).

**In the browser** (`sky-web`, the `ToolRouter` class; the wasm module is built by
`scripts/build-wasm.sh`):

```js
import init, { ToolRouter } from './pkg/sky_web.js'
await init()
const bytes = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer())
const router = new ToolRouter(await bytes('s6-route-nano.gguf'), await bytes('s6-route-embed.gguf'))
const tools = JSON.stringify([
  'lights_on: Turns the lights on in a room.',
  'get_weather: Gives the weather forecast for a city.',
])
const r = JSON.parse(router.route('turn on the hall lights', tools, false)) // true: fast mode
// r.probs: the tools then none; r.answer: a tool name or "none";
// r.decision: "tool", "none" or "ask" (r.top below r.tau_ask3); r.model, r.embedding: the parts
```

The tool list is encoded once and cached on both parts: the next request against the same list
reuses it (`static_hit`, `tools_hit`). The fast mode stops the model part at its first exit (2 of
6 block passes): 0.8836 on T2-tools, 10.5 ms per request.

**Natively** (`sky-route`, `crates/sky-infer/src/bin/sky-route.rs`):

```sh
cargo run --release -p sky-infer --bin sky-route -- fuse \
  --model s6-route-nano.gguf --embed s6-route-embed.gguf --slates slates.jsonl --out routed.jsonl
```

`slates.jsonl` holds one slate per line, `{"opts": [<tools as name: description>..., "none of these
tools fits the request"], "state": "User: <request>"}` (an optional `target`, one-hot over the
options, adds the accuracy to the summary). Each output line is `{probs, model, embedding, answer,
ask}`; `--fast` stops the model part at its first exit. From Rust, `sky_web::router::Router` gives
the same call as the browser.

## Latency and size

Apple M5, Node, the first 200 T2-tools slates, on a machine in use by its owner with a GPU run of
another model active (the numbers can only be higher than on an idle machine):

| | SIMD128 | relaxed SIMD | fast mode |
|---|---:|---:|---:|
| median per request, tool list cached | 30.9 ms | 23.5 ms | 10.5 ms |
| first request (tool list encoded) | 45.3 ms | 34.3 ms | 25.0 ms |
| max \|p_wasm − p_python\| | 1.6e-6 | 1.0e-6 | 9.0e-7 |

The embedding part takes about 0.02 ms of it; the rest is the model part (natively 24.6 ms median,
p95 45.6 ms).

## Training data and licences

**The model part** is s6-route-nano unchanged; its training is described in
[its card](https://huggingface.co/{{USER}}/s6-route-nano): from scratch in Rust on an Apple M5, the
s6-micro graph (5,587,201 ternary parameters) initialised from s6-micro's masked-language
pretraining, then 120 working minutes of decision training on routing data only: tools of
glaive-function-calling-v2, hermes-function-calling-v1, ToolACE and Seal-Tools (Apache-2.0) and
API-Bank (MIT), clustered so that no test tool has a training tool in its cluster; intents of
MASSIVE in English and French (14 of 18 scenarios) and Banking77 (CC-BY-4.0); When2Call
(CC-BY-4.0). The teacher GLM-5.3-flash (open weights, MIT, called through the z.ai API) wrote
the intent descriptions, filtered the hard distractors and rewrote the training tools'
descriptions; it never saw a test request and gives no training target. The z.ai terms of service
restrict training on its outputs; the project's choice is described in
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md). Licence of the
file: Apache-2.0.

**The embedding part** is derived from
[minishlab/potion-base-2M](https://huggingface.co/minishlab/potion-base-2M) by Stephan Tulkens and
Thomas van Dongen (Minish Lab), **MIT licence**, at commit `389b9f64`: its table packed to 4 bits,
with its tokenizer (`scripts/s6-route3-embed-export.py`, `sky-route embed-gguf`). The file carries
the licence, the authors and the commit in its metadata; it stays under the MIT licence. Its θ and
temperature were fitted on v2's validation slates.

The fresh test files were built from requests of the same sources that no v2 slate uses
(MetaTool, CLINC150 and the MASSIVE held-out scenarios are test only); they are not published
here.

## Files

- `s6-route-nano.gguf`: the model part, v2's file byte for byte. 2,418,912 bytes, sha256
  `a124dfe86f76e9547db98edc03351f66908617ae2be3f5a4f423a876b7956c82`.
- `s6-route-embed.gguf`: the embedding part. 1,273,312 bytes, sha256
  `c53fb76dc8a56a696d859e80b7383d7e9db168e54c024525f220851fbacda67a`.
