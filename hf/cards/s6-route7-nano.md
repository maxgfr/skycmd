---
license: apache-2.0
language:
  - en
  - fr
base_model: minishlab/potion-base-2M
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
  - failed-gate
  - negative-result
---

# s6-route7-nano

> **Failed its pre-registered gate: published as a result, not for use.** The gate fails (G1
> fails, G2 fails, G3 fails, G4 holds, G5 holds, G6 fails). On T5-tools, 2,000 fresh requests about
> the tools of 262 MCP servers (Toucan), a family that no training slate holds, this router picks
> the right option on **0.5610**, against **0.5675 for BM25** (−0.0065, 95% interval [−0.0235,
> +0.0120]): it does not beat a score that costs no byte (G1). Under masked names it answers *none*
> on 0.3345 of the slates, where the true share is 0.20 (G2 (c)), and its probabilities are badly
> calibrated (ECE **0.1845**, bar 0.05; G3). **Read with the request hidden**, its trained model
> part picks the right tool at chance (0.1406, chance 0.1429; G6 (a) holds), but the pair does not
> (**0.1969**, bar 0.1929, server upper bound 0.2350 against 0.2229; G6 (b) fails); the frozen
> embedding part alone already reads 0.1931 there. It runs in 13.9 ms per request in WASM from
> 3,692,288 bytes (G4), and it reads 0.8815 on the older T2-tools (G5). The route line is closed
> with this result, and the published router stays [s6-route](https://huggingface.co/{{USER}}/s6-route).
> The files are published so that the result can be checked and run (owner's decision of
> 15/11/2026); the verdict does not change. Numbers:
> [BENCH.md, section s6-route7-nano](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-route7-nano-s6-route-v7-the-router-retrained-without-its-shortcut-pre-registered-gate-failed),
> [`runs/s6-route7/gate.json`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-route7/gate.json)
> and [`readings.json`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-route7/readings.json).

**A tool router in two files, 3.7 MB, retrained to answer from the request.** Given a request and a
list of up to 7 tools, each a name and one sentence, plus the option "none of these tools fits the
request", it gives a probability to each option and says "ask the user" when the top probability
is below τ_ask7 = 0.762097. It has the shape of [s6-route](https://huggingface.co/{{USER}}/s6-route):
a 2.4 MB ternary model part (5,587,201 parameters) fused option by option with a 4-bit copy of the
static embedding [minishlab/potion-base-2M](https://huggingface.co/minishlab/potion-base-2M):

`p(i) ∝ (p_model(i) · p_embedding(i))^0.7`, renormalised over the options.

The model part is new: it was trained again, from the same pretraining, on data rebuilt against the
cues that let a rule pick the right tool without reading the request (smaller ones remain, and are
reported), and it reads each tool's BM25 score inside that tool's option text. It is the seventh
version of the s6-route line and the last.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-route7-nano](https://huggingface.co/spaces/{{USER}}/s6-route7-nano)
  runs this pair in the browser, its failed gate first.
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust; the router is
  `crates/sky-web/src/router.rs`, the engine `crates/sky-infer`)
{{LINKS}}

## The gate, as read

`runs/s6-route7/gate.json`, read once on 19/11/2026. T5-tools: 2,000 slates from 262 servers,
1,600 with a tool right and 400 with *none* right; chance c = 1/7 on the tool-right slates;
intervals are 95% bootstraps of 1,000 resamples, of the slates unless "servers" is said.

| | condition | measured | verdict |
|---|---|---|---|
| G1 | the pair − BM25, lower bound of the slate bootstrap > 0 | 0.5610 against 0.5675: −0.0065 [−0.0235, +0.0120] | **fails** |
| G1 | the pair − `potion-cos-4bit` (its embedding part), lower bound > 0 | 0.5610 against 0.5025: +0.0585 [+0.0380, +0.0800] | holds |
| G1 | the pair above its model part | 0.5610 against 0.4905 | holds |
| G2 (a) | one distractor's description rewritten by the teacher: the pair answers it on ≤ 0.1929 of 1,600 tool-gold slates, and ≤ +0.03 over the same option unrewritten | real names 0.0494, +0.0238; masked names 0.0513, +0.0287 | holds |
| G2 (b) | names masked: the pair ≥ the embedding part − 0.02 | 0.5440 against 0.4750 − 0.02 | holds |
| G2 (c) | names masked: *none* answered on 0.10 to 0.30 of the slates (true share 0.20) | 0.3345 | **fails** |
| G3 | ECE (15 bins) on T5-tools ≤ 0.05 | 0.1845 | **fails** |
| G4 | WASM against the Python reference, 200 slates: max \|Δp\| ≤ 1e-3, the same answer on ≥ 199 | 9.1e-7, 200 of 200 | holds |
| G4 | WASM SIMD128 median ≤ 60 ms per request, tool list cached | 13.87 ms | holds |
| G4 | the two files ≤ 4,000,000 bytes | 3,692,288 | holds |
| G5 | T2-tools (945 slates, v2's held-out tools) ≥ 0.8668 | 0.8815 | holds |
| G6 (a) | the model part with the request hidden: right tool on ≤ c + 0.05 = 0.1929, server upper bound ≤ c + 0.08 = 0.2229 | 0.1406; servers [0.1113, 0.1720] | holds |
| G6 (b) | the pair with the request hidden: the same | 0.1969; servers [0.1599, 0.2350] | **fails** |
| G6 (c) | with the request hidden, the *none* answer does not depend on whether *none* is right: gap ≤ 0.05, server upper bound ≤ 0.15 | the model part −0.0069 [−0.0220, +0.0081]; the pair −0.0087 [−0.0521, +0.0371] | holds |

With servers resampled, the pair − BM25 is −0.0065 [−0.0309, +0.0170] and the pair − the
embedding part +0.0585 [+0.0278, +0.0860]. all-MiniLM-L6-v2 with cosine (91.6 MB) reads 0.6630,
0.1020 above the pair; that difference is reported and was not a gate (the code gives no interval
for it). Without the 1,204 slates that show other tools of the right tool's own server, G6 fails
too (the model part 0.1692, servers upper bound 0.2421; the pair 0.3081 [0.2208, 0.3974]), so the
failure is not confined to those slates.

Announced goals (not gates): T5-tools ≥ 0.80, not met; within 0.03 of MiniLM, not met; ≥ 0.95
among answered requests at τ_ask7 with coverage ≥ 0.80, not met (0.7127 at coverage 0.5325, from
the log of the reading); the fast mode within 0.015, met (0.5595); with the request hidden, the
pair says *none* or asks on at least 0.80 of the slates, met (0.9955); G6 (a) and (b) on the masked
copy, met for the model part, not for the pair (servers upper bound 0.2241 against 0.2229).

## Results

Accuracy on T5 (`runs/s6-route7/readings.json`; *none* included; the two rewritten copies hold
the 1,600 tool-gold slates only):

| system | size | T5-tools | names masked | a distractor rewritten | masked and rewritten |
|---|---:|---:|---:|---:|---:|
| **s6-route7-nano (the pair)** | 3.7 MB | **0.5610** | 0.5440 | 0.57625 | 0.541875 |
| its fast mode | 3.7 MB | 0.5595 | 0.5395 | 0.58125 | 0.5425 |
| its model part alone | 2.4 MB | 0.4905 | 0.4610 | 0.481875 | 0.40625 |
| its embedding part alone (potion-base-2M, cosine, 4 bits) | 1.3 MB | 0.5025 | 0.4750 | 0.56875 | 0.529375 |
| s6-route (the published pair, v3) | 3.7 MB | 0.5530 | 0.2810 | 0.604375 | 0.183125 |
| BM25 (the slate's own tools) | no parameters | 0.5675 | 0.5580 | 0.58125 | 0.56375 |
| all-MiniLM-L6-v2, cosine | 91.6 MB | 0.6630 | 0.5925 | 0.734375 | 0.6300 |
| toolrouter (stanley-nv) | 7.66 MB | 0.5895 | 0.5220 | 0.685625 | 0.599375 |

On validation (`v5_tools`, 1,144 slates of the training families, read for the fit) the pair reads
0.8802 with an ECE of 0.0232, its model part 0.8418, BM25 0.8383. On T5 the pair is at 0.5610: on
the tools of a new family it loses about a third of its accuracy, and its advantage over BM25 is
gone. Why is not measured: no reading separates the new tool family from the new source of the
requests (Toucan's requests were written by language models) or from the way T5 was built.

## Read with the request hidden

The question G6 asks: does the router pick the right tool from the options alone? Each request is
replaced by `?`, so every BM25 score is 0.00 and the embedding part embeds `?`; a router that reads
only the request and the tools should then be at chance, 1/7 on these slates.

| | right tool, request hidden | servers 95% | says *none* | *none* gap (servers 95%) |
|---|---:|---|---:|---|
| this model part | 0.1406 | [0.1113, 0.1720] | 0.0405 | −0.0069 [−0.0220, +0.0081] |
| this pair | 0.1969 | [0.1599, 0.2350] | 0.8120 | −0.0087 [−0.0521, +0.0371] |
| the embedding part alone | 0.1931 | [0.1589, 0.2278] | 0.8215 | −0.0112 [−0.0507, +0.0324] |
| s6-route (v3), for comparison | 0.1794 | [0.1466, 0.2138] | 0.6710 | +0.0113 [−0.0367, +0.0608] |

- **The model part is at chance** with the request hidden, and its *none* answer does not depend on
  whether *none* is right. With the scores given in another format (B1, reported) it reads 0.1813,
  still under the bar.
- **The pair is not.** The embedding part, the static embedding that no version of this line
  trained, already reads above the bar on its own; the pair is 0.0038 above it, with no interval
  for that difference. Both are furthest from chance on the 396 slates that show no other tool of
  the right tool's server (0.3081 for the pair, 0.2904 for the embedding part).
- **What this reading cannot see:** a cue used together with the request. Hiding the request hides
  that kind of cue too.

## What it does, and does not

- **Use:** none. On tools of a new family it does not beat BM25, which needs no file at all, and
  its confidence does not mean what it says. Use BM25 there, or all-MiniLM-L6-v2 if 91.6 MB fit.
- **What it shows:** the shortcut that versions 4 to 7 of this line set out to remove is not found
  in the trained model part on fresh tools. With the request hidden it is at chance; the pair
  rarely follows a rewritten description (0.0494); masking the names costs the pair 0.0170, where it
  costs s6-route 0.2720 on the same slates.
- **What it does not show:** that the model reads the request well on new tools (it is at 0.4905
  alone, below its embedding part), that it is calibrated there, or that it has no cue used together
  with the request.
- **Under masked names** it says *none* too often: 0.3345 of the slates, where *none* is right on
  0.20.

## How it differs from the published router

| | s6-route (v3, passed on fresh slates) | s6-route7-nano (v7, failed) |
|---|---|---|
| the model part | s6-route-nano, trained on v2's data | trained again on v5's rebuilt data, with v6's and v7's fixes to When2Call's slates |
| the BM25 score | a line of the state, bound to each tool by its name | inside each option's own text: `score <s> rank <r>; <name>: <description>` |
| β, ask threshold | 0.81, τ_ask3 0.671047 | 0.7, τ_ask7 0.762097 (fitted on `v5_tools`) |
| the embedding part | potion-base-2M, 4 bits | the same table, byte for byte; only β and the threshold differ |
| its gate | T2-tools, fresh requests about v2's held-out tools | T5-tools, fresh requests about MCP tools of a new family, and the request-hidden reading G6 |
| on T5-tools (reported for v3) | 0.5530; names masked 0.2810 | 0.5610; names masked 0.5440 |

## How to use it

Both files are needed: `s6-route7-nano.gguf` (the model part, `skycmd-s2`) and
`s6-route7-embed.gguf` (the embedding part: the 4-bit table, the tokenizer of potion-base-2M, θ,
the temperature, β and the ask threshold). The model file carries the key
`skycmd-s2.route.render = scores-in-options`, which tells the engine to put each tool's BM25 score
and rank in that tool's option; nothing else is needed for that.

**The input.** A request (one user turn) and up to 7 tools, each `name: description`, the
description cut to its first sentence and to 30 words (`sky_data::route::cut_description`); do not
add *none*: the router appends `none of these tools fits the request` as the last option.

**In the browser** (`sky-web`, the `ToolRouter` class; the wasm module is built by
`scripts/build-wasm.sh`):

```js
import init, { ToolRouter } from './pkg/sky_web.js'
await init()
const bytes = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer())
const router = new ToolRouter(await bytes('s6-route7-nano.gguf'), await bytes('s6-route7-embed.gguf'))
const tools = JSON.stringify(['lights_on: Turns the lights on in a room.', 'get_weather: Gives the weather forecast for a city.'])
const r = JSON.parse(router.route('turn on the hall lights', tools, false)) // true: fast mode
// r.probs: the tools then none; r.decision: "tool", "none" or "ask" (r.top below r.tau_ask3)
```

The ask threshold is stored under the engine's key `skycmd-embed.route.tau_ask3` (the JSON field
`tau_ask3`); its value is v7's τ_ask7 (amendment 3 of the pre-registration).

**Natively** (`sky-route`):

```sh
cargo run --release -p sky-infer --bin sky-route -- fuse \
  --model s6-route7-nano.gguf --embed s6-route7-embed.gguf --slates slates.jsonl --out routed.jsonl
```

## Latency and size

Apple M5, Node, the first 200 T5-tools slates, WASM SIMD128 (`runs/s6-route7/g4.json`): median
13.87 ms per request with the tool list cached (the model part 13.86 ms, the embedding part
0.01 ms), 28.33 ms for the first request on a new tool list. WASM matches the Python reference to
9.1e-7, with the same answer on 200 of 200. 2,418,976 + 1,273,312 = 3,692,288 bytes.

## Training data and licences

**The model part.** From s6-micro's masked-language pretraining (`runs/s6-micro/pretrain/init`,
checked by sha256), in Rust (mlx-rs) on an Apple M5, in a braked mode, so the budgets are
working minutes: a pilot of 20 working minutes (3,664 steps), then one main run of 120 working
minutes (22,214 steps, seed 0, ternary QAT), the s6-micro graph (2 weight-shared blocks looped
three times, exits after 2, 4 and 6 block passes, d 256, vocabulary 4,096, a head window of 320
tokens). The last checkpoint was exported; none was chosen on a reading. Licence of the file:
Apache-2.0.

**Its training data** (182,085 records, drawn 0.60 / 0.25 / 0.15; validation 15,873): tools of
glaive-function-calling-v2, hermes-function-calling-v1, ToolACE and Seal-Tools (Apache-2.0) and
API-Bank (MIT), clustered so that no test tool has a training tool in its cluster (77,070
records); intents of MASSIVE in English and French (14 of 18 scenarios) and Banking77 (CC-BY-4.0;
91,130); When2Call (CC-BY-4.0; 13,885). Every drawn distractor is taken within the right tool's
surface cell (v5), the hard ones nearest to the request, and each description is the teacher's
rewrite with probability 0.5 (v4). When2Call's no-call slates are drawn without the request's own
tools (v6), and those whose request has no tool in the pool around the pool tool nearest to the
request (v7). **The data set:**
[skycmd-route7](https://huggingface.co/datasets/{{USER}}/skycmd-route7) holds these records, the
validation files of the pilot and the fit (with their masked, rewritten and blind copies), the
fresh test T5 with its probe and blind copies, and the teacher's rewrites, each file checked
against the sha256 that `runs/s6-route7/` records and each row under its sources' licences; every
source is in [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

**The teacher**, GLM-5.3-Flash (open weights, MIT), filtered the hard distractors that also serve
the request (v5's 58,984 answers, v6's 694 and v7's 296) and wrote the description rewrites (v4's
13,687); it gives no training target. For the two probe copies of the test only, it rewrote the
descriptions of the 1,448 Toucan tools that T5 shows, from each tool's name and description: it saw
the test's tools, never a test request.

**The embedding part** is derived from
[minishlab/potion-base-2M](https://huggingface.co/minishlab/potion-base-2M) by Stephan Tulkens and
Thomas van Dongen (Minish Lab), **MIT licence**, at commit `389b9f64`: s6-route's embedding file,
copied with v7's β and ask threshold in place of v3's (`sky-route embed-params`), every other key
and every tensor byte for byte. It carries the licence, the authors and the commit in its metadata,
and stays under the MIT licence.

**The fresh test** T5 was built from
[Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) (Apache-2.0 on its
card; its requests were written by Mistral-Small, DevStral-Small, GPT-OSS, Kimi-K2 and Qwen3-32B,
its tool descriptions are the MCP servers' own), test only, never trained. It was read once and
nothing is gated on it again, so its files are published in
[skycmd-route7](https://huggingface.co/datasets/{{USER}}/skycmd-route7) (configs `t5_*`); their
hashes are in
[`runs/s6-route7/test-files.json`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-route7/test-files.json).

## Disclosure

From §10 of the pre-registration, with v6's, v5's and v4's:

- v7 was designed after v4's, v5's and v6's data tables were read. Its one change follows v6's table
  and a count made after v6's stop, and it is the third construction of When2Call's no-call slates.
- The anchor of those slates is chosen with the request: an unanchored no-call slate's options are
  drawn around the tool nearest to the request's text. A model could, in principle, learn that a
  request close to every option is more often a no-call request; the request-hidden reading cannot
  see that cue.
- The stop rule was written before v6's rebuild and was not changed, and the rule that closes the
  line was written before any computation of v7.
- v6 relaxed what stopped v4 and v5: their data checks are reported, and the test on the model
  (G6) replaces them. Outside the rebuilt slates, the training data hold the cues that v5's table
  names.
- The teacher saw the fresh test's tools (above). The field names and split values of two T5 files
  were read once before T5 was opened, with no request text. all-MiniLM-L6-v2 defines the clusters
  that make a "new family" and is a reported baseline; the latency bar was set after v2 was
  measured.
- A reader who holds that the data should have been blind may hold v7 to that view; the tables of
  v4 to v7 are published.

## Files

- `s6-route7-nano.gguf`: the model part, ternary weights, tokenizer, temperatures,
  `skycmd-s2.route.render` and the model's own `skycmd-s2.route.tau_ask` (0.784774, not used by the
  pair). 2,418,976 bytes, sha256
  `5fbe51b451023730c0517254eed43ce5c96c9a2384563bca6d8ddac3b77c7a9d`.
- `s6-route7-embed.gguf`: the embedding part with β = 0.7 and the ask threshold 0.762097.
  1,273,312 bytes, sha256 `717dc3d89f096d063de9af52636fdafca834c53d484ecde533d2043047e7dafd`.

These are the hashes that `runs/s6-route7/FRESH-OPEN` recorded before the fresh test was read. The
result files are on GitHub:
[`runs/s6-route7/`](https://github.com/maxgfr/skycmd/tree/main/runs/s6-route7) (`GATE.md`,
`gate.json`, `readings.json`, `g4.json`, `fit.json`, `embed.json`, `FRESH-OPEN`, the validation
readings and every checkpoint's readings) and
[`runs/s6-route7-pilot/summary.json`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-route7-pilot/summary.json).
