# Compatibility of skycmd GGUFs with other runtimes

skycmd ships two model families:

- **s1** (`general.architecture = "modern-bert"`, decision type `laya`): the Laya / ModernBERT
  layout of llama.cpp [PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818). The same
  file runs in the skycmd engine (CLI, server, Python, JS/WASM) and in runtimes built on a recent
  llama.cpp.
- **s2** (`general.architecture = "skycmd-s2"`, see `docs/SPEC-S2.md`): two-stream encoder,
  native numbers, early exit, ternary weights. **s2 GGUFs are served only by the skycmd engine**
  (`skycmd serve`, `skycmd decide`, the `skycmd` Python wheel and npm package). No other runtime
  knows this architecture.

All tests below used `site/public/models/s1-pico.gguf` (`skycmd-s1-pico`, 202,305 parameters,
F16 matrices, 545,280 bytes) on an Apple M5 (macOS, arm64), on 2026-10-11.

## Summary

| runtime | version | loads s1 | `/v1/systemone` | agreement with skycmd |
|---|---|---|---|---|
| skycmd (`decide`, `serve`, Python, JS/WASM) | 0.1.0 | yes | yes | reference |
| llama.cpp `llama-server` (CPU, `-ngl 0`) | 0.6.0-dev, commit 62a6f74 | yes | yes | max abs. probability difference 4.7e-4 over 103 probabilities, same argmax everywhere |
| wllama (headless Chrome, CPU) | 3.8.1 (libllama b11364-46ca246) | yes | `createSystemOne()` works | max abs. difference 1.1e-4 (2.0e-4 against llama-server) |
| Ollama | 0.40.3 | yes (`ollama create` succeeds, capability `decision`) | endpoint exists, rejects the model: `unsupported decision encoding "laya"` | n/a |
| LM Studio | 0.4.26 (`lms` CLI present) | not tested: no inference runtime installed (`lms runtime ls`: none) | not available (no `/v1/systemone` in its API) | untested |

## Request used

The three-question Flappy Drone frame (`SAMPLE_REQUEST` in `crates/sky-serve/src/lib.rs`):

```json
{
  "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  "questions": {
    "action": {"type": "choice", "instructions": "Which input keeps the drone flying through the next gap?",
               "criteria": {"thrust": "fire the rotors for an upward impulse", "glide": "let gravity pull the drone down"}},
    "danger": {"type": "noul", "instructions": "Will the drone crash within the next half second if it keeps gliding?"},
    "margin": {"type": "score", "instructions": "How much room is left above and below the drone?",
               "criteria": ["none", "tight", "comfortable"]}
  }
}
```

Current published file (`site/public/models/s1-pico.gguf`, the DAgger-trained s1-pico shipped in
commit a6ab1fd), re-measured on 2026-10-12 with `skycmd decide` and llama-server (`-ngl 0`):

| answer | skycmd (native) | llama-server |
|---|---|---|
| `action` p(thrust) | 0.0743545 | 0.0743950 |
| `danger` noul | 0.1645060 | 0.1645364 |
| `margin` score | 1.8986974 | 1.8986165 |
| max abs. probability difference | | 6.9e-5, same `choice` (glide) |

The first compatibility run (below, and in the summary table) used the earlier, pre-DAgger
s1-pico of commit 60f1c0c. Its values are kept for the WASM and wllama columns, which were not
re-run on the new file:

| answer | skycmd (native) | skycmd (WASM, Node/Deno/Bun/Chrome) | llama-server | wllama 3.8.1 |
|---|---|---|---|---|
| `action` p(thrust) | 0.0597123 | 0.0597122 | 0.0598046 | 0.0596021 |
| `danger` noul | 0.9533952 | 0.9533952 | 0.9532889 | 0.9533593 |
| `margin` score | 1.8128099 | 1.8128095 | 1.8128024 | 1.8128236 |
| input tokens | 193 | 193 | 193 | 193 |

## (a) llama.cpp `llama-server`

```sh
llama-server -m site/public/models/s1-pico.gguf -ngl 0 --host 127.0.0.1 --port <free port>
skycmd serve --model site/public/models/s1-pico.gguf --port <another port>
```

- 18 requests were sent to both servers: 15 Flappy Drone states (choice + noul + score each), a
  6-option choice with empty and non-empty descriptions plus a noul with custom `criteria` and a
  5-level score, a JSON-object state, and a 400-token state. The response shapes are identical
  (`model`, `answers` in question order, `type`, `choice`, `probabilities`, `confidence`, `score`,
  `legend`, `noul`, `usage`) and so are the `input_tokens` counts.
- Probabilities agree within **4.7e-4** (max over 103 values), and `choice` is the same in every
  answer. `skycmd decide` against `llama-server` on the sample request: 1.1e-4. The gap comes
  from ggml's CPU path, which rounds activations to F16 for the F16 matrices; the skycmd engine
  dequantizes them to F32 at load. The sky-infer golden test checks our F32 path against the
  PyTorch reference (oscar-1-17m).
- **Long states.** `llama-server` does not cut the state: a prompt above the context (256) fails
  with `400 exceed_context_size_error`. skycmd cuts the state to the room left after the head,
  as the Laya runtime (`rl_common.build_sequence`) does, and answers.
- **Validation.** Both reject a missing `state` and an unknown `type` with a 400 and the same
  message (skycmd prefixes it with `invalid request: `). `llama-server` also rejects a `score` question with fewer than 2 or more than 10
  levels; the skycmd engine currently accepts it (open issue in `sky-infer`).
- `GET /health` returns `{"status":"ok"}` on both. `GET /v1/models` on `skycmd serve` returns the
  OpenAI list with the llama.cpp extras (`architecture.output_modalities = ["decisions"]`,
  `meta.n_vocab`, `n_ctx_train`, `n_embd`, `n_params`, `size`) plus the skycmd model description.
  llama.cpp uses the file path as the model id; skycmd uses `general.name`.
- HTTP latency of the sample request (one client, keep-alive, 200 requests): `skycmd serve`
  median 0.79 ms (p90 1.5 ms); `llama-server -ngl 0` median 4.9 ms (p90 50 ms).
  `skycmd bench`: 659 us per 3-question request on one thread (about 4,500 decisions/s).

## (b) Ollama

```sh
OLLAMA_HOST=127.0.0.1:11555 OLLAMA_MODELS=$(mktemp -d) ollama serve &
cd packaging/ollama && cp ../../site/public/models/s1-pico.gguf .
OLLAMA_HOST=127.0.0.1:11555 ollama create skycmd-s1-pico -f Modelfile
curl http://127.0.0.1:11555/v1/systemone -d '{"model": "skycmd-s1-pico", "state": "...", "questions": {...}}'
```

`ollama create` imports the file and `ollama show` reports `architecture modern-bert`,
`quantization F16` and the capability `decision`. Ollama 0.40.3 has a `/v1/systemone` route, but
it answers `400 {"error":"unsupported decision encoding \"laya\""}`: its decision runner does not
implement the Laya head yet. Use `skycmd serve` or `llama-server` to serve the file.
The server started for the test (and its temporary model store) was stopped and removed.

## (c) LM Studio

The `lms` CLI is installed (LM Studio 0.4.26), but `lms runtime ls` lists no inference runtime,
so no GGUF can be loaded on this machine and the model was **not tested** in LM Studio.
`lms import --dry-run` accepts the file. LM Studio's server has no `/v1/systemone` endpoint, so
even with a llama.cpp runtime that knows `modern-bert`, it could not answer decision requests.
See `packaging/lmstudio/README.md`.

## (d) wllama

wllama 3.8.1 (`@wllama/wllama`, libllama `b11364-46ca246`) was driven in headless Chrome by
puppeteer-core, from a temporary directory:

```js
import { Wllama } from "@wllama/wllama";
const w = new Wllama({ default: "/node_modules/@wllama/wllama/esm/wasm/wllama.wasm" });
await w.loadModelFromUrl("https://maxgfr.github.io/skycmd/models/s1-pico.gguf", { n_gpu_layers: 0, n_ctx: 256 });
const r = await w.createSystemOne({ state: "...", questions: { ... } });
```

The response has the same shape and agrees with skycmd within 1.1e-4 (same choices). It reports
`model` as wllama's internal file name. wllama runs its engine in a Web Worker and needs a
browser (or a worker-capable runtime); it was not tried in Node.

## The skycmd JS/WASM package

`packages/skycmd-js` (npm `skycmd`) was tested with `test/smoke.mjs` on Node 18.20.8, Node
24.10.0, Deno 2.9.7, Bun 1.4.3 and headless Chrome. On all of them, its probabilities match
`skycmd decide` within 1e-5 on the sample request (f32 on both sides; the SIMD128 kernels sum in a
different order). `decideRaw` takes about 145 us per 2-option question on s1-pico in Node, Deno
and Bun.

## Reproduce

- llama.cpp: build the PR #29818 branch (or a later master), then run the commands of (a).
- `cargo test -p sky-serve --release` checks the server on a random port with a random-init s1
  GGUF; `node packages/skycmd-js/test/smoke.mjs` checks the npm package;
  `python crates/sky-py/tests/smoke.py` checks the wheel.
