---
title: Install
outline: [2, 3]
---

# Install and use

A skycmd model is one `.gguf` file. You send it a **state** and typed **questions**, and it
returns a probability for every allowed answer in one forward pass. The same JSON request works
everywhere on this page: in the browser, in Node, Deno and Bun, in Python, on the command line
and over HTTP.

| where | package | s1 (`modern-bert`) | s2 and s3 (`skycmd-s2`) |
|---|---|:-:|:-:|
| browser, Deno | `https://maxgfr.github.io/skycmd/lib/skycmd.js` | yes | yes |
| Node >= 18, Bun | `packages/skycmd-js`, built from source | yes | yes |
| Python >= 3.9 | `crates/sky-py`, built from source | yes | yes |
| command line, HTTP server | `skycmd` binary, built from source with cargo | yes | yes |
| llama.cpp `llama-server`, wllama | upstream | yes | no |
| Ollama, LM Studio | upstream | no `/v1/systemone` for this head yet | no |

Nothing is published to npm, PyPI or Homebrew yet, and there are no prebuilt binaries: outside
the browser, build from this repository as shown below.

s2 and s3 models, including the default s3-pico, run only in the skycmd engine (browser, Node,
Deno, Bun, Python, CLI and server). Only s1 files run in llama.cpp and wllama. Tested versions
and numbers are in [docs/compat.md](https://github.com/maxgfr/skycmd/blob/main/docs/compat.md).

## Models

| model | architecture | parameters | file | runs in |
|---|---|---:|---|---|
| **s3-pico** (default) | s3, ternary | 0.45M | [s3-pico.gguf](https://maxgfr.github.io/skycmd/models/s3-pico.gguf) (251 KB) | skycmd engine only |
| s1-pico | s1, f16 | 0.2M | [s1-pico.gguf](https://maxgfr.github.io/skycmd/models/s1-pico.gguf) (545 KB) | skycmd engine, llama.cpp, wllama |
| s2-nano | s2, ternary | 1.5M | [s2-nano.gguf](https://maxgfr.github.io/skycmd/models/s2-nano.gguf) (702 KB) | skycmd engine only |
| s2-nano-doom | s2, ternary | 1.5M | [s2-nano-doom.gguf](https://maxgfr.github.io/skycmd/models/s2-nano-doom.gguf) (702 KB) | skycmd engine only |
| s4-micro (experimental) | s2, ternary | 5.6M | [s4-micro.gguf](https://maxgfr.github.io/skycmd/models/s4-micro.gguf) (2.4 MB) | skycmd engine only |
| s5-lite | iso, q8_0 | 17M | [s5-lite.gguf](https://maxgfr.github.io/skycmd/models/s5-lite.gguf) (20.7 MB) | skycmd engine only |

Every model is also on Hugging Face with its model card and a demo Space that runs in the browser:
[s5-lite](https://huggingface.co/maxgfr/s5-lite), [s4-micro](https://huggingface.co/maxgfr/s4-micro),
[s3-pico](https://huggingface.co/maxgfr/s3-pico), [s1-pico](https://huggingface.co/maxgfr/s1-pico),
[s2-nano](https://huggingface.co/maxgfr/s2-nano) and [s2-nano-doom](https://huggingface.co/maxgfr/s2-nano-doom).

s3-pico plays the four games of the [Play](/play/) page at the oracle's level
([chapter 21](/tutorial/21-s3-recursive-tiny)).

```sh
curl -LO https://maxgfr.github.io/skycmd/models/s3-pico.gguf
curl -LO https://maxgfr.github.io/skycmd/models/s1-pico.gguf   # for llama.cpp or wllama
```

## The request

This is the body of `POST /v1/systemone`, the same format as llama.cpp's System One server.

```json
{
  "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Which input keeps the drone flying through the next gap?",
      "criteria": {
        "thrust": "fire the rotors for an upward impulse",
        "glide": "let gravity pull the drone down"
      }
    },
    "danger": {
      "type": "noul",
      "instructions": "Will the drone crash within the next half second if it keeps gliding?"
    },
    "margin": {
      "type": "score",
      "instructions": "How much room is left above and below the drone?",
      "criteria": ["none", "tight", "comfortable"]
    }
  }
}
```

- `choice`: `criteria` maps option keys to descriptions (an empty description shows the key only).
- `score`: `criteria` lists 2 to 10 levels, from lowest to highest.
- `noul`: a yes/no question. `criteria` is optional: `{"false": "...", "true": "..."}`.

The response keeps the question order. This is s3-pico's answer (`skycmd decide`, rounded to
4 decimals):

```json
{
  "model": "skycmd-s3-pico",
  "answers": {
    "action": {
      "type": "choice",
      "choice": "thrust",
      "probabilities": { "thrust": 0.5190, "glide": 0.4810 },
      "confidence": 0.0381,
      "exit": 3,
      "layers_used": 4
    },
    "danger": { "type": "noul", "noul": 0.0563, "exit": 0, "layers_used": 1 },
    "margin": {
      "type": "score",
      "score": 1.1346,
      "legend": { "0": "none", "1": "tight", "2": "comfortable" },
      "probabilities": { "0": 0.0436, "1": 0.7782, "2": 0.1782 },
      "confidence": 0.6673,
      "exit": 0,
      "layers_used": 1
    }
  },
  "usage": { "input_tokens": 164, "output_tokens": 0 }
}
```

`choice` is the most likely key, `confidence = (pmax - 1/n) / (1 - 1/n)`, `score` is the expected
level and `noul` is the probability of "yes". s2 and s3 models also report, for each answer, the
early exit they took (`exit`, counted from 0) and the state layers it used (`layers_used`):
here the two confident answers stopped after the first pass of s3-pico's looped block, and the
close `action` call went to the last exit. s1 answers have neither field.

## Browser

No install and no bundler: import the module from this site. The WebAssembly engine (about
0.7 MB, 0.3 MB gzipped, SIMD128) loads from the same folder.

```html
<script type="module">
  import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";

  const model = await SkyCmd.load("https://maxgfr.github.io/skycmd/models/s3-pico.gguf");
  const r = model.decide({
    state: "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
    questions: {
      action: {
        type: "choice",
        instructions: "Which input keeps the drone flying through the next gap?",
        criteria: {
          thrust: "fire the rotors for an upward impulse",
          glide: "let gravity pull the drone down",
        },
      },
    },
  });
  console.log(r.answers.action.choice, r.answers.action.probabilities);
</script>
```

`SkyCmd.load` accepts a URL, a `Response`, a `Blob` (for example from a file input), an
`ArrayBuffer` or a `Uint8Array`.

For a game loop, `decideRaw` skips the JSON and caches the question and option tokens (s2 and s3
models also cache their encoding), so only the state is processed on each frame. It takes rendered options (`"key: description"`, or
`["false: ...", "true: ..."]` for noul) and returns their probabilities in order:

```js
const [pThrust, pGlide] = model.decideRaw(
  "choice",
  "Which input keeps the drone flying through the next gap?",
  ["thrust: fire the rotors for an upward impulse", "glide: let gravity pull the drone down"],
  `vy ${vy} dx ${dx} up ${up} dn ${dn} nxt ${nxt} alt ${alt}`,
);
```

## Node, Deno and Bun

Deno imports the browser module directly:

```js
// deno run --allow-read --allow-net main.js
import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";

const model = await SkyCmd.load("./s3-pico.gguf");
const r = model.decide({
  state: "alt 12 vy -3",
  questions: { climb: { type: "noul", instructions: "Should the drone climb now?" } },
});
console.log(r.answers.climb.noul);
```

Node and Bun need the package built from the repository (Rust and
[wasm-pack](https://github.com/rustwasm/wasm-pack)):

```sh
git clone https://github.com/maxgfr/skycmd && cd skycmd/packages/skycmd-js
npm run build && npm pack                # writes skycmd-0.1.0.tgz
cd /your/project && npm install /path/to/skycmd-0.1.0.tgz
```

Then `import SkyCmd from "skycmd"` works as above, and a plain path is read from disk. The package
is plain ESM with TypeScript declarations (`DecisionRequest`, `DecisionResponse`, `Answer`, ...).
Bundlers (Vite, webpack, esbuild) pick up `sky_infer_bg.wasm` through
`new URL(..., import.meta.url)`; if yours does not, serve the file yourself and pass its URL:
`SkyCmd.load(model, { wasm: "/assets/sky_infer_bg.wasm" })`.

Call `model.free()` when you are done with a model to release its WebAssembly memory.

## Python

Build the wheel from the repository (needs Rust, and pip >= 21.3 or uv):

```sh
pip install "skycmd @ git+https://github.com/maxgfr/skycmd#subdirectory=crates/sky-py"
```

```python
import skycmd

model = skycmd.Model("s3-pico.gguf")      # a path, or the file content as bytes
r = model.decide({
    "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Which input keeps the drone flying through the next gap?",
            "criteria": {
                "thrust": "fire the rotors for an upward impulse",
                "glide": "let gravity pull the drone down",
            },
        },
        "margin": {
            "type": "score",
            "instructions": "How much room is left above and below the drone?",
            "criteria": ["none", "tight", "comfortable"],
        },
    },
})
print(r["answers"]["action"]["choice"], r["answers"]["margin"]["score"])

# rendered options in, probabilities out (fast path for loops)
p = model.decide_raw("noul", "Should the drone climb now?",
                     ["false: no, the statement does not hold", "true: yes, the statement holds"],
                     "alt 12 vy -3")
```

`decide` also takes the request as JSON text, and `decide_json` returns JSON text. Calls release
the GIL, and one `Model` can be shared between threads.

## Command line

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve   # installs the `skycmd` binary
```

### Commands

```sh
skycmd decide --model s3-pico.gguf --request req.json --pretty   # one request from a file
cat req.json | skycmd decide --model s3-pico.gguf -               # or from stdin
skycmd info s3-pico.gguf                                          # sizes, temperatures, exit thresholds (JSON)
skycmd bench s3-pico.gguf                                         # latency on a 3-question frame
```

## HTTP server

```sh
skycmd serve --model s3-pico.gguf --port 8080
```

It serves `POST /v1/systemone`, `GET /health` and `GET /v1/models`, with the same request and
response as llama.cpp, so clients written for one work with the other. Options: `--host`
(default `127.0.0.1`), `--threads N` (one engine per worker), `--no-cors` (CORS is on by default
so pages can call it).

```sh
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  "questions": {
    "danger": {
      "type": "noul",
      "instructions": "Will the drone crash within the next half second if it keeps gliding?",
      "criteria": {"false": "the drone stays in the gap", "true": "the drone hits a wall or the ground"}
    }
  }
}'
```

Errors use HTTP 400 with `{"error": {"code": 400, "message": "...", "type": "invalid_request_error"}}`.

## llama.cpp, wllama, Ollama, LM Studio (s1 only)

s1 files use llama.cpp's `modern-bert` architecture with the `laya` decision head
([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818)), so recent llama.cpp builds read
them without conversion. s2 and s3 files (`skycmd-s2`: s3-pico, s2-nano, s2-nano-doom) do not
run there: use `s1-pico.gguf` with these runtimes.

### llama.cpp

```sh
llama-server -m s1-pico.gguf -ngl 0 --port 8080
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d @req.json
```

Probabilities match the skycmd engine within about 5e-4 (llama.cpp rounds activations to F16 on
the CPU; measured on the s1-pico checkpoint published before the DAgger rounds, `docs/compat.md`). One difference: llama.cpp rejects a request whose state does not fit in the context
(256 tokens), while skycmd cuts the state to fit, as the models were trained.

### wllama (browser)

```js
import { Wllama } from "@wllama/wllama";

const wllama = new Wllama({ default: "/path/to/@wllama/wllama/esm/wasm/wllama.wasm" });
await wllama.loadModelFromUrl("https://maxgfr.github.io/skycmd/models/s1-pico.gguf", {
  n_gpu_layers: 0,
  n_ctx: 256,
});
const r = await wllama.createSystemOne({
  state: "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  questions: {
    danger: {
      type: "noul",
      instructions: "Will the drone crash within the next half second if it keeps gliding?",
    },
  },
});
```

Tested with wllama 3.8.1 in Chrome. For a 0.5 MB model the skycmd package is a smaller download.

### Ollama

Ollama 0.40.3 imports the file (`ollama create` with
[packaging/ollama/Modelfile](https://github.com/maxgfr/skycmd/blob/main/packaging/ollama/Modelfile))
and lists the `decision` capability, but its `/v1/systemone` answers
`unsupported decision encoding "laya"`. Serve the file with `skycmd serve` or `llama-server`
instead.

### LM Studio

LM Studio can import the file (`lms import --copy s1-pico.gguf`), but its server has no
`/v1/systemone` endpoint; see
[packaging/lmstudio](https://github.com/maxgfr/skycmd/blob/main/packaging/lmstudio/README.md).
