---
title: Install
outline: [2, 3]
---

# Install and use

A skycmd model is one `.gguf` file. You send it a **state** and typed **questions**, and it
returns a probability for every allowed answer in one forward pass. The same JSON request works
everywhere on this page: in the browser, in Node, Deno and Bun, in Python, on the command line
and over HTTP.

| where | package | s1 (`modern-bert`) | s2 (`skycmd-s2`) |
|---|---|:-:|:-:|
| browser, Node >= 18, Deno, Bun | npm `skycmd` or `https://maxgfr.github.io/skycmd/lib/skycmd.js` | yes | yes |
| Python >= 3.9 | PyPI `skycmd` (wheel) | yes | yes |
| command line, HTTP server | `skycmd` binary (Homebrew, release archives, cargo) | yes | yes |
| llama.cpp `llama-server`, wllama | upstream | yes | no |
| Ollama, LM Studio | upstream | no `/v1/systemone` for this head yet | no |

s2 models run only in the skycmd engine. Tested versions and numbers are in
[docs/compat.md](https://github.com/maxgfr/skycmd/blob/main/docs/compat.md).

## Models

| model | parameters | file |
|---|---:|---|
| s1-pico | 0.2M | [s1-pico.gguf](https://maxgfr.github.io/skycmd/models/s1-pico.gguf) (545 KB) |

Every [GitHub release](https://github.com/maxgfr/skycmd/releases) also carries the model files
and a `SHA256SUMS` file.

```sh
curl -LO https://maxgfr.github.io/skycmd/models/s1-pico.gguf
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

The response keeps the question order:

```json
{
  "model": "skycmd-s1-pico",
  "answers": {
    "action": {
      "type": "choice",
      "choice": "glide",
      "probabilities": { "thrust": 0.0597, "glide": 0.9403 },
      "confidence": 0.8806
    },
    "danger": { "type": "noul", "noul": 0.9534 },
    "margin": {
      "type": "score",
      "score": 1.8128,
      "legend": { "0": "none", "1": "tight", "2": "comfortable" },
      "probabilities": { "0": 0.0399, "1": 0.1073, "2": 0.8527 },
      "confidence": 0.7192
    }
  },
  "usage": { "input_tokens": 193, "output_tokens": 0 }
}
```

`choice` is the most likely key, `confidence = (pmax - 1/n) / (1 - 1/n)`, `score` is the expected
level and `noul` is the probability of "yes". s2 models also report the early exit they took.

## Browser

No install and no bundler: import the module from this site. The WebAssembly engine (about
0.6 MB, SIMD128) loads from the same folder.

```html
<script type="module">
  import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";

  const model = await SkyCmd.load("https://maxgfr.github.io/skycmd/models/s1-pico.gguf");
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

For a game loop, `decideRaw` skips the JSON and caches the question and option tokens, so only
the state is tokenized on each frame. It takes rendered options (`"key: description"`, or
`["false: ...", "true: ..."]` for noul) and returns their probabilities in order:

```js
const [pThrust, pGlide] = model.decideRaw(
  "choice",
  "Which input keeps the drone flying through the next gap?",
  ["thrust: fire the rotors for an upward impulse", "glide: let gravity pull the drone down"],
  `vy ${vy} dx ${dx} up ${up} dn ${dn} nxt ${nxt} alt ${alt}`,
);
```

## npm

```sh
npm install skycmd
```

```js
import SkyCmd from "skycmd";

const model = await SkyCmd.load(new URL("./s1-pico.gguf", import.meta.url));
```

The package is plain ESM with TypeScript declarations (`DecisionRequest`, `DecisionResponse`,
`Answer`, ...). Bundlers (Vite, webpack, esbuild) pick up `sky_infer_bg.wasm` through
`new URL(..., import.meta.url)`; if yours does not, serve the file yourself and pass its URL:
`SkyCmd.load(model, { wasm: "/assets/sky_infer_bg.wasm" })`.

## Node, Deno and Bun

The same package runs server-side. A plain path is read from disk.

::: code-group

```js [Node >= 18]
// npm install skycmd
import SkyCmd from "skycmd";

const model = await SkyCmd.load("./s1-pico.gguf");
const r = model.decide({
  state: "alt 12 vy -3",
  questions: { climb: { type: "noul", instructions: "Should the drone climb now?" } },
});
console.log(r.answers.climb.noul);
```

```js [Deno]
// deno run --allow-read --allow-net main.js
import SkyCmd from "npm:skycmd";
// or, without npm: import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";

const model = await SkyCmd.load("./s1-pico.gguf");
const r = model.decide({
  state: "alt 12 vy -3",
  questions: { climb: { type: "noul", instructions: "Should the drone climb now?" } },
});
console.log(r.answers.climb.noul);
```

```js [Bun]
// bun add skycmd && bun main.js
import SkyCmd from "skycmd";

const model = await SkyCmd.load("./s1-pico.gguf");
const r = model.decide({
  state: "alt 12 vy -3",
  questions: { climb: { type: "noul", instructions: "Should the drone climb now?" } },
});
console.log(r.answers.climb.noul);
```

:::

Call `model.free()` when you are done with a model to release its WebAssembly memory.

## Python

```sh
pip install skycmd        # or: uv add skycmd
```

One abi3 wheel per platform covers CPython 3.9 and later (macOS arm64 and x86_64, manylinux
x86_64 and aarch64, Windows x86_64). The wheels are also attached to each GitHub release.

```python
import skycmd

model = skycmd.Model("s1-pico.gguf")      # a path, or the file content as bytes
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

### Homebrew (macOS, Linux)

```sh
brew install maxgfr/tap/skycmd
```

The formula also installs `s1-pico.gguf` under `$(brew --prefix)/share/skycmd/models/`.

### Release binaries

Download the archive for your platform from the
[releases](https://github.com/maxgfr/skycmd/releases): `skycmd-<version>-<target>.tar.gz` for
`aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`,
`aarch64-unknown-linux-gnu`, and a `.zip` for `x86_64-pc-windows-msvc`. Check it against
`SHA256SUMS`:

```sh
v=0.1.0; t=aarch64-apple-darwin
curl -LO https://github.com/maxgfr/skycmd/releases/download/v$v/skycmd-$v-$t.tar.gz
curl -LO https://github.com/maxgfr/skycmd/releases/download/v$v/SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing
tar xzf skycmd-$v-$t.tar.gz && sudo mv skycmd-$v-$t/skycmd /usr/local/bin/
```

### From source

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve   # installs the `skycmd` binary
```

### Commands

```sh
skycmd decide --model s1-pico.gguf --request req.json --pretty   # one request from a file
cat req.json | skycmd decide --model s1-pico.gguf -               # or from stdin
skycmd info s1-pico.gguf                                          # sizes, temperatures (JSON)
skycmd bench s1-pico.gguf                                         # latency on a 3-question frame
```

## HTTP server

```sh
skycmd serve --model s1-pico.gguf --port 8080
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
them without conversion. s2 files do not run there.

### llama.cpp

```sh
llama-server -m s1-pico.gguf -ngl 0 --port 8080
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d @req.json
```

Probabilities match the skycmd engine within about 5e-4 (llama.cpp rounds activations to F16 on
the CPU). One difference: llama.cpp rejects a request whose state does not fit in the context
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
