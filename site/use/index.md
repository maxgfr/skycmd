---
title: Use it for real
description: "Which skycmd files can be used in a real project outside the games, and how: the end-of-turn detector s6-ear2-nano and the tool router s6-route, with working code, their limits, and when BM25 or a larger model is the better choice."
outline: [2, 3]
---

# Use it for real

Most files skycmd publishes are results or demos. Two can go into a real project outside the games: **s6-ear2-nano**, which tells a voice agent when the speaker has finished, and **s6-route**, which picks a tool from a list you write. A third, s6-secrets-verify-pico, already runs inside a real tool, secretgate, as an experiment you have to switch on. This page says which file fits which job, shows the code, and says when not to use them.

Each model was judged by a **gate**: a bar written down in a pre-registration before the model was trained, and read once on test data. A model that misses its gate is still published, as a result. The numbers below are copied from [Results](/results/), the model cards and the result files in [`runs/`](https://github.com/maxgfr/skycmd/tree/main/runs).

## Which file for which job {#which-file}

| tier | model | the job | why this tier |
|---|---|---|---|
| **Use it in a project, with a safety net** | [s6-ear2-nano](#end-of-turn) | End of turn for a voice agent: has the speaker finished? | It passed its gate. On real conversations it misses about half of the real ends of turn at its cautious threshold, so it needs a fallback timer next to it. |
| | [s6-route](#tool-routing) | Pick a tool from a list written at run time, or answer "none of these tools", or "ask the user". | It passed its gate on fresh requests, but only under [conditions](#s6-route-or-bm25): distinctive tool names, tools of familiar kinds. Otherwise take BM25 or MiniLM. |
| **Already used in a real tool, as an opt-in experiment** | [s6-secrets-verify-pico](#secrets) | Judge whether a value found by a secret-scanning rule is a real credential. | It **failed** its pre-registered gate and stays a published result. secretgate (from version 1.9.0) offers it off by default, and it can only add redactions to what the rules find. |
| **Games and demos** | s3-pico, s6-nano, s6-micro; the earlier s1-pico, s2-nano and s2-nano-doom | Play the browser games from the game state; answer the typed questions of the demos. | Not for free-text questions in a product. s6-nano's own gate was too low: answering each question with its most common training answer scores 0.4815, against s6-nano's 0.4805. s6-micro: expect mistakes on free-text questions, 0.401 on When2Call and 0.092 on NOTAM subjects. s3-pico (0.382 on the typed questions) stays below the majority class (0.398). s1-pico and the s2 files were replaced by s3-pico. |
| **Playground** | s5-lite | Answers questions you write yourself, each option defined in plain words. | No gate was fixed for this file. 26% of its answers flip when the definitions are paraphrased, and four test sets fall below their majority class. |
| **Results, not for use** | every other file | | Failed gates and replaced versions, each published with its numbers: see [Models](/models/#results). |

The [Models](/models/) page has a card for every file, with its status and its key number.

## End of turn in a voice agent: s6-ear2-nano {#end-of-turn}

A voice agent has to know when you have finished speaking. **End of turn** is that moment. Answer too early and the agent cuts you off; answer too late and it feels slow. A **VAD** (voice activity detector) only tells speech from silence: it cannot tell a thinking pause from the end of a sentence. An end-of-turn model listens to the last seconds of audio and gives a probability that the turn is finished.

**s6-ear2-nano** is a 631,936-byte file. On the 31,527 test clips of Smart Turn v3.2, the open model it learned from (its teacher), it is right on **0.8993**; the teacher scores 0.9260 on the same clips. It takes **2.39 ms** per decision in WebAssembly. Its gate, with every number: [Results](/results/#s6-ear2-nano-passed) and [`runs/s6-ear2-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear2-nano/GATE.md).

::: warning Its limit: real conversations
On real conversational speech ([LiveKit's eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data), 17,934 cuts that the design never read) both models are far weaker than on the test clips: AUC 0.7299 for s6-ear2-nano against 0.7838 for its teacher. At its cautious threshold τ₈₅, s6-ear2-nano answers "finished" on only **0.5253** of the real ends of turn. So a **fallback timer is required**: without it, a speaker who has finished could wait for an answer that never comes.
:::

The probability is **calibrated**: a calibrated model that says 0.9 is right about 9 times in 10. Its calibration error (ECE, the average gap between what it says and how often it is right) is 0.0160 on the test clips. The file stores two thresholds fitted on validation: `tau50()` (0.5, the accuracy reading) and `tau85()` (0.7244, the highest threshold that still catches 85% of the finished turns on validation; it cuts fewer people off).

**When to take Smart Turn instead**, in the words of the gate report: "Take v2 when size and speed matter: it gives up 2.7 points of accuracy against Smart Turn for a file of 0.63 MB instead of 8.7 MB and about 2.4 ms per decision in WASM instead of about 32 ms natively; take [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)'s file when those points matter more." Its successor, s6-ear3-nano, [failed its gate](/results/#s6-ear3-nano-failed): take s6-ear2-nano.

### The policy

The [Listen](/ear/) demo runs the model with this policy (the defaults of `site/.vitepress/theme/lib/turn.ts`, from the operating notes of the pre-registration):

1. Your VAD marks each 10 ms frame as speech or silence. Every frame goes into the detector with `push`, speech or not.
2. After at least **150 ms** of speech, a pause starts to count.
3. After **200 ms** of silence, ask the model once (`probEnd()`). At or above the threshold (τ₈₅ by default), the turn is finished.
4. Below it, keep listening and ask again every **200 ms** while the pause lasts. If speech resumes, it is the same turn.
5. After **2,500 ms** of silence, the fallback timer ends the turn, whatever the model says.
6. When a turn ends, call `reset()`: the next speech starts a new turn. The demo also calls `reset()` when it starts listening.

### The code

The browser or Deno (`deno run --allow-net --allow-import`), with the module this site deploys. The detector reads 16 kHz mono audio as a `Float32Array` in [−1, 1]; if your audio comes at another rate, resample it first (the Listen demo uses a linear resampler).

```js
import init, { TurnDetector } from 'https://maxgfr.github.io/skycmd/lib/sky_infer.js'

const SITE = 'https://maxgfr.github.io/skycmd'
await init({ module_or_path: `${SITE}/lib/sky_infer_bg.wasm` })
const gguf = new Uint8Array(await (await fetch(`${SITE}/models/s6-ear2-nano.gguf`)).arrayBuffer())
const det = new TurnDetector(gguf)

const TAU = det.tau85()          // the file's cautious threshold
const MIN_SPEECH_MS = 150        // speech needed before a pause counts
const STOP_MS = 200              // silence before the model is asked
const RECHECK_MS = 200           // asked again while the pause lasts
const FALLBACK_MS = 2500         // the timer ends the turn, whatever the model says

let state = 'waiting'            // 'waiting', 'speaking' or 'holding'
let speechMs = 0, silenceMs = 0, sinceAskMs = 0

// Call it for every 10 ms frame (160 samples at 16 kHz, mono, Float32 in [-1, 1]),
// with your VAD's answer for that frame. Returns 'model' or 'timer' when the turn ends.
function onFrame(frame, isSpeech) {
  det.push(frame)
  if (isSpeech) {
    silenceMs = sinceAskMs = 0
    if (state === 'holding') state = 'speaking'  // speech resumed: the same turn goes on
    else if (state === 'waiting' && (speechMs += 10) >= MIN_SPEECH_MS) state = 'speaking'
    return null
  }
  speechMs = 0
  if (state === 'waiting') return null           // silence before any speech asks nothing
  silenceMs += 10
  sinceAskMs += 10
  if (silenceMs >= FALLBACK_MS) return endTurn('timer')
  const due = state === 'speaking' ? silenceMs >= STOP_MS : sinceAskMs >= RECHECK_MS
  if (!due) return null
  sinceAskMs = 0
  if (det.probEnd() >= TAU) return endTurn('model')
  state = 'holding'                              // not finished: keep listening
  return null
}

function endTurn(by) {
  state = 'waiting'
  speechMs = silenceMs = sinceAskMs = 0
  det.reset()                                    // the next speech starts a new turn
  return by
}
```

`isSpeech` comes from your VAD, for example [Silero VAD](https://github.com/snakers4/silero-vad) or WebRTC VAD. To try the code without one, a crude energy check works on clean audio: a frame is speech when its RMS is above a fixed level (`Math.sqrt(frame.reduce((s, x) => s + x * x, 0) / frame.length) > 0.01`). The Listen demo uses an energy VAD that tracks the noise floor (`EnergyVad` in `lib/turn.ts`).

In **Node**, an `https:` import does not work: download `sky_infer.js` and `sky_infer_bg.wasm` from `/lib/`, and the `.gguf`, then give the bytes:

```js
import { readFile } from 'node:fs/promises'
import init, { TurnDetector } from './sky_infer.js'

await init({ module_or_path: await readFile('./sky_infer_bg.wasm') })
const det = new TurnDetector(await readFile('./s6-ear2-nano.gguf'))
```

The rest of the API: `probEndPcm(clip)` gives one decision on a whole clip (its last 4 s) without touching the stream; `hopCount()` is the number of 10 ms hops kept (at most 400, the last 4 s); `info()` returns the file's name, size, temperature and thresholds as JSON. The class is `TurnDetector` in [`crates/sky-infer/src/wasm.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/wasm.rs).

## Tool routing: s6-route {#tool-routing}

An assistant with tools has to choose, for each request, which tool to call, or none. **s6-route** does that from a list you write at run time: up to 7 tools, each a name and one sentence (`name: description`). It adds the option "none of these tools fits the request" itself. It gives a probability to every option and answers one of three things: a tool, `none`, or `ask` (ask the user) when its top probability is below the threshold τ_ask3 = 0.671047.

It is 3.7 MB in two files: `s6-route-nano.gguf` (2.4 MB, the model part) and `s6-route-embed.gguf` (1.3 MB, the embedding part). With the tool list cached, it takes 30.9 ms per request in WebAssembly (measured under Node on a machine in use). Numbers: [Results](/results/#s6-route-passed) and [`runs/s6-nano3/eval-fresh/results.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-nano3/eval-fresh/results.md). Try your own list on the [Router](/router/) page.

```js
import init, { ToolRouter } from 'https://maxgfr.github.io/skycmd/pkg/sky_web.js'

const SITE = 'https://maxgfr.github.io/skycmd'
await init({ module_or_path: `${SITE}/pkg/sky_web_bg.wasm` })
const bytes = async (file) => new Uint8Array(await (await fetch(`${SITE}/models/${file}`)).arrayBuffer())
const router = new ToolRouter(await bytes('s6-route-nano.gguf'), await bytes('s6-route-embed.gguf'))

// Up to 7 tools, each "name: description" (one sentence). Do not add "none": the router does.
const tools = JSON.stringify([
  'lights_on: Turns the lights on in a room.',
  'get_weather: Gives the weather forecast for a city.',
  'send_email: Sends an email to a contact.',
])

const r = JSON.parse(router.route('turn on the hall lights', tools, false)) // true: fast mode
r.decision   // 'tool', 'none' or 'ask' (r.top below router.tauAsk(), 0.671047)
r.answer     // the top option's name ('lights_on'), or 'none'
r.probs      // one probability per tool, then one for "none"
```

With these three tools, `turn on the hall lights` gives `tool` `lights_on`, `will it rain in Paris tomorrow?` gives `tool` `get_weather`, and `what is the capital of Peru?` gives `none`. The tool list is encoded once and cached: the next request against the same list reuses it. The fast mode (`true`) stops the model part early: 0.8836 on the fresh test instead of 0.8868, about 10.5 ms per request. In Node, load the files as in the [end-of-turn example](#end-of-turn) (`pkg/sky_web.js` and `pkg/sky_web_bg.wasm`). The class is `ToolRouter` in [`crates/sky-web/src/router.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-web/src/router.rs).

### s6-route or BM25? {#s6-route-or-bm25}

**BM25** is a classic search score. For each tool, it counts the words the request shares with the tool's name and description, and gives more weight to rare words. It needs no training and no file: 0 bytes. Plain BM25 gives scores, not probabilities, and has no "none" answer. In the project's tests, the BM25 baseline (`bm25s`, Lucene variant, over the tools of each list, the request as the query) gets the "none" option as one more score, a fixed value θ: it answers "none" when no tool scores above θ. θ was chosen on the validation lists to maximise accuracy, and the scores were turned into probabilities by a softmax with a temperature fitted on validation ([`scripts/route-baselines.py`](https://github.com/maxgfr/skycmd/blob/main/scripts/route-baselines.py), the "bm25" system; its fitted values come from validation and did not change for the test, [`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md) §3). It has no "ask the user" answer.

**s6-route** is two parts whose probabilities are multiplied option by option:

`p(i) ∝ (p_model(i) · p_embedding(i))^0.81`

- the **model part**, s6-route-nano, a small trained model that reads the request, the tools, and each tool's BM25 score and rank in its input;
- the **embedding part**, a 4-bit copy of [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M), a static embedding: a table of word vectors that places texts of similar meaning close together, so it matches "rain" with "weather" where BM25 sees no shared word.

What it adds to BM25: its probabilities are better **calibrated** (ECE 0.0162 on the fresh test, against 0.0779 for BM25's fitted ones), it answers "none" more reliably (F1 0.7732 against 0.6368), and it can answer "ask the user".

On 945 fresh requests about tools never seen in training (the test called T2-tools):

| system | size | right option, T2-tools | tool names masked (`tool_<k>`) | MCP tools, a new family (2,000 fresh requests) | ECE, T2-tools | "none" F1, T2-tools |
|---|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2, cosine | 91.6 MB | **0.9048** | 0.9005 | **0.6630** | 0.0254 | 0.8010 |
| **s6-route** | 3.7 MB | 0.8868 | 0.2497 | 0.5530 | **0.0162** | 0.7732 |
| BM25 | 0 bytes | 0.8169 | 0.8138 | 0.5675 | 0.0779 | 0.6368 |
| its model part alone | 2.4 MB | 0.7926 | 0.0720 | | 0.0234 | 0.6220 |
| its embedding part alone | 1.3 MB | 0.7386 | 0.7280 | | 0.0243 | 0.5480 |

The MCP column is a later test, read for s6-route's retrained successor, [s6-route7-nano](/results/#s6-route7-nano-failed), which failed its own gate; blank cells were not reported there.

**all-MiniLM-L6-v2** (MiniLM for short) is itself a small model: a sentence encoder of 91.6 MB that turns a text into one vector, compared here by cosine, not a large language model. s6-route comes close to it (−0.0180 [−0.0381, +0.0032] on T2-tools, for 25 times fewer bytes) only on tools of familiar kinds with distinctive names. On the MCP tools, a new family, MiniLM reads 0.6630 on the same requests where s6-route reads 0.5530, and BM25, at 0.5675, is level with s6-route or slightly above it (no interval was computed for that difference).

The gate passed, but the advantage over BM25 is fragile:

- **It leans on three tool families.** Without the three largest families of the test, the gain over BM25 is +0.0202 [−0.0062, +0.0481]. Resampling families instead of requests (a cluster bootstrap) gives +0.0698 [−0.0079, +0.1404]. Both intervals hold zero.
- **It needs distinctive tool names.** With every tool renamed `tool_<k>`, s6-route falls to 0.2497, while BM25 keeps 0.8138.
- **On tools of a new family it does not beat BM25.** On 2,000 fresh requests about the tools of 262 MCP servers (MCP, the Model Context Protocol, is a common way to plug tools into an assistant), a family no training data held, s6-route reads 0.5530 against BM25's 0.5675.

The project's own conclusion ([the article](/articles/a-3-7-mb-tool-router)): on tools of a new family, the honest advice is BM25, or MiniLM when 92 MB are affordable.

**A decision rule:**

- **s6-route** when your tools have distinctive names, are of familiar kinds (lights, weather, email, calendar…), and you want "none" and "ask the user" with probabilities you can trust.
- **BM25** when the tool names are generic or numbered, or the tools are a new family (MCP servers, for example). Fit its "none" threshold on a few examples of your own.
- **all-MiniLM-L6-v2** when accuracy matters most and about 92 MB per client are fine.
- Whatever you pick, **keep BM25 as a cross-check**: it costs almost nothing, and when it disagrees with the router the request is worth a second look, or a question to the user.

## Secrets: through secretgate {#secrets}

[secretgate](https://github.com/maxgfr/secretgate) is a tool that redacts secrets in what a coding agent reads and writes. Since version 1.9.0 it can run **s6-secrets-verify-pico** after its rules, as an experimental option called the verifier: a rule proposes a candidate value, and the model answers whether that value is a real credential.

::: danger It failed its pre-registered gate
It had to beat the best rule scanner by 0.15 of line F1 over all the test repositories, and reached +0.123. It stays a published result, not a validated detector ([Results](/results/#s6-secrets-verify-pico-failed)). secretgate offers it **off by default**, and it can **only add redactions**: the rules decide first and their result never changes.
:::

What secretgate's [verifier page](https://github.com/maxgfr/secretgate/blob/main/docs/verifier.md) reports:

- **Line F1 0.856** on 221 real test repositories (Samsung CredData), against 0.733 for the best rule scanner and 0.704 for Gitleaks. F1 balances the secrets found against the false alarms.
- **False redactions** on 8,825 clean real files: something is redacted in 0.46% of them without a file path (the output of a shell command) and 0.27% with one (a file read).
- **Time.** It runs in a child process killed at 1 s, and adds at most about 1.25 s to a hook (median 44 to 59 ms measured). It never blocks a prompt, and any failure leaves the rules' result unchanged.

To switch it on:

```sh
secretgate verifier fetch     # download the four data files (pinned, sha256-checked)
secretgate verifier on        # threshold f1 (default); or --threshold f05 | f2
secretgate verifier           # status
secretgate verifier off
```

More: the [model card](https://huggingface.co/maxgfr/s6-secrets-verify-pico) and the [Secrets verifier](/secrets-verify/) page, which runs the model in your browser. Its first version, s6-secrets-pico, is **not a scanner** on its own: it flagged 59.8% of clean real files at p ≥ 0.9 ([Results](/results/#s6-secrets-real-text)).

## In production {#in-production}

- **Host the files yourself.** The examples above load the `.js`, `.wasm` and `.gguf` files from this site to stay short. These files change with each deploy of the site, and the JavaScript glue must match its `.wasm`. Copy the ones you use into your project and serve them with it.
- **Check each `.gguf`** against its sha256 in [`models/manifest.json`](https://maxgfr.github.io/skycmd/models/manifest.json) (for s6-route, the two files are under `files`), for example with `shasum -a 256 s6-ear2-nano.gguf`. The [Install](/install/#models) page lists where to download each file.
- **Other runtimes.** `TurnDetector` and `ToolRouter` are in the WebAssembly modules (`lib/sky_infer.js` has `TurnDetector`; `pkg/sky_web.js` has both) and in Rust: `sky_infer::turn` and `sky_web::router::Router` (`crates/sky-web/src/router.rs`), usable as a git dependency on this repository. Natively, `sky-turn` and `sky-route fuse` run them on files (the model cards show how). The Python wheel and the `skycmd` command line and HTTP server expose only the generic decision API (`Model.decide`, `POST /v1/systemone`), not these two.
- **Nothing is on npm or PyPI**: outside the browser, build from the repository ([Install](/install/)).

## What none of them is {#what-none-of-them-is}

- **Not flight software.** In the games the model only advises, and coded rules (geofence, ceiling, battery, collision reflex) keep a veto.
- **Not a standalone secret scanner.** The secrets verifier judges only the candidates that rules propose, and it failed its gate.
- **Not a general free-text classifier.** The models that answer typed questions (s6-nano, s6-micro, s5-lite) are demos and a playground, with the limits in the table [above](#which-file).
