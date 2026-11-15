# s6-route-nano: what the site's router playground needs from the engine

A note for item 7 of section 14 of `docs/s6-nano2-prereg.md` (the page itself is not written
yet). The page is published with its playground only if the gate passes (section 11); otherwise
it shows the result without the playground.

## What the engine already gives (`crates/sky-web`, `site/public/pkg`)

- **One call per decision.** `engine.decide("choice", INSTRUCTION, JSON.stringify(options), state)`
  returns `{probs, exit, n_exits, layers_used, static_hit, state_hit, micros}`. The page must send
  exactly what the model was trained on (section 3):
  - `INSTRUCTION` = `Which tool should handle the user's request?`;
  - each option `name: description`, the description cut to its first sentence (up to the first
    `.`, `!` or `?` followed by a space or the end) and to 30 words, as
    `sky_data::route::cut_description` does; with no description, the name alone;
  - the option `none of these tools fits the request`, always last;
  - `state` = `User: ` + the request.
- **The tool list is cached.** The static stream (instruction and options) is kept in an LRU of
  16 (`prefix_cache: 16`): editing a tool changes the key and the next call encodes the list
  again (`static_hit: false`, the "first call" latency); typing in the request box reuses it
  (`static_hit: true`, the per-request latency). `parity-node.mjs --router` measures both
  (section 7.4).
- **Full depth.** `engine.setExitThreshold(1.0)`: the published file already carries exit
  thresholds 1, but the call makes the page independent of the file.
- **Sizes.** `engine.info()` gives `head_max_len` (the head window of section 3), `max_len`, the
  parameter count and `file_bytes`; `engine.file_bytes()` the GGUF size for the first line.

## What it lacks

1. **tau_ask.** The threshold is stored in the GGUF under `skycmd-s2.route.tau_ask`, but
   `ModelInfo` (`crates/sky-infer/src/engine.rs`) exposes no custom metadata key. Either a getter
   (`engine.metadataF32(key)`) or the value copied into `site/public/models/manifest.json` when
   the file is published; the slider starts there, and "ask the user" shows when the top
   probability is below it.
2. **Truncation.** With 7 tools the head may exceed `head_max_len`, and the engine then shrinks
   every option evenly without saying so. A count of the static stream's tokens (or a
   `truncated` flag in the answer of `decide`) would let the page warn that a description is
   cut.
3. **The description cut** is in Rust in `sky-data`; the page needs the same rule in JavaScript
   (about 15 lines) or a helper exported by `sky-web`, so that what a visitor types is what the
   model was trained on.

## What is page logic only

- The editable list of up to 7 `name: description` lines, presets (smart home, developer
  tools, shop, a French agenda app: French requests against English tool text, which section 4.3
  trains and reports but does not promise), the request box, one probability bar per tool and one
  for *none*, updated as the visitor types.
- "Mask names": every name replaced by `tool_<k>` before the call; "swap descriptions": two
  tools' descriptions exchanged (the reading checks of 7.3, live).
- Latency: `micros` of the first call after an edit (tool list encoded) and of the following
  ones (cached).
- The first line: the verdict of the gate and its numbers (`runs/s6-nano2/eval-test/gate.json`),
  copied into the page when it is built.
- The model file served from `site/public/models/` (with its `manifest.json` entry), no other
  network call.

## Measured so far (no trained model yet)

On 200 validation slates (`v_tools`), with the published s6-nano v1 file (not a router; its head
window is 160) at full depth, SIMD128 build, Node, on a machine that was also running a GPU
training and a CPU baseline at the time: median 2,339 µs per request with the tool list cached,
5,070 µs for the first request, largest probability gap to the native engine 8.0e-7, every argmax
equal. These are plumbing numbers, not the G4 reading, which is taken on the trained file on an
idle machine.
