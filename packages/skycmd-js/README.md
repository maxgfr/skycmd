# skycmd (JavaScript / WebAssembly)

Run [skycmd](https://github.com/maxgfr/skycmd) decision models in the browser, Node >= 18, Deno
and Bun. skycmd models are tiny "System One" models: a bidirectional encoder with a typed decision
head that scores options in one forward pass, without generating any text. The package is the
Rust engine compiled to WebAssembly (SIMD128) plus a small ESM wrapper; it needs no bundler and
loads both `s1` (`modern-bert`) and `s2` (`skycmd-s2`) GGUF files.

```sh
npm run build && npm pack     # not on npm yet: build from this folder (Rust + wasm-pack), then npm install ./skycmd-0.1.0.tgz
```

```js
import SkyCmd from "skycmd";

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
    danger: {
      type: "noul",
      instructions: "Will the drone crash within the next half second if it keeps gliding?",
    },
  },
});
r.answers.action.choice; // "thrust" | "glide"
r.answers.danger.noul;   // p(yes)
```

Without npm, import it straight from the project site:

```html
<script type="module">
  import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";
  const model = await SkyCmd.load("https://maxgfr.github.io/skycmd/models/s1-pico.gguf");
</script>
```

## API

- `await SkyCmd.load(source, { q8?, wasm? })`: `source` is a URL, a file path (Node, Deno, Bun),
  a `Response`, a `Blob`, an `ArrayBuffer` or a `Uint8Array`.
- `model.decide(request)`: `request` is the body of `POST /v1/systemone` (object or JSON text);
  returns `{ model, answers, usage }`, the same JSON as the `skycmd serve` and llama.cpp servers.
- `model.decideRaw(type, instructions, options, state)`: probabilities of already rendered
  options (`"key: description"`, noul `["false: ...", "true: ..."]`), in order. The question and
  option tokens are cached, so a game loop that only changes `state` is the fastest path.
- `model.scores(...)`: same, before the temperature. `model.info`, `model.name`, `model.nParams`,
  `model.free()`.

Types are in `skycmd.d.ts`. Install guide: <https://maxgfr.github.io/skycmd/install/>.
License: Apache-2.0.
