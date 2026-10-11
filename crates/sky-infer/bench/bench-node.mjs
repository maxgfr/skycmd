// WASM latency of the sky-infer engine in Node (the same measurement as `sky-bench` natively):
// a Flappy-like game loop where the question and options stay the same and the state changes
// every frame (`<state> t N`, N = frame % 50). Build the module first:
//
//   RUSTFLAGS="-C target-feature=+simd128" wasm-pack build crates/sky-infer --target nodejs \
//     --release --out-dir <pkg> --no-typescript -- --features wasm,timing
//   (add +relaxed-simd to RUSTFLAGS for the relaxed_madd kernels)
//
//   node crates/sky-infer/bench/bench-node.mjs --pkg <pkg> [--iters 3000] [--repeat 5]
//        [--questions 1|2] [--exit-threshold T] [--no-static-cache] [--packed]
//        [--game flappy|doom] model.gguf...
//
// `--game doom`: a Doom state (`Doom::state_text`, 14 `key value` pairs), the action question
// with its 8 actions and the danger question (same frames as `sky-bench --game doom`).
//
// `--no-static-cache` cycles through 17 questions of the same length, one more than the LRU,
// so every question encodes its static stream (SkyEngine has no option for it). `--packed`
// keeps ternary matrices packed (add/sub kernels) through `SkyEngine.withOptions`.
// Prints one JSON line per model: median / p95 / min µs per frame (median over the repeats).
import { readFileSync } from 'node:fs'
import { basename, resolve } from 'node:path'
import { createRequire } from 'node:module'

const args = process.argv.slice(2)
let pkg
let iters = 3000
let repeat = 5
let questions = 1
let tau
let staticCache = true
let packed = false
let state = 'vy -2 dx 35 up 20 dn -5 nxt -18 alt 32'
let doom = false
const models = []
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--pkg') pkg = args[++i]
  else if (a === '--iters') iters = Number(args[++i])
  else if (a === '--repeat') repeat = Number(args[++i])
  else if (a === '--questions') questions = Number(args[++i])
  else if (a === '--exit-threshold') tau = Number(args[++i])
  else if (a === '--no-static-cache') staticCache = false
  else if (a === '--packed') packed = true
  else if (a === '--state') state = args[++i]
  else if (a === '--game') doom = args[++i] === 'doom'
  else models.push(a)
}
if (!pkg || !models.length) {
  console.error('usage: bench-node.mjs --pkg <wasm-pack nodejs dir> [--iters N] [--repeat R] [--questions 1|2] [--exit-threshold T] [--no-static-cache] [--packed] [--state TEXT] model.gguf...')
  process.exit(2)
}
const { SkyEngine } = createRequire(import.meta.url)(resolve(pkg, 'sky_infer.js'))

const INS = doom
  ? 'Which action should the player take now to survive, clear the level and reach the exit?'
  : 'Which action keeps the bird alive?'
const OPTS = doom
  ? ['forward: move forward', 'turn_left: turn to the left', 'turn_right: turn to the right',
      'strafe_left: sidestep to the left', 'strafe_right: sidestep to the right',
      'shoot: fire the weapon at the crosshair', 'use: open the door or press the switch',
      'back: step backward']
  : ['flap: push the bird up', 'wait: let the bird fall']
const DANGER = doom
  ? 'Will the player take damage in the next second if nothing changes?'
  : 'Will the drone crash within half a second if it keeps gliding?'
const DANGER_OPTS = ['false: no, the statement does not hold', 'true: yes, the statement holds']
// the frames of sky-bench: `<state> t N`, or the Doom state of frame i
const stateOf = (i) =>
  doom
    ? `ea ${(i % 7) - 3} ed ${5 + (i % 4)} al ${i % 2} en ${1 + (i % 3)} hp 8 am ${20 - (i % 5)} wf ${6 + (i % 3)} wl 2 wr 3 dr 0 gb ${4 - (i % 9)} gd 12 pb 0 pd 0`
    : `${state} t ${i % 50}`
const pct = (s, q) => s[Math.round((s.length - 1) * q)]

for (const path of models) {
  const bytes = new Uint8Array(readFileSync(path))
  const engine = packed ? SkyEngine.withOptions(bytes, true, true) : new SkyEngine(bytes)
  if (tau !== undefined) engine.setExitThreshold(tau)
  const info = JSON.parse(engine.model_info())
  const frame = (i) => {
    const s = stateOf(i)
    const ins = staticCache ? INS : `${INS} (v${10 + (i % 17)})`
    let p = engine.decide('choice', ins, OPTS, s)
    if (questions > 1) p = engine.decide('noul', DANGER, DANGER_OPTS, s)
    return p[0]
  }
  let sink = 0
  for (let i = 0; i < 300; i++) sink += frame(i) // warm-up (JIT, caches)
  const runs = []
  let layers = 0
  for (let r = 0; r < repeat; r++) {
    const t = new Float64Array(iters)
    for (let i = 0; i < iters; i++) {
      const t0 = performance.now()
      sink += frame(i)
      t[i] = (performance.now() - t0) * 1000
      if (r === repeat - 1) layers += JSON.parse(engine.lastStats()).layers_used ?? 0
    }
    t.sort()
    runs.push({ median: pct(t, 0.5), p95: pct(t, 0.95), min: t[0] })
  }
  const med = (k) => [...runs.map((x) => x[k])].sort((a, b) => a - b)[Math.floor((runs.length - 1) / 2)]
  console.log(
    JSON.stringify({
      model: basename(path),
      game: doom ? 'doom' : 'flappy',
      arch: info.arch,
      params: info.n_params,
      weight_format: info.weight_format,
      ternary_packed: info.ternary_packed ?? false,
      static_cache: staticCache,
      questions,
      exit_threshold: tau ?? null,
      iters,
      repeat,
      median_us: +med('median').toFixed(2),
      p95_us: +med('p95').toFixed(2),
      min_us: +med('min').toFixed(2),
      mean_layers_used: info.exits ? +(layers / iters).toFixed(3) : null,
      checksum: +sink.toFixed(3),
    }),
  )
  engine.free()
}
