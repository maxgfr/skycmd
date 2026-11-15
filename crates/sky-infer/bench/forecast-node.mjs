// WASM latency of a forecaster in Node, through sky-web's `Forecaster` (the class the /forecast/
// page calls): the measurement of docs/s6-cast2-prereg.md §7.2, next to `sky-cast --bench`
// natively. One forecast of `--horizon` steps after a `--len`-point history (the generator's
// `tsi` series of `--seed`, the same one `sky-cast --synth tsi --seed S --len N` reads), timed
// from JavaScript around each call, after a warm-up. Node runs WASM on one thread: one core.
// Build the module first:
//
//   RUSTFLAGS="-C target-feature=+simd128" wasm-pack build crates/sky-web --target nodejs \
//     --release --out-dir <pkg> --no-typescript
//
//   node crates/sky-infer/bench/forecast-node.mjs --pkg <pkg> [--iters 1000] [--horizon 64]
//        [--len 2048] [--seed 7] [--warmup 50] model.gguf...
//
// Full depth is the file without its calibrated exits (`<name>-full.gguf`, exit tolerance 0, so
// every exit runs); the calibrated file gives the calibrated exits. TinyCast has no exits.
// Prints one JSON line per model: median and p95 µs over the iterations.
import { readFileSync } from 'node:fs'
import { basename, resolve } from 'node:path'
import { createRequire } from 'node:module'

const args = process.argv.slice(2)
let pkg
let iters = 1000
let horizon = 64
let len = 2048
let seed = 7
let warmup = 50
const models = []
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--pkg') pkg = args[++i]
  else if (a === '--iters') iters = Number(args[++i])
  else if (a === '--horizon') horizon = Number(args[++i])
  else if (a === '--len') len = Number(args[++i])
  else if (a === '--seed') seed = Number(args[++i])
  else if (a === '--warmup') warmup = Number(args[++i])
  else models.push(a)
}
if (!pkg || !models.length) {
  console.error('usage: forecast-node.mjs --pkg <wasm-pack nodejs dir of sky-web> [--iters N] [--horizon H] [--len N] [--seed S] [--warmup W] model.gguf...')
  process.exit(2)
}
const { Forecaster, synthSeries } = createRequire(import.meta.url)(resolve(pkg, 'sky_web.js'))
const history = synthSeries('tsi', BigInt(seed), len)
const pct = (s, q) => s[Math.min(s.length - 1, Math.floor(s.length * q))]

for (const path of models) {
  const f = new Forecaster(new Uint8Array(readFileSync(path)))
  let sink = 0
  for (let i = 0; i < warmup; i++) sink += f.forecast(history, horizon, false)[0]
  const t = new Float64Array(iters)
  let exits = 0
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now()
    sink += f.forecast(history, horizon, false)[0]
    t[i] = (performance.now() - t0) * 1000
    exits += JSON.parse(f.info()).exits_used
  }
  t.sort()
  const info = JSON.parse(f.info())
  console.log(
    JSON.stringify({
      model: basename(path),
      name: info.name,
      params: info.params,
      bytes: info.bytes,
      periods: info.periods,
      history: len,
      horizon,
      iters,
      passes: info.passes,
      mean_exits_used: +(exits / iters).toFixed(3),
      median_us: +pct(t, 0.5).toFixed(2),
      p95_us: +pct(t, 0.95).toFixed(2),
      checksum: +sink.toFixed(3),
    }),
  )
  f.free()
}
