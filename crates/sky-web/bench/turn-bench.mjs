// WASM latency of an s6-ear v2 end-of-turn file in Node (docs/s6-ear2-prereg.md §4.4, §8.3),
// through sky-infer's `TurnDetector` (exported by sky-web's module), next to `sky-turn --bench`
// natively. The input is `turnSynthSpeech(4.5, 1, 1)`, the signal `sky-turn --bench` reads.
//
// * Front end: the clip pushed in 10 ms chunks (160 samples), the time per hop computed (median
//   of `--passes` passes).
// * Decision: `probEnd()` on the stream's full 4 s window (the hops already computed), timed from
//   JavaScript around each call after a warm-up; median and p95 over `--iters` decisions.
//
// Node runs WASM on one thread: one core. Build the module first:
//
//   RUSTFLAGS="-C target-feature=+simd128" wasm-pack build crates/sky-web --target nodejs \
//     --release --out-dir <pkg> --no-typescript
//
//   node crates/sky-web/bench/turn-bench.mjs --pkg <pkg> [--iters 1000] [--warmup 50] [--passes 5]
//        model.gguf...
//
// Prints one JSON line per model.
import { readFileSync, statSync } from 'node:fs'
import { basename, resolve } from 'node:path'
import { createRequire } from 'node:module'

const args = process.argv.slice(2)
let pkg
let iters = 1000
let warmup = 50
let passes = 5
const models = []
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--pkg') pkg = args[++i]
  else if (a === '--iters') iters = Number(args[++i])
  else if (a === '--warmup') warmup = Number(args[++i])
  else if (a === '--passes') passes = Number(args[++i])
  else models.push(a)
}
if (!pkg || !models.length) {
  console.error('usage: turn-bench.mjs --pkg <wasm-pack nodejs dir of sky-web> [--iters N] [--warmup W] [--passes P] model.gguf...')
  process.exit(2)
}
const { TurnDetector, turnSynthSpeech } = createRequire(import.meta.url)(resolve(pkg, 'sky_web.js'))
const pcm = turnSynthSpeech(4.5, 1n, 1.0)
const pct = (s, q) => s[Math.min(s.length - 1, Math.round((s.length - 1) * q))]

for (const path of models) {
  const det = new TurnDetector(new Uint8Array(readFileSync(path)))
  // the front end, hop by hop as audio arrives
  const perHop = []
  for (let p = 0; p < passes; p++) {
    det.reset()
    const t0 = performance.now()
    for (let i = 0; i + 160 <= pcm.length; i += 160) det.push(pcm.subarray(i, i + 160))
    const dt = performance.now() - t0
    const hops = Math.floor((pcm.length - 640) / 160) + 1
    perHop.push((dt * 1000) / hops)
  }
  perHop.sort((a, b) => a - b)
  let sink = 0
  for (let i = 0; i < warmup; i++) sink += det.probEnd()
  const t = new Float64Array(iters)
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now()
    sink += det.probEnd()
    t[i] = (performance.now() - t0) * 1000
  }
  t.sort()
  const info = JSON.parse(det.info())
  console.log(
    JSON.stringify({
      model: basename(path),
      path,
      name: info.name,
      params: info.params,
      file_bytes: statSync(path).size,
      hops: det.hopCount(),
      iters,
      median_us: pct(t, 0.5),
      p95_us: pct(t, 0.95),
      frontend_us_per_hop: pct(perHop, 0.5),
      prob: det.probEnd(),
      sink: Number.isFinite(sink),
    }),
  )
  det.free()
}
