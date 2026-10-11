// WASM latency benchmark in Node: loads the browser module (site/public/pkg, built by
// scripts/build-wasm.sh) and runs `bench()` (Flappy Drone decisions: both questions on the same
// state + physics, timed in batches of 10) for every model given on the command line.
//
//   node crates/sky-web/bench/bench-node.mjs [--n 2000] [--repeat 5] [--no-static-cache] [--packed]
//        [--exit-threshold T] model.gguf...
//
// Prints one JSON line per model: the median over `--repeat` runs of the bench median, plus the
// depth and cache statistics reported by `bench()`.
import { readFileSync } from 'node:fs'
import { dirname, join, basename } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../../../site/public/pkg')

const args = process.argv.slice(2)
let n = 2000
let repeat = 5
let staticCache = true
let packed = false
let tau
const models = []
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--n') n = Number(args[++i])
  else if (a === '--repeat') repeat = Number(args[++i])
  else if (a === '--no-static-cache') staticCache = false
  else if (a === '--packed') packed = true
  else if (a === '--exit-threshold') tau = Number(args[++i])
  else models.push(a)
}
if (!models.length) {
  console.error('usage: bench-node.mjs [--n N] [--repeat R] [--no-static-cache] [--packed] [--exit-threshold T] model.gguf...')
  process.exit(2)
}

const mod = await import(join(pkg, 'sky_web.js'))
await mod.default({ module_or_path: readFileSync(join(pkg, 'sky_web_bg.wasm')) })

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.floor((s.length - 1) / 2)]
}

for (const path of models) {
  const engine = mod.Engine.withOptions(new Uint8Array(readFileSync(path)), staticCache, packed)
  if (tau !== undefined) engine.setExitThreshold(tau)
  mod.bench(engine, 200) // warm-up (JIT, caches)
  const runs = []
  for (let r = 0; r < repeat; r++) runs.push(JSON.parse(mod.bench(engine, n)))
  const last = runs[runs.length - 1]
  console.log(
    JSON.stringify({
      model: basename(path),
      family: last.family,
      params: last.params,
      file_bytes: last.file_bytes,
      static_cache: staticCache,
      ternary_packed: packed,
      exit_threshold: tau ?? null,
      n,
      repeat,
      median_us: median(runs.map((x) => x.median_us)),
      p95_us: median(runs.map((x) => x.p95_us)),
      mean_layers_used: last.mean_layers_used,
      max_layers: last.max_layers,
      static_hit_rate: last.static_hit_rate,
      state_reuse_rate: last.state_reuse_rate,
    }),
  )
  engine.free()
}
