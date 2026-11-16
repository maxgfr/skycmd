// s6-nano v3 (docs/s6-nano3-prereg.md §5 G4): the fused tool router in WASM against the Python
// reference, and its latency, in Node.
//
//   node crates/sky-web/bench/route3-node.mjs model.gguf embedding.gguf slates.jsonl fused.jsonl \
//     [--n 200] [--relaxed] [--fast] [--pkg <nodejs pkg>] [--ref-model M] [--ref-embedding E]
//
// slates.jsonl: routing slates ({opts, state = "User: <request>"}, none last); fused.jsonl: the
// reference fused probabilities, one JSON line with `probs` per slate in slate order
// (scripts/s6-route3-reference.py). The module is site/public/pkg (scripts/build-wasm.sh; set
// SKY_WASM_OUT to build it elsewhere): sky_web_bg.wasm (SIMD128) by default, --relaxed loads
// sky_web_relaxed_bg.wasm; --pkg loads a `wasm-pack --target nodejs` build instead.
//
// Each of the first N slates is routed twice through `ToolRouter.route`: the first call meets its
// tool list for the first time (both parts encode it: the "first request" latency) and is the one
// compared with the reference; the second asks the same tool list about another request (the
// request with " please" appended: the "cached" latency). --ref-model / --ref-embedding (dumps in
// the same format) add the parity of each part.
//
// Prints one JSON line: {n, max_abs_diff, argmax_agree, median_cached_us, median_first_us,
// bytes_model, bytes_embedding, build, fast, model: {...}, embedding: {...}, static_hits,
// tools_hits}; `model` and `embedding` hold each part's median latencies (first and cached) and,
// with a reference, its parity. Exit status 1 when the G4 parity (max |dp| <= 1e-3, at least
// 199/200 of the answers) does not hold.
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../../../site/public/pkg')

const args = process.argv.slice(2)
let n = 200
let relaxed = false
let fast = false
let nodePkg
const refs = {}
const files = []
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--n') n = Number(args[++i])
  else if (args[i] === '--relaxed') relaxed = true
  else if (args[i] === '--fast') fast = true
  else if (args[i] === '--pkg') nodePkg = args[++i]
  else if (args[i] === '--ref-model') refs.model = args[++i]
  else if (args[i] === '--ref-embedding') refs.embedding = args[++i]
  else files.push(args[i])
}
if (files.length !== 4) {
  console.error('usage: route3-node.mjs model.gguf embedding.gguf slates.jsonl fused.jsonl [--n 200] [--relaxed] [--fast] [--pkg dir] [--ref-model M] [--ref-embedding E]')
  process.exit(2)
}
const [modelPath, embedPath, slatesPath, fusedPath] = files

let mod
if (nodePkg) {
  mod = createRequire(import.meta.url)(resolve(nodePkg, 'sky_web.js'))
} else {
  mod = await import(join(pkg, 'sky_web.js'))
  await mod.default({ module_or_path: readFileSync(join(pkg, relaxed ? 'sky_web_relaxed_bg.wasm' : 'sky_web_bg.wasm')) })
}

const jsonl = (p) => readFileSync(p, 'utf8').split('\n').filter(Boolean).slice(0, n).map((l) => JSON.parse(l))
const slates = jsonl(slatesPath)
const fused = jsonl(fusedPath)
const parts = Object.fromEntries(Object.entries(refs).map(([k, p]) => [k, jsonl(p)]))
if (fused.length < slates.length) throw new Error('the reference is shorter than the slates')

const modelBytes = readFileSync(modelPath)
const embedBytes = readFileSync(embedPath)
const router = new mod.ToolRouter(new Uint8Array(modelBytes), new Uint8Array(embedBytes))

const argmax = (v) => v.indexOf(Math.max(...v))
const median = (v) => {
  const s = [...v].sort((x, y) => x - y)
  return s[Math.floor(s.length / 2)]
}
const requestOf = (state) => {
  const s = state.includes('\nBM25: ') ? state.slice(0, state.lastIndexOf('\nBM25: ')) : state
  return s.startsWith('User: ') ? s.slice(6) : s
}
const gap = () => ({ max_abs_diff: 0, argmax_agree: 0 })
const cmp = (acc, got, want) => {
  if (got.length !== want.length) throw new Error('a reference has another number of options')
  for (let i = 0; i < got.length; i++) acc.max_abs_diff = Math.max(acc.max_abs_diff, Math.abs(got[i] - want[i]))
  acc.argmax_agree += argmax(got) === argmax(want) ? 1 : 0
}

const all = gap()
const partGap = { model: gap(), embedding: gap() }
const t = { first: [], cached: [], model_first: [], model_cached: [], embedding_first: [], embedding_cached: [] }
let staticHits = 0
let toolsHits = 0
slates.forEach((r, k) => {
  const tools = JSON.stringify(r.opts.slice(0, -1))
  const request = requestOf(r.state)
  const a = JSON.parse(router.route(request, tools, fast))
  cmp(all, a.probs, fused[k].probs)
  for (const p of ['model', 'embedding']) if (parts[p]) cmp(partGap[p], a[p], parts[p][k].probs)
  t.first.push(a.micros)
  t.model_first.push(a.micros_model)
  t.embedding_first.push(a.micros_embedding)
  const b = JSON.parse(router.route(`${request} please`, tools, fast))
  t.cached.push(b.micros)
  t.model_cached.push(b.micros_model)
  t.embedding_cached.push(b.micros_embedding)
  staticHits += b.static_hit ? 1 : 0
  toolsHits += b.tools_hit ? 1 : 0
})

const part = (p) => ({
  median_first_us: median(t[`${p}_first`]),
  median_cached_us: median(t[`${p}_cached`]),
  ...(parts[p] ? partGap[p] : {}),
})
const out = {
  n: slates.length,
  max_abs_diff: all.max_abs_diff,
  argmax_agree: all.argmax_agree,
  median_cached_us: median(t.cached),
  median_first_us: median(t.first),
  bytes_model: modelBytes.length,
  bytes_embedding: embedBytes.length,
  build: nodePkg ? 'nodejs' : relaxed ? 'relaxed' : 'simd128',
  fast,
  model: part('model'),
  embedding: part('embedding'),
  static_hits: staticHits,
  tools_hits: toolsHits,
}
console.log(JSON.stringify(out))
const need = Math.ceil((199 / 200) * slates.length)
process.exit(all.max_abs_diff <= 1e-3 && all.argmax_agree >= need ? 0 : 1)
