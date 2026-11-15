// The secrets verifier (s6-secrets-verify-pico) called from Node through the glue-free host
// (crates/sky-web/js/verify-host.mjs), with the SIMD128 module and with the relaxed-SIMD one, and
// timed:
//
//   node crates/sky-web/bench/verify-node.mjs [--data DIR] [--probs P] [--words dictionary.txt]
//        [--assets DIR] [--model M.gguf] [--n N] [--glue]
//
//   --data    the data folder of s6-secrets v2: the candidates are the CredData validation
//             candidates of build/tables/creddata-val.jsonl, rebuilt from the generator's raw
//             export (gen/candidates-validation.raw.jsonl) with the scope words of their stored
//             path, and every state is checked against the table's. Without it: the invented
//             candidates of crates/sky-infer/tests/fixtures/secret_cand.json, with the fixture's
//             word list, and P(real) is checked against crates/sky-web/tests/parity/verify-expected.json
//   --words   the word list (default: the build's dictionary.txt of --data, or the fixture's)
//   --probs   the model's validation dump (val-creddata-probs.jsonl): P(real) is compared
//   --assets  where sky_web_bg.wasm and sky_web_relaxed_bg.wasm are (default site/public/pkg,
//             as scripts/build-wasm.sh writes them; or target/verify-assets)
//   --model   default site/public/models/s6-secrets-verify-pico.gguf
//   --n       time only the first N candidates
//   --glue    also run the official glue (sky_web.js of --assets) on the first 50 candidates and
//             require the same answers, bit for bit
//
// Prints one JSON line per module, then a Markdown table. Only counts and timings are printed,
// never a value or a line. Exit status 1 when a state differs or a probability is off by more
// than 1e-5.
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { verifierFromModule } from '../js/verify-host.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '../../..')
const TOL = 1e-5
const BATCH = 16

const opt = { assets: join(root, 'site/public/pkg'), model: join(root, 'site/public/models/s6-secrets-verify-pico.gguf'), n: Infinity }
const args = process.argv.slice(2)
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--glue') opt.glue = true
  else if (['--words', '--data', '--probs', '--assets', '--model', '--n'].includes(a)) opt[a.slice(2)] = args[++i]
  else {
    console.error(`unknown argument ${a}`)
    process.exit(2)
  }
}
opt.n = Number(opt.n)

const jsonl = (path) => readFileSync(path, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))

// The candidates, with the state each must give (null: not checked) and the native P(real).
function candidates() {
  if (!opt.data) {
    const f = JSON.parse(readFileSync(join(root, 'crates/sky-infer/tests/fixtures/secret_cand.json'), 'utf8'))
    const want = JSON.parse(readFileSync(join(root, 'crates/sky-web/tests/parity/verify-expected.json'), 'utf8'))
    if (want.length !== f.candidates.length) throw new Error('verify-expected.json is not the answers of the fixture')
    return {
      source: 'fixture (invented values)',
      words: new TextEncoder().encode(f.dictionary.join('\n')),
      // the expected answers hold for the fixture's word list only
      items: f.candidates.map((c, k) => ({ cand: c.record, state: null, p: opt.words ? null : want[k].p_real })),
    }
  }
  const raw = jsonl(join(opt.data, 'gen/candidates-validation.raw.jsonl'))
  const table = jsonl(join(opt.data, 'build/tables/creddata-val.jsonl'))
  const probs = opt.probs ? jsonl(opt.probs).filter((d) => d.src === 'creddata:real').map((d) => d.probs[1]) : null
  if (probs && probs.length !== table.length) throw new Error('the dump is not the dump of the table')
  const items = []
  table.forEach((row, k) => {
    const m = /^cred:val:(\d+)$/.exec(row.id)
    const rec = m && raw[Number(m[1])]
    if (!rec) return
    // CredData's stored path data/<repo>/<scope...>/<file>: its directories are the scope words
    const dirs = rec.file.split('/').slice(2, -1)
    const cand = { ...rec, scope: dirs.length === 1 && dirs[0] === '_' ? [] : dirs }
    items.push({ cand, state: row.state, p: probs ? probs[k] : null })
  })
  const words = readFileSync(opt.words ?? join(opt.data, 'build/dictionary.txt'))
  return { source: `CredData validation (${table.length} rows of the table)`, words, items }
}

const quantile = (v, q) => {
  const s = [...v].sort((a, b) => a - b)
  return s[Math.round((s.length - 1) * q)]
}

async function glueVerifier(name, wasmBytes, model, words) {
  // one instance of the glue per module: the query string keeps Node from reusing the first
  const mod = await import(`${pathToFileURL(join(opt.assets, 'sky_web.js')).href}?${name}`)
  await mod.default({ module_or_path: wasmBytes })
  return new mod.SecretVerifier(model, words)
}

const { source, items, words: listed } = candidates()
const model = new Uint8Array(readFileSync(opt.model))
const words = new Uint8Array(opt.words ? readFileSync(opt.words) : listed)
const timed = items.slice(0, Math.min(items.length, opt.n))
let failed = false
const rows = []

for (const [name, file] of [['simd128', 'sky_web_bg.wasm'], ['relaxed', 'sky_web_relaxed_bg.wasm']]) {
  const wasmBytes = new Uint8Array(readFileSync(join(opt.assets, file)))
  // load: compile the module, instantiate it, read the model and the word list
  const t0 = performance.now()
  const module = await WebAssembly.compile(wasmBytes)
  const t1 = performance.now()
  const v = verifierFromModule(module, model, words)
  const t2 = performance.now()
  const info = v.info()

  // correctness on every candidate: the state, and P(real) against the native dump
  let statesChecked = 0
  let statesEqual = 0
  let maxDiff = 0
  let compared = 0
  let truncated = 0
  const answers = []
  for (const it of items) {
    if (it.state !== null) {
      statesChecked++
      statesEqual += v.state(it.cand) === it.state
    }
    const a = v.judge(it.cand)
    answers.push(a)
    truncated += a.truncated
    if (it.p !== null) {
      compared++
      maxDiff = Math.max(maxDiff, Math.abs(a.p_real - it.p))
    }
  }
  if (statesEqual !== statesChecked || maxDiff > TOL) failed = true

  // latency: one call per candidate (warm: every candidate was judged once above)
  const perCall = []
  const inside = []
  for (const it of timed) {
    const s = performance.now()
    const a = v.judge(it.cand)
    perCall.push((performance.now() - s) * 1000)
    inside.push(a.micros)
  }
  const perBatch = []
  for (let i = 0; i + BATCH <= timed.length; i += BATCH) {
    const batch = timed.slice(i, i + BATCH).map((it) => it.cand)
    const s = performance.now()
    const out = v.judgeMany(batch)
    perBatch.push((performance.now() - s) * 1000)
    out.forEach((a, j) => {
      if (a.p_real !== answers[i + j].p_real) failed = true
    })
  }

  let glue = null
  if (opt.glue) {
    const g = await glueVerifier(name, wasmBytes, model, words)
    const n = Math.min(50, items.length)
    let same = 0
    for (let i = 0; i < n; i++) same += JSON.parse(g.judge(JSON.stringify(items[i].cand))).p_real === answers[i].p_real
    glue = { n, same }
    if (same !== n) failed = true
    g.free()
  }
  v.free()

  const row = {
    module: name,
    wasm_bytes: wasmBytes.length,
    source,
    candidates: items.length,
    words: info.words,
    load_ms: { compile: +(t1 - t0).toFixed(1), verifier: +(t2 - t1).toFixed(1), total: +(t2 - t0).toFixed(1) },
    states: { checked: statesChecked, equal: statesEqual },
    p_real: { compared, max_abs_diff: maxDiff },
    truncated,
    timed: timed.length,
    per_candidate_us: { median: Math.round(quantile(perCall, 0.5)), p95: Math.round(quantile(perCall, 0.95)), inside_median: Math.round(quantile(inside, 0.5)) },
    per_batch_16_us: perBatch.length ? { batches: perBatch.length, median: Math.round(quantile(perBatch, 0.5)), p95: Math.round(quantile(perBatch, 0.95)) } : null,
    glue,
  }
  rows.push(row)
  console.log(JSON.stringify(row))
}

console.log(`\nnode ${process.version}, ${process.platform}/${process.arch}; ${source}, ${timed.length} timed\n`)
console.log('| module | load (compile + model + words) | per candidate, median | p95 | per batch of 16, median | p95 | states equal | max |P - native| |')
console.log('|---|---|---|---|---|---|---|---|')
for (const r of rows) {
  const b = r.per_batch_16_us
  console.log(
    `| ${r.module} | ${r.load_ms.total} ms (${r.load_ms.compile} + ${r.load_ms.verifier}) | ${r.per_candidate_us.median} µs | ${r.per_candidate_us.p95} µs | ` +
      `${b ? `${b.median} µs` : '-'} | ${b ? `${b.p95} µs` : '-'} | ${r.states.equal}/${r.states.checked} | ${r.p_real.compared ? r.p_real.max_abs_diff.toExponential(2) : '-'} |`,
  )
}
if (failed) {
  console.error('verify-node: a state, a probability or a batch answer differs')
  process.exit(1)
}
