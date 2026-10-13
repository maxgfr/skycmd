// WASM vs native parity in Node: replays the first N requests of a `sky-eval --dump` file
// (native probabilities of the shipping engine) through the browser module's `Engine.decide`
// (site/public/pkg, built by scripts/build-wasm.sh) and reports the largest probability gap.
//
//   node crates/sky-web/bench/parity-node.mjs model.gguf dump.jsonl [--n 20]
//
// Prints one JSON line: {n, max_abs_diff, argmax_agree, median_wasm_us}. Exit status 1 when a
// gap exceeds 1e-4 or an argmax differs.
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../../../site/public/pkg')

const args = process.argv.slice(2)
let n = 20
const files = []
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--n') n = Number(args[++i])
  else files.push(args[i])
}
if (files.length !== 2) {
  console.error('usage: parity-node.mjs model.gguf dump.jsonl [--n 20]')
  process.exit(2)
}
const [modelPath, dumpPath] = files

const mod = await import(join(pkg, 'sky_web.js'))
await mod.default({ module_or_path: readFileSync(join(pkg, 'sky_web_bg.wasm')) })
const engine = new mod.Engine(new Uint8Array(readFileSync(modelPath)))

const rows = readFileSync(dumpPath, 'utf8').split('\n').filter(Boolean).slice(0, n).map((l) => JSON.parse(l))
let maxDiff = 0
let agree = 0
const us = []
const argmax = (v) => v.indexOf(Math.max(...v))
for (const r of rows) {
  const a = JSON.parse(engine.decide(r.t, r.ins, JSON.stringify(r.opts), r.state))
  us.push(a.micros)
  for (let i = 0; i < r.probs.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a.probs[i] - r.probs[i]))
  agree += argmax(a.probs) === argmax(r.probs) ? 1 : 0
}
us.sort((x, y) => x - y)
const out = { n: rows.length, max_abs_diff: maxDiff, argmax_agree: agree, median_wasm_us: us[Math.floor(us.length / 2)] }
console.log(JSON.stringify(out))
process.exit(maxDiff <= 1e-4 && agree === rows.length ? 0 : 1)
