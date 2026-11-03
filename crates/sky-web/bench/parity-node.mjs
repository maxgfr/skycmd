// WASM vs native parity in Node: replays the first N requests of a `sky-eval --dump` file
// (native probabilities of the shipping engine) through the browser module's `Engine.decide`
// (site/public/pkg, built by scripts/build-wasm.sh) and reports the largest probability gap.
//
//   node crates/sky-web/bench/parity-node.mjs model.gguf dump.jsonl [--n 20]
//
// s6-ear dumps (`sky-eval --dump` on an audio model) carry a `wav` path per row: those rows go
// through `Engine.decideAudio` on the clip's samples (16-bit PCM WAV at 16 kHz).
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

// 16-bit PCM WAV (mono or downmixed) at 16 kHz -> Float32Array in [-1, 1]
function readWav(path) {
  const b = readFileSync(path)
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw new Error(`${path}: not a WAV file`)
  let fmt = null
  for (let i = 12; i + 8 <= b.length; ) {
    const id = b.toString('ascii', i, i + 4)
    const len = b.readUInt32LE(i + 4)
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(i + 8), ch: b.readUInt16LE(i + 10), rate: b.readUInt32LE(i + 12), bits: b.readUInt16LE(i + 22) }
    if (id === 'data') {
      if (!fmt || fmt.tag !== 1 || fmt.bits !== 16 || fmt.rate !== 16000) throw new Error(`${path}: need 16-bit PCM at 16 kHz`)
      const n = Math.floor(len / (2 * fmt.ch))
      const out = new Float32Array(n)
      for (let f = 0; f < n; f++) {
        let s = 0
        for (let c = 0; c < fmt.ch; c++) s += b.readInt16LE(i + 8 + 2 * (f * fmt.ch + c)) / 32768
        out[f] = s / fmt.ch
      }
      return out
    }
    i += 8 + len + (len & 1)
  }
  throw new Error(`${path}: no data chunk`)
}

for (const r of rows) {
  const opts = JSON.stringify(r.opts)
  const a = JSON.parse(r.wav ? engine.decideAudio(r.t, r.ins, opts, readWav(r.wav)) : engine.decide(r.t, r.ins, opts, r.state))
  us.push(a.micros)
  for (let i = 0; i < r.probs.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a.probs[i] - r.probs[i]))
  agree += argmax(a.probs) === argmax(r.probs) ? 1 : 0
}
us.sort((x, y) => x - y)
const out = { n: rows.length, max_abs_diff: maxDiff, argmax_agree: agree, median_wasm_us: us[Math.floor(us.length / 2)] }
console.log(JSON.stringify(out))
process.exit(maxDiff <= 1e-4 && agree === rows.length ? 0 : 1)
