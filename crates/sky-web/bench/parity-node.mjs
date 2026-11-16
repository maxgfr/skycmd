// WASM vs native parity in Node: replays the first N requests of a `sky-eval --dump` file
// (native probabilities of the shipping engine) through the browser module's `Engine.decide`
// (site/public/pkg, built by scripts/build-wasm.sh) and reports the largest probability gap.
//
//   node crates/sky-web/bench/parity-node.mjs model.gguf dump.jsonl [--n 20] [--relaxed]
//
// `--relaxed` loads the relaxed-SIMD module (sky_web_relaxed_bg.wasm) instead of the SIMD128 one.
//
// s6-ear dumps (`sky-eval --dump` on an audio model) carry a `wav` path per row: those rows go
// through `Engine.decideAudio` on the clip's samples (16-bit PCM WAV at 16 kHz).
//
// `--turn` (s6-ear v2): the dump is `sky-turn <model> --records ... --out dump.jsonl` and each
// clip goes through `TurnDetector.probEndPcm`; `--pkg <dir>` loads a `wasm-pack --target nodejs`
// build of sky-web instead of the site's module.
//
// Prints one JSON line: {n, max_abs_diff, argmax_agree, median_wasm_us}. Exit status 1 when a
// gap exceeds 1e-4 or an argmax differs.
//
// `--router` (s6-route-nano, docs/s6-nano2-prereg.md 7.4): each request is decided twice. The
// first call meets its tool list for the first time (the static stream is encoded: the
// "first request" latency) and is the one compared with the native probabilities; the second
// asks the same tool list about another request text (the request with " please" appended, so
// that the state is encoded again and only the tool list is cached: the "per request with the
// tool list cached" latency). The line adds {median_first_us, median_cached_us, static_hits,
// state_hits}; median_wasm_us is the cached median.
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../../../site/public/pkg')

const args = process.argv.slice(2)
let n = 20
let relaxed = false
let router = false
let turn = false
let nodePkg
const files = []
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--n') n = Number(args[++i])
  else if (args[i] === '--relaxed') relaxed = true
  else if (args[i] === '--router') router = true
  else if (args[i] === '--turn') turn = true
  else if (args[i] === '--pkg') nodePkg = args[++i]
  else files.push(args[i])
}
if (files.length !== 2) {
  console.error('usage: parity-node.mjs model.gguf dump.jsonl [--n 20] [--relaxed] [--router] [--turn] [--pkg <nodejs pkg>]')
  process.exit(2)
}
const [modelPath, dumpPath] = files

// `--pkg` loads a `wasm-pack --target nodejs` build of sky-web instead of the site's module
let mod
if (nodePkg) {
  mod = createRequire(import.meta.url)(resolve(nodePkg, 'sky_web.js'))
} else {
  mod = await import(join(pkg, 'sky_web.js'))
  await mod.default({ module_or_path: readFileSync(join(pkg, relaxed ? 'sky_web_relaxed_bg.wasm' : 'sky_web_bg.wasm')) })
}

const rows = readFileSync(dumpPath, 'utf8').split('\n').filter(Boolean).slice(0, n).map((l) => JSON.parse(l))

// s6-ear v2 (docs/s6-ear2-prereg.md §7): rows of `sky-turn <model> --records ... --out dump.jsonl`
// ({id, wav, prob}); each clip goes through `TurnDetector.probEndPcm` (its last 4 s) and must give
// the native probability within 1e-4 and the same decision at tau50.
if (turn) {
  const det = new mod.TurnDetector(new Uint8Array(readFileSync(modelPath)))
  const tau = det.tau50()
  let maxDiff = 0
  let same = 0
  const us = []
  const used = rows.filter((r) => Number.isFinite(r.prob))
  for (const r of used) {
    const x = readWav(r.wav)
    const t0 = performance.now()
    const p = det.probEndPcm(x)
    us.push((performance.now() - t0) * 1000)
    maxDiff = Math.max(maxDiff, Math.abs(p - r.prob))
    same += (p >= tau) === (r.prob >= tau) ? 1 : 0
  }
  us.sort((a, b) => a - b)
  const out = { mode: 'turn', n: used.length, max_abs_diff: maxDiff, same_decision: same, tau50: tau, median_wasm_us: us[Math.floor(us.length / 2)] }
  console.log(JSON.stringify(out))
  process.exit(maxDiff <= 1e-4 && same === used.length && used.length > 0 ? 0 : 1)
}
const engine = new mod.Engine(new Uint8Array(readFileSync(modelPath)))
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

const first = []
let staticHits = 0
let stateHits = 0
for (const r of rows) {
  const opts = JSON.stringify(r.opts)
  const a = JSON.parse(r.wav ? engine.decideAudio(r.t, r.ins, opts, readWav(r.wav)) : engine.decide(r.t, r.ins, opts, r.state))
  for (let i = 0; i < r.probs.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a.probs[i] - r.probs[i]))
  agree += argmax(a.probs) === argmax(r.probs) ? 1 : 0
  if (router) {
    first.push(a.micros)
    const b = JSON.parse(engine.decide(r.t, r.ins, opts, `${r.state} please`))
    us.push(b.micros)
    staticHits += b.static_hit ? 1 : 0
    stateHits += b.state_hit ? 1 : 0
  } else {
    us.push(a.micros)
  }
}
const median = (v) => {
  const s = [...v].sort((x, y) => x - y)
  return s[Math.floor(s.length / 2)]
}
const out = { n: rows.length, max_abs_diff: maxDiff, argmax_agree: agree, median_wasm_us: median(us) }
if (router) Object.assign(out, { median_first_us: median(first), median_cached_us: median(us), static_hits: staticHits, state_hits: stateHits })
console.log(JSON.stringify(out))
process.exit(maxDiff <= 1e-4 && agree === rows.length ? 0 : 1)
