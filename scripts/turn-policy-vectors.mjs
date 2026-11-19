// Records test vectors of the end-of-turn demo's energy VAD and turn policy
// (site/.vitepress/theme/lib/turn.ts, `EnergyVad` and `TurnPolicy` with `TURN_DEFAULTS`) for the Rust
// port of docs/s6-ear3-prereg.md §3.2 item 2 and §16 (crates/sky-data/src/turn_policy.rs), which
// replays them bit for bit: the VAD's answer and noise floor after every frame, and every event of
// the policy at the same frame.
//
//   node scripts/turn-policy-vectors.mjs   (Node >= 22.18: turn.ts is loaded by type stripping)
//
// Writes crates/sky-data/tests/fixtures/turn-policy.json. The frames are not stored: each vector is
// a list of segments [frames, amplitude], and frame samples are amplitude x a 32-bit LCG noise
// value in [-1, 1) computed with integer arithmetic and exact powers of two, then rounded to f32 by
// the Float32Array, so the Rust test rebuilds the same samples bit for bit.
import { writeFileSync } from 'node:fs'
import { EnergyVad, FRAME_SAMPLES, TurnPolicy } from '../site/.vitepress/theme/lib/turn.ts'

/** The LCG of the vectors (Numerical Recipes' constants), one state per vector. */
function lcg(seed) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return (s >>> 8) / 16777216 * 2 - 1
  }
}

/** Segments [frames, amplitude]: speech-like bursts over a noise floor, with the cases the policy
 * distinguishes (short blips, pauses under and over stopMs, long pauses past the fallback timer,
 * a rising noise floor, digital silence). */
const VECTORS = [
  {
    name: 'turns and pauses over a low floor',
    seed: 1,
    probs: [],
    segments: [[50, 0.001], [40, 0.2], [10, 0.001], [60, 0.15], [25, 0.001], [30, 0.3], [80, 0.001], [5, 0.25],
      [3, 0.001], [40, 0.2], [270, 0.0005], [20, 0.2], [12, 0.001], [100, 0.12], [300, 0.001]],
  },
  {
    name: 'short blips, a rising floor and digital silence',
    seed: 7,
    probs: [],
    segments: [[30, 0], [8, 0.3], [30, 0.0], [14, 0.2], [2, 0.001], [16, 0.2], [100, 0.004], [60, 0.02],
      [40, 0.006], [30, 0.5], [21, 0.0], [30, 0.4], [400, 0.0], [10, 0.3], [5, 0], [16, 0.3], [260, 0.002]],
  },
  {
    name: 'a scripted model: finished at some asks',
    seed: 23,
    // the probability returned at each successive ask (0 once the list is used up)
    probs: [0.1, 0.2, 0.9, 0.05, 0.6, 0.4, 0.4, 0.95],
    segments: [[20, 0.001], [40, 0.2], [70, 0.001], [40, 0.25], [30, 0.001], [5, 0.2], [60, 0.001], [50, 0.2],
      [90, 0.001], [40, 0.3], [300, 0.001], [40, 0.2], [45, 0.001], [40, 0.2], [120, 0.001]],
  },
]

const vectors = VECTORS.map((v) => {
  const noise = lcg(v.seed)
  const vad = new EnergyVad()
  const events = []
  let ask = 0
  const policy = new TurnPolicy({
    probEnd: () => (ask < v.probs.length ? v.probs[ask++] : (ask++, 0)),
    onEvent: (e) => events.push({ type: e.type, atMs: e.atMs, ...(e.p !== undefined ? { p: e.p } : {}), ...(e.by ? { by: e.by } : {}) }),
  })
  const speech = []
  const floors = []
  const rms = []
  const frame = new Float32Array(FRAME_SAMPLES)
  for (const [frames, amp] of v.segments) {
    for (let f = 0; f < frames; f++) {
      for (let i = 0; i < FRAME_SAMPLES; i++) frame[i] = amp * noise()
      const s = vad.isSpeech(frame)
      speech.push(s ? 1 : 0)
      rms.push(vad.rms)
      floors.push(vad.noiseFloor)
      policy.feed(s)
    }
  }
  return { name: v.name, seed: v.seed, probs: v.probs, segments: v.segments, speech: speech.join(''), rms, noiseFloor: floors, events }
})

const out = new URL('../crates/sky-data/tests/fixtures/turn-policy.json', import.meta.url)
writeFileSync(
  out,
  JSON.stringify({
    about: 'Vectors of site/.vitepress/theme/lib/turn.ts (EnergyVad, TurnPolicy, TURN_DEFAULTS) recorded by scripts/turn-policy-vectors.mjs for the Rust port of docs/s6-ear3-prereg.md §3.2 item 2 (crates/sky-data/src/turn_policy.rs)',
    frameSamples: FRAME_SAMPLES,
    vectors,
  }) + '\n',
)
console.log(vectors.map((v) => `${v.name}: ${v.speech.length} frames, ${v.events.length} events`).join('\n'))
