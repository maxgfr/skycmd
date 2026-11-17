// Tests of the end-of-turn demo's front end and policy (lib/turn.ts): the resampler to 16 kHz,
// the energy VAD and the turn state machine, with a fake `probEnd` and synthetic frames. Pure
// node: no DOM, no wasm.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  FRAME_SAMPLES,
  RATE_16K,
  EnergyVad,
  TurnPolicy,
  makeResampler16k,
  resampleTo16k,
  type TurnEvent,
} from '../.vitepress/theme/lib/turn.ts'

/** A sine at `freq` Hz, amplitude `amp`, at 16 kHz. */
function sine(secs: number, freq = 220, amp = 0.2): Float32Array {
  const n = Math.round(secs * RATE_16K)
  const x = new Float32Array(n)
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * freq * i) / RATE_16K)
  return x
}

/** A constant frame of 10 ms: its RMS is the constant. */
function frame(value: number): Float32Array {
  return new Float32Array(FRAME_SAMPLES).fill(value)
}

// ------------------------------------------------------------------------------------------
// The resampler.

test('the resampler is the identity at 16 kHz and lands on the right length at other rates', () => {
  const x = sine(1)
  assert.equal(resampleTo16k(x, RATE_16K), x, 'identity: the input array itself')
  const step48 = resampleTo16k(x, 48000)
  assert.equal(step48.length, Math.ceil(x.length / 3))
  const step44 = resampleTo16k(x, 44100)
  assert.equal(step44.length, Math.ceil(x.length / (44100 / RATE_16K)))
  for (const v of step44) assert.ok(Number.isFinite(v))
})

test('the stateful resampler agrees with the one-shot one across chunk boundaries', () => {
  const x = sine(0.5, 137, 0.9)
  const whole = resampleTo16k(x, 48000)
  const r = makeResampler16k(48000)
  const parts: Float32Array[] = []
  for (let i = 0; i < x.length; i += 128) parts.push(r(x.subarray(i, i + 128)))
  let n = 0
  for (const p of parts) {
    for (let j = 0; j < p.length; j++)
      assert.ok(Math.abs(p[j] - whole[n + j]) < 1e-6, `sample ${n + j}: ${p[j]} vs ${whole[n + j]}`)
    n += p.length
  }
  assert.equal(n, whole.length)
})

test('the stateful resampler keeps the rate across many chunks at a fractional ratio', () => {
  const x = sine(5, 97, 0.5)
  const r = makeResampler16k(44100)
  let n = 0
  for (let i = 0; i < x.length; i += 128) n += r(x.subarray(i, i + 128)).length
  assert.equal(n, Math.ceil(x.length / (44100 / RATE_16K)))
})

// ------------------------------------------------------------------------------------------
// The energy VAD.

test('the VAD calls a loud frame speech and quiet frames silence', () => {
  const v = new EnergyVad()
  assert.equal(v.isSpeech(sine(FRAME_SAMPLES / RATE_16K)), true, 'a 0.2 sine, RMS about 0.14')
  assert.equal(v.isSpeech(frame(0.001)), false, 'below the absolute floor')
  assert.equal(v.isSpeech(frame(0.02)), true, 'above the absolute floor, whatever the floor was')
  const v2 = new EnergyVad({ noiseFloor: 0.05, ratio: 3.5 })
  assert.equal(v2.isSpeech(frame(0.02)), false, 'under 3.5 x the noise floor')
})

test('the VAD tracks the noise floor on quiet frames, and its threshold adapts', () => {
  const v = new EnergyVad()
  const mid = frame(0.01) // speech at first: above the absolute floor of 0.008
  assert.equal(v.isSpeech(mid), true)
  const room = frame(0.005) // quiet, but well above the initial floor of 0.002
  for (let i = 0; i < 400; i++) v.isSpeech(room)
  assert.ok(Math.abs(v.noiseFloor - 0.005) < 0.001, `floor ${v.noiseFloor}`)
  assert.equal(v.isSpeech(mid), false, 'the same frame is silence once the floor has risen')
  // digital silence pulls the floor to its minimum, and the absolute floor holds the line
  const v2 = new EnergyVad()
  for (let i = 0; i < 2000; i++) v2.isSpeech(new Float32Array(FRAME_SAMPLES))
  assert.equal(v2.isSpeech(new Float32Array(FRAME_SAMPLES)), false)
  assert.equal(v2.isSpeech(frame(0.009)), true)
})

// ------------------------------------------------------------------------------------------
// The turn policy.

/** A policy with a scripted model: `ps` answered in order, the last one repeats. */
function script(ps: number[], tau = 0.5) {
  const events: TurnEvent[] = []
  const calls: number[] = []
  let i = 0
  const policy = new TurnPolicy({
    probEnd: () => {
      calls.push(policy.tMs)
      return ps[Math.min(i++, ps.length - 1)]
    },
    tau,
    onEvent: (e) => events.push(e),
  })
  return { policy, events, calls }
}

function speak(policy: TurnPolicy, frames: number) {
  for (let i = 0; i < frames; i++) policy.feed(true)
}

function hush(policy: TurnPolicy, frames: number) {
  for (let i = 0; i < frames; i++) policy.feed(false)
}

test('speech then silence with p above tau: one decision, finished by the model, then nothing', () => {
  const { policy, events, calls } = script([0.9])
  speak(policy, 16) // 160 ms of speech: past minSpeechMs (150)
  hush(policy, 40) // 400 ms of silence
  assert.deepEqual(events.map((e) => e.type), ['speech-start', 'decision', 'finished'])
  assert.equal(events[0].atMs, 150, 'minSpeechMs (150) reached on the 15th frame')
  assert.equal(events[1].p, 0.9)
  assert.equal(events[1].atMs, 360, 'asked after stopMs (200) of silence')
  assert.ok((events[1].decisionMs ?? -1) >= 0, 'the decision is timed')
  assert.equal(events[2].by, 'model')
  assert.equal(events[2].p, 0.9)
  assert.equal(calls.length, 1, 'the model is asked once')
  assert.equal(policy.state, 'waiting')
  hush(policy, 200) // silence after the finished: no speech before it, nothing asked
  assert.equal(calls.length, 1)
})

test('p below tau: holding, asked again every recheckMs, finished by the timer at fallbackMs', () => {
  const { policy, events, calls } = script([0.2])
  speak(policy, 16)
  hush(policy, 300) // 3 s of silence
  assert.deepEqual(events.filter((e) => e.type === 'speech-start').length, 1)
  assert.equal(events.filter((e) => e.type === 'decision').length, calls.length)
  assert.equal(calls.length, 12, 'asked at 200, 400, ..., 2400 ms of silence')
  assert.equal(calls[0], 360)
  assert.equal(calls[1], 560, 'recheckMs later')
  const fin = events.find((e) => e.type === 'finished')!
  assert.equal(fin.by, 'timer')
  assert.equal(fin.atMs, 160 + 2500, 'fallbackMs of silence, the timer, no p')
  assert.equal(fin.p, undefined)
  assert.equal(policy.state, 'waiting')
})

test('speech resuming during a holding pause: resumed, no finished, the turn goes on', () => {
  const { policy, events, calls } = script([0.2, 0.2, 0.9])
  speak(policy, 16)
  hush(policy, 20) // decision p = 0.2: holding
  speak(policy, 10) // resumed
  hush(policy, 20) // decision p = 0.2: holding again
  speak(policy, 10) // resumed
  hush(policy, 20) // decision p = 0.9: finished by the model
  assert.deepEqual(events.map((e) => e.type), [
    'speech-start',
    'decision',
    'resumed',
    'decision',
    'resumed',
    'decision',
    'finished',
  ])
  assert.equal(events[6].by, 'model')
  assert.equal(calls.length, 3)
  assert.equal(policy.state, 'waiting')
})

test('a pause shorter than stopMs asks nothing and stays in the same turn', () => {
  const { policy, events, calls } = script([0.9])
  speak(policy, 16)
  for (let i = 0; i < 5; i++) {
    hush(policy, 19) // 190 ms of silence, under stopMs (200)
    speak(policy, 4)
  }
  assert.deepEqual(events.map((e) => e.type), ['speech-start'])
  assert.equal(calls.length, 0)
  assert.equal(policy.state, 'speaking')
})

test('speech shorter than minSpeechMs is not a turn: silence after it asks nothing', () => {
  const { policy, events, calls } = script([0.9])
  speak(policy, 14) // 140 ms, under minSpeechMs (150)
  hush(policy, 40)
  assert.deepEqual(events, [])
  assert.equal(calls.length, 0)
  speak(policy, 16) // speech that reaches minSpeechMs starts the turn
  assert.deepEqual(events.map((e) => e.type), ['speech-start'])
})

test('silence with no speech before it asks nothing, the fallback included', () => {
  const { policy, events, calls } = script([0.9])
  hush(policy, 400) // 4 s of silence: past fallbackMs too
  assert.deepEqual(events, [])
  assert.equal(calls.length, 0)
  assert.equal(policy.state, 'waiting')
})

test('the fallback timer ends the turn whatever stopMs says, the model unasked', () => {
  const events: TurnEvent[] = []
  const policy = new TurnPolicy({
    probEnd: () => {
      throw new Error('the model must not be asked')
    },
    tau: 0.5,
    stopMs: 10000,
    fallbackMs: 1000,
    onEvent: (e) => events.push(e),
  })
  speak(policy, 16)
  hush(policy, 120) // 1.2 s of silence
  assert.deepEqual(events.map((e) => e.type), ['speech-start', 'finished'])
  assert.equal(events[1].by, 'timer')
  assert.equal(policy.state, 'waiting')
})

test('reset clears the timeline and the state', () => {
  const { policy, events } = script([0.9])
  speak(policy, 16)
  hush(policy, 20) // one decision at 360 ms
  assert.equal(events.length, 3)
  policy.reset()
  assert.equal(policy.tMs, 0)
  assert.equal(policy.state, 'waiting')
  const n = events.length
  speak(policy, 16)
  hush(policy, 20)
  assert.equal(events.length, n + 3, 'the new turn starts over')
  assert.equal(events[n].type, 'speech-start')
  assert.equal(events[n].atMs, 150, 'the timeline restarted')
})
