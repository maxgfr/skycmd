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
  Framer,
  TurnPolicy,
  TURN_DEFAULTS,
  makeResampler16k,
  resampleTo16k,
  validMs,
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

/** A deterministic generator of chunk sizes (0 and 1 included), as an audio callback might give. */
function chunkSizes(seed: number, n: number, max: number): number[] {
  let s = seed
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) % 2147483648
    out.push(s % (max + 1))
  }
  return out
}

/** Linear interpolation at input position k * rate / 16000, written out: the reference. */
function reference(x: Float32Array, rate: number): number[] {
  const out: number[] = []
  for (let k = 0; ; k++) {
    const pos = (k * rate) / RATE_16K
    if (pos > x.length - 1) break
    const i = Math.floor(pos)
    const f = pos - i
    out.push(f === 0 ? x[i] : x[i] + (x[i + 1] - x[i]) * f)
  }
  return out
}

test('chunk by chunk, the resampler gives exactly the reference samples, boundaries included', () => {
  for (const rate of [44100, 48000, 22050, 8000, 96000]) {
    const x = sine(0.7, 137, 0.9)
    const ref = reference(x, rate)
    const whole = resampleTo16k(x, rate)
    assert.equal(whole.length, ref.length, `${rate}: one-shot length`)
    for (const sizes of [[128], [4096], chunkSizes(rate, 400, 300)]) {
      const r = makeResampler16k(rate)
      const got: number[] = []
      let at = 0
      for (let c = 0; at < x.length; c++) {
        const n = sizes[c % sizes.length]
        for (const v of r(x.subarray(at, at + n))) got.push(v)
        at += n
      }
      assert.equal(got.length, ref.length, `${rate}: no sample dropped or duplicated`)
      for (let k = 0; k < ref.length; k++) {
        // Float32 storage of the same arithmetic: equal to the reference rounded to f32
        assert.equal(got[k], Math.fround(ref[k]), `${rate}, chunks ${sizes.length}: sample ${k}`)
        assert.equal(whole[k], Math.fround(ref[k]))
      }
    }
  }
})

test('the resampler does not drift over a long stream: the count is the exact one', () => {
  const rate = 44100
  const r = makeResampler16k(rate)
  const chunk = new Float32Array(441_000) // 10 s per chunk, 10 minutes in all
  let n = 0
  for (let c = 0; c < 60; c++) n += r(chunk).length
  const total = 60 * chunk.length
  assert.equal(n, Math.floor(((total - 1) * RATE_16K) / rate) + 1)
})

// ------------------------------------------------------------------------------------------
// The framer: every sample to the detector once and in order, a frame decided at its own end.

test('the framer pushes every sample once and in order, the tail of a chunk included', () => {
  const x = sine(1.3, 311, 0.5)
  const pushed: number[] = []
  const atFrame: number[] = []
  const f = new Framer(
    (s) => {
      for (const v of s) pushed.push(v)
    },
    (frame) => {
      atFrame.push(pushed.length) // what the detector holds when the frame is decided
      assert.equal(frame.length, FRAME_SAMPLES)
      const start = pushed.length - FRAME_SAMPLES
      for (let j = 0; j < FRAME_SAMPLES; j++) assert.equal(frame[j], x[start + j])
    },
  )
  let at = 0
  for (const n of chunkSizes(5, 10_000, 700)) {
    if (at >= x.length) break
    f.feed(x.subarray(at, at + n))
    at += n
  }
  assert.deepEqual(pushed, Array.from(x.subarray(0, at)))
  assert.equal(atFrame.length, Math.floor(at / FRAME_SAMPLES))
  atFrame.forEach((n, i) => assert.equal(n, (i + 1) * FRAME_SAMPLES, `frame ${i} decided at its end`))
})

/** A fake detector: the samples pushed since the last reset, and what each decision read. */
function fakeDetector(p: number) {
  const det = { held: 0, total: 0, resets: 0, reads: [] as number[] }
  return {
    det,
    push: (s: Float32Array) => {
      det.held += s.length
      det.total += s.length
    },
    probEnd: () => {
      det.reads.push(det.held)
      return p
    },
    reset: () => {
      det.held = 0
      det.resets++
    },
  }
}

test('one chunk holding a whole clip (the self-test): each decision reads up to its own frame', () => {
  // 1 s of speech then 3 s of zeros, in ONE chunk: the first version pushed the whole chunk before
  // feeding its frames, so the decision at 1.2 s read the stream up to 4 s, mostly silence
  const clip = new Float32Array(4 * RATE_16K)
  clip.set(sine(1, 220, 0.3))
  const d = fakeDetector(0.2)
  const events: TurnEvent[] = []
  const vad = new EnergyVad()
  const policy = new TurnPolicy({
    probEnd: d.probEnd,
    onEvent: (e) => {
      events.push(e)
      if (e.type === 'finished') d.reset()
    },
  })
  const f = new Framer(d.push, (frame) => policy.feed(vad.isSpeech(frame)))
  f.feed(clip)
  const decisions = events.filter((e) => e.type === 'decision')
  assert.ok(decisions.length > 0)
  decisions.forEach((e, i) =>
    assert.equal(d.det.reads[i], e.atMs * (RATE_16K / 1000), `decision at ${e.atMs} ms reads ${d.det.reads[i]} samples`),
  )
  assert.equal(decisions[0].atMs, 1000 + TURN_DEFAULTS.stopMs)
  const fin = events.find((e) => e.type === 'finished')!
  assert.equal(fin.by, 'timer')
  // the reset at the finished turn keeps what follows it for the next turn, nothing before it
  assert.equal(d.det.held, clip.length - fin.atMs * (RATE_16K / 1000))
  assert.equal(d.det.total, clip.length)
})

test('a duration field being edited keeps the default instead of zero', () => {
  assert.equal(validMs(300, 200), 300)
  assert.equal(validMs(0, 200), 0)
  assert.equal(validMs('', 200), 200, "Vue's v-model.number hands '' over while the field is empty")
  assert.equal(validMs(-50, 200), 200)
  assert.equal(validMs(Number.NaN, 200), 200)
  assert.equal(validMs(Number.POSITIVE_INFINITY, 2500), 2500)
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

test('speech resuming on the very frame a recheck was due: resumed, not asked', () => {
  const { policy, events, calls } = script([0.2])
  speak(policy, 16)
  hush(policy, 20) // the decision at 200 ms of silence: holding
  hush(policy, 19) // the recheck is due on the next silent frame
  speak(policy, 1) // speech comes instead
  assert.equal(calls.length, 1, 'not asked on a speech frame')
  assert.equal(events.at(-1)!.type, 'resumed')
  hush(policy, 19)
  assert.equal(calls.length, 1, 'the next pause waits stopMs again')
  hush(policy, 1)
  assert.equal(calls.length, 2)
})

test('settings changed mid-run apply from the next frame', () => {
  const { policy, events, calls } = script([0.6], 0.7)
  speak(policy, 16)
  hush(policy, 20) // p 0.6 under tau 0.7: holding
  assert.equal(policy.state, 'holding')
  policy.tau = 0.5 // the threshold control
  hush(policy, 20) // the recheck now finishes the turn
  assert.equal(events.at(-1)!.type, 'finished')
  assert.equal(events.at(-1)!.by, 'model')
  speak(policy, 16)
  policy.fallbackMs = 300 // lowered under the pause already under way
  policy.stopMs = 10000
  hush(policy, 30)
  assert.equal(events.at(-1)!.by, 'timer')
  assert.equal(calls.length, 2)
})
