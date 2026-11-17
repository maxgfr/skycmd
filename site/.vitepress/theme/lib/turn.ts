// The end-of-turn demo's front end and turn policy (the Listen page): a linear resampler to
// 16 kHz, an energy voice-activity detector over 10 ms frames, and the state machine that asks
// the wasm detector (`TurnDetector`) for its probability of "finished" only when it needs a
// decision: after `stopMs` of silence following speech, then every `recheckMs` while the pause
// lasts, with a fallback timer that ends the turn after `fallbackMs` of silence whatever the
// model says (docs/s6-ear2-prereg.md §10, the operating notes; Pipecat's `stop_secs` 0.2). Pure
// logic, no DOM and no wasm import, so tests/turn.test.ts runs it in node.

/** The rate the detector and the VAD work at. */
export const RATE_16K = 16000

/** One 10 ms frame at 16 kHz. */
export const FRAME_SAMPLES = 160

/** The defaults of the VAD and the policy. They are documented here, on the Listen page and in
 * the operating notes of docs/s6-ear2-prereg.md §10 (Pipecat's `stop_secs` 0.2 and its longer
 * `SmartTurnParams.stop_secs` fallback). */
export const TURN_DEFAULTS = {
  /** Speech needed before a pause counts. */
  minSpeechMs: 150,
  /** Silence after speech before the model is asked. */
  stopMs: 200,
  /** Asked again while the pause lasts. */
  recheckMs: 200,
  /** Silence after which the turn is finished by the timer whatever the model says. */
  fallbackMs: 2500,
  /** The threshold applied to the probability (the demo defaults to the file's tau85). */
  tau: 0.5,
  /** The duration of one frame fed to the policy. */
  frameMs: 10,
  /** A frame is speech only above this RMS, whatever the noise floor. */
  absoluteFloor: 0.008,
  /** A frame is speech when its RMS is above this times the noise floor. */
  ratio: 3.5,
} as const

/** Linear resampling to 16 kHz: the sample at output k interpolates the input at k·rate/16000,
 * for every such position up to the last input sample. At 16 kHz the identity (the input array
 * itself, no copy). */
export function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === RATE_16K) return input
  return makeResampler16k(inputRate)(input)
}

/** A resampler that gives, chunk by chunk, exactly the samples the one-shot resampler gives on the
 * concatenated input: output k sits at input position k·rate/16000, computed from integer counts
 * (no drift however long the stream), and a position between the last sample of a chunk and the
 * first of the next waits for that next sample instead of holding the last one. At 16 kHz it
 * passes chunks through untouched. */
export function makeResampler16k(inputRate: number): (chunk: Float32Array) => Float32Array {
  if (inputRate === RATE_16K) return (chunk) => chunk
  if (!(inputRate > 0)) throw new Error(`a sample rate must be positive, not ${inputRate}`)
  let k = 0 // index of the next output sample
  let base = 0 // input samples before the current chunk
  let prev = 0 // the last sample of the previous chunk
  return (chunk) => {
    const len = chunk.length
    const end = base + len - 1 // the last input position this chunk can serve
    const out = new Float32Array(Math.max(0, Math.ceil(((len + 1) * RATE_16K) / inputRate) + 1))
    let n = 0
    for (;;) {
      const pos = (k * inputRate) / RATE_16K // exact up to the last bit: k * rate is an integer
      if (pos > end) break
      const i = Math.floor(pos)
      const f = pos - i
      const a = i < base ? prev : chunk[i - base] // i is base - 1 at most once per chunk
      out[n++] = f === 0 ? a : a + (chunk[i - base + 1] - a) * f
      k++
    }
    if (len > 0) prev = chunk[len - 1]
    base += len
    return out.subarray(0, n)
  }
}

/** Cuts a stream of 16 kHz chunks into 10 ms frames for the VAD and the policy, and hands every
 * sample to `push` (the detector's stream) exactly once and in order, as it arrives: the tail of a
 * chunk that does not fill a frame is pushed at once and completed by the next chunk. `onFrame`
 * runs on each frame right after its last sample was pushed, so a decision taken there reads the
 * stream up to that frame's end, never past it, and a `reset` of the detector there drops nothing
 * that comes after it. */
export class Framer {
  private readonly buf = new Float32Array(FRAME_SAMPLES)
  private len = 0
  private readonly push: (samples: Float32Array) => void
  private readonly onFrame: (frame: Float32Array) => void

  constructor(push: (samples: Float32Array) => void, onFrame: (frame: Float32Array) => void) {
    this.push = push
    this.onFrame = onFrame
  }

  feed(chunk: Float32Array): void {
    let i = 0
    while (i < chunk.length) {
      const take = Math.min(FRAME_SAMPLES - this.len, chunk.length - i)
      const seg = chunk.subarray(i, i + take)
      this.push(seg)
      this.buf.set(seg, this.len)
      this.len += take
      i += take
      if (this.len === FRAME_SAMPLES) {
        this.len = 0
        this.onFrame(this.buf)
      }
    }
  }
}

/** A duration typed in a number field: a finite value of at least 0, else `fallback`. Vue's
 * `v-model.number` hands over '' while the field is being edited (and a negative number can be
 * typed despite `min`): taken as 0, it would end every pause on its first silent frame. */
export function validMs(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : fallback
}

/** An energy voice-activity detector over 10 ms frames: the RMS of each frame against a noise
 * floor tracked slowly on quiet frames. A frame is speech when its RMS is above
 * `max(absoluteFloor, ratio x noiseFloor)`. It is the energy VAD of the operating notes
 * (docs/s6-ear2-prereg.md §10), not Silero: the demo says so. */
export class EnergyVad {
  /** RMS of the last frame. */
  rms = 0
  /** The noise floor, followed slowly while frames are quiet (never below `FLOOR_MIN`, so
   * digital silence cannot pull it to zero: the absolute floor holds the line there). */
  noiseFloor: number
  private readonly absoluteFloor: number
  private readonly ratio: number
  private readonly adapt: number

  constructor(opts: { absoluteFloor?: number; ratio?: number; adapt?: number; noiseFloor?: number } = {}) {
    this.absoluteFloor = opts.absoluteFloor ?? TURN_DEFAULTS.absoluteFloor
    this.ratio = opts.ratio ?? TURN_DEFAULTS.ratio
    this.adapt = opts.adapt ?? 0.05
    this.noiseFloor = opts.noiseFloor ?? 0.002
  }

  /** Whether one 10 ms frame counts as speech; a quiet frame lowers (or raises) the floor. */
  isSpeech(frame: Float32Array): boolean {
    let sum = 0
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i]
    this.rms = Math.sqrt(sum / Math.max(1, frame.length))
    const speech = this.rms > Math.max(this.absoluteFloor, this.ratio * this.noiseFloor)
    if (!speech) this.noiseFloor += this.adapt * (this.rms - this.noiseFloor)
    this.noiseFloor = Math.max(1e-4, this.noiseFloor)
    return speech
  }
}

export type TurnState = 'waiting' | 'speaking' | 'holding' | 'finished'

export interface TurnEvent {
  type: 'speech-start' | 'decision' | 'finished' | 'resumed'
  /** Milliseconds on the audio timeline, from the policy's start or last `reset`. */
  atMs: number
  /** The probability of "finished" the model returned (`decision`, and the `finished` it caused). */
  p?: number
  /** What declared the turn finished (`finished` only). */
  by?: 'model' | 'timer'
  /** Wall-clock milliseconds spent inside `probEnd()` (`decision` only). */
  decisionMs?: number
}

export interface TurnPolicyOptions {
  /** The detector's probability of "finished". Called only when a decision is needed: at
   * `stopMs` of silence after speech, then every `recheckMs` while the pause lasts. */
  probEnd: () => number
  /** The threshold applied to the probability. */
  tau?: number
  minSpeechMs?: number
  stopMs?: number
  recheckMs?: number
  fallbackMs?: number
  /** The duration of one `feed`, in ms. */
  frameMs?: number
  /** The events, in order. */
  onEvent?: (e: TurnEvent) => void
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now()
}

/** The turn state machine, fed one frame at a time. `waiting` until `minSpeechMs` of speech,
 * then `speaking`; after `stopMs` of silence it asks the model once, then every `recheckMs`
 * while the pause lasts (`holding`); speech during the pause is a `resumed` and the turn goes
 * on; at `fallbackMs` of silence the turn is finished by the timer whatever the model says. A
 * `finished` (the `finished` state is never kept) returns to `waiting`: the caller resets the
 * detector, and the next speech starts a new turn. */
export class TurnPolicy {
  state: TurnState = 'waiting'
  /** The threshold applied to the probability (live: the demo's threshold control). */
  tau: number
  /** Silence after speech before the model is asked (live: the demo's control). */
  stopMs: number
  /** Silence after which the timer ends the turn (live: the demo's control). */
  fallbackMs: number
  /** Milliseconds on the audio timeline. */
  tMs = 0

  private readonly probEnd: () => number
  private readonly minSpeechMs: number
  private readonly recheckMs: number
  private readonly frameMs: number
  private readonly onEvent: (e: TurnEvent) => void
  private speechMs = 0 // the current run of speech frames, while waiting
  private silenceMs = 0 // the current run of silence frames, once speech began
  private sinceAskMs = 0 // silence since the last `probEnd` call

  constructor(options: TurnPolicyOptions) {
    this.probEnd = options.probEnd
    this.tau = options.tau ?? TURN_DEFAULTS.tau
    this.minSpeechMs = options.minSpeechMs ?? TURN_DEFAULTS.minSpeechMs
    this.stopMs = options.stopMs ?? TURN_DEFAULTS.stopMs
    this.recheckMs = options.recheckMs ?? TURN_DEFAULTS.recheckMs
    this.fallbackMs = options.fallbackMs ?? TURN_DEFAULTS.fallbackMs
    this.frameMs = options.frameMs ?? TURN_DEFAULTS.frameMs
    this.onEvent = options.onEvent ?? (() => {})
  }

  /** One frame: `speech` is the VAD's answer for it. Emits the events as they happen. */
  feed(speech: boolean): void {
    this.tMs += this.frameMs
    if (speech) {
      this.silenceMs = 0
      this.sinceAskMs = 0
      if (this.state === 'waiting') {
        this.speechMs += this.frameMs
        if (this.speechMs >= this.minSpeechMs) {
          this.state = 'speaking'
          this.emit({ type: 'speech-start', atMs: this.tMs })
        }
      } else if (this.state === 'holding') {
        this.state = 'speaking'
        this.emit({ type: 'resumed', atMs: this.tMs })
      }
      return
    }
    this.speechMs = 0
    if (this.state === 'waiting') return // silence with no speech before it asks nothing
    this.silenceMs += this.frameMs
    this.sinceAskMs += this.frameMs
    if (this.silenceMs >= this.fallbackMs) {
      this.finish({ type: 'finished', atMs: this.tMs, by: 'timer' })
      return
    }
    const due = this.state === 'speaking' ? this.silenceMs >= this.stopMs : this.sinceAskMs >= this.recheckMs
    if (due) this.ask()
  }

  /** Back to the start: a new stream and a new timeline. */
  reset(): void {
    this.state = 'waiting'
    this.tMs = 0
    this.speechMs = this.silenceMs = this.sinceAskMs = 0
  }

  /** One decision, timed around `probEnd()`: finished at or above `tau`, else the pause is
   * held and the model is asked again every `recheckMs`. */
  private ask(): void {
    const t0 = nowMs()
    const p = this.probEnd()
    const decisionMs = nowMs() - t0
    this.sinceAskMs = 0
    this.emit({ type: 'decision', atMs: this.tMs, p, decisionMs })
    if (p >= this.tau) this.finish({ type: 'finished', atMs: this.tMs, by: 'model', p })
    else this.state = 'holding'
  }

  private finish(e: TurnEvent & { type: 'finished' }): void {
    this.state = 'waiting'
    this.speechMs = this.silenceMs = this.sinceAskMs = 0
    this.emit(e)
  }

  private emit(e: TurnEvent): void {
    this.onEvent(e)
  }
}
