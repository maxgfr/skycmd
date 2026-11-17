<script setup lang="ts">
// The end-of-turn demo (s6-ear2-nano, the Listen page): an energy VAD (lib/turn.ts) watches 10 ms
// frames of the microphone (or of a synthetic clip), and a `TurnPolicy` asks the wasm detector
// (`TurnDetector`, crates/sky-infer/src/wasm.rs) for the probability that the turn is finished
// only when it needs a decision: after `stopMs` of silence (200 ms, Pipecat's `stop_secs`), then
// every `recheckMs` while the pause lasts. A fallback timer ends the turn after 2.5 s of silence
// whatever the model says, because on real conversations the model answers "finished" on about
// half of the real ends of turn at the cautious threshold (the file's tau85, the default here).
// Everything runs in the page: no audio is sent anywhere.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  engineLabel,
  formatInt,
  formatKB,
  formatMicros,
  loadTurnDetector,
  loadTurnEntry,
  turnSynthSpeech,
  type WasmTurnDetector,
} from '../lib/wasm'
import {
  EnergyVad,
  Framer,
  RATE_16K,
  TurnPolicy,
  TURN_DEFAULTS,
  makeResampler16k,
  validMs,
  type TurnEvent,
  type TurnState,
} from '../lib/turn'

/** Speech needed before a pause counts, and the recheck pace of a held pause (lib/turn.ts). */
const MIN_SPEECH = TURN_DEFAULTS.minSpeechMs
const RECHECK = TURN_DEFAULTS.recheckMs

const det = ref<WasmTurnDetector | null>(null)
/** The facts line, built once at load (the model, its size, its window, the wasm build). */
const status = ref('Loading…')
const missing = ref(false)
const error = ref('')
const running = ref<'' | 'mic'>('')
const phase = ref<TurnState>('waiting')
const doneBy = ref<'model' | 'timer'>('model')
const lastP = ref<number | null>(null)
const lastDecisionMs = ref<number | null>(null)
const level = ref(0)
const heardMs = ref(0)
const log = ref<LogRow[]>([])

interface LogRow {
  id: number
  atMs: number
  text: string
  p?: number
  decisionMs?: number
}

let logSeq = 0
function pushLog(text: string, p?: number, decisionMs?: number) {
  log.value = [{ id: ++logSeq, atMs: policy ? policy.tMs : 0, text, p, decisionMs }, ...log.value].slice(0, 8)
}

// The controls. The two thresholds come from the model file, read at load; the times feed the
// policy live. Defaults: tau85 (cautious: fewer cut-offs), 200 ms, 2500 ms (lib/turn.ts).
const tau85 = ref(0.724)
const tau50 = ref(0.5)
const tauChoice = ref<'tau85' | 'tau50'>('tau85')
const stopMs = ref(TURN_DEFAULTS.stopMs)
const fallbackMs = ref(TURN_DEFAULTS.fallbackMs)
const tau = computed(() => (tauChoice.value === 'tau85' ? tau85.value : tau50.value))
// A field being edited holds '' (or a negative number): the policy keeps the default meanwhile
const stopMsOk = computed(() => validMs(stopMs.value, TURN_DEFAULTS.stopMs))
const fallbackMsOk = computed(() => validMs(fallbackMs.value, TURN_DEFAULTS.fallbackMs))
/** The silence the timer waited for, kept with the outcome it produced. */
const doneAfterMs = ref(0)
watch(tau, (v) => {
  if (policy) policy.tau = Math.max(0, v)
})
watch(stopMsOk, (v) => {
  if (policy) policy.stopMs = v
})
watch(fallbackMsOk, (v) => {
  if (policy) policy.fallbackMs = v
})

/** The live status line, announced by screen readers as it changes. */
const phaseText = computed(() => {
  if (phase.value === 'speaking') return 'Speaking…'
  if (phase.value === 'holding')
    return `Pause: the model says not finished (p = ${p2(lastP.value)}), still listening`
  if (phase.value === 'finished')
    return doneBy.value === 'timer'
      ? `Finished by the fallback timer after ${(doneAfterMs.value / 1000).toFixed(1)} s of silence`
      : `Finished (p = ${p2(lastP.value)}): an agent would answer now`
  return 'Waiting for speech'
})

/** A probability in two decimals, as the status line shows it. */
function p2(v: number | null): string {
  return (v ?? 0).toFixed(2)
}

// The pipeline state. `gen` tags the current run: callbacks of an earlier run (a worklet message
// in flight, a synthetic clip still feeding) carry an older value and are ignored.
let gen = 0
let policy: TurnPolicy | null = null
let vad: EnergyVad | null = null
let ctx: AudioContext | null = null
let stream: MediaStream | null = null
let node: AudioWorkletNode | ScriptProcessorNode | null = null
let mute: GainNode | null = null
let src: AudioNode | null = null
let resample: ((x: Float32Array) => Float32Array) | null = null
let framer: Framer | null = null

/** One resampled chunk, cut into 10 ms frames: each frame's samples go into the detector, then
 * the frame into the VAD and the policy, so a decision reads the audio up to its own frame and the
 * reset at a finished turn keeps what follows it for the next turn. Errors land in the UI; the run
 * stops, it does not throw. */
function onChunk(x: Float32Array, g: number) {
  if (g !== gen || !policy || !vad || !framer || !det.value) return
  try {
    const y = resample ? resample(x) : x
    heardMs.value += (y.length * 1000) / RATE_16K
    let peak = 0
    for (let i = 0; i < y.length; i++) {
      const a = Math.abs(y[i])
      if (a > peak) peak = a
    }
    level.value = Math.max(peak, level.value * 0.85)
    framer.feed(y)
  } catch (err) {
    error.value = `The detector failed: ${(err as Error).message ?? err}`
    stop()
  }
}

const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true }
}
registerProcessor('turn-tap', Tap)`

/** The audio graph: the tap that forwards each input block to the main thread, an AudioWorklet
 * where the browser has one, a ScriptProcessorNode (kept pulling by a muted gain) otherwise.
 * Returns the node, for `src.connect`. */
async function addTap(c: AudioContext): Promise<AudioWorkletNode | ScriptProcessorNode> {
  const g = gen
  if (c.audioWorklet) {
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
    let loaded = false
    try {
      await c.audioWorklet.addModule(url)
      loaded = true
    } catch {
      // a blob: module refused (a content security policy, an older engine): the fallback below
    } finally {
      URL.revokeObjectURL(url)
    }
    if (loaded) {
      const w = new AudioWorkletNode(c, 'turn-tap')
      w.port.onmessage = (ev) => onChunk(ev.data as Float32Array, g)
      return w
    }
  }
  const sp = c.createScriptProcessor(4096, 1, 1)
  sp.onaudioprocess = (ev) => onChunk(new Float32Array(ev.inputBuffer.getChannelData(0)), g)
  return sp
}

/** A new run: the VAD and the policy, and the readouts back to their start. */
function beginPipeline() {
  const d = det.value!
  // a new stream: the audio of an earlier run (stopped mid-turn) must not enter this one's window
  d.reset()
  const v = new EnergyVad()
  const p = new TurnPolicy({
    probEnd: () => d.probEnd(),
    tau: Math.max(0, tau.value),
    stopMs: stopMsOk.value,
    recheckMs: RECHECK,
    fallbackMs: fallbackMsOk.value,
    minSpeechMs: MIN_SPEECH,
    onEvent: handleEvent,
  })
  vad = v
  policy = p
  framer = new Framer(
    (samples) => d.push(samples),
    (frame) => p.feed(v.isSpeech(frame)),
  )
  log.value = []
  lastP.value = null
  lastDecisionMs.value = null
  phase.value = 'waiting'
  doneBy.value = 'model'
}

function handleEvent(e: TurnEvent) {
  if (e.type === 'speech-start') {
    phase.value = 'speaking'
    pushLog('speech began')
  } else if (e.type === 'resumed') {
    phase.value = 'speaking'
    pushLog('speech resumed: the same turn goes on')
  } else if (e.type === 'decision') {
    lastP.value = e.p ?? null
    lastDecisionMs.value = e.decisionMs ?? null
    if ((e.p ?? 0) < (policy?.tau ?? tau.value)) {
      phase.value = 'holding'
      pushLog('the model says not finished: still listening', e.p, e.decisionMs)
    }
    // at or above tau the `finished` right after carries the outcome; its log line shows the numbers
  } else if (e.type === 'finished') {
    det.value?.reset() // the caller resets the detector: the next speech starts a new turn
    phase.value = 'finished'
    doneBy.value = e.by ?? 'model'
    doneAfterMs.value = policy?.fallbackMs ?? fallbackMsOk.value
    if (e.by === 'timer') pushLog('the fallback timer ends the turn')
    else pushLog('the model says finished: an agent would answer now', e.p, lastDecisionMs.value)
  }
}

async function startMic() {
  error.value = ''
  stop()
  if (!det.value) return
  const md = navigator.mediaDevices
  if (!md?.getUserMedia) {
    error.value = 'This browser exposes no microphone API: navigator.mediaDevices is missing.'
    return
  }
  // This start's run. A stop (the Stop button, leaving the page) or another start while a promise
  // below is pending bumps `gen`: this start then releases what it holds and goes no further, so
  // no microphone or audio context outlives the run it was opened for.
  const g = gen
  let s: MediaStream
  try {
    s = await md.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
  } catch (err) {
    if (g === gen) error.value = `Microphone unavailable or refused: ${(err as Error).message ?? err}`
    return
  }
  if (g !== gen) {
    s.getTracks().forEach((t) => t.stop())
    return
  }
  stream = s
  try {
    // a 16 kHz context resamples the microphone itself where the browser allows it
    let c: AudioContext
    try {
      c = new AudioContext({ sampleRate: RATE_16K })
    } catch {
      c = new AudioContext()
    }
    ctx = c // from here a stop() closes it
    let tap: AudioWorkletNode | ScriptProcessorNode
    let source: MediaStreamAudioSourceNode
    try {
      tap = await addTap(c)
      if (g !== gen) return
      source = c.createMediaStreamSource(s)
    } catch {
      if (g !== gen) return
      // Firefox refuses a microphone in a context at another rate: the default rate, resampled
      await c.close().catch(() => {})
      if (g !== gen) return
      c = new AudioContext()
      ctx = c
      tap = await addTap(c)
      if (g !== gen) return
      source = c.createMediaStreamSource(s)
    }
    resample = c.sampleRate === RATE_16K ? null : makeResampler16k(c.sampleRate)
    node = tap
    src = source
    src.connect(node)
    if (!('port' in node)) {
      // a ScriptProcessorNode only runs when it drains to the destination: through a muted gain
      mute = c.createGain()
      mute.gain.value = 0
      node.connect(mute)
      mute.connect(c.destination)
    }
    beginPipeline()
    running.value = 'mic'
  } catch (err) {
    error.value = `Audio: ${(err as Error).message ?? err}`
    stop()
  }
}

// The self-test (not speech: the engine's deterministic test signal, a pulse train and noise):
// 3 s of it then 3 s of zeros (longer than the fallback timer, so the test ends a turn), run through exactly the same pipeline as the microphone (same
// VAD, same policy, same detector) as fast as possible, so the events appear at once. Its RMS is
// well above the VAD's absolute floor (0.008), so the VAD calls it speech; this checks that the
// pipeline and the engine run in this browser, not that the model understands speech.
async function runSelfTest() {
  error.value = ''
  stop() // also clears `resample`: the signal is already 16 kHz, nothing to resample
  if (!det.value) return
  const g = gen
  try {
    const speech = await turnSynthSpeech(3, 1n, 1)
    if (g !== gen) return // a start or a stop came in meanwhile: this test is no longer the run
    const clip = new Float32Array(speech.length + Math.round(3 * RATE_16K)) // the tail is zeros, longer than the fallback timer
    clip.set(speech)
    beginPipeline()
    onChunk(clip, gen)
  } catch (err) {
    error.value = `Self-test: ${(err as Error).message ?? err}`
    stop()
  }
}

/** The audio teardown. The readouts (phase, probability, log) stay as the run left them, so the
 * outcome can be read afterwards; the next start resets them. */
function stop() {
  gen += 1
  try {
    src?.disconnect()
  } catch {
    /* already disconnected */
  }
  src = null
  if (mute) {
    try {
      mute.disconnect()
    } catch {
      /* already disconnected */
    }
    mute = null
  }
  if (node) {
    // duck-typed: a browser without one of the two classes must still reach the teardown below
    if ('port' in node) node.port.close()
    else node.onaudioprocess = null
    try {
      node.disconnect()
    } catch {
      /* already disconnected */
    }
    node = null
  }
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
  ctx?.close().catch(() => {})
  ctx = null
  resample = null
  policy = null
  vad = null
  framer = null
  running.value = ''
  level.value = 0
  heardMs.value = 0
}

onMounted(async () => {
  try {
    const found = await loadTurnEntry()
    if (!found) {
      missing.value = true
      status.value = ''
      return
    }
    const d = await loadTurnDetector(found)
    tau85.value = d.tau85()
    tau50.value = d.tau50()
    const inf = JSON.parse(d.info()) as {
      name: string
      params: number
      file_bytes: number
      window_hops: number
    }
    const line = [
      inf.name,
      `${formatInt(inf.params)} parameters`,
      formatKB(inf.file_bytes),
      // hops of 10 ms: the window is 4 s
      `window ${((inf.window_hops * TURN_DEFAULTS.frameMs) / 1000).toFixed(1)} s`,
    ]
    const eng = engineLabel()
    if (eng) line.push(eng)
    status.value = line.join(' · ')
    det.value = d // last: the demo appears only once everything above held
  } catch (err) {
    error.value = String(err)
    status.value = ''
  }
})

onBeforeUnmount(() => {
  // no free() here: loadTurnDetector caches the instance for the whole page, and a freed one
  // would be handed back on the next visit
  stop()
})
</script>

<template>
  <div class="turn">
    <p v-if="status" class="turn-status">{{ status }}</p>
    <p v-if="missing" class="turn-note">The end-of-turn model (s6-ear2-nano) is not published on this site.</p>
    <template v-if="det">
      <div class="turn-controls">
        <button @click="running === 'mic' ? stop() : startMic()">
          {{ running === 'mic' ? 'Stop the microphone' : 'Use the microphone' }}
        </button>
        <button :disabled="running === 'mic'" @click="runSelfTest()">Self-test without a microphone</button>
      </div>
      <div class="turn-tunables">
        <label>threshold
          <select v-model="tauChoice">
            <option value="tau85">cautious: fewer cut-offs (τ85 = {{ tau85.toFixed(3) }})</option>
            <option value="tau50">accuracy (τ50 = {{ tau50.toFixed(3) }})</option>
          </select>
        </label>
        <label>ask after <input v-model.number="stopMs" type="number" min="0" step="50" /> ms of pause</label>
        <label>end after <input v-model.number="fallbackMs" type="number" min="0" step="250" /> ms of silence</label>
      </div>
      <div
        class="turn-meter"
        role="meter"
        aria-label="input level"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuenow="Math.round(Math.min(100, level * 140))"
      >
        <div class="turn-meter-bar" :style="{ width: Math.min(100, level * 140) + '%' }" />
        <span class="turn-meter-text">{{ (heardMs / 1000).toFixed(1) }} s heard</span>
      </div>
      <p class="turn-phase" :class="phase" aria-live="polite">{{ phaseText }}</p>
      <!-- drawn only: the line below states p and the threshold in words -->
      <div class="turn-bar" aria-hidden="true">
        <div class="turn-bar-fill" :style="{ width: (lastP ?? 0) * 100 + '%' }" />
        <div class="turn-bar-tau" :style="{ left: tau * 100 + '%' }" :title="`threshold τ = ${tau.toFixed(3)}`" />
      </div>
      <p v-if="lastP !== null" class="turn-prob">
        p(finished) = {{ lastP.toFixed(3) }} ·
        <template v-if="lastP >= tau">at or above τ = {{ tau.toFixed(3) }}: finished</template>
        <template v-else>below τ = {{ tau.toFixed(3) }}: the pause is held</template>
        <template v-if="lastDecisionMs !== null"> · {{ formatMicros(lastDecisionMs * 1000) }} in the model</template>
      </p>
      <ul v-if="log.length" class="turn-log">
        <li v-for="r in log" :key="r.id">
          <span class="turn-log-at">{{ (r.atMs / 1000).toFixed(2) }} s</span>
          <span>{{ r.text }}</span>
          <span v-if="r.p !== undefined" class="turn-log-p">
            p = {{ r.p.toFixed(3) }}<template v-if="r.decisionMs !== undefined"> ·
              {{ formatMicros(r.decisionMs * 1000) }}</template>
          </span>
        </li>
      </ul>
      <p class="turn-foot">
        An energy VAD (not a neural one) cuts the audio into 10 ms frames and says speech or silence; only at a
        pause does the model read its last 4 s, about 2.4 ms per decision in WebAssembly, nothing between pauses.
        The self-test runs the engine's test signal, not speech: it only checks that the pipeline and the engine
        run in this browser. Everything runs in your browser: no audio is sent anywhere.
      </p>
    </template>
    <p v-if="error" class="turn-error" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
.turn { border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 16px; margin: 16px 0; }
.turn-note { margin: 0; }
.turn-status { margin: 0 0 12px; color: var(--vp-c-text-2); font-size: 0.9em; }
.turn-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
.turn-controls button {
  border: 1px solid var(--vp-c-brand-1); border-radius: 6px; padding: 4px 12px; color: var(--vp-c-brand-1);
}
.turn-controls button:disabled { opacity: 0.5; cursor: not-allowed; }
.turn-tunables {
  display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 12px;
  color: var(--vp-c-text-2); font-size: 0.9em;
}
.turn-tunables label { display: inline-flex; gap: 6px; align-items: center; }
.turn-tunables select,
.turn-tunables input {
  font: inherit; font-size: 0.95em; color: var(--vp-c-text-1); background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 2px 6px;
}
.turn-tunables input { width: 5.5em; }
.turn-meter { position: relative; height: 18px; background: var(--vp-c-bg-soft); border-radius: 4px; margin-bottom: 14px; }
.turn-meter-bar { position: absolute; inset: 0 auto 0 0; background: var(--vp-c-brand-soft); border-radius: 4px; }
.turn-meter-text { position: relative; font-size: 0.8em; padding-left: 8px; line-height: 18px; color: var(--vp-c-text-2); }
.turn-phase { margin: 0 0 6px; font-weight: 600; }
.turn-bar { position: relative; height: 10px; background: var(--vp-c-bg-soft); border-radius: 5px; overflow: hidden; margin: 0 0 12px; }
.turn-bar-fill { position: absolute; inset: 0 auto 0 0; background: var(--vp-c-brand-1); }
.turn-bar-tau { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--vp-c-warning-1); }
.turn-phase.holding { color: var(--vp-c-warning-1); }
.turn-phase.finished { color: var(--vp-c-green-1, var(--vp-c-brand-1)); }
.turn-prob { margin: 0 0 10px; color: var(--vp-c-text-2); font-size: 0.9em; font-variant-numeric: tabular-nums; }
.turn-log {
  list-style: none; margin: 0 0 12px; padding: 0; display: grid; gap: 2px;
  font-size: 0.85em; color: var(--vp-c-text-2); font-variant-numeric: tabular-nums;
}
.turn-log-at { display: inline-block; min-width: 5em; color: var(--vp-c-text-3); }
.turn-log-p { color: var(--vp-c-text-3); }
.turn-foot { margin: 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.turn-error { color: var(--vp-c-danger-1); }
</style>
