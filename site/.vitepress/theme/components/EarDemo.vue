<script setup lang="ts">
// s6-ear in the browser (docs/SPEC-S6.md §16): the microphone (or a sample clip) goes into a
// sliding 10.24 s buffer at 16 kHz, and every 80 ms the wasm engine answers typed questions about
// it (`Engine.decideAudio`): has the speaker finished their turn, which word was said. The
// log-mel front-end runs inside the wasm module, the same code as natively. Nothing leaves the
// page. Below the System 2 threshold the demo says it would hand the clip to a larger model.
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { formatInt, loadManifest, loadWasm, type ModelEntry, type WasmEngine } from '../lib/wasm'

const RATE = 16000
const WINDOW = 163840 // 128 frames of 80 ms = 10.24 s
const FRAME = 1280 // one 80 ms state frame
const TICK_MS = 80

interface Answer {
  probs: number[]
  exit: number | null
  n_exits: number
  layers_used: number | null
  micros: number
}

interface Q {
  id: string
  type: 'noul' | 'choice'
  ins: string
  keys: string[]
  options: string[]
}

/** The pre-registered questions of the audio jev (crates/sky-data/src/audio.rs). */
const QUESTIONS: Q[] = [
  {
    id: 'turn_end',
    type: 'noul',
    ins: 'Has the speaker finished their turn?',
    keys: ['still talking', 'finished'],
    options: ['false: the speaker is still talking or will go on', 'true: the speaker has finished their turn'],
  },
  {
    id: 'keyword',
    type: 'choice',
    ins: 'Which word was said?',
    keys: ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown'],
    options: [
      'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
      'silence: nothing was said', 'unknown: another word',
    ],
  },
]

const SAMPLES = [
  { file: 'ear/yes.wav', label: '“yes” (Speech Commands)' },
  { file: 'ear/fr-mid-turn.wav', label: 'French, mid-turn (Smart Turn)' },
  { file: 'ear/fr-turn-end.wav', label: 'French, end of turn (Smart Turn)' },
]

const entry = ref<ModelEntry | null>(null)
const engine = ref<WasmEngine | null>(null)
const status = ref('Loading…')
const error = ref('')
const running = ref<'' | 'mic' | 'sample'>('')
const answers = ref<Record<string, Answer | null>>({})
const level = ref(0)
const filled = ref(0)
const median = ref<number | null>(null)
/** System 2 hand-off threshold: from the manifest (conformal, `route`), else 0.5. */
const rho = computed<Record<string, number>>(() => ({
  noul: entry.value?.route?.noul ?? 0.5,
  choice: entry.value?.route?.choice ?? 0.5,
}))

// the sliding window, filled from the right
const buf = new Float32Array(WINDOW)
let ctx: AudioContext | null = null
let stream: MediaStream | null = null
let node: AudioWorkletNode | null = null
let src: AudioNode | null = null
let timer: number | null = null
let resample: ((x: Float32Array) => Float32Array) | null = null
const times: number[] = []
/** Run counter: callbacks of an earlier run (a clip still ending, worklet messages in flight)
 * carry an older value and are ignored. */
let gen = 0

function push(x: Float32Array, g: number) {
  if (g !== gen) return
  const y = resample ? resample(x) : x
  if (y.length >= WINDOW) {
    buf.set(y.subarray(y.length - WINDOW))
  } else {
    buf.copyWithin(0, y.length)
    buf.set(y, WINDOW - y.length)
  }
  filled.value = Math.min(WINDOW, filled.value + y.length)
  let peak = 0
  for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]))
  level.value = Math.max(peak, level.value * 0.85)
}

/** Linear resampler from `from` Hz to 16 kHz, carrying its phase across chunks. */
function makeResampler(from: number): (x: Float32Array) => Float32Array {
  const step = from / RATE
  let pos = 0
  return (x) => {
    const out: number[] = []
    while (pos < x.length) {
      const i = Math.floor(pos)
      const f = pos - i
      const b = i + 1 < x.length ? x[i + 1] : x[i]
      out.push(x[i] + (b - x[i]) * f)
      pos += step
    }
    pos -= x.length
    return Float32Array.from(out)
  }
}

function decide() {
  const e = engine.value
  if (!e || filled.value < FRAME) return
  const pcm = buf.subarray(WINDOW - filled.value)
  const next: Record<string, Answer | null> = {}
  let total = 0
  try {
    for (const q of QUESTIONS) {
      const a = JSON.parse(e.decideAudio(q.type, q.ins, JSON.stringify(q.options), pcm)) as Answer
      next[q.id] = a
      total += a.micros
    }
  } catch (err) {
    error.value = String(err)
    stop()
    return
  }
  answers.value = next
  times.push(total)
  if (times.length > 50) times.shift()
  const s = [...times].sort((a, b) => a - b)
  median.value = s[Math.floor(s.length / 2)]
}

const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true }
}
registerProcessor('ear-tap', Tap)`

async function startContext(): Promise<AudioContext> {
  // a 16 kHz context resamples the microphone itself where the browser allows it
  let c: AudioContext
  try {
    c = new AudioContext({ sampleRate: RATE })
  } catch {
    c = new AudioContext()
  }
  resample = c.sampleRate === RATE ? null : makeResampler(c.sampleRate)
  const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
  await c.audioWorklet.addModule(url)
  URL.revokeObjectURL(url)
  node = new AudioWorkletNode(c, 'ear-tap')
  const g = gen
  node.port.onmessage = (ev) => push(ev.data as Float32Array, g)
  return c
}

function reset() {
  buf.fill(0)
  filled.value = 0
  answers.value = {}
  times.length = 0
  median.value = null
}

async function startMic() {
  error.value = ''
  stop()
  reset()
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    ctx = await startContext()
    try {
      src = ctx.createMediaStreamSource(stream)
    } catch {
      // Firefox refuses a microphone in a context at another rate: resample in JS instead
      await ctx.close()
      ctx = new AudioContext()
      resample = makeResampler(ctx.sampleRate)
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
      await ctx.audioWorklet.addModule(url)
      URL.revokeObjectURL(url)
      node = new AudioWorkletNode(ctx, 'ear-tap')
      const g = gen
      node.port.onmessage = (ev) => push(ev.data as Float32Array, g)
      src = ctx.createMediaStreamSource(stream)
    }
    src.connect(node!)
    running.value = 'mic'
    timer = window.setInterval(decide, TICK_MS)
  } catch (err) {
    error.value = `Microphone: ${err}`
    stop()
  }
}

async function playSample(file: string) {
  error.value = ''
  stop()
  reset()
  try {
    ctx = await startContext()
    const res = await fetch(withBase('/' + file))
    const audio = await ctx.decodeAudioData(await res.arrayBuffer())
    const s = ctx.createBufferSource()
    s.buffer = audio
    s.connect(node!)
    s.connect(ctx.destination)
    const g = gen
    s.onended = () => {
      // one last decision on the whole clip, then stop (unless another run started meanwhile)
      window.setTimeout(() => {
        if (g !== gen) return
        decide()
        stop(true)
      }, 120)
    }
    src = s
    s.start()
    running.value = 'sample'
    timer = window.setInterval(decide, TICK_MS)
  } catch (err) {
    error.value = `Sample: ${err}`
    stop()
  }
}

function stop(keep = false) {
  gen += 1
  if (timer !== null) window.clearInterval(timer)
  timer = null
  try {
    src?.disconnect()
  } catch {
    /* already disconnected */
  }
  src = null
  node?.port.close()
  node = null
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
  ctx?.close().catch(() => {})
  ctx = null
  running.value = ''
  if (!keep) level.value = 0
}

function top(a: Answer | null | undefined): { i: number; p: number } {
  if (!a) return { i: -1, p: 0 }
  let i = 0
  a.probs.forEach((p, j) => {
    if (p > a.probs[i]) i = j
  })
  return { i, p: a.probs[i] }
}

onMounted(async () => {
  try {
    const params = new URLSearchParams(window.location.search)
    const list = await loadManifest()
    const dev = params.get('model')
    const found = dev
      ? ({ name: dev.split('/').pop() ?? dev, file: dev, params: 0, bytes: 0, task: 'ear' } as ModelEntry)
      : list.find((m) => m.task === 'ear') ?? null
    entry.value = found
    if (!found) {
      status.value = ''
      return
    }
    const mod = await loadWasm()
    const res = await fetch(withBase(dev ? dev : `/models/${found.file}`))
    if (!res.ok) throw new Error(`${found.file}: HTTP ${res.status}`)
    const e = new mod.Engine(new Uint8Array(await res.arrayBuffer()))
    if (!e.isAudio()) throw new Error(`${found.name} is not an audio (s6-ear) model`)
    engine.value = e
    status.value = `${e.name()} · ${formatInt(e.n_params())} parameters · ${(e.file_bytes() / 1024).toFixed(0)} KB`
  } catch (err) {
    error.value = String(err)
    status.value = ''
  }
})

onBeforeUnmount(() => {
  stop()
  engine.value?.free()
})
</script>

<template>
  <div class="ear">
    <p v-if="!entry && !status" class="ear-note">
      <strong>No s6-ear model is published.</strong> The first one failed its
      <a href="https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md">pre-registered gates</a>. This demo is
      wired to the engine and will answer here as soon as a model passes them.
    </p>
    <template v-else>
      <p class="ear-status">{{ status }}</p>
      <div class="ear-controls">
        <button :disabled="!engine" @click="running === 'mic' ? stop() : startMic()">
          {{ running === 'mic' ? '■ Stop the microphone' : '🎙 Use the microphone' }}
        </button>
        <span class="ear-or">or a sample clip:</span>
        <button v-for="s in SAMPLES" :key="s.file" :disabled="!engine" class="ear-sample" @click="playSample(s.file)">
          ▶ {{ s.label }}
        </button>
      </div>
      <div class="ear-meter" aria-label="input level">
        <div class="ear-meter-bar" :style="{ width: Math.min(100, level * 140) + '%' }" />
        <span class="ear-meter-text">{{ (filled / RATE).toFixed(2) }} s of 10.24 s heard</span>
      </div>
      <div v-for="q in QUESTIONS" :key="q.id" class="ear-q">
        <div class="ear-q-head">
          <strong>{{ q.ins }}</strong>
          <span v-if="answers[q.id]" class="ear-q-meta">
            exit {{ (answers[q.id]!.exit ?? 0) + 1 }} / {{ answers[q.id]!.n_exits }} ·
            {{ answers[q.id]!.micros.toFixed(0) }} µs
          </span>
        </div>
        <template v-if="answers[q.id]">
          <p
            class="ear-verdict"
            :class="{ 'ear-handoff': top(answers[q.id]).p < rho[q.type] }"
          >
            <template v-if="top(answers[q.id]).p >= rho[q.type]">
              → {{ q.keys[top(answers[q.id]).i] }} ({{ (top(answers[q.id]).p * 100).toFixed(0) }} %)
            </template>
            <template v-else>
              → not sure ({{ (top(answers[q.id]).p * 100).toFixed(0) }} % &lt; {{ (rho[q.type] * 100).toFixed(0) }} %): I would hand this
              clip to a larger model
            </template>
          </p>
          <div v-for="(k, i) in q.keys" :key="k" class="ear-bar">
            <span class="ear-bar-key">{{ k }}</span>
            <span class="ear-bar-track"><span class="ear-bar-fill" :style="{ width: answers[q.id]!.probs[i] * 100 + '%' }" /></span>
            <span class="ear-bar-p">{{ (answers[q.id]!.probs[i] * 100).toFixed(1) }} %</span>
          </div>
        </template>
        <p v-else class="ear-wait">Waiting for 80 ms of audio…</p>
      </div>
      <p v-if="median !== null" class="ear-foot">
        Median {{ (median / 1000).toFixed(2) }} ms for both questions, every 80 ms, in WebAssembly on this page; the
        audio never leaves your browser.
      </p>
    </template>
    <p v-if="error" class="ear-error">{{ error }}</p>
  </div>
</template>

<style scoped>
.ear { border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 16px; margin: 16px 0; }
.ear-note { margin: 0; }
.ear-status { margin: 0 0 12px; color: var(--vp-c-text-2); font-size: 0.9em; }
.ear-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
.ear-controls button {
  border: 1px solid var(--vp-c-brand-1); border-radius: 6px; padding: 4px 12px; color: var(--vp-c-brand-1);
}
.ear-controls button:disabled { opacity: 0.5; cursor: not-allowed; }
.ear-or { color: var(--vp-c-text-2); font-size: 0.9em; }
.ear-meter { position: relative; height: 18px; background: var(--vp-c-bg-soft); border-radius: 4px; margin-bottom: 16px; }
.ear-meter-bar { position: absolute; inset: 0 auto 0 0; background: var(--vp-c-brand-soft); border-radius: 4px; }
.ear-meter-text { position: relative; font-size: 0.8em; padding-left: 8px; line-height: 18px; color: var(--vp-c-text-2); }
.ear-q { margin-bottom: 16px; }
.ear-q-head { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.ear-q-meta { color: var(--vp-c-text-2); font-size: 0.85em; font-variant-numeric: tabular-nums; }
.ear-verdict { margin: 4px 0 8px; font-weight: 600; }
.ear-handoff { color: var(--vp-c-warning-1); }
.ear-bar { display: grid; grid-template-columns: 7em 1fr 4.5em; gap: 8px; align-items: center; font-size: 0.85em; }
.ear-bar-track { height: 8px; background: var(--vp-c-bg-soft); border-radius: 4px; overflow: hidden; }
.ear-bar-fill { display: block; height: 100%; background: var(--vp-c-brand-1); }
.ear-bar-p { text-align: right; font-variant-numeric: tabular-nums; }
.ear-wait, .ear-foot { color: var(--vp-c-text-2); font-size: 0.9em; }
.ear-error { color: var(--vp-c-danger-1); }
</style>
