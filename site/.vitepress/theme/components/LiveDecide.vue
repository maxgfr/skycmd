<script setup lang="ts">
// The /live/ demos of the generic s4 decision model: free-text options at inference time,
// calibrated probabilities from one forward pass, adaptive depth (early exit) and an
// abstention option. Every decision runs inside the wasm module (`Engine.decide`).
import { computed, onMounted, ref, watch } from 'vue'
import { formatInt, loadEngine, loadManifest, type ModelEntry, type WasmEngine } from '../lib/wasm'

type Mode = 'playground' | 'tools' | 'triage' | 'guard'
type QT = 'choice' | 'score' | 'noul'

const props = defineProps<{ mode: Mode }>()

interface Answer {
  probs: number[]
  exit: number | null
  n_exits: number
  layers_used: number | null
  static_hit: boolean
  micros: number
}

const NOUL = ['false: no, the statement does not hold', 'true: yes, the statement holds']
const NO_TOOL = 'no_tool: no tool is needed, reply directly or ask for details'
const UNSURE = "unsure: I don't know, none of these fits"

const PRESETS: Record<Mode, { qtype: QT; ins: string; options: string; state: string }> = {
  playground: {
    qtype: 'choice',
    ins: 'What should the on-call engineer do first?',
    options: [
      'rollback: revert the last deploy',
      'scale: add more servers',
      'wait: keep watching, nothing is wrong',
      'escalate: page the database team',
    ].join('\n'),
    state:
      'Error rate jumped from 0.1% to 12% two minutes after the 14:02 deploy. CPU and memory are normal. ' +
      'The database latency is unchanged.',
  },
  tools: {
    qtype: 'choice',
    ins: 'Which tool should the assistant call to handle this request?',
    options: [
      'get_weather: current weather and forecast for a city',
      'send_email: send an email to a contact',
      'create_event: add an event to the calendar',
      'search_web: search the web for recent information',
      'calculator: evaluate an arithmetic expression',
    ].join('\n'),
    state: 'Can you put lunch with Sarah on my calendar for Friday at noon?',
  },
  triage: {
    qtype: 'choice',
    ins: 'Which category does this customer message belong to?',
    options: [
      'billing: invoices, charges and refunds',
      'bug: something in the app is broken',
      'feature request: an idea for the product',
      'account: login, password or profile',
    ].join('\n'),
    state: 'Hi, I was charged twice for my March subscription, can you send the money back?',
  },
  guard: {
    qtype: 'noul',
    ins: 'No sharing of phone numbers, emails or home addresses.',
    options: '',
    state: 'Call me tonight at 06 12 34 56 78, I live at 4 rue des Lilas.',
  },
}

const qtype = ref<QT>(PRESETS[props.mode].qtype)
const ins = ref(PRESETS[props.mode].ins)
const optionsText = ref(PRESETS[props.mode].options)
const state = ref(PRESETS[props.mode].state)
const withUnsure = ref(props.mode === 'triage' || props.mode === 'playground')
const adaptive = ref(true)

const models = ref<ModelEntry[]>([])
const model = ref<ModelEntry | null>(null)
const engine = ref<WasmEngine | null>(null)
const status = ref('Loading the model…')
const answer = ref<Answer | null>(null)
const error = ref('')
const median = ref<number | null>(null)
const samples: number[] = []

/** Rendered option texts, as `sky_data::record::Question::options`. */
const rendered = computed<string[]>(() => {
  if (props.mode === 'guard' || qtype.value === 'noul') return NOUL
  const lines = optionsText.value
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
  let opts = qtype.value === 'score' ? lines.map((l, i) => `level ${i}: ${l}`) : lines
  if (props.mode === 'tools') opts = [...opts, NO_TOOL]
  else if (withUnsure.value && qtype.value === 'choice') opts = [...opts, UNSURE]
  return opts
})

const instructions = computed(() =>
  props.mode === 'guard' ? `The message violates this rule: "${ins.value.trim()}"` : ins.value.trim(),
)

const rows = computed(() => {
  const a = answer.value
  if (!a) return []
  const best = a.probs.indexOf(Math.max(...a.probs))
  return rendered.value.map((o, i) => {
    const [key, ...rest] = o.split(': ')
    return { key, desc: rest.join(': '), p: a.probs[i] ?? 0, best: i === best }
  })
})

/** The answer is an abstention: the abstention option wins, or no option is clearly ahead. */
const abstains = computed(() => {
  const r = rows.value
  if (!r.length || props.mode === 'guard' || qtype.value !== 'choice') return false
  const top = r.find((x) => x.best)!
  return top.key === 'unsure' || top.key === 'no_tool' || top.p < 0.4
})

const violation = computed(() => (props.mode === 'guard' && answer.value ? answer.value.probs[1] : null))

onMounted(async () => {
  const all = await loadManifest()
  models.value = all.filter((m) => m.name.startsWith('s4'))
  model.value = models.value.find((m) => m.weight_format === 'ternary') ?? models.value[0] ?? null
  if (!model.value) {
    status.value = 'No s4 model is published on this site yet.'
    return
  }
  await load()
})

async function load() {
  if (!model.value) return
  status.value = `Loading ${model.value.name} (${formatInt(model.value.bytes / 1024)} KB)…`
  try {
    engine.value = await loadEngine(model.value)
    status.value = ''
    run()
  } catch (e) {
    status.value = `Could not load the model: ${(e as Error).message}`
  }
}

function run() {
  const e = engine.value
  if (!e) return
  error.value = ''
  if (rendered.value.length < 2) {
    error.value = 'Give at least two options.'
    return
  }
  try {
    e.setExitThreshold(adaptive.value ? undefined : 1.0)
    const a = JSON.parse(
      e.decide(qtype.value, instructions.value, JSON.stringify(rendered.value), state.value),
    ) as Answer
    answer.value = a
    samples.push(a.micros)
    if (samples.length > 50) samples.shift()
    const s = [...samples].sort((x, y) => x - y)
    median.value = s[Math.floor(s.length / 2)]
  } catch (err) {
    error.value = (err as Error).message
  }
}

let timer: ReturnType<typeof setTimeout> | null = null
watch([qtype, ins, optionsText, state, withUnsure, adaptive, model], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(run, 120)
})
watch(model, () => load())

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="live">
    <div class="live-form">
      <div v-if="mode === 'playground'" class="live-row">
        <label>Type
          <select v-model="qtype">
            <option value="choice">choice</option>
            <option value="score">score</option>
            <option value="noul">noul (true / false)</option>
          </select>
        </label>
        <label v-if="qtype === 'choice'" class="live-check">
          <input v-model="withUnsure" type="checkbox" /> add “I don't know”
        </label>
      </div>
      <label v-if="mode === 'guard'">Rule
        <input v-model="ins" type="text" />
      </label>
      <label v-else>{{ mode === 'playground' && qtype === 'noul' ? 'Statement' : 'Question' }}
        <input v-model="ins" type="text" />
      </label>
      <label v-if="mode !== 'guard' && !(mode === 'playground' && qtype === 'noul')">
        {{ mode === 'tools' ? 'Tools (name: description, one per line)' : qtype === 'score' ? 'Levels (lowest first, one per line)' : 'Options (key: description, one per line)' }}
        <textarea v-model="optionsText" rows="5" spellcheck="false" />
      </label>
      <label>{{ mode === 'tools' ? 'User request' : mode === 'guard' ? 'Message' : mode === 'triage' ? 'Message' : 'State' }}
        <textarea v-model="state" rows="3" />
      </label>
      <div class="live-row">
        <label class="live-check">
          <input v-model="adaptive" type="checkbox" /> adaptive depth (early exit)
        </label>
        <label v-if="models.length > 1">Model
          <select v-model="model">
            <option v-for="m in models" :key="m.file" :value="m">{{ m.name }} · {{ m.weight_format }}</option>
          </select>
        </label>
      </div>
    </div>

    <p v-if="status" class="live-status">{{ status }}</p>
    <p v-if="error" class="live-error">{{ error }}</p>

    <div v-if="answer && !status" class="live-out">
      <div v-if="violation !== null" class="live-verdict" :class="{ bad: violation >= 0.5 }">
        p(violates the rule) = <strong>{{ pct(violation) }}</strong>
        <span>{{ violation >= 0.5 ? 'flag the message' : 'let it through' }}</span>
      </div>
      <div v-else-if="abstains" class="live-verdict unsure">The model abstains: <strong>“I don't know”</strong></div>
      <ul class="live-bars">
        <li v-for="r in rows" :key="r.key" :class="{ best: r.best }">
          <span class="k">{{ r.key }}</span>
          <span class="bar"><span :style="{ width: pct(r.p) }" /></span>
          <span class="p">{{ pct(r.p) }}</span>
        </li>
      </ul>
      <p class="live-meta">
        exit {{ (answer.exit ?? 0) + 1 }} / {{ answer.n_exits }}
        <template v-if="answer.layers_used !== null"> · {{ answer.layers_used }} block passes</template>
        · {{ formatInt(answer.micros) }} µs<template v-if="median !== null"> (median {{ formatInt(median) }} µs)</template>
        · question {{ answer.static_hit ? 'cached' : 'encoded' }}
        <template v-if="model"> · {{ model.name }}, {{ formatInt(model.bytes / 1024) }} KB</template>
      </p>
    </div>
  </div>
</template>

<style scoped>
.live {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0 24px;
  background: var(--vp-c-bg-soft);
}
.live-form {
  display: grid;
  gap: 10px;
}
.live-form label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.live-form input[type='text'],
.live-form textarea,
.live-form select {
  font: inherit;
  font-weight: 400;
  font-size: 14px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 6px 10px;
  width: 100%;
}
.live-form textarea {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  resize: vertical;
}
.live-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: end;
}
.live-row select {
  width: auto;
}
.live-check {
  display: flex !important;
  align-items: center;
  gap: 6px;
  font-weight: 400 !important;
}
.live-status,
.live-error {
  font-size: 14px;
  margin: 12px 0 0;
}
.live-error {
  color: var(--vp-c-danger-1);
}
.live-out {
  margin-top: 14px;
}
.live-bars {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 6px;
}
.live-bars li {
  display: grid;
  grid-template-columns: minmax(80px, 160px) 1fr 56px;
  gap: 10px;
  align-items: center;
  font-size: 14px;
}
.live-bars .k {
  font-family: var(--vp-font-family-mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-bars .bar {
  height: 10px;
  border-radius: 5px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.live-bars .bar span {
  display: block;
  height: 100%;
  background: var(--vp-c-text-3);
  transition: width 0.15s ease-out;
}
.live-bars li.best .bar span {
  background: var(--vp-c-brand-1);
}
.live-bars li.best .k {
  font-weight: 700;
}
.live-bars .p {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.live-verdict {
  margin-bottom: 10px;
  font-size: 15px;
}
.live-verdict span {
  margin-left: 8px;
  color: var(--vp-c-text-2);
}
.live-verdict.bad strong {
  color: var(--vp-c-danger-1);
}
.live-verdict.unsure strong {
  color: var(--vp-c-warning-1);
}
.live-meta {
  margin: 12px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
</style>
