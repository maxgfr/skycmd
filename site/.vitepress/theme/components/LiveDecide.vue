<script setup lang="ts">
// The /live/ demos of the generic decision models: free-text options at inference time and
// calibrated probabilities, computed in the browser by the wasm module (`Engine.decide`).
// s5-lite (iso architecture) reads each option by its definition only: the keys never reach the
// model, and a state is encoded once for every question asked about it. The presets are the
// demo cases the models are evaluated on (eval/s5/demo-cases.jsonl, lib/demo-cases.ts).
import { computed, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import { formatInt, loadEngine, loadManifest, type ModelEntry, type WasmEngine } from '../lib/wasm'
import { DEMO_CASES, type DemoCase } from '../lib/demo-cases'

type Mode = 'playground' | 'tools' | 'triage' | 'guard'
type QT = 'choice' | 'score' | 'noul'

const props = defineProps<{ mode: Mode }>()

interface Answer {
  probs: number[]
  exit: number | null
  n_exits: number
  layers_used: number | null
  static_hit: boolean
  state_hit?: boolean
  micros: number
}

/** Demo-case sources of each mode (English: s5 is a small English model, with no French version). */
const SOURCES: Record<Mode, string[]> = {
  playground: ['demo_playground_en', 'demo_score_en'],
  tools: ['demo_tools_en'],
  triage: ['demo_triage_en'],
  guard: ['demo_guard_en'],
}
const examples: DemoCase[] = DEMO_CASES.filter((c) => SOURCES[props.mode].includes(c.src))

/** The `noul` options of s4-micro, which was trained with these texts. */
const S4_NOUL = ['false: no, the statement does not hold', 'true: yes, the statement holds']

/** Option lines of a case as the form shows them (`key: definition`; levels for `score`). */
function linesOf(c: DemoCase): string {
  if (c.t === 'score') return c.opts.map(([, d]) => d).join('\n')
  return c.opts.map(([k, d]) => (d ? `${k}: ${d}` : k)).join('\n')
}

const first = examples[0]
const caseIdx = ref(0)
const qtype = ref<QT>(first.t)
const ins = ref(first.ins)
const optionsText = ref(linesOf(first))
const state = ref(first.state)
const lang = ref<'en' | 'fr'>('en')
const adaptive = ref(true)

const models = ref<ModelEntry[]>([])
const model = ref<ModelEntry | null>(null)
const engine = ref<WasmEngine | null>(null)
const status = ref('Loading the model…')
const answer = ref<Answer | null>(null)
const error = ref('')
const median = ref<number | null>(null)
const samples: number[] = []

const isIso = computed(() => model.value?.arch === 'skycmd-iso')

function pick(i: number) {
  const c = examples[i]
  caseIdx.value = i
  qtype.value = c.t
  ins.value = c.ins
  optionsText.value = linesOf(c)
  state.value = c.state
}

/** Rendered option texts, as `sky-eval` renders s5 records (`key: definition`, `level i: …`). */
const rendered = computed<string[]>(() => {
  if (qtype.value === 'noul') {
    if (props.mode === 'guard') {
      return optionsText.value
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean)
    }
    // s5 (iso and Laya formats): `false` / `true` without a definition, as in training (iso reads
    // them as "The statement is false/true."); s4-micro was trained with its own texts.
    return model.value?.arch === 'skycmd-s2' ? S4_NOUL : ['false', 'true']
  }
  const lines = optionsText.value
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
  return qtype.value === 'score' ? lines.map((l, i) => `level ${i}: ${l}`) : lines
})

const rows = computed(() => {
  const a = answer.value
  if (!a) return []
  const best = a.probs.indexOf(Math.max(...a.probs))
  return rendered.value.map((o, i) => {
    const [key, ...rest] = o.split(': ')
    return { key, desc: rest.join(': '), p: a.probs[i] ?? 0, best: i === best }
  })
})

/** The expected option while the form still holds an unmodified demo case. */
const expected = computed<number | null>(() => {
  const c = examples[caseIdx.value]
  if (!c || c.t !== qtype.value || c.ins !== ins.value || c.state !== state.value) return null
  if (linesOf(c) !== optionsText.value) return null
  return c.gold
})

/** The answer is an abstention: the abstention option wins, or no option is clearly ahead. */
const abstains = computed(() => {
  const r = rows.value
  if (!r.length || qtype.value !== 'choice') return false
  const top = r.find((x) => x.best)!
  return top.key === 'unsure' || top.key === 'no_tool' || top.p < 0.4
})

const violation = computed(() => (props.mode === 'guard' && answer.value ? answer.value.probs[1] : null))

onMounted(async () => {
  const all = await loadManifest()
  models.value = all.filter((m) => m.task === 'generic')
  model.value = models.value.find((m) => m.arch === 'skycmd-iso') ?? models.value[0] ?? null
  if (!model.value) {
    status.value = 'No generic decision model is published on this site yet.'
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
      e.decide(qtype.value, ins.value.trim(), JSON.stringify(rendered.value), state.value),
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
watch([qtype, ins, optionsText, state, adaptive, model], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(run, 120)
})
watch(model, () => load())
// s5 has no French version (Phase 6 was closed): the French demos fall back on the English presets.
watch(lang, (l) => {
  if (l === 'fr') pick(caseIdx.value)
})

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}

function short(s: string, n = 46) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s
}
</script>

<template>
  <div class="live">
    <div class="live-row live-top">
      <label>Language
        <select v-model="lang">
          <option value="en">English</option>
          <option value="fr">Français</option>
        </select>
      </label>
      <label v-if="models.length > 1">Model
        <select v-model="model">
          <option v-for="m in models" :key="m.file" :value="m">{{ m.name }} · {{ m.weight_format }}</option>
        </select>
      </label>
      <label v-if="mode === 'playground'">Type
        <select v-model="qtype">
          <option value="choice">choice</option>
          <option value="score">score</option>
          <option value="noul">noul (true / false)</option>
        </select>
      </label>
      <label v-if="model && !isIso" class="live-check">
        <input v-model="adaptive" type="checkbox" /> adaptive depth (early exit)
      </label>
    </div>

    <p v-if="lang === 'fr'" class="live-banner">
      s5 est un petit modèle anglais ; il n'a pas de version française, voir
      <a :href="withBase('/results/')">les résultats</a>. La démo reste sur les presets anglais.
    </p>

    <div class="live-examples">
      <span>Examples:</span>
      <button
        v-for="(c, i) in examples"
        :key="i"
        type="button"
        :class="{ on: i === caseIdx && expected !== null }"
        @click="pick(i)"
      >{{ short(c.state) }}</button>
    </div>

    <div class="live-form">
      <label>{{ mode === 'guard' ? 'Statement to check' : qtype === 'noul' ? 'Statement' : 'Question' }}
        <input v-model="ins" type="text" />
      </label>
      <label v-if="qtype !== 'noul' || mode === 'guard'">
        {{
          mode === 'tools'
            ? 'Tools (name: description, one per line)'
            : mode === 'guard'
              ? 'Answers (false: …, true: …)'
              : qtype === 'score'
                ? 'Levels (lowest first, one per line)'
                : 'Options (key: description, one per line)'
        }}
        <textarea v-model="optionsText" rows="5" spellcheck="false" />
      </label>
      <label>{{ mode === 'tools' ? 'User request' : mode === 'triage' || mode === 'guard' ? 'Message' : 'State' }}
        <textarea v-model="state" rows="3" />
      </label>
    </div>

    <p v-if="status" class="live-status">{{ status }}</p>
    <p v-if="error" class="live-error">{{ error }}</p>

    <div v-if="answer && !status" class="live-out">
      <div v-if="violation !== null" class="live-verdict" :class="{ bad: violation >= 0.5 }">
        p(true) = <strong>{{ pct(violation) }}</strong>
        <span>{{ violation >= 0.5 ? 'flag it' : 'let it through' }}</span>
      </div>
      <div v-else-if="abstains" class="live-verdict unsure">The model abstains: <strong>“I don't know”</strong></div>
      <ul class="live-bars">
        <li v-for="(r, i) in rows" :key="r.key + i" :class="{ best: r.best }">
          <span class="k" :title="r.desc">{{ r.key }}<template v-if="expected === i"> ✓</template></span>
          <span class="bar"><span :style="{ width: pct(r.p) }" /></span>
          <span class="p">{{ pct(r.p) }}</span>
        </li>
      </ul>
      <p v-if="expected !== null" class="live-meta">
        Expected answer: <strong>{{ rows[expected]?.key }}</strong> ·
        {{ rows[expected]?.best ? 'the model agrees' : 'the model disagrees' }}
      </p>
      <p class="live-meta">
        <template v-if="isIso">
          options read by their definitions only · state {{ answer.state_hit ? 'cached' : 'encoded' }}
        </template>
        <template v-else>
          exit {{ (answer.exit ?? 0) + 1 }} / {{ answer.n_exits }}
          <template v-if="answer.layers_used !== null"> · {{ answer.layers_used }} block passes</template>
          · question {{ answer.static_hit ? 'cached' : 'encoded' }}
        </template>
        · {{ formatInt(answer.micros) }} µs<template v-if="median !== null"> (median {{ formatInt(median) }} µs)</template>
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
.live label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.live input[type='text'],
.live textarea,
.live select {
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
.live textarea {
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
.live-top {
  margin-bottom: 10px;
}
.live-check {
  display: flex !important;
  align-items: center;
  gap: 6px;
  font-weight: 400 !important;
}
.live-banner {
  margin: 0 0 10px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  background: var(--vp-c-warning-soft);
}
.live-examples {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.live-examples button {
  font-size: 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 2px 10px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
}
.live-examples button.on {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
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
