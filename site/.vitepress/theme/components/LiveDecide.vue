<script setup lang="ts">
// The /live/ demos of the generic decision models: free-text options at inference time and
// calibrated probabilities, computed in the browser by the wasm module (`Engine.decide`).
// s5-lite (iso architecture) reads each option by its definition only: the keys never reach the
// model, and a state is encoded once for every question asked about it. The presets are the
// demo cases the models are evaluated on (eval/s5/demo-cases.jsonl, lib/demo-cases.ts). The `s6`
// mode asks s6-nano the game and numeric questions it was trained for (lib/s6-cases.ts), with the
// options of each record, `noul` answers included. The `secrets` mode asks s6-secrets-pico the four
// questions of its jev about a pasted text, with the generator's own texts (lib/s6-cases.ts).
// Times: the first call of a demo pays the module start-up and the first encodings, so it is shown
// apart, and the median covers the warm calls after it (lib/wasm.ts, DecisionTimes). The timing
// lines name the build of the module in use (relaxed SIMD or SIMD128, lib/wasm.ts loadWasm).
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import {
  clockStepMicros,
  DecisionTimes,
  engineLabel,
  formatInt,
  formatMicros,
  loadEngine,
  loadManifest,
  questionModels,
  type ModelEntry,
  type WasmEngine,
} from '../lib/wasm'
import { DEMO_CASES, type DemoCase } from '../lib/demo-cases'
import {
  S6_CASES,
  SECRET_CASES,
  SECRET_SERVICES,
  secretQuestions,
  type SecretQuestion,
  type SecretQuestionId,
} from '../lib/s6-cases'

type Mode = 'playground' | 'tools' | 'triage' | 'guard' | 's6' | 'secrets'
type QT = 'choice' | 'score' | 'noul'

/** `prefer`: the model selected first, by name (default: the iso model, else the first one). */
const props = defineProps<{ mode: Mode; prefer?: string }>()

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
const SOURCES: Record<Exclude<Mode, 's6' | 'secrets'>, string[]> = {
  playground: ['demo_playground_en', 'demo_score_en'],
  tools: ['demo_tools_en'],
  triage: ['demo_triage_en'],
  guard: ['demo_guard_en'],
}
const isSecrets = props.mode === 'secrets'
const examples: DemoCase[] =
  props.mode === 's6'
    ? S6_CASES
    : isSecrets
      ? []
      : DEMO_CASES.filter((c) => SOURCES[props.mode as Exclude<Mode, 's6' | 'secrets'>].includes(c.src))
/** Modes whose `noul` answers are written in the form (the records' own texts) instead of fixed. */
const ownNoul = props.mode === 'guard' || props.mode === 's6'

/** The `noul` options of s4-micro, which was trained with these texts. */
const S4_NOUL = ['false: no, the statement does not hold', 'true: yes, the statement holds']

/** Option lines of a case as the form shows them (`key: definition`; levels for `score`). */
function linesOf(c: DemoCase): string {
  if (c.t === 'score') return c.opts.map(([, d]) => d).join('\n')
  return c.opts.map(([k, d]) => (d ? `${k}: ${d}` : k)).join('\n')
}

const first: DemoCase | undefined = examples[0]
const caseIdx = ref(0)
const qtype = ref<QT>(first?.t ?? 'choice')
const ins = ref(first?.ins ?? '')
const optionsText = ref(first ? linesOf(first) : '')
const state = ref(first?.state ?? '')
const lang = ref<'en' | 'fr'>('en')
const adaptive = ref(true)

const models = ref<ModelEntry[]>([])
const model = ref<ModelEntry | null>(null)
const engine = ref<WasmEngine | null>(null)
const status = ref('Loading the model…')
const answer = ref<Answer | null>(null)
const error = ref('')
const times = reactive(new DecisionTimes())
/** The browser clock's step in µs, shown when it is coarse (measured on the client, at mount). */
const clockStep = ref(0)

const isIso = computed(() => model.value?.arch === 'skycmd-iso')
/** Early exit can be switched on: an s2 file published with its exit thresholds. */
const canExit = computed(() => !!model.value && !isIso.value && !model.value.full_depth)

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
    if (ownNoul) {
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
  if (!r.length || qtype.value !== 'choice' || props.mode === 's6') return false
  const top = r.find((x) => x.best)!
  return top.key === 'unsure' || top.key === 'no_tool' || top.p < 0.4
})

const violation = computed(() => (props.mode === 'guard' && answer.value ? answer.value.probs[1] : null))

onMounted(async () => {
  clockStep.value = clockStepMicros()
  const all = await loadManifest()
  models.value = isSecrets ? all.filter((m) => m.task === 'secrets') : questionModels(all)
  model.value =
    models.value.find((m) => m.name === props.prefer) ??
    models.value.find((m) => m.arch === 'skycmd-iso') ??
    models.value[0] ??
    null
  if (!model.value) {
    status.value = isSecrets
      ? 'The secret-detection model is not published on this site.'
      : 'No generic decision model is published on this site yet.'
    return
  }
  await load()
})

async function load() {
  if (!model.value) return
  // The times are per model.
  times.reset()
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
  if (isSecrets) return runSecrets()
  const e = engine.value
  if (!e) return
  error.value = ''
  if (rendered.value.length < 2) {
    error.value = 'Give at least two options.'
    return
  }
  try {
    e.setExitThreshold(adaptive.value && canExit.value ? undefined : 1.0)
    const a = JSON.parse(
      e.decide(qtype.value, ins.value.trim(), JSON.stringify(rendered.value), state.value),
    ) as Answer
    answer.value = a
    times.add(a.micros)
  } catch (err) {
    error.value = (err as Error).message
  }
}

// ---- secrets mode: one pasted text, the four questions of the jev

const secretIdx = ref(0)
const snippet = ref(SECRET_CASES[0]?.state ?? '')
interface SecretAnswer {
  q: SecretQuestion
  a: Answer
}
const secretAnswers = ref<SecretAnswer[]>([])
/** The text the shown answers are about (the textarea may be ahead of them while debounced). */
const answeredState = ref('')
/** The text as the model reads it: trailing blank lines dropped, so they do not count as lines. */
const secretState = computed(() => snippet.value.replace(/\s+$/, ''))
const secretLines = computed(() => secretState.value.split('\n').length)
/** The preset while the text is unmodified. */
const secretCase = computed(() => {
  const c = SECRET_CASES[secretIdx.value]
  return c && c.state === secretState.value ? c : null
})
/** The preset the shown answers are about, if any: expected answers are marked only then. */
const answeredCase = computed(() => {
  const c = secretCase.value
  return c && c.state === answeredState.value ? c : null
})
const QUESTION_LABEL: Record<SecretQuestionId, string> = {
  secret: 'Is a usable secret exposed?',
  service: 'Which service issued it?',
  kind: 'What kind of value?',
  line: 'Which line?',
}

function pickSecret(i: number) {
  secretIdx.value = i
  snippet.value = SECRET_CASES[i].state
}

function runSecrets() {
  const e = engine.value
  if (!e) return
  error.value = ''
  if (!secretState.value.trim()) {
    secretAnswers.value = []
    answeredState.value = ''
    error.value = 'Paste a text first.'
    return
  }
  try {
    e.setExitThreshold(1.0)
    const out: SecretAnswer[] = []
    for (const q of secretQuestions(secretState.value)) {
      const a = JSON.parse(e.decide(q.t, q.ins, JSON.stringify(q.opts), secretState.value)) as Answer
      times.add(a.micros)
      out.push({ q, a })
    }
    secretAnswers.value = out
    answeredState.value = secretState.value
  } catch (err) {
    error.value = (err as Error).message
  }
}

/** One answer card: the options sorted by probability (at most `top`), the expected one marked. */
function secretRows(sa: SecretAnswer, top = 3) {
  const g = answeredCase.value?.gold[sa.q.id]
  const gold = g !== undefined && g < sa.q.opts.length ? g : null
  const best = sa.a.probs.indexOf(Math.max(...sa.a.probs))
  const all = sa.q.opts.map((o, i) => {
    const [key, ...rest] = o.split(': ')
    return { i, key, desc: rest.join(': '), p: sa.a.probs[i] ?? 0, best: i === best, gold: i === gold }
  })
  const shown = [...all].sort((x, y) => y.p - x.p).slice(0, top)
  // the expected option stays visible even when the model ranks it low
  if (gold !== null && !shown.some((r) => r.gold)) shown.push(all[gold])
  return { rows: shown, best: all[best], gold: gold === null ? null : all[gold], hidden: all.length - shown.length }
}

const secretTotal = computed(() => secretAnswers.value.reduce((s, x) => s + x.a.micros, 0))

/** The scanner's verdict on the preset, judged against the expected answers. */
const scannerVerdict = computed(() => {
  const c = answeredCase.value
  if (!c) return null
  const flagged = c.scanner !== null
  const secretOk = flagged === (c.gold.secret === 1)
  const service = c.scanner ?? 'none'
  const serviceOk = service === SECRET_SERVICES[c.gold.service].split(':')[0]
  return { flagged, secretOk, service, serviceOk }
})

let timer: ReturnType<typeof setTimeout> | null = null
watch(snippet, () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(run, 120)
})
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

/** Preset button text: the state (s5 cases), or the source and type (s6 cases, whose states are numbers). */
function caseLabel(c: DemoCase) {
  return props.mode === 's6' ? `${c.src.replace(/^(game|numeric):/, '')} · ${c.t}` : short(c.state)
}
</script>

<template>
  <div class="live">
    <div class="live-row live-top">
      <label v-if="mode !== 's6' && !isSecrets">Language
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
      <label v-if="mode === 'playground' || mode === 's6'">Type
        <select v-model="qtype">
          <option value="choice">choice</option>
          <option value="score">score</option>
          <option value="noul">noul (true / false)</option>
        </select>
      </label>
      <label v-if="canExit" class="live-check">
        <input v-model="adaptive" type="checkbox" /> adaptive depth (early exit)
      </label>
      <span v-else-if="model && model.full_depth" class="live-note">full depth, as published</span>
    </div>

    <p v-if="lang === 'fr'" class="live-banner">
      s5 est un petit modèle anglais ; il n'a pas de version française, voir
      <a :href="withBase('/results/')">les résultats</a>. La démo reste sur les presets anglais.
    </p>

    <template v-if="isSecrets">
      <p class="live-banner">
        Not a scanner: the presets are texts of the model's own generator. On real files with no secret, the model
        flags 59.8% of the files at p ≥ 0.9 and 49.4% at p ≥ 0.999
        (<a href="https://github.com/maxgfr/secretgate/pull/19">measurement</a>), so its answer on a text of your
        own cannot be trusted.
      </p>
      <p class="live-banner live-safe">
        The keys in the presets are fake: each is a provider's public format with a random body, as the bench
        generator writes them, and the Stripe-format ones are drawn again in your browser at each page load. Everything runs locally in your browser; nothing you paste is sent anywhere.
      </p>
      <div class="live-examples">
        <span>Bench texts:</span>
        <button
          v-for="(c, i) in SECRET_CASES"
          :key="i"
          type="button"
          :class="{ on: i === secretIdx && secretCase !== null }"
          :title="c.why"
          @click="pickSecret(i)"
        >{{ c.label }}</button>
      </div>
      <div class="live-form">
        <label>Text to check (a config file, a .env, code)
          <textarea v-model="snippet" rows="8" spellcheck="false" autocomplete="off" />
        </label>
      </div>
      <p v-if="secretCase" class="live-meta">Preset: {{ secretCase.why }}.</p>
      <p v-if="secretLines > 7" class="live-meta">
        {{ secretLines }} lines: the training texts have 2 to 7, and the state window holds 320 tokens, so the line
        question (and a long text) is outside what the model has seen.
      </p>
    </template>

    <div v-if="!isSecrets" class="live-examples">
      <span>Examples:</span>
      <button
        v-for="(c, i) in examples"
        :key="i"
        type="button"
        :class="{ on: i === caseIdx && expected !== null }"
        :title="c.state"
        @click="pick(i)"
      >{{ caseLabel(c) }}</button>
    </div>

    <div v-if="!isSecrets" class="live-form">
      <label>{{ mode === 'guard' ? 'Statement to check' : qtype === 'noul' ? 'Statement' : 'Question' }}
        <input v-model="ins" type="text" />
      </label>
      <label v-if="qtype !== 'noul' || ownNoul">
        {{
          mode === 'tools'
            ? 'Tools (name: description, one per line)'
            : mode === 'guard' || qtype === 'noul'
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

    <div v-if="isSecrets && secretAnswers.length && !status" class="live-out">
      <div class="live-cards">
        <section v-for="sa in secretAnswers" :key="sa.q.id" class="live-card">
          <h4>{{ QUESTION_LABEL[sa.q.id] }}</h4>
          <template v-for="v in [secretRows(sa, sa.q.id === 'secret' ? 2 : 3)]" :key="sa.q.id">
            <p class="live-answer">
              <strong>{{ sa.q.id === 'secret' ? (v.best.i === 1 ? 'yes' : 'no') : v.best.key }}</strong>
              {{ pct(v.best.p) }}
              <template v-if="v.gold">
                · expected <strong>{{ sa.q.id === 'secret' ? (v.gold.i === 1 ? 'yes' : 'no') : v.gold.key }}</strong>
                · <span :class="v.gold.best ? 'ok' : 'miss'">{{ v.gold.best ? 'the model agrees' : 'the model disagrees' }}</span>
              </template>
            </p>
            <ul class="live-bars">
              <li v-for="r in v.rows" :key="r.i" :class="{ best: r.best }">
                <span class="k" :title="r.desc">{{ sa.q.id === 'secret' ? (r.i === 1 ? 'yes' : 'no') : r.key }}<template v-if="r.gold"> ✓</template></span>
                <span class="bar"><span :style="{ width: pct(r.p) }" /></span>
                <span class="p">{{ pct(r.p) }}</span>
              </li>
            </ul>
            <p class="live-meta">
              <template v-if="v.hidden > 0">{{ v.hidden }} other options below · </template>{{ formatMicros(sa.a.micros) }}
              · question {{ sa.a.static_hit ? 'cached' : 'encoded' }}
            </p>
          </template>
        </section>
        <section v-if="secretLines < 2" class="live-card">
          <h4>{{ QUESTION_LABEL.line }}</h4>
          <p class="live-meta">Not asked: a one-line text, as in the generator.</p>
        </section>
      </div>
      <p v-if="scannerVerdict" class="live-meta">
        Rule scanner (<code>sky-data secret-baseline</code>) on this text:
        <template v-if="scannerVerdict.flagged">flags it as <strong>{{ scannerVerdict.service }}</strong></template>
        <template v-else>flags nothing</template>
        · secret question <span :class="scannerVerdict.secretOk ? 'ok' : 'miss'">{{ scannerVerdict.secretOk ? 'right' : 'wrong' }}</span>
        · service <span :class="scannerVerdict.serviceOk ? 'ok' : 'miss'">{{ scannerVerdict.serviceOk ? 'right' : 'wrong' }}</span>
        <template v-if="!scannerVerdict.flagged && !scannerVerdict.serviceOk"> (it names a service only for what it flags)</template>
      </p>
      <p class="live-meta">
        {{ secretAnswers.length }} decisions, {{ formatMicros(secretTotal) }} for this text · all 4 block passes (full depth, as published)
        <template v-if="model"> · {{ model.name }}, {{ formatInt(model.bytes / 1024) }} KB</template>
      </p>
      <p class="live-meta live-times">
        <template v-if="times.warmMedian !== null">
          warm median <strong>{{ formatMicros(times.warmMedian) }}</strong> per decision ({{ times.warmCount }} {{ times.warmCount === 1 ? 'decision' : 'decisions' }} after the first)
          ·
        </template>
        first call {{ times.first !== null ? formatMicros(times.first) : '…' }} (module start-up and first encodings included)
        <template v-if="engineLabel()"> · {{ engineLabel() }}</template>
        <template v-if="clockStep >= 20"> · this browser's clock counts in steps of {{ formatInt(clockStep) }} µs, so each time is rounded to a step</template>
      </p>
    </div>

    <div v-if="!isSecrets && answer && !status" class="live-out">
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
        <template v-if="model"> · {{ model.name }}, {{ formatInt(model.bytes / 1024) }} KB</template>
      </p>
      <p class="live-meta live-times">
        this decision <strong>{{ formatMicros(answer.micros) }}</strong>
        <template v-if="times.warmCount === 0 && times.first !== null">
          (the first call: it includes the module start-up and the first encodings)
        </template>
        <template v-else>
          <template v-if="times.warmMedian !== null">
            · warm median {{ formatMicros(times.warmMedian) }} ({{ times.warmCount }} {{ times.warmCount === 1 ? 'decision' : 'decisions' }} after the first)
          </template>
          · first call {{ formatMicros(times.first ?? 0) }}
        </template>
        <template v-if="engineLabel()"> · {{ engineLabel() }}</template>
        <template v-if="clockStep >= 20"> · this browser's clock counts in steps of {{ formatInt(clockStep) }} µs, so each time is rounded to a step</template>
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
.live-note {
  font-size: 13px;
  color: var(--vp-c-text-2);
  padding-bottom: 6px;
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
.live-times {
  margin-top: 4px;
}
.live-safe {
  background: var(--vp-c-tip-soft);
}
.live-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.live-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--vp-c-bg);
}
.live-card h4 {
  margin: 0 0 6px;
  font-size: 14px;
}
.live-card .live-meta {
  margin-top: 8px;
}
.live-answer {
  margin: 0 0 8px;
  font-size: 14px;
}
.live .ok {
  color: var(--vp-c-green-1, var(--vp-c-brand-1));
}
.live .miss {
  color: var(--vp-c-danger-1);
}
</style>
