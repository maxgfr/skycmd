<script setup lang="ts">
// The Router page and the s6-route Space: a tool list and a request, routed in the browser by
// `ToolRouter` (sky-web, crates/sky-web/src/router.rs), the s6-route-nano file and the 4-bit static
// embedding fused option by option, `p ∝ (p_model · p_embedding)^β`, with "ask the user" below
// τ_ask3 (both values come from the embedding file). The tool list is cached on both parts: the
// first request on a new list encodes it, the next ones reuse it, and the two times are shown
// apart. "Mask names" renames every tool `tool_<k>`, the names-masked reading of the
// pre-registration (§6), where the system is known to fail. The lines a visitor types are cut as
// the slates were (lib/router.ts).
import { computed, onMounted, ref, watch } from 'vue'
import {
  clockStepMicros,
  engineLabel,
  formatInt,
  formatMicros,
  loadRouter,
  loadRouters,
  type RouterEntry,
  type WasmToolRouter,
} from '../lib/wasm'
import {
  MAX_TOOLS,
  ROUTER_PRESETS,
  maskNames,
  parseTools,
  renderTools,
  type RouterPreset,
} from '../lib/router'

/** `prefer`: the router entry to load, by name (default: the first one of the manifest). */
const props = defineProps<{ prefer?: string }>()

interface Routed {
  probs: number[]
  index: number
  answer: string
  none: boolean
  ask: boolean
  decision: 'tool' | 'none' | 'ask'
  top: number
  tau_ask3: number
  model: number[]
  embedding: number[]
  fast: boolean
  exit: number | null
  layers_used: number | null
  static_hit: boolean
  tools_hit: boolean
  micros: number
  micros_model: number
  micros_embedding: number
}

const presetIdx = ref(0)
const toolsText = ref(ROUTER_PRESETS[0].tools)
const request = ref(ROUTER_PRESETS[0].requests[0].text)
const fast = ref(false)
const masked = ref(false)

const entry = ref<RouterEntry | null>(null)
const router = ref<WasmToolRouter | null>(null)
const tau = ref<number | null>(null)
const status = ref('Loading the router…')
const error = ref('')
const answer = ref<Routed | null>(null)
const clockStep = ref(0)

/** The last request that met its tool list for the first time (both parts encode it). */
const encodedUs = ref<number | null>(null)
/** The requests that reused a cached tool list (the last 50). */
const cachedUs = ref<number[]>([])
const cachedMedian = computed(() => {
  const v = cachedUs.value
  if (!v.length) return null
  const s = [...v].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)]
})

const parsed = computed(() => parseTools(toolsText.value))
/** The tools as the router reads them: masked when asked. */
const tools = computed(() => (masked.value ? maskNames(parsed.value.tools) : parsed.value.tools))
const rendered = computed(() => renderTools(tools.value))
const cutCount = computed(() => parsed.value.tools.filter((t) => t.cut).length)

const preset = computed<RouterPreset | null>(() => {
  const p = ROUTER_PRESETS[presetIdx.value]
  return p && p.tools === toolsText.value ? p : null
})
/** What the preset request was written for, while the form holds an unmodified preset. */
const intended = computed<string | null>(() => {
  const p = preset.value
  if (!p) return null
  const r = p.requests.find((x) => x.text === request.value.trim())
  if (!r) return null
  if (r.intended === 'none' || !masked.value) return r.intended
  const i = parsed.value.tools.findIndex((t) => t.name === r.intended)
  return i >= 0 ? tools.value[i].name : null
})

const rows = computed(() => {
  const a = answer.value
  if (!a) return []
  const out = tools.value.map((t, i) => ({
    key: t.name,
    real: masked.value ? parsed.value.tools[i]?.name : undefined,
    desc: t.desc,
    p: a.probs[i] ?? 0,
    pm: a.model[i] ?? 0,
    pe: a.embedding[i] ?? 0,
    best: i === a.index,
    none: false,
  }))
  const n = tools.value.length
  out.push({
    key: 'none',
    real: undefined,
    desc: 'none of these tools fits the request',
    p: a.probs[n] ?? 0,
    pm: a.model[n] ?? 0,
    pe: a.embedding[n] ?? 0,
    best: a.index === n,
    none: true,
  })
  return out
})

const toolsLabel = `Tools (one per line, name: description, up to ${MAX_TOOLS})`
const maskLabel = computed(() => (masked.value ? 'Show the real names' : 'Mask names (tool_<k>)'))

const sizeKB = computed(() => (entry.value ? formatInt(entry.value.bytes / 1024) : ''))

onMounted(async () => {
  clockStep.value = clockStepMicros()
  const list = await loadRouters()
  entry.value = list.find((m) => m.name === props.prefer) ?? list[0] ?? null
  if (!entry.value) {
    status.value = 'The router is not published on this site.'
    return
  }
  status.value = `Loading ${entry.value.name} (${sizeKB.value} KB in two files)…`
  try {
    router.value = await loadRouter(entry.value)
    tau.value = router.value.tauAsk()
    status.value = ''
    run()
  } catch (e) {
    status.value = `Could not load the router: ${(e as Error).message}`
  }
})

function run() {
  const r = router.value
  if (!r) return
  error.value = ''
  if (!rendered.value.length) {
    answer.value = null
    error.value = 'Give at least one tool, one per line as name: description.'
    return
  }
  if (!request.value.trim()) {
    answer.value = null
    error.value = 'Type a request.'
    return
  }
  try {
    const a = JSON.parse(r.route(request.value.trim(), JSON.stringify(rendered.value), fast.value)) as Routed
    answer.value = a
    if (a.static_hit && a.tools_hit) {
      cachedUs.value = [...cachedUs.value, a.micros].slice(-50)
    } else {
      encodedUs.value = a.micros
    }
  } catch (err) {
    error.value = (err as Error).message
  }
}

function pickPreset(i: number) {
  presetIdx.value = i
  toolsText.value = ROUTER_PRESETS[i].tools
  request.value = ROUTER_PRESETS[i].requests[0].text
}

function pickRequest(text: string) {
  request.value = text
}

// The times are per mode: the fast mode runs a third of the model part.
watch(fast, () => {
  cachedUs.value = []
  encodedUs.value = null
})

let timer: ReturnType<typeof setTimeout> | null = null
watch([toolsText, request, fast, masked], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(run, 150)
})

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="router">
    <div class="router-row">
      <span class="router-label" id="router-presets">Tool lists:</span>
      <div class="router-chips" role="group" aria-labelledby="router-presets">
        <button
          v-for="(p, i) in ROUTER_PRESETS"
          :key="p.id"
          type="button"
          :class="{ on: i === presetIdx && preset !== null }"
          :aria-pressed="i === presetIdx && preset !== null"
          @click="pickPreset(i)"
        >{{ p.label }}</button>
      </div>
    </div>

    <div class="router-form">
      <label>{{ toolsLabel }}
        <textarea v-model="toolsText" rows="7" spellcheck="false" autocomplete="off" />
      </label>
      <p v-if="parsed.dropped > 0" class="router-meta router-warn">
        {{ parsed.dropped }} {{ parsed.dropped === 1 ? 'line' : 'lines' }} after the {{ MAX_TOOLS }}th left out: the
        router was trained and tested on lists of up to {{ MAX_TOOLS }} tools.
      </p>
      <p v-if="cutCount > 0" class="router-meta">
        {{ cutCount }} {{ cutCount === 1 ? 'description' : 'descriptions' }} cut to the first sentence and 30 words, as
        every training and test slate was.
      </p>
      <p v-if="parsed.duplicates.length" class="router-meta router-warn">
        Two tools share a name ({{ parsed.duplicates.join(', ') }}): the router reads them as two options.
      </p>
      <label>Request
        <input v-model="request" type="text" autocomplete="off" />
      </label>
      <div v-if="preset" class="router-row">
        <span class="router-label" id="router-requests">Requests:</span>
        <div class="router-chips" role="group" aria-labelledby="router-requests">
          <button
            v-for="r in preset.requests"
            :key="r.text"
            type="button"
            :class="{ on: r.text === request.trim() }"
            :aria-pressed="r.text === request.trim()"
            @click="pickRequest(r.text)"
          >{{ r.text }}</button>
        </div>
      </div>
    </div>

    <div class="router-row router-toggles">
      <label class="router-check">
        <input v-model="fast" type="checkbox" /> fast mode (the model part stops at its first exit, 2 of 6 block passes)
      </label>
      <button type="button" class="router-mask" :aria-pressed="masked" @click="masked = !masked">
        {{ maskLabel }}
      </button>
    </div>

    <p v-if="masked" class="router-banner router-failed">
      <strong>Names masked.</strong> Every tool is renamed <code>tool_&lt;k&gt;</code> and keeps its description, as in
      the names-masked reading of the pre-registration: there s6-route fell to 0.2497, where BM25 kept 0.8138 and
      the embedding part alone 0.7280. It needs distinctive tool names.
    </p>

    <p v-if="status" class="router-status">{{ status }}</p>
    <p v-if="error" class="router-error" role="alert">{{ error }}</p>

    <div v-if="answer && !status" class="router-out" aria-live="polite">
      <p class="router-verdict" :class="answer.decision">
        <template v-if="answer.decision === 'ask'">
          <strong>Ask the user.</strong>
          <span>The top option, {{ answer.answer }}, has {{ pct(answer.top) }}, below τ_ask3 = {{ pct(answer.tau_ask3) }}.</span>
        </template>
        <template v-else-if="answer.decision === 'none'">
          <strong>None of these tools fits the request.</strong>
          <span>{{ pct(answer.top) }}, at or above τ_ask3 = {{ pct(answer.tau_ask3) }}.</span>
        </template>
        <template v-else>
          Route to <strong><code>{{ answer.answer }}</code></strong>
          <span>{{ pct(answer.top) }}, at or above τ_ask3 = {{ pct(answer.tau_ask3) }}.</span>
        </template>
      </p>
      <ul class="router-bars" aria-label="Fused probability of each option">
        <li v-for="r in rows" :key="r.key + (r.none ? '-none' : '')" :class="{ best: r.best, none: r.none }">
          <span class="k" :title="r.desc">{{ r.key }}<template v-if="intended === r.key"> ✓</template><span v-if="r.real" class="real">&nbsp;({{ r.real }})</span></span>
          <span class="bar" aria-hidden="true"><span :style="{ width: pct(r.p) }" /></span>
          <span class="p">{{ pct(r.p) }}</span>
          <span class="parts">model {{ pct(r.pm) }} · embedding {{ pct(r.pe) }}</span>
        </li>
      </ul>
      <p v-if="intended" class="router-meta">
        This preset request was written for <strong>{{ intended }}</strong> (our label, not a test answer):
        <span :class="answer.decision !== 'ask' && answer.answer === intended ? 'ok' : 'miss'">{{
          answer.decision === 'ask'
            ? 'the router asks instead'
            : answer.answer === intended
              ? 'the router agrees'
              : 'the router disagrees'
        }}</span>.
      </p>
      <p class="router-meta">
        {{ answer.fast ? 'fast mode' : 'full depth' }}<template v-if="answer.layers_used !== null"> · {{ answer.layers_used }} block passes</template>
        · tool list {{ answer.static_hit && answer.tools_hit ? 'cached' : 'encoded' }}
        · model part {{ formatMicros(answer.micros_model) }}, embedding part {{ formatMicros(answer.micros_embedding) }}
        <template v-if="entry"> · {{ entry.name }}, {{ sizeKB }} KB in two files</template>
      </p>
      <p class="router-meta router-times">
        this request <strong>{{ formatMicros(answer.micros) }}</strong>
        <template v-if="cachedMedian !== null">
          · cached median {{ formatMicros(cachedMedian) }} ({{ cachedUs.length }} {{ cachedUs.length === 1 ? 'request' : 'requests' }} on a cached tool list)
        </template>
        <template v-if="encodedUs !== null"> · first request on the last new tool list {{ formatMicros(encodedUs) }}</template>
        <template v-if="engineLabel()"> · {{ engineLabel() }}</template>
        <template v-if="clockStep >= 20"> · this browser's clock counts in steps of {{ formatInt(clockStep) }} µs</template>
      </p>
    </div>
    <p v-if="tau !== null" class="router-meta">
      The decision: the option with the highest fused probability, unless that probability is below τ_ask3 =
      {{ tau.toFixed(6) }} (fixed on validation), in which case the router asks the user. Everything runs in your
      browser; nothing you type is sent anywhere.
    </p>
  </div>
</template>

<style scoped>
.router {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0 24px;
  background: var(--vp-c-bg-soft);
}
.router-form {
  display: grid;
  gap: 10px;
}
.router label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.router input[type='text'],
.router textarea {
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
.router textarea {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  resize: vertical;
}
.router input:focus-visible,
.router textarea:focus-visible,
.router button:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 1px;
}
.router-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  align-items: center;
  margin-bottom: 10px;
}
.router-form .router-row {
  margin-bottom: 0;
}
.router-label {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.router-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.router-chips button,
.router-mask {
  font-size: 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 2px 10px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
  text-align: left;
}
.router-chips button.on,
.router-mask[aria-pressed='true'] {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.router-toggles {
  margin-top: 12px;
}
.router-check {
  display: flex !important;
  align-items: center;
  gap: 6px;
  font-weight: 400 !important;
}
.router-banner {
  margin: 0 0 10px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  background: var(--vp-c-warning-soft);
}
.router-failed {
  background: var(--vp-c-danger-soft);
}
.router-status,
.router-error {
  font-size: 14px;
  margin: 12px 0 0;
}
.router-error {
  color: var(--vp-c-danger-1);
}
.router-out {
  margin-top: 14px;
}
.router-verdict {
  margin: 0 0 10px;
  font-size: 15px;
}
.router-verdict span {
  margin-left: 8px;
  color: var(--vp-c-text-2);
}
.router-verdict.ask strong {
  color: var(--vp-c-warning-1);
}
.router-bars {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 6px;
}
.router-bars li {
  display: grid;
  grid-template-columns: minmax(90px, 190px) 1fr 56px;
  grid-template-areas: 'k bar p' '. parts parts';
  column-gap: 10px;
  align-items: center;
  font-size: 14px;
}
.router-bars .k {
  grid-area: k;
  font-family: var(--vp-font-family-mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.router-bars .real {
  color: var(--vp-c-text-3);
  font-weight: 400;
}
.router-bars .bar {
  grid-area: bar;
  height: 10px;
  border-radius: 5px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.router-bars .bar span {
  display: block;
  height: 100%;
  background: var(--vp-c-text-3);
  transition: width 0.15s ease-out;
}
.router-bars li.best .bar span {
  background: var(--vp-c-brand-1);
}
.router-bars li.best .k {
  font-weight: 700;
}
.router-bars li.none .k {
  font-style: italic;
}
.router-bars .p {
  grid-area: p;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.router-bars .parts {
  grid-area: parts;
  font-size: 12px;
  color: var(--vp-c-text-3);
  font-variant-numeric: tabular-nums;
}
.router-meta {
  margin: 12px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.router-form .router-meta {
  margin: 0;
}
.router-warn {
  color: var(--vp-c-warning-1);
}
.router-times {
  margin-top: 4px;
}
.router .ok {
  color: var(--vp-c-green-1, var(--vp-c-brand-1));
}
.router .miss {
  color: var(--vp-c-danger-1);
}
@media (max-width: 560px) {
  .router-bars li {
    grid-template-columns: minmax(70px, 1fr) 1fr 50px;
  }
}
</style>
