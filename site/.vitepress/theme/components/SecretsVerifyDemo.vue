<script setup lang="ts">
// The secrets verifier demo (s6-secrets-verify-pico, the Secrets verifier page and its Space): a
// pasted text goes through a SIMPLIFIED proposer (lib/secrets-cand.ts, one rule, not secretgate's
// rule engine), and each candidate record is judged by the wasm `SecretVerifier`
// (crates/sky-web/src/verify.rs) at full depth: P(real), the decision at one of the three
// thresholds stored in the file, the time per candidate and the state text the model reads.
// Everything runs in the page: nothing typed is sent anywhere.
import { computed, onMounted, ref, shallowRef, watch } from 'vue'
import {
  clockStepMicros,
  engineLabel,
  formatInt,
  formatKB,
  formatMicros,
  loadSecretVerifier,
  loadVerifierEntry,
  type WasmSecretVerifier,
} from '../lib/wasm'
import { MIN_VALUE_CHARS, SECRETS_PRESETS, proposeCandidates, type CandidateRecord } from '../lib/secrets-cand'

type TauKey = 'tau_f1' | 'tau_f05' | 'tau_f2'

/** The three thresholds of the file, in the order of the menu (tau_f1 is the one every test
 * reading used). */
const TAUS: { key: TauKey; label: string }[] = [
  { key: 'tau_f1', label: 'F1-optimal' },
  { key: 'tau_f05', label: 'fewer alarms' },
  { key: 'tau_f2', label: 'fewer misses' },
]

interface Judged {
  rec: CandidateRecord
  p: number
  micros: number
  truncated: boolean
  stateTokens: number
  state: string
}

const presetIdx = ref(0)
const text = ref(SECRETS_PRESETS[0].text)
const file = ref(SECRETS_PRESETS[0].file)
const tauKey = ref<TauKey>('tau_f1')
const thresholds = ref<Record<TauKey, number> | null>(null)

const ver = shallowRef<WasmSecretVerifier | null>(null)
const status = ref('Loading the verifier…')
const missing = ref(false)
const error = ref('')
const results = ref<Judged[]>([])
const clockStep = ref(0)

const preset = computed(() => {
  const p = SECRETS_PRESETS[presetIdx.value]
  return p && p.text === text.value ? p : null
})
const candidates = computed(() => proposeCandidates(text.value, file.value))
const tau = computed(() => thresholds.value?.[tauKey.value] ?? null)
const accepted = computed(() => (tau.value === null ? 0 : results.value.filter((r) => r.p >= tau.value!).length))
const medianUs = computed(() => {
  const v = results.value.map((r) => r.micros).sort((a, b) => a - b)
  return v.length ? v[Math.floor(v.length / 2)] : null
})

onMounted(async () => {
  clockStep.value = clockStepMicros()
  try {
    const entry = await loadVerifierEntry()
    if (!entry) {
      missing.value = true
      status.value = ''
      return
    }
    status.value = `Loading ${entry.name} (${formatKB(entry.bytes)}) and its word list…`
    const v = await loadSecretVerifier(entry)
    const t = JSON.parse(v.thresholds()) as Record<TauKey, number | null>
    if (TAUS.some((x) => typeof t[x.key] !== 'number')) throw new Error('the file does not store its three thresholds')
    thresholds.value = t as Record<TauKey, number>
    const inf = JSON.parse(v.info()) as { n_params: number; file_bytes: number; window: number; words: number }
    const line = [
      entry.name,
      `${formatInt(inf.n_params)} parameters`,
      formatKB(inf.file_bytes),
      `state window ${formatInt(inf.window)} tokens`,
      `${formatInt(inf.words)} words in the word list`,
      'full depth',
    ]
    const eng = engineLabel()
    if (eng) line.push(eng)
    status.value = line.join(' · ')
    ver.value = v // last: the demo appears only once everything above held
    run()
  } catch (err) {
    error.value = `Could not load the verifier: ${(err as Error).message ?? err}`
    status.value = ''
  }
})

function run() {
  const v = ver.value
  if (!v) return
  error.value = ''
  const c = candidates.value
  if (!c.length) {
    results.value = []
    return
  }
  try {
    const answers = JSON.parse(v.judgeMany(JSON.stringify(c))) as {
      p_real: number
      micros: number
      truncated: boolean
      state_tokens: number
    }[]
    results.value = c.map((rec, i) => ({
      rec,
      p: answers[i].p_real,
      micros: answers[i].micros,
      truncated: answers[i].truncated,
      stateTokens: answers[i].state_tokens,
      state: v.state(JSON.stringify(rec)),
    }))
  } catch (err) {
    results.value = []
    error.value = (err as Error).message ?? String(err)
  }
}

function pickPreset(i: number) {
  presetIdx.value = i
  text.value = SECRETS_PRESETS[i].text
  file.value = SECRETS_PRESETS[i].file
}

let timer: ReturnType<typeof setTimeout> | null = null
watch([text, file], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(run, 250)
})

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}

function tauText(v: number) {
  return v.toFixed(4)
}
</script>

<template>
  <div class="sv">
    <p class="sv-privacy">
      Everything runs in your browser: nothing you paste is sent anywhere. Do not paste real credentials into any
      web page.
    </p>
    <p v-if="status" class="sv-status">{{ status }}</p>
    <p v-if="missing" class="sv-note">The secrets verifier (s6-secrets-verify-pico) is not published on this site.</p>

    <template v-if="ver && thresholds">
      <div class="sv-row">
        <span class="sv-label" id="sv-presets">Presets:</span>
        <div class="sv-chips" role="group" aria-labelledby="sv-presets">
          <button
            v-for="(p, i) in SECRETS_PRESETS"
            :key="p.id"
            type="button"
            :class="{ on: i === presetIdx && preset !== null }"
            :aria-pressed="i === presetIdx && preset !== null"
            @click="pickPreset(i)"
          >{{ p.label }}</button>
        </div>
      </div>

      <div class="sv-form">
        <label>File name (path in the repository, optional)
          <input v-model="file" type="text" spellcheck="false" autocomplete="off" placeholder=".env" />
        </label>
        <label>Text
          <textarea v-model="text" rows="8" spellcheck="false" autocomplete="off" />
        </label>
        <label class="sv-inline">Threshold
          <select v-model="tauKey">
            <option v-for="t in TAUS" :key="t.key" :value="t.key">
              {{ t.label }} ({{ t.key }} = {{ tauText(thresholds[t.key]) }})
            </option>
          </select>
        </label>
      </div>

      <p v-if="error" class="sv-error" role="alert">{{ error }}</p>

      <div aria-live="polite">
        <p v-if="!results.length && !error" class="sv-meta">
          No candidate. This demo proposes a value of at least {{ MIN_VALUE_CHARS }} characters assigned to a name
          that contains password, passwd, secret, token, api key, apikey, auth, credential or private key.
        </p>
        <template v-else-if="results.length && tau !== null">
          <p class="sv-summary">
            {{ results.length }} {{ results.length === 1 ? 'candidate' : 'candidates' }}; {{ accepted }} accepted as real
            at {{ tauKey }} = {{ tauText(tau) }}
          </p>
          <ol class="sv-list">
            <li v-for="(r, i) in results" :key="`${r.rec.line}:${r.rec.start}:${i}`" :class="{ real: r.p >= tau }">
              <div class="sv-head">
                <code class="sv-name">{{ r.rec.name }}</code>
                <span class="sv-line">line {{ r.rec.line }}</span>
              </div>
              <code class="sv-value">{{ r.rec.value }}</code>
              <div class="sv-bar" role="img" :aria-label="`P(real) ${pct(r.p)}, threshold ${tauText(tau)}`">
                <span class="sv-fill" :style="{ width: pct(r.p) }" />
                <span class="sv-tau" :style="{ left: pct(tau) }" />
              </div>
              <p class="sv-verdict">
                P(real) = {{ r.p.toFixed(4) }} ·
                <strong v-if="r.p >= tau">accepted as real</strong>
                <span v-else>rejected</span>
                ({{ r.p >= tau ? 'at or above' : 'below' }} {{ tauKey }})
                · {{ formatMicros(r.micros) }}
                <template v-if="r.truncated"> · state cut to the window ({{ formatInt(r.stateTokens) }} tokens)</template>
              </p>
              <details class="sv-state">
                <summary>The text the model reads</summary>
                <pre>{{ r.state }}</pre>
              </details>
            </li>
          </ol>
          <p class="sv-meta">
            <template v-if="medianUs !== null">median {{ formatMicros(medianUs) }} per candidate on this text</template>
            <template v-if="engineLabel()"> · {{ engineLabel() }}</template>
            <template v-if="clockStep >= 20"> · this browser's clock counts in steps of {{ formatInt(clockStep) }} µs</template>
          </p>
        </template>
      </div>

      <p class="sv-foot">
        The candidates come from a simplified proposer, not from secretgate's rule engine: the real generator is
        secretgate's, frozen as sg-cand-1, and this demo proposes with one simple rule (the rule id
        <code>generic-keyword</code>) so that the model's judgement can be seen in a browser. The four filter flags of
        secretgate's default engine are not computed here: they are sent as false.
      </p>
    </template>
  </div>
</template>

<style scoped>
.sv {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0 24px;
  background: var(--vp-c-bg-soft);
}
.sv-privacy {
  margin: 0 0 12px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  background: var(--vp-c-warning-soft);
}
.sv-status {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.sv-note {
  margin: 0;
}
.sv-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  align-items: center;
  margin-bottom: 10px;
}
.sv-label {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.sv-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.sv-chips button {
  font-size: 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 2px 10px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
}
.sv-chips button.on {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.sv-form {
  display: grid;
  gap: 10px;
}
.sv label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.sv label.sv-inline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.sv input[type='text'],
.sv textarea,
.sv select {
  font: inherit;
  font-weight: 400;
  font-size: 14px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 6px 10px;
}
.sv input[type='text'],
.sv textarea {
  width: 100%;
}
.sv select {
  max-width: 100%;
}
.sv textarea {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  resize: vertical;
}
.sv input:focus-visible,
.sv textarea:focus-visible,
.sv select:focus-visible,
.sv button:focus-visible,
.sv summary:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 1px;
}
.sv-error {
  margin: 12px 0 0;
  font-size: 14px;
  color: var(--vp-c-danger-1);
}
.sv-summary {
  margin: 14px 0 8px;
  font-size: 15px;
  font-variant-numeric: tabular-nums;
}
.sv-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
}
.sv-list li {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--vp-c-bg);
  min-width: 0;
}
.sv-list li.real {
  border-color: var(--vp-c-danger-2, var(--vp-c-danger-1));
}
.sv-head {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  align-items: baseline;
}
.sv-name {
  font-weight: 700;
}
.sv-line {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.sv-value {
  display: block;
  margin: 4px 0 8px;
  font-size: 13px;
  word-break: break-all;
  color: var(--vp-c-text-2);
}
.sv-bar {
  position: relative;
  height: 10px;
  border-radius: 5px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.sv-fill {
  position: absolute;
  inset: 0 auto 0 0;
  background: var(--vp-c-text-3);
  transition: width 0.15s ease-out;
}
.sv-list li.real .sv-fill {
  background: var(--vp-c-danger-1);
}
.sv-tau {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--vp-c-text-1);
}
.sv-verdict {
  margin: 6px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.sv-list li.real .sv-verdict strong {
  color: var(--vp-c-danger-1);
}
.sv-state {
  margin-top: 6px;
  font-size: 13px;
}
.sv-state summary {
  cursor: pointer;
  color: var(--vp-c-text-2);
}
.sv-state pre {
  margin: 6px 0 0;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
}
.sv-meta {
  margin: 12px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.sv-foot {
  margin: 12px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
</style>
