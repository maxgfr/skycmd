<script setup lang="ts">
// Proof demos of s5-lite on /live/, computed in the browser:
// - `guarantee`: renaming the option keys or reversing their order cannot change what the iso
//   model gives each definition (the keys are never read, the options never see each other);
//   the same edit moves a model that reads the keys (s4-micro).
// - `paraphrase`: the same options described in other words; robustness here is learned, not
//   guaranteed, and the measured rate is on the results page.
// - `dutch`: the model's own probabilities that each option is the answer ("The correct answer
//   to q is: x"), their negations and their sum; a sum away from 1 is a Dutch book.
import { onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import { loadEngine, loadManifest, type ModelEntry, type WasmEngine } from '../lib/wasm'

type Mode = 'guarantee' | 'paraphrase' | 'dutch'
const props = defineProps<{ mode: Mode }>()

const S4_NOUL = ['false: no, the statement does not hold', 'true: yes, the statement holds']

const ins = ref('Which category does this customer message belong to?')
const optionsText = ref(
  [
    'billing: Invoices, charges, payments or refunds.',
    'bug: Something in the app is broken or crashes.',
    'feature_request: An idea or a wish for the product.',
    'account: Login, password or profile problems.',
  ].join('\n'),
)
const paraText = ref(
  [
    'billing: Anything about money: a bill, a payment, a charge or getting money back.',
    'bug: A feature of the app misbehaves, errors out or stops working.',
    'feature_request: The customer suggests something new for the product.',
    'account: Signing in, credentials or the user profile.',
  ].join('\n'),
)
const state = ref('Hi, I was charged twice for my March subscription, can you send the money back?')

const iso = ref<WasmEngine | null>(null)
const keyed = ref<WasmEngine | null>(null)
const keyedName = ref('')
const status = ref('Loading the models…')
const error = ref('')

onMounted(async () => {
  const all = await loadManifest()
  const s5 = all.find((m) => m.arch === 'skycmd-iso')
  const s4 = all.find((m: ModelEntry) => m.task === 'generic' && m.arch !== 'skycmd-iso')
  if (!s5) {
    status.value = 's5-lite is not published on this site yet.'
    return
  }
  try {
    iso.value = await loadEngine(s5)
    if (props.mode === 'guarantee' && s4) {
      keyed.value = await loadEngine(s4)
      keyedName.value = s4.name
    }
    status.value = ''
    recompute()
  } catch (e) {
    status.value = `Could not load the models: ${(e as Error).message}`
  }
})

function lines(text: string): [string, string][] {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      const i = l.indexOf(': ')
      return i > 0 ? [l.slice(0, i), l.slice(i + 2)] : [l, '']
    })
}

function decide(e: WasmEngine, qtype: string, q: string, options: string[], s: string): number[] {
  return (JSON.parse(e.decide(qtype, q, JSON.stringify(options), s)) as { probs: number[] }).probs
}

interface Guarantee {
  defs: string[]
  iso: { a: number[]; b: number[]; c: number[]; diff: number }
  keyed: { a: number[]; b: number[]; c: number[]; diff: number } | null
}
interface Paraphrase {
  keys: string[]
  pa: number[]
  pb: number[]
  flip: boolean
  tv: number
}
interface Dutch {
  keys: string[]
  c: number[]
  s: number[]
  n: number[]
  sum: number
  mad: number
}

const guarantee = ref<Guarantee | null>(null)
const paraphrase = ref<Paraphrase | null>(null)
const dutch = ref<Dutch | null>(null)

/** guarantee: per definition, p with the given keys, moved keys and the reversed order. */
function runGuarantee(e: WasmEngine, k: WasmEngine | null): Guarantee | null {
  const o = lines(optionsText.value)
  if (o.length < 2) return null
  const keys = o.map(([key]) => key)
  // Every key moves to the next definition (no definition keeps its key).
  const moved = o.map(([, d], i) => `${keys[(i + 1) % keys.length]}: ${d}`)
  const given = o.map(([key, d]) => `${key}: ${d}`)
  const reversed = [...given].reverse()
  const run = (m: WasmEngine) => {
    const a = decide(m, 'choice', ins.value, given, state.value)
    const b = decide(m, 'choice', ins.value, moved, state.value)
    const c = decide(m, 'choice', ins.value, reversed, state.value).reverse()
    const diff = Math.max(...a.map((x, i) => Math.max(Math.abs(x - b[i]), Math.abs(x - c[i]))))
    return { a, b, c, diff }
  }
  return { defs: o.map(([key, d]) => d || key), iso: run(e), keyed: k ? run(k) : null }
}

/** paraphrase: the same keys with two wordings of the definitions. */
function runParaphrase(e: WasmEngine): Paraphrase | null {
  const a = lines(optionsText.value)
  const b = lines(paraText.value)
  if (a.length < 2 || a.length !== b.length) throw new Error('Give the same number of options in both columns.')
  const pa = decide(e, 'choice', ins.value, a.map(([k, d]) => `${k}: ${d}`), state.value)
  const pb = decide(e, 'choice', ins.value, b.map(([k, d]) => `${k}: ${d}`), state.value)
  const argmax = (v: number[]) => v.indexOf(Math.max(...v))
  const tv = 0.5 * pa.reduce((s, x, i) => s + Math.abs(x - pb[i]), 0)
  return { keys: a.map(([k]) => k), pa, pb, flip: argmax(pa) !== argmax(pb), tv }
}

/** The text of an option inside a statement, as in the coherence families of the training data. */
function label(k: string, d: string) {
  return d ? d.trim().replace(/\.+$/, '') : k.replace(/_/g, ' ')
}

/** dutch: c_k (choice), s_k ("is: x_k") and n_k ("is not: x_k"), each its own question. */
function runDutch(e: WasmEngine): Dutch | null {
  const o = lines(optionsText.value)
  if (o.length < 2) return null
  const c = decide(e, 'choice', ins.value, o.map(([k, d]) => `${k}: ${d}`), state.value)
  const yes = (q: string) => decide(e, 'noul', q, ['false', 'true'], state.value)[1]
  const s = o.map(([k, d]) => yes(`The correct answer to "${ins.value}" is: ${label(k, d)}.`))
  const n = o.map(([k, d]) => yes(`The correct answer to "${ins.value}" is not: ${label(k, d)}.`))
  const sum = s.reduce((x, y) => x + y, 0)
  const mad = s.reduce((x, v, i) => x + Math.abs(v + n[i] - 1), 0) / s.length
  return { keys: o.map(([k]) => k), c, s, n, sum, mad }
}

function recompute() {
  const e = iso.value
  if (!e) return
  error.value = ''
  try {
    if (props.mode === 'guarantee') guarantee.value = runGuarantee(e, keyed.value)
    else if (props.mode === 'paraphrase') paraphrase.value = runParaphrase(e)
    else dutch.value = runDutch(e)
  } catch (err) {
    error.value = (err as Error).message
  }
}

let timer: ReturnType<typeof setTimeout> | null = null
watch([ins, optionsText, paraText, state], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(recompute, 150)
})

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}
function sci(x: number) {
  return x === 0 ? '0' : x.toExponential(1)
}
</script>

<template>
  <div class="proof">
    <div class="proof-form">
      <label>Question <input v-model="ins" type="text" /></label>
      <div :class="{ cols: mode === 'paraphrase' }">
        <label>Options (key: definition, one per line)
          <textarea v-model="optionsText" rows="5" spellcheck="false" />
        </label>
        <label v-if="mode === 'paraphrase'">Same keys, other words
          <textarea v-model="paraText" rows="5" spellcheck="false" />
        </label>
      </div>
      <label>Message <textarea v-model="state" rows="2" /></label>
    </div>

    <p v-if="status" class="proof-status">{{ status }}</p>
    <p v-if="error" class="proof-error">{{ error }}</p>

    <template v-if="!status">
      <table v-if="guarantee" class="proof-table">
        <thead>
          <tr>
            <th>definition</th>
            <th>s5-lite: keys given</th>
            <th>keys moved</th>
            <th>order reversed</th>
            <th v-if="guarantee.keyed">{{ keyedName }}: given</th>
            <th v-if="guarantee.keyed">keys moved</th>
            <th v-if="guarantee.keyed">reversed</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(d, i) in guarantee.defs" :key="i">
            <td :title="d">{{ d }}</td>
            <td>{{ pct(guarantee.iso.a[i]) }}</td>
            <td>{{ pct(guarantee.iso.b[i]) }}</td>
            <td>{{ pct(guarantee.iso.c[i]) }}</td>
            <template v-if="guarantee.keyed">
              <td>{{ pct(guarantee.keyed.a[i]) }}</td>
              <td>{{ pct(guarantee.keyed.b[i]) }}</td>
              <td>{{ pct(guarantee.keyed.c[i]) }}</td>
            </template>
          </tr>
        </tbody>
        <tfoot>
          <tr>
            <td>largest change</td>
            <td colspan="3"><strong>{{ sci(guarantee.iso.diff) }}</strong> (guaranteed: float rounding only)</td>
            <td v-if="guarantee.keyed" colspan="3">{{ pct(guarantee.keyed.diff) }}</td>
          </tr>
        </tfoot>
      </table>

      <div v-if="paraphrase" class="proof-out">
        <table class="proof-table">
          <thead>
            <tr><th>key</th><th>first wording</th><th>other words</th></tr>
          </thead>
          <tbody>
            <tr v-for="(k, i) in paraphrase.keys" :key="k">
              <td>{{ k }}</td>
              <td>{{ pct(paraphrase.pa[i]) }}</td>
              <td>{{ pct(paraphrase.pb[i]) }}</td>
            </tr>
          </tbody>
        </table>
        <p class="proof-meta">
          {{ paraphrase.flip ? 'The answer changes.' : 'Same answer.' }} Total variation
          {{ pct(paraphrase.tv) }}. This is learned, not guaranteed: the measured flip rate on held-out
          paraphrases is on <a :href="withBase('/results/')">the results page</a>.
        </p>
      </div>

      <div v-if="dutch" class="proof-out">
        <table class="proof-table">
          <thead>
            <tr><th>option</th><th>choice c</th><th>“is: x” s</th><th>“is not: x” n</th><th>|s + n − 1|</th></tr>
          </thead>
          <tbody>
            <tr v-for="(k, i) in dutch.keys" :key="k">
              <td>{{ k }}</td>
              <td>{{ pct(dutch.c[i]) }}</td>
              <td>{{ pct(dutch.s[i]) }}</td>
              <td>{{ pct(dutch.n[i]) }}</td>
              <td>{{ pct(Math.abs(dutch.s[i] + dutch.n[i] - 1)) }}</td>
            </tr>
          </tbody>
        </table>
        <p class="proof-meta">
          Σ s = <strong>{{ dutch.sum.toFixed(3) }}</strong> (coherent: 1).
          <template v-if="dutch.sum > 1.005">
            Selling a 1-unit bet on every option at the model's prices s collects
            {{ dutch.sum.toFixed(3) }} and pays exactly 1: a sure gain of {{ (dutch.sum - 1).toFixed(3) }}.
          </template>
          <template v-else-if="dutch.sum < 0.995">
            Buying a 1-unit bet on every option at the model's prices costs {{ dutch.sum.toFixed(3) }} and
            pays exactly 1: a sure gain of {{ (1 - dutch.sum).toFixed(3) }}.
          </template>
          Mean |s + n − 1| = {{ dutch.mad.toFixed(3) }} on this question.
        </p>
      </div>
    </template>
  </div>
</template>

<style scoped>
.proof {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0 24px;
  background: var(--vp-c-bg-soft);
}
.proof-form {
  display: grid;
  gap: 10px;
}
.proof-form .cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
@media (max-width: 640px) {
  .proof-form .cols {
    grid-template-columns: 1fr;
  }
}
.proof label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.proof input[type='text'],
.proof textarea {
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
.proof textarea {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  resize: vertical;
}
.proof-status,
.proof-error {
  font-size: 14px;
  margin: 12px 0 0;
}
.proof-error {
  color: var(--vp-c-danger-1);
}
.proof-out {
  margin-top: 12px;
}
.proof-table {
  margin-top: 12px;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  display: block;
  overflow-x: auto;
}
.proof-table td:first-child {
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.proof-meta {
  margin: 10px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
</style>
