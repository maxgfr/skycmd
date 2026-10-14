<script setup lang="ts">
// The cascade of bet 3 (exploratory, amendment 4), replayed from public/results/cascade.json:
// `sky-s5-report pari3` recorded 200 steps of one stream of bench questions for every (α, π).
// s5 answers alone when its max p reaches the threshold λ of the question type, and escalates
// otherwise; the auditor (the dataset gold) checks every escalated answer and a random share π of
// the accepted ones, and λ moves with the estimated error of the accepted answers. Nothing is
// computed or fetched here beyond that file: no model call, no network call, no GLM.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'

interface Item {
  pos: number
  t: string
  k: number
  probs: number[]
  gold: number
  pred: number
  drift: boolean
}
interface Step {
  lambda: number
  accepted: boolean
  audited: boolean
  est: number | null
  wrong: number | null
}
interface Cell {
  alpha: number
  pi: number
  lambda0: Record<string, number>
  steps: Step[]
  after: { error_est: number | null; error_true: number | null; escalations: number; fixed_escalations: number }
}
interface Cascade {
  run: string
  format: string
  eta: number
  stream_len: number
  drift_at: number
  items: Item[]
  cells: Cell[]
}

const data = ref<Cascade | null>(null)
const failed = ref('')
const alpha = ref(0.05)
const pi = ref(0.2)
const at = ref(0)
const playing = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

onMounted(async () => {
  try {
    const r = await fetch(withBase('/results/cascade.json'))
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    data.value = await r.json()
  } catch (e) {
    failed.value = (e as Error).message
  }
})
onBeforeUnmount(() => stop())

const alphas = computed(() => [...new Set(data.value?.cells.map((c) => c.alpha) ?? [])])
const pis = computed(() => [...new Set(data.value?.cells.map((c) => c.pi) ?? [])])
const cell = computed(() => data.value?.cells.find((c) => c.alpha === alpha.value && c.pi === pi.value) ?? null)
const n = computed(() => data.value?.items.length ?? 0)
const item = computed(() => data.value?.items[at.value] ?? null)
const step = computed(() => cell.value?.steps[at.value] ?? null)
const driftIdx = computed(() => data.value?.items.findIndex((i) => i.drift) ?? -1)
const types = computed(() => [...new Set(data.value?.items.map((i) => i.t) ?? [])].sort())

function stop() {
  playing.value = false
  if (timer) clearInterval(timer)
  timer = null
}
function play() {
  if (playing.value) return stop()
  if (at.value >= n.value - 1) at.value = 0
  playing.value = true
  timer = setInterval(() => {
    if (at.value >= n.value - 1) stop()
    else at.value += 1
  }, 120)
}
watch([alpha, pi], () => stop())

function pct(x: number | null | undefined, d = 1): string {
  return typeof x === 'number' && Number.isFinite(x) ? `${(x * 100).toFixed(d)}%` : '–'
}

// Charts: x is the step (0..n-1), y a rate clipped to its axis.
const W = 600
const H = 150
const PAD = 28
function x(i: number): number {
  return PAD + (i / Math.max(1, n.value - 1)) * (W - 2 * PAD)
}
function y(v: number, max: number): number {
  return H - PAD + 8 - (Math.min(Math.max(v, 0), max) / max) * (H - PAD - 8)
}
const ERR_MAX = 0.5
function line(vals: (number | null)[], max: number): string {
  return vals
    .map((v, i) => (v === null || !Number.isFinite(v) ? null : `${x(i).toFixed(1)},${y(v, max).toFixed(1)}`))
    .filter((p) => p !== null)
    .join(' ')
}
const estLine = computed(() => line(cell.value?.steps.map((s) => s.est) ?? [], ERR_MAX))
const trueLine = computed(() => line(cell.value?.steps.map((s) => s.wrong) ?? [], ERR_MAX))
/** λ of each type at the steps where a question of that type came in. */
const lambdaPoints = computed(() =>
  types.value.map((t) => ({
    t,
    pts: (cell.value?.steps ?? [])
      .map((s, i) => ({ i, l: s.lambda, ok: data.value?.items[i].t === t }))
      .filter((p) => p.ok)
      .map((p) => `${x(p.i).toFixed(1)},${y(p.l, 1).toFixed(1)}`)
      .join(' '),
  })),
)
const TYPE_CLASS = ['c0', 'c1', 'c2', 'c3']
</script>

<template>
  <div class="casc">
    <p v-if="failed" class="casc-note">The cascade trajectory is not published yet ({{ failed }}).</p>
    <template v-else-if="data && cell">
      <div class="casc-row">
        <label>Error level α
          <select v-model.number="alpha">
            <option v-for="a in alphas" :key="a" :value="a">{{ pct(a, 0) }}</option>
          </select>
        </label>
        <label>Audited share π of the accepted answers
          <select v-model.number="pi">
            <option v-for="p in pis" :key="p" :value="p">{{ p }}</option>
          </select>
        </label>
        <button type="button" class="casc-btn" @click="play">{{ playing ? 'Pause' : 'Play' }}</button>
        <label class="casc-slider">Step {{ at + 1 }} / {{ n }}
          <input v-model.number="at" type="range" min="0" :max="n - 1" @input="stop" />
        </label>
      </div>

      <div v-if="item && step" class="casc-item" :class="{ drift: item.drift }">
        <div class="casc-head">
          <span class="casc-tag">{{ item.t }}</span>
          <span class="casc-tag">{{ item.k }} options</span>
          <span v-if="item.drift" class="casc-tag warn">after the drift</span>
          <span class="casc-pos">stream question {{ item.pos + 1 }} of {{ data.stream_len }}</span>
        </div>
        <div class="casc-bars">
          <div v-for="(p, j) in item.probs" :key="j" class="casc-bar">
            <span class="casc-opt">option {{ j + 1 }}<template v-if="j === item.gold && step.audited"> · gold</template></span>
            <span class="casc-track"><span class="casc-fill" :class="{ top: j === item.pred }" :style="{ width: `${p * 100}%` }" /></span>
            <span class="casc-p">{{ pct(p) }}</span>
          </div>
        </div>
        <p class="casc-decision">
          max p {{ pct(Math.max(...item.probs)) }} against λ = {{ step.lambda.toFixed(3) }} for {{ item.t }}:
          <strong v-if="step.accepted" class="ok">s5 answers alone</strong>
          <strong v-else class="ko">escalated to the auditor</strong>
          <template v-if="step.accepted && step.audited"> · <strong>drawn for a random audit</strong></template>
          <template v-if="step.audited">
            · the auditor finds it {{ item.pred === item.gold ? 'right' : 'wrong' }}
          </template>
          <template v-else> · not audited: the system never learns whether it was right</template>
        </p>
      </div>

      <figure class="casc-fig">
        <figcaption>
          Error of the accepted answers since the start of the half: <span class="k-est">estimated (ê)</span> and
          <span class="k-true">true</span>, against <span class="k-alpha">α + 0.01</span>
        </figcaption>
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Error of the accepted answers over the stream">
          <line :x1="PAD" :x2="W - PAD" :y1="y(alpha + 0.01, ERR_MAX)" :y2="y(alpha + 0.01, ERR_MAX)" class="alpha" />
          <line v-if="driftIdx > 0" :x1="x(driftIdx)" :x2="x(driftIdx)" :y1="8" :y2="H - PAD + 8" class="drift" />
          <polyline :points="trueLine" class="true" />
          <polyline :points="estLine" class="est" />
          <line :x1="x(at)" :x2="x(at)" :y1="8" :y2="H - PAD + 8" class="cursor" />
          <text :x="PAD - 4" :y="y(0, ERR_MAX)" class="ax" text-anchor="end">0</text>
          <text :x="PAD - 4" :y="y(ERR_MAX, ERR_MAX) + 4" class="ax" text-anchor="end">{{ pct(ERR_MAX, 0) }}</text>
          <text v-if="driftIdx > 0" :x="x(driftIdx) + 4" :y="18" class="ax">drift</text>
        </svg>
      </figure>

      <figure class="casc-fig">
        <figcaption>
          Threshold λ of each type:
          <span v-for="(t, j) in types" :key="t" :class="['k-type', TYPE_CLASS[j % 4]]">{{ t }}</span>
        </figcaption>
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Threshold of each question type over the stream">
          <line v-if="driftIdx > 0" :x1="x(driftIdx)" :x2="x(driftIdx)" :y1="8" :y2="H - PAD + 8" class="drift" />
          <polyline v-for="(l, j) in lambdaPoints" :key="l.t" :points="l.pts" :class="['lam', TYPE_CLASS[j % 4]]" />
          <line :x1="x(at)" :x2="x(at)" :y1="8" :y2="H - PAD + 8" class="cursor" />
          <text :x="PAD - 4" :y="y(0, 1)" class="ax" text-anchor="end">0</text>
          <text :x="PAD - 4" :y="y(1, 1) + 4" class="ax" text-anchor="end">1</text>
        </svg>
      </figure>

      <p class="casc-note">
        On this whole stream ({{ data.stream_len }} questions, every 20th shown), after the drift: estimated error of the
        accepted answers {{ pct(cell.after.error_est) }}, true error {{ pct(cell.after.error_true) }}, against
        α = {{ pct(alpha, 0) }}; {{ cell.after.escalations }} escalations against {{ cell.after.fixed_escalations }} for
        the fixed threshold. Model {{ data.run }} ({{ data.format }}), step η = {{ data.eta }}. A replay of a recorded
        run: no model and no network call here, and no GLM anywhere. The auditor is the dataset gold, standing in for a
        human. Bet 3 fails in every (α, π) cell: see <a :href="withBase('/results/')">the results</a>.
      </p>
    </template>
    <p v-else class="casc-note">Loading the cascade…</p>
  </div>
</template>

<style scoped>
.casc {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0 24px;
  background: var(--vp-c-bg-soft);
}
.casc-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: end;
}
.casc label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.casc select {
  font: inherit;
  font-weight: 400;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 4px 8px;
}
.casc-slider {
  flex: 1 1 200px;
}
.casc-slider input {
  width: 100%;
}
.casc-btn {
  font-size: 14px;
  font-weight: 600;
  border: 1px solid var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  border-radius: 8px;
  padding: 4px 14px;
  min-width: 72px;
}
.casc-btn:hover {
  background: var(--vp-c-brand-soft);
}
.casc-item {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg);
}
.casc-item.drift {
  border-color: var(--vp-c-warning-1);
}
.casc-head {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  font-size: 13px;
}
.casc-tag {
  border-radius: 6px;
  padding: 1px 8px;
  background: var(--vp-c-default-soft);
}
.casc-tag.warn {
  background: var(--vp-c-warning-soft);
}
.casc-pos {
  margin-left: auto;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.casc-bars {
  display: grid;
  gap: 4px;
  margin: 10px 0;
}
.casc-bar {
  display: grid;
  grid-template-columns: 120px 1fr 56px;
  gap: 8px;
  align-items: center;
  font-size: 13px;
}
.casc-track {
  height: 10px;
  border-radius: 5px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.casc-fill {
  display: block;
  height: 100%;
  background: var(--vp-c-text-3);
}
.casc-fill.top {
  background: var(--vp-c-brand-1);
}
.casc-p {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.casc-decision {
  font-size: 14px;
  margin: 0;
}
.casc .ok {
  color: var(--vp-c-green-1, #16a34a);
}
.casc .ko {
  color: var(--vp-c-warning-1);
}
.casc-fig {
  margin: 14px 0 0;
}
.casc-fig figcaption {
  font-size: 13px;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.casc-fig svg {
  width: 100%;
  height: auto;
  display: block;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}
.casc-fig polyline {
  fill: none;
  stroke-width: 2;
  stroke-linejoin: round;
}
.casc-fig .est,
.k-est {
  stroke: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.casc-fig .true,
.k-true {
  stroke: var(--vp-c-text-2);
  color: var(--vp-c-text-2);
}
.casc-fig .true {
  stroke-dasharray: 4 3;
}
.casc-fig .alpha,
.k-alpha {
  stroke: var(--vp-c-danger-1);
  color: var(--vp-c-danger-1);
  stroke-dasharray: 2 4;
}
.casc-fig .drift {
  stroke: var(--vp-c-warning-1);
  stroke-dasharray: 6 4;
}
.casc-fig .cursor {
  stroke: var(--vp-c-text-3);
}
.casc-fig .ax {
  font-size: 11px;
  fill: var(--vp-c-text-2);
}
.k-type {
  margin-left: 8px;
  font-weight: 600;
}
.c0 {
  stroke: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.c1 {
  stroke: var(--vp-c-warning-1);
  color: var(--vp-c-warning-1);
}
.c2 {
  stroke: var(--vp-c-green-1, #16a34a);
  color: var(--vp-c-green-1, #16a34a);
}
.c3 {
  stroke: var(--vp-c-text-2);
  color: var(--vp-c-text-2);
}
.casc-note {
  color: var(--vp-c-text-2);
  font-size: 14px;
  margin: 12px 0 0;
}
@media (max-width: 640px) {
  .casc-bar {
    grid-template-columns: 90px 1fr 48px;
  }
  .casc-pos {
    margin-left: 0;
  }
}
</style>
