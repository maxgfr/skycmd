<script setup lang="ts">
// The /forecast/ page: pick a series, forecast it in the browser with every published forecaster
// (s6-cast once trained, TinyCast in wasm, Timestral-5M in onnxruntime-web, seasonal naive),
// and compare the median and the 10–90 % band, the latency and, in backtest mode, the MASE.
import { computed, onMounted, ref, shallowRef, watch } from 'vue'
import {
  coverage,
  levelIndex,
  loadDemoSeries,
  loadForecastModels,
  loadForecastWasm,
  loadOnnxForecaster,
  loadWasmForecaster,
  runOnnxForecaster,
  runWasmForecaster,
  type DemoSeries,
  type ForecastModelEntry,
  type ForecastWeb,
  type OnnxForecaster,
  type QuantileForecast,
  type WasmForecaster,
} from '../lib/forecast'
import { formatInt, formatKB } from '../lib/wasm'

type ModelKey = 's6-cast' | 'TinyCast' | 'Timestral-5M' | 'Seasonal naive'

// Series colors: categorical slots 1–4, light and dark steps (validated for both modes).
const SLOT: Record<ModelKey, number> = { 's6-cast': 1, TinyCast: 2, 'Timestral-5M': 3, 'Seasonal naive': 4 }
const MODEL_KEYS: ModelKey[] = ['s6-cast', 'TinyCast', 'Timestral-5M', 'Seasonal naive']
const SYNTH = [
  { kind: 'tsi', label: 'Trend + seasons + noise' },
  { kind: 'sarima', label: 'SARIMA process' },
  { kind: 'gp', label: 'Gaussian-process kernels' },
  { kind: 'pulse', label: 'Pulse train' },
]
const HORIZONS = [16, 32, 64, 128]

const mod = shallowRef<ForecastWeb | null>(null)
const entries = ref<ForecastModelEntry[]>([])
const demos = ref<DemoSeries[]>([])
const source = ref('synth:tsi')
const seed = ref(7)
const csv = ref('')
const horizon = ref(64)
const backtest = ref(true)
const enabled = ref<Record<ModelKey, boolean>>({
  's6-cast': true,
  TinyCast: true,
  'Timestral-5M': false,
  'Seasonal naive': true,
})
const status = ref('Loading the wasm module…')
const error = ref('')
const results = ref<Partial<Record<ModelKey, QuantileForecast>>>({})
const loading = ref<Partial<Record<ModelKey, string>>>({})
const wasmModels = new Map<string, WasmForecaster>()
let onnx: OnnxForecaster | null = null
let runId = 0

const entryOf = (k: ModelKey) => entries.value.find((e) => (k === 's6-cast' ? e.name.startsWith('s6-cast') : e.name === k))
const available = (k: ModelKey) => k === 'Seasonal naive' || !!entryOf(k)

/** The full series of the selected source. */
const series = computed<number[]>(() => {
  const m = mod.value
  if (source.value.startsWith('synth:')) {
    if (!m) return []
    return Array.from(m.synthSeries(source.value.slice(6), BigInt(Math.max(0, Math.floor(seed.value))), 1200))
  }
  if (source.value === 'csv') {
    return csv.value
      .split(/[\s,;]+/)
      .filter((t) => t.trim() !== '')
      .map((t) => (/^(nan|null|na)$/i.test(t) ? NaN : Number(t)))
  }
  const d = demos.value.find((s) => s.id === source.value)
  return d ? d.values.map((v) => (v == null ? NaN : v)) : []
})
const demo = computed(() => demos.value.find((s) => s.id === source.value) ?? null)
/** History given to the models, and the hidden truth in backtest mode. */
const history = computed(() => (backtest.value ? series.value.slice(0, -horizon.value) : series.value))
const truth = computed(() => (backtest.value ? series.value.slice(-horizon.value) : []))
const season = computed(() => {
  if (demo.value?.season) return demo.value.season
  const m = mod.value
  if (!m || history.value.length < 8) return 1
  return m.dominantPeriod(Float32Array.from(history.value)) || 1
})
const enough = computed(() => history.value.filter((v) => Number.isFinite(v)).length >= 8)

async function ensureModel(k: ModelKey): Promise<boolean> {
  const e = entryOf(k)
  if (!e || !mod.value) return false
  if (e.kind === 'onnx') {
    if (onnx) return true
    loading.value = { ...loading.value, [k]: `Loading onnxruntime-web and ${formatKB(e.bytes)}…` }
    try {
      onnx = await loadOnnxForecaster(e)
      return true
    } finally {
      loading.value = { ...loading.value, [k]: undefined }
    }
  }
  if (wasmModels.has(e.name)) return true
  loading.value = { ...loading.value, [k]: `Downloading ${formatKB(e.bytes)}…` }
  try {
    wasmModels.set(e.name, await loadWasmForecaster(mod.value, e))
    return true
  } finally {
    loading.value = { ...loading.value, [k]: undefined }
  }
}

async function run() {
  const m = mod.value
  if (!m) return
  const id = ++runId
  error.value = ''
  if (!enough.value) {
    results.value = {}
    return
  }
  const hist = history.value
  const h = horizon.value
  const next: Partial<Record<ModelKey, QuantileForecast>> = {}
  for (const k of MODEL_KEYS) {
    if (!enabled.value[k] || !available(k)) continue
    try {
      if (k === 'Seasonal naive') {
        const t0 = performance.now()
        const sn = Array.from(m.seasonalNaive(Float32Array.from(hist), season.value, h))
        next[k] = { levels: [0.5], q: sn.map((v) => [v]), micros: (performance.now() - t0) * 1000 }
        continue
      }
      if (!(await ensureModel(k))) continue
      const e = entryOf(k)!
      next[k] =
        e.kind === 'onnx' && onnx
          ? await runOnnxForecaster(onnx, hist, h)
          : runWasmForecaster(wasmModels.get(e.name)!, hist, h)
    } catch (err) {
      error.value = `${k}: ${(err as Error).message}`
    }
    if (id !== runId) return
  }
  if (id === runId) results.value = next
}

watch([source, seed, horizon, backtest, csv, enabled], () => run(), { deep: true })

onMounted(async () => {
  try {
    mod.value = await loadForecastWasm()
  } catch (e) {
    status.value = ''
    error.value = `The wasm module could not be loaded (${(e as Error).message}).`
    return
  }
  ;[entries.value, demos.value] = await Promise.all([loadForecastModels(), loadDemoSeries()])
  status.value = ''
  await run()
})

// --- chart --------------------------------------------------------------------------------

const W = 760
const H = 300
const PAD = { l: 52, r: 16, t: 14, b: 28 }
// Twice the horizon of history (at least 96 points): enough context to read, room for the forecast.
const shown = computed(() => Math.min(history.value.length, Math.max(2 * horizon.value, 96)))
const xs = computed(() => {
  const n = shown.value + horizon.value
  return (i: number) => PAD.l + ((W - PAD.l - PAD.r) * i) / Math.max(1, n - 1)
})
const visibleHistory = computed(() => history.value.slice(-shown.value))
const extent = computed(() => {
  const vals: number[] = [...visibleHistory.value, ...truth.value]
  for (const r of Object.values(results.value)) {
    if (!r) continue
    const lo = levelIndex(r.levels, 0.1)
    const hi = levelIndex(r.levels, 0.9)
    for (const row of r.q) vals.push(row[lo], row[hi])
  }
  const fin = vals.filter((v) => Number.isFinite(v))
  if (!fin.length) return [0, 1]
  let lo = Math.min(...fin)
  let hi = Math.max(...fin)
  if (hi - lo < 1e-9) {
    lo -= 1
    hi += 1
  }
  const pad = (hi - lo) * 0.06
  return [lo - pad, hi + pad]
})
const ys = computed(() => {
  const [lo, hi] = extent.value
  return (v: number) => PAD.t + (H - PAD.t - PAD.b) * (1 - (v - lo) / (hi - lo))
})
const ticks = computed(() => {
  const [lo, hi] = extent.value
  const raw = (hi - lo) / 4
  const p = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(+v.toPrecision(10))
  return { values: out, decimals: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) }
})
/** Tick labels with the decimals of the step (0, 0.5, 1.0 — not 0.500 next to 1.00). */
function tickLabel(v: number): string {
  const d = ticks.value.decimals
  return Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: d }) : v.toFixed(d)
}

function path(points: [number, number][]): string {
  let d = ''
  let open = false
  for (const [x, y] of points) {
    if (!Number.isFinite(y)) {
      open = false
      continue
    }
    d += `${open ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
    open = true
  }
  return d
}

const historyPath = computed(() => path(visibleHistory.value.map((v, i) => [xs.value(i), ys.value(v)])))
const truthPath = computed(() => {
  const n0 = visibleHistory.value.length - 1
  const pts: [number, number][] = [[xs.value(n0), ys.value(visibleHistory.value[n0])]]
  truth.value.forEach((v, i) => pts.push([xs.value(n0 + 1 + i), ys.value(v)]))
  return path(pts)
})
const cutX = computed(() => xs.value(visibleHistory.value.length - 1))

interface Drawn {
  key: ModelKey
  median: string
  band: string | null
}
const drawn = computed<Drawn[]>(() => {
  const n0 = visibleHistory.value.length
  return MODEL_KEYS.flatMap((k) => {
    const r = results.value[k]
    if (!r) return []
    const mi = levelIndex(r.levels, 0.5)
    const median = path(r.q.map((row, i) => [xs.value(n0 + i), ys.value(row[mi])]))
    let band: string | null = null
    if (r.levels.length > 1) {
      const lo = levelIndex(r.levels, 0.1)
      const hi = levelIndex(r.levels, 0.9)
      const up = r.q.map((row, i) => `${xs.value(n0 + i).toFixed(1)},${ys.value(row[hi]).toFixed(1)}`)
      const down = r.q.map((row, i) => `${xs.value(n0 + i).toFixed(1)},${ys.value(row[lo]).toFixed(1)}`).reverse()
      band = `M${up.join('L')}L${down.join('L')}Z`
    }
    return [{ key: k, median, band }]
  })
})

// Hover: crosshair at the nearest forecast step, tooltip with every model's median.
const hoverStep = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const svg = ev.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const x = ((ev.clientX - rect.left) / rect.width) * W
  const n0 = visibleHistory.value.length
  const i = Math.round(((x - PAD.l) / (W - PAD.l - PAD.r)) * (shown.value + horizon.value - 1)) - n0
  hoverStep.value = i >= 0 && i < horizon.value ? i : null
}
const tooltip = computed(() => {
  const i = hoverStep.value
  if (i == null) return null
  const rows = MODEL_KEYS.flatMap((k) => {
    const r = results.value[k]
    if (!r || !r.q[i]) return []
    const v = r.q[i][levelIndex(r.levels, 0.5)]
    const band =
      r.levels.length > 1 ? `${fmt(r.q[i][levelIndex(r.levels, 0.1)])} – ${fmt(r.q[i][levelIndex(r.levels, 0.9)])}` : ''
    return [{ key: k, v: fmt(v), band }]
  })
  return { step: i + 1, x: xs.value(visibleHistory.value.length + i), truth: truth.value[i], rows }
})

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  return a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toPrecision(3)
}

// --- results table --------------------------------------------------------------------------

const rows = computed(() =>
  MODEL_KEYS.map((k) => {
    const e = entryOf(k)
    const r = results.value[k]
    let maseV: number | null = null
    let cov: number | null = null
    if (r && backtest.value && mod.value && truth.value.length) {
      const med = r.q.map((row) => row[levelIndex(r.levels, 0.5)])
      const m = mod.value.mase(Float32Array.from(history.value), season.value, Float32Array.from(truth.value), Float32Array.from(med))
      maseV = Number.isFinite(m) ? m : null
      cov = r.levels.length > 1 ? coverage(r, truth.value) : null
    }
    return {
      key: k,
      entry: e,
      result: r,
      available: available(k),
      params: k === 'Seasonal naive' ? 0 : e?.params ?? null,
      bytes: k === 'Seasonal naive' ? 0 : e?.bytes ?? null,
      mase: maseV,
      cov,
    }
  }),
)

function latency(us: number): string {
  return us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : `${formatInt(us)} µs`
}
</script>

<template>
  <div class="forecast">
    <div class="toolbar">
      <label class="field">
        <span>Series</span>
        <select v-model="source">
          <optgroup label="Synthetic (s6-cast's generators)">
            <option v-for="s in SYNTH" :key="s.kind" :value="`synth:${s.kind}`">{{ s.label }}</option>
          </optgroup>
          <optgroup v-if="demos.length" label="Real series">
            <option v-for="d in demos" :key="d.id" :value="d.id">{{ d.label }}</option>
          </optgroup>
          <option value="csv">Paste your own…</option>
        </select>
      </label>
      <label v-if="source.startsWith('synth:')" class="field">
        <span>Seed</span>
        <input v-model.number="seed" type="number" min="0" step="1" class="seed" />
      </label>
      <label class="field">
        <span>Horizon</span>
        <select v-model.number="horizon">
          <option v-for="h in HORIZONS" :key="h" :value="h">{{ h }} steps</option>
        </select>
      </label>
      <div class="seg" role="group" aria-label="Mode">
        <button type="button" :class="{ on: backtest }" @click="backtest = true">Backtest</button>
        <button type="button" :class="{ on: !backtest }" @click="backtest = false">Forecast</button>
      </div>
    </div>
    <textarea
      v-if="source === 'csv'"
      v-model="csv"
      class="csv"
      rows="3"
      placeholder="Numbers separated by commas, spaces or new lines (NaN for a missing value), oldest first"
    />
    <div class="models" role="group" aria-label="Models">
      <label v-for="k in MODEL_KEYS" :key="k" class="model" :class="{ off: !available(k) }">
        <!-- an untrained model stays enabled (it runs as soon as it ships) but shows unchecked -->
        <input
          type="checkbox"
          :checked="enabled[k] && available(k)"
          :disabled="!available(k)"
          @change="enabled[k] = ($event.target as HTMLInputElement).checked"
        />
        <span class="key" :class="`s${SLOT[k]}`" aria-hidden="true" />
        <span>{{ k }}</span>
        <span v-if="!available(k)" class="badge">not trained yet</span>
        <span v-else-if="loading[k]" class="badge">{{ loading[k] }}</span>
        <span v-else-if="k === 'Timestral-5M' && !enabled[k]" class="badge">23 MB, on demand</span>
      </label>
    </div>
    <p v-if="status" class="status">{{ status }}</p>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="!status && !enough" class="status">Give at least 8 numbers of history.</p>

    <figure class="chart">
      <svg
        :viewBox="`0 0 ${W} ${H}`"
        role="img"
        :aria-label="`Forecast of the next ${horizon} steps`"
        @pointermove="onMove"
        @pointerleave="hoverStep = null"
      >
        <g class="grid">
          <line v-for="t in ticks.values" :key="t" :x1="PAD.l" :x2="W - PAD.r" :y1="ys(t)" :y2="ys(t)" />
          <text v-for="t in ticks.values" :key="`l${t}`" :x="PAD.l - 8" :y="ys(t) + 4" text-anchor="end">{{ tickLabel(t) }}</text>
        </g>
        <line class="cut" :x1="cutX" :x2="cutX" :y1="PAD.t" :y2="H - PAD.b" />
        <text class="axis-note" :x="cutX + 6" :y="H - PAD.b + 18">now</text>
        <path v-for="d in drawn" :key="`b${d.key}`" class="band" :class="`s${SLOT[d.key]}`" :d="d.band ?? ''" />
        <path class="history" :d="historyPath" />
        <path v-if="backtest" class="truth" :d="truthPath" />
        <path v-for="d in drawn" :key="`m${d.key}`" class="median" :class="`s${SLOT[d.key]}`" :d="d.median" />
        <g v-if="tooltip">
          <line class="crosshair" :x1="tooltip.x" :x2="tooltip.x" :y1="PAD.t" :y2="H - PAD.b" />
        </g>
      </svg>
      <div v-if="tooltip" class="tip" :style="{ left: `${(tooltip.x / W) * 100}%` }">
        <div class="tip-head">step +{{ tooltip.step }}</div>
        <div v-if="backtest && Number.isFinite(tooltip.truth)" class="tip-row"><span class="key truth-key" />actual <b>{{ fmt(tooltip.truth) }}</b></div>
        <div v-for="r in tooltip.rows" :key="r.key" class="tip-row">
          <span class="key" :class="`s${SLOT[r.key]}`" />{{ r.key }} <b>{{ r.v }}</b><span v-if="r.band" class="muted"> ({{ r.band }})</span>
        </div>
      </div>
      <figcaption>
        <span><span class="key history-key" />history</span>
        <span v-if="backtest"><span class="key truth-key" />actual (hidden from the models)</span>
        <span v-for="d in drawn" :key="`c${d.key}`"
          ><span class="key" :class="`s${SLOT[d.key]}`" />{{ d.key }}: {{ d.band ? 'median, band 10–90 %' : 'point forecast' }}</span
        >
      </figcaption>
    </figure>

    <table class="results">
      <thead>
        <tr>
          <th>Model</th>
          <th class="num">Params</th>
          <th class="num">Size</th>
          <th class="num">Latency</th>
          <th v-if="backtest" class="num">MASE</th>
          <th v-if="backtest" class="num">In 10–90 %</th>
          <th>License</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in rows" :key="r.key" :class="{ off: !r.available }">
          <td><span class="key" :class="`s${SLOT[r.key]}`" />{{ r.key }}<span v-if="r.result?.exits" class="muted"> · {{ r.result.exits }}</span></td>
          <td class="num">{{ r.params == null ? '—' : r.params === 0 ? '0' : formatInt(r.params) }}</td>
          <td class="num">{{ r.bytes == null ? '—' : r.bytes === 0 ? '—' : formatKB(r.bytes) }}</td>
          <td class="num">{{ r.result ? latency(r.result.micros) : r.available ? '—' : 'not trained yet' }}</td>
          <td v-if="backtest" class="num">{{ r.mase == null ? '—' : r.mase.toFixed(3) }}</td>
          <td v-if="backtest" class="num">{{ r.cov == null ? '—' : `${Math.round(r.cov * 100)} %` }}</td>
          <td>
            <a v-if="r.entry" :href="r.entry.url" target="_blank" rel="noopener">{{ r.entry.license }}</a>
            <span v-else-if="r.key === 'Seasonal naive'" class="muted">no model</span>
            <span v-else class="muted">Apache-2.0</span>
          </td>
        </tr>
      </tbody>
    </table>
    <p class="note">
      MASE divides the mean absolute error of the median by the history's own seasonal-naive error (season
      {{ season }}{{ demo ? '' : ', from the periodogram' }}); below 1 beats repeating the last season. Every model runs
      on this page: the wasm module for s6-cast, TinyCast and the seasonal naive, onnxruntime-web for Timestral.
      <span v-if="demo"> Series: {{ demo.credit }}.</span>
    </p>
  </div>
</template>

<style scoped>
.forecast {
  --s1: #2a78d6;
  --s2: #eb6834;
  --s3: #1baf7a;
  --s4: #eda100;
  --ink: var(--vp-c-text-1);
  --ink-2: var(--vp-c-text-2);
  --muted: #898781;
  --grid: #e1e0d9;
  --surface: var(--vp-c-bg);
  margin: 16px 0 32px;
}
:global(.dark) .forecast {
  --s1: #3987e5;
  --s2: #d95926;
  --s3: #199e70;
  --s4: #c98500;
  --grid: #2c2c2a;
}
.toolbar,
.models {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  align-items: center;
}
.models {
  margin: 12px 0 4px;
}
.field {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}
.field span {
  color: var(--ink-2);
}
select,
input.seed,
.csv {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 5px 8px;
  background: var(--vp-c-bg-soft);
  color: var(--ink);
  font: inherit;
  font-size: 14px;
}
input.seed {
  width: 72px;
}
.csv {
  width: 100%;
  margin-top: 10px;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
}
.seg {
  display: inline-flex;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  overflow: hidden;
}
.seg button {
  padding: 5px 14px;
  font-size: 14px;
  color: var(--ink-2);
}
.seg button.on {
  background: var(--vp-c-brand-soft);
  color: var(--ink);
}
.model {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  cursor: pointer;
}
.model.off {
  opacity: 0.6;
  cursor: default;
}
.badge {
  font-size: 12px;
  color: var(--ink-2);
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  padding: 1px 8px;
}
.key {
  display: inline-block;
  width: 14px;
  height: 3px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
  background: var(--muted);
}
.key.s1 { background: var(--s1); }
.key.s2 { background: var(--s2); }
.key.s3 { background: var(--s3); }
.key.s4 { background: var(--s4); }
.history-key { background: var(--ink-2); }
.truth-key { background: var(--ink); }
.status,
.error,
.note {
  font-size: 14px;
  color: var(--ink-2);
}
.error {
  color: var(--vp-c-danger-1);
}
.chart {
  position: relative;
  margin: 12px 0 0;
}
svg {
  width: 100%;
  height: auto;
  display: block;
  touch-action: none;
}
.grid line {
  stroke: var(--grid);
  stroke-width: 1;
}
.grid text,
.axis-note {
  fill: var(--muted);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.cut,
.crosshair {
  stroke: var(--muted);
  stroke-width: 1;
}
.history {
  fill: none;
  stroke: var(--ink-2);
  stroke-width: 1.5;
  stroke-linejoin: round;
}
.truth {
  fill: none;
  stroke: var(--ink);
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.median {
  fill: none;
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.median.s1 { stroke: var(--s1); }
.median.s2 { stroke: var(--s2); }
.median.s3 { stroke: var(--s3); }
.median.s4 { stroke: var(--s4); }
.band { stroke: none; opacity: 0.12; }
.band.s1 { fill: var(--s1); }
.band.s2 { fill: var(--s2); }
.band.s3 { fill: var(--s3); }
.band.s4 { fill: var(--s4); }
.tip {
  position: absolute;
  top: 8px;
  transform: translateX(8px);
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
  box-shadow: var(--vp-shadow-2);
}
.tip-head {
  color: var(--ink-2);
  margin-bottom: 2px;
}
.tip-row b {
  font-variant-numeric: tabular-nums;
}
.muted {
  color: var(--ink-2);
}
figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  font-size: 12px;
  color: var(--ink-2);
  margin-top: 6px;
}
.results {
  width: 100%;
  margin-top: 16px;
  font-size: 14px;
}
.results .num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.results tr.off td {
  color: var(--ink-2);
}
@media (max-width: 640px) {
  .tip {
    display: none;
  }
}
</style>
