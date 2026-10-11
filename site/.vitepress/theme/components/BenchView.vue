<script setup lang="ts">
// The /bench/ page: size vs accuracy vs latency per model (public/bench.json), plus an
// in-browser benchmark of every published model through the wasm `bench()` export.
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { familyOf, formatInt, loadEngine, loadManifest, loadWasm, type ModelEntry } from '../lib/wasm'

interface BenchRow {
  name: string
  /** `general.architecture` (`modern-bert` or `skycmd-s2`), when known. */
  arch?: string | null
  params: number | null
  bytes: number | null
  accuracy: number | null
  games_accuracy: number | null
  flappy_accuracy: number | null
  rescue_accuracy: number | null
  dodge_accuracy: number | null
  nll: number | null
  ece: number | null
  native_us: number | null
  wasm_us: number | null
}

interface Local {
  median_us: number
  p95_us: number
  decisions_per_s: number
  family?: string
  /** s2: state-encoder blocks per question (early exit); null for s1. */
  mean_layers_used?: number | null
  max_layers?: number | null
}

const rows = ref<BenchRow[]>([])
const note = ref('')
const generated = ref('')
const loaded = ref(false)
const local = ref<Record<string, Local>>({})
const running = ref(false)
const progress = ref('')
const manifest = ref<ModelEntry[]>([])

onMounted(async () => {
  try {
    const r = await fetch(withBase('/bench.json'))
    if (r.ok) {
      const data = await r.json()
      rows.value = data.models ?? []
      note.value = data.note ?? ''
      generated.value = data.generated ?? ''
    }
  } catch {
    /* no bench.json yet */
  }
  manifest.value = await loadManifest()
  // Models published without metrics still get a row.
  for (const m of manifest.value) {
    if (!rows.value.some((r) => r.name === m.name)) {
      rows.value.push({
        name: m.name,
        arch: m.arch ?? null,
        params: m.params,
        bytes: m.bytes,
        accuracy: null,
        games_accuracy: null,
        flappy_accuracy: null,
        rescue_accuracy: null,
        dodge_accuracy: null,
        nll: null,
        ece: null,
        native_us: null,
        wasm_us: null,
      })
    }
  }
  loaded.value = true
})

async function runBench() {
  running.value = true
  try {
    const mod = await loadWasm()
    for (const m of manifest.value) {
      progress.value = `Loading ${m.name}…`
      const engine = await loadEngine(m)
      progress.value = `Running ${m.name}…`
      // Let the browser paint the progress text before the synchronous run.
      await new Promise((r) => setTimeout(r, 30))
      local.value = { ...local.value, [m.name]: JSON.parse(mod.bench(engine, 300)) }
    }
    progress.value = manifest.value.length
      ? `Done: 300 decisions per model on ${navigator.hardwareConcurrency ?? '?'}-thread ${navigator.platform || 'device'}.`
      : 'No model is published yet: nothing to benchmark.'
  } catch (e) {
    progress.value = `Benchmark failed: ${(e as Error).message}`
  } finally {
    running.value = false
  }
}

const fmtPct = (v: number | null) => (v == null ? 'TBD' : `${(v * 100).toFixed(1)}%`)
const fmtNum = (v: number | null, d = 3) => (v == null ? 'TBD' : v.toFixed(d))
const fmtUs = (v: number | null | undefined) => (v == null ? 'TBD' : `${formatInt(v)} µs`)
const fmtKB = (v: number | null) => (v == null ? 'TBD' : `${formatInt(v / 1024)} KB`)
const fmtInt = (v: number | null) => (v == null ? 'TBD' : formatInt(v))

function family(r: BenchRow): string {
  const l = local.value[r.name]
  if (l?.family) return l.family
  const m = manifest.value.find((e) => e.name === r.name)
  return familyOf({ name: r.name, arch: r.arch ?? m?.arch })
}

function depth(r: BenchRow): string {
  const l = local.value[r.name]
  if (!l) return '—'
  if (l.mean_layers_used == null) return 'all'
  return `${l.mean_layers_used.toFixed(2)} / ${l.max_layers ?? '?'}`
}

function wasmUs(r: BenchRow): number | null {
  return local.value[r.name]?.median_us ?? r.wasm_us
}

// SVG scatter: x = file size (log scale), y = accuracy on game decisions, bubble = latency.
const W = 640
const H = 320
const PAD = { l: 56, r: 24, t: 20, b: 44 }
const plotted = computed(() => rows.value.filter((r) => r.bytes != null && (r.games_accuracy ?? r.accuracy) != null))
const chart = computed(() => {
  const pts = plotted.value
  if (!pts.length) return null
  const xs = pts.map((r) => Math.log10(r.bytes!))
  const ys = pts.map((r) => (r.games_accuracy ?? r.accuracy)!)
  const x0 = Math.min(...xs) - 0.25
  const x1 = Math.max(...xs) + 0.25
  const y0 = Math.max(0, Math.floor((Math.min(...ys) - 0.05) * 10) / 10)
  const y1 = Math.min(1, Math.ceil((Math.max(...ys) + 0.05) * 10) / 10)
  const X = (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * (W - PAD.l - PAD.r)
  const Y = (v: number) => H - PAD.b - ((v - y0) / (y1 - y0 || 1)) * (H - PAD.t - PAD.b)
  const yTicks: number[] = []
  for (let v = y0; v <= y1 + 1e-9; v += 0.1) yTicks.push(Math.round(v * 10) / 10)
  const xTicks: number[] = []
  for (let e = Math.ceil(x0 * 2) / 2; e <= x1; e += 0.5) xTicks.push(e)
  const points = pts.map((r) => {
    const us = wasmUs(r) ?? r.native_us
    return {
      name: r.name,
      x: X(Math.log10(r.bytes!)),
      y: Y((r.games_accuracy ?? r.accuracy)!),
      r: us == null ? 7 : Math.max(6, Math.min(26, 4 + Math.sqrt(us) / 2)),
      label: `${r.name} · ${us == null ? 'latency TBD' : `${formatInt(us)} µs`}`,
    }
  })
  return { X, Y, yTicks, xTicks, points }
})

function kbLabel(e: number) {
  const kb = 10 ** e / 1024
  return kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${Math.round(kb)} KB`
}
</script>

<template>
  <div class="bench">
    <div class="actions">
      <button type="button" class="run" :disabled="running || !loaded" @click="runBench">
        {{ running ? 'Running…' : 'Run benchmark in your browser' }}
      </button>
      <span class="progress">{{ progress }}</span>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>model</th>
            <th>family</th>
            <th>params</th>
            <th>file</th>
            <th>accuracy (all)</th>
            <th>games</th>
            <th>flappy</th>
            <th>rescue</th>
            <th>dodge</th>
            <th>NLL</th>
            <th>ECE</th>
            <th>native / question</th>
            <th>WASM / decision</th>
            <th>decisions/s here</th>
            <th>layers / question here</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in rows" :key="r.name">
            <td><strong>{{ r.name }}</strong></td>
            <td>{{ family(r) }}</td>
            <td>{{ fmtInt(r.params) }}</td>
            <td>{{ fmtKB(r.bytes) }}</td>
            <td>{{ fmtPct(r.accuracy) }}</td>
            <td>{{ fmtPct(r.games_accuracy) }}</td>
            <td>{{ fmtPct(r.flappy_accuracy) }}</td>
            <td>{{ fmtPct(r.rescue_accuracy) }}</td>
            <td>{{ fmtPct(r.dodge_accuracy) }}</td>
            <td>{{ fmtNum(r.nll) }}</td>
            <td>{{ fmtNum(r.ece) }}</td>
            <td>{{ fmtUs(r.native_us) }}</td>
            <td :class="{ fresh: local[r.name] }">{{ fmtUs(wasmUs(r)) }}</td>
            <td>{{ local[r.name] ? formatInt(local[r.name].decisions_per_s) : '—' }}</td>
            <td>{{ depth(r) }}</td>
          </tr>
          <tr v-if="loaded && !rows.length">
            <td colspan="15" class="empty">No model is published yet. Every number is TBD until training finishes.</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="note">
      <template v-if="generated">Data from {{ generated }}. </template><template v-if="note">{{ note }}. </template>A WASM decision answers both questions of
      Flappy Drone and steps the physics: s1 runs two full forward passes, s2 encodes the state once for both questions (the
      question and options come from its cache) and may stop at an early exit. The native column times a single question.
    </p>

    <figure v-if="chart" class="chart">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Accuracy on game decisions versus model file size">
        <g class="grid">
          <line v-for="t in chart.yTicks" :key="`y${t}`" :x1="PAD.l" :x2="W - PAD.r" :y1="chart.Y(t)" :y2="chart.Y(t)" />
        </g>
        <g class="axis">
          <text v-for="t in chart.yTicks" :key="`yl${t}`" :x="PAD.l - 8" :y="chart.Y(t) + 4" text-anchor="end">
            {{ Math.round(t * 100) }}%
          </text>
          <text v-for="t in chart.xTicks" :key="`xl${t}`" :x="chart.X(t)" :y="H - PAD.b + 18" text-anchor="middle">
            {{ kbLabel(t) }}
          </text>
          <text :x="(W + PAD.l) / 2" :y="H - 6" text-anchor="middle" class="title">model file size (log scale)</text>
          <text :x="14" :y="H / 2" text-anchor="middle" class="title" :transform="`rotate(-90 14 ${H / 2})`">
            accuracy on game decisions
          </text>
        </g>
        <g v-for="p in chart.points" :key="p.name" class="pt">
          <circle :cx="p.x" :cy="p.y" :r="p.r" />
          <text :x="p.x + p.r + 6" :y="p.y + 4">{{ p.label }}</text>
        </g>
      </svg>
      <figcaption>Bubble size grows with latency per decision (WASM when measured here, else native).</figcaption>
    </figure>
    <p v-else-if="loaded" class="note">The chart appears once at least one model has evaluation results.</p>
  </div>
</template>

<style scoped>
.actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin: 16px 0;
}
.run {
  padding: 8px 16px;
  border-radius: 10px;
  font-weight: 700;
  color: #fff;
  background: var(--vp-c-brand-1);
}
.run:disabled {
  opacity: 0.6;
  cursor: progress;
}
.progress {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.table-wrap {
  overflow-x: auto;
}
table {
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
td:first-child {
  white-space: nowrap;
}
td.fresh {
  color: var(--vp-c-brand-1);
  font-weight: 700;
}
.empty {
  color: var(--vp-c-text-3);
}
.note {
  font-size: 13px;
  color: var(--vp-c-text-3);
}
.chart {
  margin: 24px 0 0;
}
.chart svg {
  width: 100%;
  height: auto;
  max-width: 720px;
}
.grid line {
  stroke: var(--vp-c-divider);
}
.axis text {
  font-size: 11px;
  fill: var(--vp-c-text-2);
}
.axis .title {
  font-size: 12px;
  fill: var(--vp-c-text-3);
}
.pt circle {
  fill: var(--vp-c-brand-1);
  fill-opacity: 0.35;
  stroke: var(--vp-c-brand-1);
  stroke-width: 2;
}
.pt text {
  font-size: 12px;
  font-weight: 600;
  fill: var(--vp-c-text-1);
}
figcaption {
  font-size: 12px;
  color: var(--vp-c-text-3);
}
</style>
