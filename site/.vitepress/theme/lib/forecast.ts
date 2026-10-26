// The /forecast/ page's runtime: the wasm forecasters (s6-cast, TinyCast) of sky-web, the
// zero-parameter references it exports, and Timestral-5M through onnxruntime-web (loaded from
// the CDN only when that model is selected).
import { withBase } from 'vitepress'
import { loadWasm, type SkyWeb } from './wasm'

export interface WasmForecaster {
  /** `[horizon × levels]` quantiles, row-major, in the units of the history. */
  forecast(history: Float32Array, horizon: number, flip: boolean): Float32Array
  levels(): Float32Array
  /** JSON: name, params, bytes, levels, pass_horizon, micros, exits_used, passes. */
  info(): string
  free(): void
}

export interface ForecastWeb extends SkyWeb {
  Forecaster: new (bytes: Uint8Array) => WasmForecaster
  seasonalNaive(history: Float32Array, season: number, horizon: number): Float32Array
  synthSeries(kind: string, seed: bigint, len: number): Float32Array
  dominantPeriod(history: Float32Array): number
  mase(history: Float32Array, season: number, target: Float32Array, point: Float32Array): number
}

export interface ForecastModelEntry {
  name: string
  kind: 'gguf' | 'onnx'
  file: string
  spec?: string
  params: number
  bytes: number
  license: string
  notice: string
  url: string
  credit: string
  gift_eval?: { nGMASE: number; nWQL: number; source: string }
}

export interface DemoSeries {
  id: string
  label: string
  credit: string
  freq: string
  season: number
  values: (number | null)[]
}

/** A forecast as the page draws it: levels and `[horizon][levels]` quantiles. */
export interface QuantileForecast {
  levels: number[]
  q: number[][]
  micros: number
  exits?: string
}

export function loadForecastWasm(): Promise<ForecastWeb> {
  return loadWasm() as Promise<ForecastWeb>
}

async function json<T>(path: string, fallback: T): Promise<T> {
  try {
    const r = await fetch(withBase(path))
    return r.ok ? ((await r.json()) as T) : fallback
  } catch {
    return fallback
  }
}

export async function loadForecastModels(): Promise<ForecastModelEntry[]> {
  const doc = await json<{ models?: ForecastModelEntry[] }>('/models/forecast.json', {})
  return (doc.models ?? []).filter((m) => m && m.name && m.file)
}

export async function loadDemoSeries(): Promise<DemoSeries[]> {
  const doc = await json<{ series?: DemoSeries[] }>('/forecast/demo-series.json', {})
  return doc.series ?? []
}

export async function loadWasmForecaster(mod: ForecastWeb, entry: ForecastModelEntry): Promise<WasmForecaster> {
  const res = await fetch(withBase(`/models/${entry.file}`))
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return new mod.Forecaster(new Uint8Array(await res.arrayBuffer()))
}

export function runWasmForecaster(f: WasmForecaster, history: number[], horizon: number): QuantileForecast {
  const out = f.forecast(Float32Array.from(history, (v) => (Number.isFinite(v) ? v : NaN)), horizon, false)
  const levels = Array.from(f.levels())
  const info = JSON.parse(f.info())
  const q: number[][] = []
  for (let h = 0; h < horizon; h++) q.push(Array.from(out.subarray(h * levels.length, (h + 1) * levels.length)))
  // Early exits are s6-cast's: TinyCast reports one "exit" per pass.
  const exits = String(info.name).startsWith('s6-cast') ? `${info.exits_used} exits used` : undefined
  return { levels, q, micros: info.micros, exits }
}

// --- Timestral-5M (onnxruntime-web) -------------------------------------------------------------

// Pinned: released 2026-09-14, the wasm backend runs the whole exported graph.
const ORT_VERSION = '1.30.0'
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`

interface OrtTensor {
  data: Float32Array
  dims: number[]
}
interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>
}
interface Ort {
  env: { wasm: { wasmPaths: string; numThreads: number } }
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown
  InferenceSession: { create(url: string, opts: unknown): Promise<OrtSession> }
}

let ortPromise: Promise<Ort> | null = null

function loadOrt(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = new Promise<Ort>((resolve, reject) => {
      const w = window as unknown as { ort?: Ort }
      if (w.ort) return resolve(w.ort)
      const s = document.createElement('script')
      s.src = `${ORT_BASE}ort.min.js`
      s.async = true
      s.onload = () => (w.ort ? resolve(w.ort) : reject(new Error('onnxruntime-web did not load')))
      s.onerror = () => reject(new Error('onnxruntime-web could not be fetched'))
      document.head.appendChild(s)
    }).then((ort) => {
      ort.env.wasm.wasmPaths = ORT_BASE
      // GitHub Pages is not cross-origin isolated: no wasm threads.
      ort.env.wasm.numThreads = 1
      return ort
    })
    ortPromise.catch(() => {
      ortPromise = null
    })
  }
  return ortPromise
}

export interface OnnxForecaster {
  session: OrtSession
  ort: Ort
  context: number
  horizon: number
  levels: number[]
}

export async function loadOnnxForecaster(entry: ForecastModelEntry): Promise<OnnxForecaster> {
  const ort = await loadOrt()
  const spec = await json<{ context_length?: number; horizon?: number; quantile_levels?: number[] }>(
    `/models/${entry.spec}`,
    {},
  )
  // 'basic' graph optimization: the session is ready in ~3 s instead of ~35 s with the default
  // full optimization (measured on an M5 in Chrome), for the same outputs and ~0.2 s per forecast.
  const session = await ort.InferenceSession.create(withBase(`/models/${entry.file}`), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
  })
  return {
    session,
    ort,
    context: spec.context_length ?? 2048,
    horizon: spec.horizon ?? 128,
    levels: spec.quantile_levels ?? [],
  }
}

/** The ONNX graph takes exactly `context` raw values, NaN-padded on the left, and returns
 * `[1, horizon, levels]` raw-unit quantiles (normalization is inside the graph). */
export async function runOnnxForecaster(f: OnnxForecaster, history: number[], horizon: number): Promise<QuantileForecast> {
  const ctx = new Float32Array(f.context).fill(NaN)
  const tail = history.slice(-f.context)
  ctx.set(Float32Array.from(tail, (v) => (Number.isFinite(v) ? v : NaN)), f.context - tail.length)
  const t0 = performance.now()
  const out = await f.session.run({ context: new f.ort.Tensor('float32', ctx, [1, f.context]) })
  const micros = (performance.now() - t0) * 1000
  const data = out.quantiles.data
  const nl = f.levels.length
  const q: number[][] = []
  for (let h = 0; h < Math.min(horizon, f.horizon); h++) q.push(Array.from(data.subarray(h * nl, (h + 1) * nl)))
  return { levels: f.levels, q, micros }
}

// --- helpers ---------------------------------------------------------------------------------

/** The index of the level closest to `p`. */
export function levelIndex(levels: number[], p: number): number {
  let best = 0
  for (let i = 1; i < levels.length; i++) if (Math.abs(levels[i] - p) < Math.abs(levels[best] - p)) best = i
  return best
}

/** Share of `target` steps inside the [10 %, 90 %] band. */
export function coverage(f: QuantileForecast, target: number[]): number | null {
  const lo = levelIndex(f.levels, 0.1)
  const hi = levelIndex(f.levels, 0.9)
  let n = 0
  let inside = 0
  f.q.forEach((row, h) => {
    const y = target[h]
    if (!Number.isFinite(y)) return
    n++
    if (y >= row[lo] && y <= row[hi]) inside++
  })
  return n ? inside / n : null
}
