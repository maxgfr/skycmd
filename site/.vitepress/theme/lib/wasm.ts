// Runtime loading of the sky-web wasm module (built into public/pkg by scripts/build-wasm.sh),
// the model manifest and the GGUF models. Everything is fetched lazily on the client.
import { withBase } from 'vitepress'

export interface WasmEngine {
  info(): string
  n_params(): number
  file_bytes(): number
  name(): string
  free(): void
}

export interface WasmSession {
  kind(): string
  step_human(key: string): string
  step_ai(engine: WasmEngine): string
  step_oracle(): string
  snapshot(): string
  reset(): void
  score(): number
  done(): boolean
  state_text(): string
  set_decide_every(n: number): void
  decide_every(): number
  set_veto(on: boolean): void
  timing(): string
  action_keys(): string
  questions(): string
  free(): void
}

export interface SkyWeb {
  Engine: new (bytes: Uint8Array) => WasmEngine
  GameSession: new (kind: string, seed: number) => WasmSession
  bench(engine: WasmEngine, n: number): string
  version(): string
}

export interface ModelEntry {
  name: string
  file: string
  params: number
  bytes: number
}

export interface OptionProb {
  key: string
  p: number
}

export interface StepReport {
  policy?: 'model' | 'oracle'
  decided?: boolean
  action?: string
  executed: string
  probs?: OptionProb[]
  secondary?: { id: string; type: string; value: number; max: number; probs: OptionProb[] } | null
  vetoed: boolean
  micros?: number
  state_text?: string
  event: string
  reward: number
  done: boolean
  score: number
}

export interface Question {
  id: string
  type: 'choice' | 'score' | 'noul'
  instructions: string
  keys: string[]
  options: string[]
}

let modPromise: Promise<SkyWeb> | null = null

/** Loads and instantiates the wasm module once. */
export function loadWasm(): Promise<SkyWeb> {
  if (!modPromise) {
    modPromise = (async () => {
      const url = withBase('/pkg/sky_web.js')
      const mod = await import(/* @vite-ignore */ url)
      await mod.default({ module_or_path: withBase('/pkg/sky_web_bg.wasm') })
      return mod as unknown as SkyWeb
    })()
    modPromise.catch(() => {
      modPromise = null
    })
  }
  return modPromise
}

let manifestPromise: Promise<ModelEntry[]> | null = null

/** The models listed in public/models/manifest.json (empty when absent). */
export function loadManifest(): Promise<ModelEntry[]> {
  if (!manifestPromise) {
    manifestPromise = fetch(withBase('/models/manifest.json'))
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => (Array.isArray(list) ? (list as ModelEntry[]) : []))
      .catch(() => [])
  }
  return manifestPromise
}

const engines = new Map<string, Promise<WasmEngine>>()

/** Downloads a GGUF model and loads it (cached per file). */
export function loadEngine(entry: ModelEntry): Promise<WasmEngine> {
  let p = engines.get(entry.file)
  if (!p) {
    p = (async () => {
      const mod = await loadWasm()
      const res = await fetch(withBase(`/models/${entry.file}`))
      if (!res.ok) throw new Error(`${entry.file}: HTTP ${res.status}`)
      const bytes = new Uint8Array(await res.arrayBuffer())
      return new mod.Engine(bytes)
    })()
    p.catch(() => engines.delete(entry.file))
    engines.set(entry.file, p)
  }
  return p
}

/** Small local storage helpers: storage may be unavailable (private mode, sandboxed iframes). */
export function readNumber(key: string): number {
  try {
    const v = Number(window.localStorage.getItem(key))
    return Number.isFinite(v) ? v : 0
  } catch {
    return 0
  }
}

export function writeNumber(key: string, value: number): void {
  try {
    window.localStorage.setItem(key, String(value))
  } catch {
    /* storage unavailable: keep the value in memory only */
  }
}

/** Rolling counter of events per second over a sliding window. */
export class RateMeter {
  private stamps: number[] = []
  constructor(private windowMs = 1000) {}
  tick(now: number, n = 1): void {
    for (let i = 0; i < n; i++) this.stamps.push(now)
  }
  rate(now: number): number {
    const from = now - this.windowMs
    while (this.stamps.length && this.stamps[0] < from) this.stamps.shift()
    return (this.stamps.length * 1000) / this.windowMs
  }
}

export function formatInt(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}

export function formatKB(bytes: number): string {
  return `${Math.round(bytes / 1024).toLocaleString('en-US')} KB`
}
