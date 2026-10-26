// Runtime loading of the sky-web wasm module (built into public/pkg by scripts/build-wasm.sh),
// the model manifest and the GGUF models. Everything is fetched lazily on the client.
import { withBase } from 'vitepress'

export interface WasmEngine {
  info(): string
  n_params(): number
  file_bytes(): number
  name(): string
  /** `s1` (ModernBERT + decision head) or `s2` (static cache + early exits). */
  family(): string
  /** s2: early-exit threshold override (undefined = calibrated, 1 = full depth). */
  setExitThreshold(tau?: number): void
  /** Generic decision over rendered option texts (JSON array); returns the JSON answer
   * `{probs, exit, n_exits, layers_used, static_hit, micros}`. */
  decide(qtype: string, instructions: string, optionsJson: string, state: string): string
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
  benchGame(engine: WasmEngine, game: string, n: number): string
  version(): string
}

export type Family = 's1' | 's2'

export interface ModelEntry {
  name: string
  file: string
  params: number
  bytes: number
  /** `general.architecture`: `modern-bert` (s1) or `skycmd-s2` (s2). */
  arch?: string
  /** `f32`, `f16`, `q8_0` or `ternary`. */
  weight_format?: string
  /** `generic`: a decision model for free-text questions (s4, the Live page), not for the games. */
  task?: string
  /** The games the model was trained and measured on (`sky-arena`); absent = every game. */
  games?: string[]
}

/** Whether a game model knows `game` (a model never trained on a game plays it at random). */
export function playsGame(entry: Pick<ModelEntry, 'games'>, game: string): boolean {
  return !entry.games || entry.games.includes(game)
}

/** The models that play the games (every manifest entry but the generic ones). */
export function gameModels(list: ModelEntry[]): ModelEntry[] {
  return list.filter((m) => m.task !== 'generic')
}

/** Model family of a manifest entry (from `arch`, else from the name). */
export function familyOf(entry: Pick<ModelEntry, 'name' | 'arch'>): Family {
  if (entry.arch) return entry.arch === 'skycmd-s2' ? 's2' : 's1'
  return entry.name.startsWith('s2') ? 's2' : 's1'
}

export const FAMILY_LABEL: Record<Family, string> = {
  s1: 's1 · ModernBERT encoder + decision head',
  s2: 's2 · cached question, early exit, native numbers',
}

/** How the model answered a decision frame (see `sky_web::session::DecisionInfo`). */
export interface DecisionInfo {
  family: Family
  exit: number | null
  n_exits: number
  layers_used: number | null
  mean_layers: number | null
  max_layers: number | null
  static_hit: boolean
  state_reused: boolean
}

export interface OptionProb {
  key: string
  p: number
}

export interface Secondary {
  id: string
  type: string
  value: number
  max: number
  probs: OptionProb[]
}

export interface StepReport {
  policy?: 'model' | 'oracle'
  decided?: boolean
  action?: string
  executed: string
  probs?: OptionProb[]
  secondary?: Secondary | null
  others?: Secondary[]
  info?: DecisionInfo | null
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
      .then((list) => (Array.isArray(list) ? (list as ModelEntry[]).filter((m) => m && m.file && m.name) : []))
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
