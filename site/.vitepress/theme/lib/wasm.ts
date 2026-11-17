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
  /** s6-ear: the same decision about 16 kHz mono PCM (the last 10.24 s count). */
  decideAudio(qtype: string, instructions: string, optionsJson: string, pcm: Float32Array): string
  /** True for an s6-ear (audio state) model. */
  isAudio(): boolean
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

export type Family = 's1' | 's2' | 'iso'

export interface ModelEntry {
  name: string
  file: string
  params: number
  bytes: number
  /** `general.architecture`: `modern-bert` (s1) or `skycmd-s2` (s2). */
  arch?: string
  /** `f32`, `f16`, `q8_0` or `ternary`. */
  weight_format?: string
  /** `generic`: a decision model for free-text questions (s4, s5: the Live page), not for the games.
   * `general`: both, the games and free-text questions (s6-nano, s6-micro, s6-micro-v1). `secrets`: the four
   * questions of the API-key jev only (s6-secrets-pico, its own demo). `secrets-verify`: one rule
   * candidate judged in its context (s6-secrets-verify-pico, run by `SecretVerifier` with its word
   * list `dictionary`: the Secrets verifier page). `route`: a tool picked from a run-time list, with
   * a BM25 line in the state (s6-route-nano; its Space runs it through s6-route's router). `router`: two
   * files fused by `ToolRouter` (s6-route: the s6-route-nano file and a static embedding, the
   * Router page); such an entry has `files` and no `file`, so `loadManifest` leaves it out and only
   * `loadRouters` lists it. `ear`: an audio state
   * (s6-ear, the Listen page). `forecast`: a forecaster, not a decision engine (s6-cast, the
   * Forecast page; `forecast.json` lists it there). Absent: the games only. */
  task?: string
  /** Published at full depth: the file's exit thresholds are 1, so early exit is off. */
  full_depth?: boolean
  /** The pre-registered gate of the model, for the s6 models: `passed`, or `failed`, in which case
   * its file is published as a result, not for use (owner's decision of 15/11/2026). Every list
   * says so, and no picker selects a failed model by default. */
  gate?: 'passed' | 'failed'
  /** The gate and the numbers, in one line (shown next to a failed model). */
  gate_note?: string
  /** Where the result is written (a section of BENCH.md). */
  gate_url?: string
  /** s6-ear: the System 2 hand-off thresholds per question type (conformal, `ear-gate.py crc`). */
  route?: Record<string, number>
  /** The games the model was trained and measured on (`sky-arena`); absent = every game. */
  games?: string[]
  /** `secrets-verify`: the word list the verifier reads (`dictionary.txt` of the engine), a file of
   * public/models/ next to the model. */
  dictionary?: string
}

/** Whether a game model knows `game` (a model never trained on a game plays it at random). */
export function playsGame(entry: Pick<ModelEntry, 'games'>, game: string): boolean {
  return !entry.games || entry.games.includes(game)
}

/** Whether the model failed its pre-registered gate (its file is published as a result only). */
export function isFailed(entry: Pick<ModelEntry, 'gate'> | null | undefined): boolean {
  return entry?.gate === 'failed'
}

/** ` · failed its gate` for a failed model, else empty: appended to every model label. */
export function gateSuffix(entry: Pick<ModelEntry, 'gate'> | null | undefined): string {
  return isFailed(entry) ? ' · failed its gate' : ''
}

/** The list in its order, failed models last (a stable sort). */
export function failedLast<T extends Pick<ModelEntry, 'gate'>>(list: T[]): T[] {
  return [...list].sort((a, b) => Number(isFailed(a)) - Number(isFailed(b)))
}

/** The model a picker selects by default: `prefer` by name if listed, else the first model that
 * did not fail its gate, else the first one. */
export function defaultModel<T extends Pick<ModelEntry, 'name' | 'gate'>>(list: T[], prefer?: string): T | null {
  return list.find((m) => m.name === prefer) ?? list.find((m) => !isFailed(m)) ?? list[0] ?? null
}

/** The models that play the games: entries without a task, and the `general` ones. */
export function gameModels(list: ModelEntry[]): ModelEntry[] {
  return failedLast(list.filter((m) => !m.task || m.task === 'general'))
}

/** The models that answer free-text questions (the Live page): `generic` and `general` entries. */
export function questionModels(list: ModelEntry[]): ModelEntry[] {
  return failedLast(list.filter((m) => m.task === 'generic' || m.task === 'general'))
}

/** Model family of a manifest entry (from `arch`, else from the name). */
export function familyOf(entry: Pick<ModelEntry, 'name' | 'arch'>): Family {
  if (entry.arch === 'skycmd-s2') return 's2'
  if (entry.arch === 'skycmd-iso') return 'iso'
  if (entry.arch) return 's1'
  return entry.name.startsWith('s2') ? 's2' : 's1'
}

export const FAMILY_LABEL: Record<Family, string> = {
  s1: 's1 · ModernBERT encoder + decision head',
  s2: 's2 · cached question, early exit, native numbers',
  iso: 'iso · options read by definition, state encoded once',
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

/** The build of the module in use: `relaxed` (SIMD128 + relaxed SIMD, `f32x4.relaxed_madd`) or
 * `simd128` (the baseline every browser with WebAssembly SIMD runs). Both come from the same
 * sources and share one JS glue (scripts/build-wasm.sh). */
export type WasmBuild = 'relaxed' | 'simd128'

export const WASM_BUILD_LABEL: Record<WasmBuild, string> = {
  relaxed: 'relaxed SIMD',
  simd128: 'SIMD128',
}

let build: WasmBuild | null = null

/** The build `loadWasm` instantiated (null before it resolves). */
export function wasmBuild(): WasmBuild | null {
  return build
}

/** `engine: relaxed SIMD` or `engine: SIMD128`, for the timing lines; empty before the load. */
export function engineLabel(): string {
  return build ? `engine: ${WASM_BUILD_LABEL[build]}` : ''
}

// A function `() -> v128` returning `i8x16.relaxed_swizzle` of two `i8x16.splat`: it validates only
// where relaxed SIMD is supported.
const RELAXED_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15,
  253, 128, 2, 11,
])

/** Whether this engine validates relaxed-SIMD instructions. */
export function relaxedSimdSupported(): boolean {
  try {
    return typeof WebAssembly === 'object' && WebAssembly.validate(RELAXED_PROBE)
  } catch {
    return false
  }
}

let modPromise: Promise<SkyWeb> | null = null

/** Loads and instantiates the wasm module once: the relaxed-SIMD build where the browser supports
 * it, else (or if that file cannot be fetched or instantiated: an older deploy, a missing file)
 * the SIMD128 build. */
export function loadWasm(): Promise<SkyWeb> {
  if (!modPromise) {
    modPromise = (async () => {
      const url = withBase('/pkg/sky_web.js')
      const mod = await import(/* @vite-ignore */ url)
      if (relaxedSimdSupported()) {
        try {
          await mod.default({ module_or_path: withBase('/pkg/sky_web_relaxed_bg.wasm') })
          build = 'relaxed'
          return mod as unknown as SkyWeb
        } catch (e) {
          console.warn('sky-web: the relaxed-SIMD module did not load, using the SIMD128 one.', e)
        }
      }
      await mod.default({ module_or_path: withBase('/pkg/sky_web_bg.wasm') })
      build = 'simd128'
      return mod as unknown as SkyWeb
    })()
    modPromise.catch(() => {
      modPromise = null
    })
  }
  return modPromise
}

let manifestPromise: Promise<ModelEntry[]> | null = null

/** The single-file models listed in public/models/manifest.json (empty when absent). A two-file
 * router entry (`files`, no `file`) is left out: see `loadRouters`. */
export function loadManifest(): Promise<ModelEntry[]> {
  if (!manifestPromise) {
    manifestPromise = fetch(withBase('/models/manifest.json'))
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => (Array.isArray(list) ? (list as ModelEntry[]).filter((m) => m && m.file && m.name) : []))
      .catch(() => [])
  }
  return manifestPromise
}

/** One file of a two-file entry (`task: router`). */
export interface RouterFile {
  /** `model` (the s2 router file) or `embedding` (the `skycmd-embed` file). */
  role: 'model' | 'embedding'
  file: string
  bytes: number
  sha256?: string
}

/** A manifest entry run by `ToolRouter` (s6-route): no `file`, two `files`. */
export interface RouterEntry {
  name: string
  task: 'router'
  bytes: number
  files: RouterFile[]
  full_depth?: boolean
  gate?: 'passed' | 'failed'
  gate_note?: string
  gate_url?: string
}

/** The fused tool router of sky-web (`crates/sky-web/src/router.rs`). */
export interface WasmToolRouter {
  /** `tools_json`: rendered `name: description` options, without *none*. Returns the JSON answer
   * `{probs, index, answer, none, ask, decision, top, tau_ask3, model, embedding, fast, exit,
   * layers_used, static_hit, tools_hit, micros, micros_model, micros_embedding}`. */
  route(request: string, toolsJson: string, fast: boolean): string
  tauAsk(): number
  info(): string
  free(): void
}

let routersPromise: Promise<RouterEntry[]> | null = null

/** The two-file router entries of public/models/manifest.json (`task: router`, a `model` and an
 * `embedding` file). The pickers of the other pages read `loadManifest`, which skips them. */
export function loadRouters(): Promise<RouterEntry[]> {
  if (!routersPromise) {
    routersPromise = fetch(withBase('/models/manifest.json'))
      .then((r) => (r.ok ? r.json() : []))
      .then((list) =>
        Array.isArray(list)
          ? (list as RouterEntry[]).filter(
              (m) =>
                m &&
                m.name &&
                m.task === 'router' &&
                Array.isArray(m.files) &&
                m.files.some((f) => f.role === 'model') &&
                m.files.some((f) => f.role === 'embedding'),
            )
          : [],
      )
      .catch(() => [])
  }
  return routersPromise
}

async function fetchBytes(file: string): Promise<Uint8Array> {
  const res = await fetch(withBase(`/models/${file}`))
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`)
  return new Uint8Array(await res.arrayBuffer())
}

const routers = new Map<string, Promise<WasmToolRouter>>()

/** Downloads the two files of a router entry and loads them in one `ToolRouter` (cached). */
export function loadRouter(entry: RouterEntry): Promise<WasmToolRouter> {
  let p = routers.get(entry.name)
  if (!p) {
    p = (async () => {
      const mod = (await loadWasm()) as unknown as {
        ToolRouter: new (model: Uint8Array, embedding: Uint8Array) => WasmToolRouter
      }
      const of = (role: RouterFile['role']) => entry.files.find((f) => f.role === role)!.file
      const [model, embedding] = await Promise.all([fetchBytes(of('model')), fetchBytes(of('embedding'))])
      return new mod.ToolRouter(model, embedding)
    })()
    p.catch(() => routers.delete(entry.name))
    routers.set(entry.name, p)
  }
  return p
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

/** The end-of-turn detector of sky-web (crates/sky-infer/src/wasm.rs, `TurnDetector`): the
 * s6-ear2-nano model over a stream of 16 kHz mono samples, pushed as it arrives. */
export interface WasmTurnDetector {
  /** Appends 16 kHz samples to the stream. */
  push(pcm: Float32Array): void
  /** Forgets the stream (a new turn). */
  reset(): void
  /** Hops kept by the stream (at most the window's 400). */
  hopCount(): number
  /** The calibrated probability that the turn is finished, from the stream's last 4 s. */
  probEnd(): number
  /** τ₀.₅ (0.5 on the calibrated probability). */
  tau50(): number
  /** τ₈₅: the highest validation threshold with an END recall of at least 0.85. */
  tau85(): number
  /** `{name, params, file_bytes, window_hops, hops_per_position, temperature, tau50, tau85}`. */
  info(): string
  free(): void
}

/** The manifest entry of the end-of-turn detector (s6-ear2-nano); null when not published.
 * The Listen demo of v1 picks `s6-ear-nano` by name, so this entry stays invisible to it. */
export async function loadTurnEntry(): Promise<ModelEntry | null> {
  const list = await loadManifest()
  return list.find((m) => m.task === 'ear' && m.name === 's6-ear2-nano') ?? null
}

const turnDetectors = new Map<string, Promise<WasmTurnDetector>>()

/** Downloads the GGUF of an end-of-turn entry and loads it in one `TurnDetector` (cached). */
export function loadTurnDetector(entry: ModelEntry): Promise<WasmTurnDetector> {
  let p = turnDetectors.get(entry.name)
  if (!p) {
    p = (async () => {
      const mod = (await loadWasm()) as unknown as {
        TurnDetector: new (gguf: Uint8Array) => WasmTurnDetector
      }
      return new mod.TurnDetector(await fetchBytes(entry.file))
    })()
    p.catch(() => turnDetectors.delete(entry.name))
    turnDetectors.set(entry.name, p)
  }
  return p
}

/** The deterministic speech-like test signal of `sky_infer::audio::synth_speech` (not speech:
 * it only checks that a pipeline and the engine run). */
export async function turnSynthSpeech(secs: number, seed: bigint, amp: number): Promise<Float32Array> {
  const mod = (await loadWasm()) as unknown as {
    turnSynthSpeech(secs: number, seed: bigint, amp: number): Float32Array
  }
  return mod.turnSynthSpeech(secs, seed, amp)
}

/** The secrets verifier of sky-web (crates/sky-web/src/verify.rs, `SecretVerifier`):
 * s6-secrets-verify-pico judging candidate records (`sg-cand-1`) at full depth. */
export interface WasmSecretVerifier {
  /** One candidate (JSON) -> `{p_real, state_tokens, truncated, micros}` as JSON. */
  judge(candidateJson: string): string
  /** A JSON array of candidates -> a JSON array of answers, in the same order. */
  judgeMany(candidatesJson: string): string
  /** `{tau_f1, tau_f05, tau_f2}` as JSON: a candidate is accepted when `p_real >= tau`. */
  thresholds(): string
  /** The state text the model reads for a candidate (JSON). */
  state(candidateJson: string): string
  /** `{name, n_params, file_bytes, window, words}` as JSON. */
  info(): string
  free(): void
}

/** The manifest entry of the secrets verifier (`task: secrets-verify`, with its word list); null
 * when not published. */
export async function loadVerifierEntry(): Promise<(ModelEntry & { dictionary: string }) | null> {
  const list = await loadManifest()
  const e = list.find((m) => m.task === 'secrets-verify' && m.dictionary)
  return e ? (e as ModelEntry & { dictionary: string }) : null
}

const verifiers = new Map<string, Promise<WasmSecretVerifier>>()

/** Downloads the GGUF and the word list of a verifier entry and loads them in one
 * `SecretVerifier` (cached). */
export function loadSecretVerifier(entry: ModelEntry & { dictionary: string }): Promise<WasmSecretVerifier> {
  let p = verifiers.get(entry.name)
  if (!p) {
    p = (async () => {
      const mod = (await loadWasm()) as unknown as {
        SecretVerifier: new (gguf: Uint8Array, words: Uint8Array) => WasmSecretVerifier
      }
      const [gguf, words] = await Promise.all([fetchBytes(entry.file), fetchBytes(entry.dictionary)])
      return new mod.SecretVerifier(gguf, words)
    })()
    p.catch(() => verifiers.delete(entry.name))
    verifiers.set(entry.name, p)
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

let clockStep: number | null = null

/** The step of the browser clock in µs (`performance.now`). Browsers coarsen it: Chrome counts in
 * 100 µs steps on a page that is not cross-origin isolated, such as a GitHub Pages site, so a
 * single decision faster than that reads as 0 or 100 µs. */
export function clockStepMicros(): number {
  if (clockStep === null) {
    let step = Infinity
    let prev = performance.now()
    for (let i = 0; i < 20000 && step > 1; i++) {
      const t = performance.now()
      if (t > prev) step = Math.min(step, t - prev)
      prev = t
    }
    clockStep = Number.isFinite(step) ? Math.round(step * 1000) : 0
  }
  return clockStep
}

/** A duration given in µs, readable: `412 µs`, `1.42 ms`, `22.1 ms`. A duration under the step of
 * the browser clock reads `< 100 µs` (or the step there), never `0 µs`. */
export function formatMicros(us: number): string {
  const step = typeof performance === 'undefined' ? 0 : clockStepMicros()
  if (step > 1 && us < step) return `< ${formatMicros(step)}`
  if (us < 1000) return `${Math.round(us)} µs`
  const ms = us / 1000
  return `${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms`
}

/** Decision times of one demo: the first call apart (it pays the module start-up, compilation and
 * the first encodings), and the median of the warm calls after it (the last 50). */
export class DecisionTimes {
  first: number | null = null
  last: number | null = null
  private warm: number[] = []
  add(us: number): void {
    this.last = us
    if (this.first === null) this.first = us
    else {
      this.warm.push(us)
      if (this.warm.length > 50) this.warm.shift()
    }
  }
  get warmCount(): number {
    return this.warm.length
  }
  get warmMedian(): number | null {
    if (!this.warm.length) return null
    const s = [...this.warm].sort((x, y) => x - y)
    return s[Math.floor(s.length / 2)]
  }
  reset(): void {
    this.first = this.last = null
    this.warm = []
  }
}
