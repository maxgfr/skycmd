// Drives one game session on a canvas: fixed 60 Hz physics (flappy, dodge) or timed grid steps
// (rescue), AI or human control, rendering with requestAnimationFrame, and a HUD state object.
//
// The wasm module runs on the main thread: a full AI frame (state text, tokenization, two
// forward passes, safety reflex, physics) measures ~0.7 ms with s1-pico in V8 (s2 models encode
// only the state per frame, once for both questions), well inside a 16 ms frame, so a Web Worker
// would add latency (one frame of delay for the snapshot) without making rendering smoother.
// Slower models are throttled with `set_decide_every` instead.
import type { DecisionInfo, Question, StepReport, SkyWeb, WasmEngine, WasmSession } from './wasm'
import { RateMeter, readNumber, writeNumber } from './wasm'
import {
  currentTheme,
  drawDodge,
  drawFlappy,
  drawRescue,
  fitCanvas,
  newFx,
  type DodgeView,
  type FlappyView,
  type FxState,
  type RescueView,
} from './render'

export type GameKind = 'flappy' | 'rescue' | 'dodge'
export type Mode = 'ai' | 'human'

export interface Hud {
  policy: string
  medianUs: number
  meanUs: number
  decisionsPerS: number
  decideEvery: number
  probs: { key: string; p: number }[]
  action: string
  executed: string
  secondary: StepReport['secondary']
  vetoCount: number
  vetoAt: number
  score: number
  best: number
  episodes: number
  done: boolean
  stateText: string
  /** How the model answered the last decision (null for the oracle). */
  info: DecisionInfo | null
  /** s2: mean state-encoder blocks used per question over the recent decisions. */
  meanLayers: number
  /** Share of the recent decisions whose question part came from the cache. */
  staticHitRate: number
}

export function newHud(): Hud {
  return {
    policy: '',
    medianUs: 0,
    meanUs: 0,
    decisionsPerS: 0,
    decideEvery: 1,
    probs: [],
    action: '',
    executed: '',
    secondary: null,
    vetoCount: 0,
    vetoAt: 0,
    score: 0,
    best: 0,
    episodes: 0,
    done: false,
    stateText: '',
    info: null,
    meanLayers: 0,
    staticHitRate: 0,
  }
}

/** Decisions kept for the rolling depth and cache statistics of the HUD. */
const DEPTH_WINDOW = 120

/** Per-decision time budget before the runner starts deciding every N frames. */
const BUDGET_US: Record<GameKind, number> = { flappy: 6000, dodge: 12000, rescue: 60000 }
/** Rescue: grid steps per second in AI mode. */
const RESCUE_AI_RATE = 5

export interface RunnerOptions {
  kind: GameKind
  seed?: number
  mode?: Mode
  /** Restart automatically this long after an episode ends (AI mode always, human mode never). */
  autoResetMs?: number
  showScore?: boolean
  /** localStorage key for the high score (omit to disable). */
  bestKey?: string
  onHud?: (hud: Hud) => void
}

export class GameRunner {
  readonly kind: GameKind
  mode: Mode
  hud: Hud = newHud()
  private session: WasmSession
  private engine: WasmEngine | null = null
  private fx: FxState = newFx()
  private raf = 0
  private last = 0
  private acc = 0
  private doneAt = 0
  private rate = new RateMeter(1000)
  private queue: string[] = []
  private heldKey = 'continue'
  private hudAt = 0
  private decisions = 0
  private running = false
  private layers: number[] = []
  private hits: number[] = []

  constructor(
    private mod: SkyWeb,
    private canvas: HTMLCanvasElement,
    private opts: RunnerOptions,
  ) {
    this.kind = opts.kind
    this.mode = opts.mode ?? 'ai'
    this.session = new mod.GameSession(opts.kind, opts.seed ?? Math.floor(Math.random() * 1e9))
    this.hud.best = opts.bestKey ? readNumber(opts.bestKey) : 0
    this.hud.policy = 'oracle (no model loaded)'
  }

  /** Uses a model, or the game's oracle when `engine` is null. */
  setEngine(engine: WasmEngine | null, label?: string) {
    this.engine = engine
    this.session.set_decide_every(1)
    this.decisions = 0
    this.layers = []
    this.hits = []
    this.hud.info = null
    this.hud.meanLayers = 0
    this.hud.staticHitRate = 0
    this.hud.policy = engine ? label ?? engine.name() : 'oracle (no model loaded)'
  }

  setMode(mode: Mode) {
    this.mode = mode
    this.queue = []
    this.heldKey = 'continue'
  }

  setVeto(on: boolean) {
    this.session.set_veto(on)
  }

  /** Human input: an action key (`thrust`, `north`, `over`, ...). Restarts a finished episode. */
  input(key: string) {
    if (this.session.done()) {
      if (this.mode === 'human' && performance.now() - this.doneAt > 400) this.reset()
      return
    }
    if (this.kind === 'dodge') this.heldKey = key
    else this.queue.push(key)
  }

  reset() {
    this.session.reset()
    this.doneAt = 0
    this.acc = 0
    this.queue = []
    this.heldKey = 'continue'
    this.fx.tween = 1
    this.hud.done = false
  }

  start() {
    if (this.running) return
    this.running = true
    this.last = performance.now()
    const loop = (now: number) => {
      if (!this.running) return
      this.frame(now)
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  stop() {
    this.running = false
    cancelAnimationFrame(this.raf)
  }

  /** The game's System One questions (canonical phrasing). */
  questions(): Question[] {
    return JSON.parse(this.session.questions())
  }

  get isRunning() {
    return this.running
  }

  destroy() {
    this.stop()
    this.session.free()
  }

  /** Draws the current state once (used when paused). */
  render() {
    const { ctx, w, h } = fitCanvas(this.canvas)
    const snap = JSON.parse(this.session.snapshot())
    const theme = currentTheme()
    if (this.kind === 'flappy') drawFlappy(ctx, w, h, snap.view as FlappyView, this.fx, theme, { showScore: this.opts.showScore })
    else if (this.kind === 'rescue') drawRescue(ctx, w, h, snap.view as RescueView, this.fx, theme)
    else drawDodge(ctx, w, h, snap.view as DodgeView, this.fx, theme)
  }

  private stepAi(now: number) {
    const raw = this.engine ? this.session.step_ai(this.engine) : this.session.step_oracle()
    const r = JSON.parse(raw) as StepReport
    if (r.decided) {
      this.rate.tick(now)
      this.decisions++
      this.hud.probs = r.probs ?? []
      this.hud.action = r.action ?? ''
      this.hud.secondary = r.secondary ?? null
      this.hud.stateText = r.state_text ?? ''
      this.hud.info = r.info ?? null
      if (r.info) this.trackDepth(r.info)
      if (this.engine && this.decisions % 30 === 0) this.retune()
    }
    this.afterStep(r, now)
  }

  private stepHuman(key: string, now: number) {
    const r = JSON.parse(this.session.step_human(key)) as StepReport
    if (key) this.hud.action = key
    this.afterStep(r, now)
  }

  private afterStep(r: StepReport, now: number) {
    if (r.executed && r.executed !== 'none') this.hud.executed = r.executed
    if (r.vetoed) {
      this.hud.vetoCount++
      this.hud.vetoAt = now
    }
    if (r.done && !this.doneAt) {
      this.doneAt = now
      this.fx.crashAt = this.fx.t
      this.hud.episodes++
      this.hud.done = true
      const score = this.session.score()
      if (score > this.hud.best) {
        this.hud.best = score
        if (this.opts.bestKey) writeNumber(this.opts.bestKey, score)
      }
    }
  }

  private trackDepth(info: DecisionInfo) {
    const push = (a: number[], v: number) => {
      a.push(v)
      if (a.length > DEPTH_WINDOW) a.shift()
    }
    if (info.mean_layers != null) push(this.layers, info.mean_layers)
    push(this.hits, info.static_hit ? 1 : 0)
    const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0)
    this.hud.meanLayers = mean(this.layers)
    this.hud.staticHitRate = mean(this.hits)
  }

  /** Decide every N frames when the rolling median exceeds the frame budget. */
  private retune() {
    const t = JSON.parse(this.session.timing())
    const n = Math.min(8, Math.max(1, Math.ceil(t.median_us / BUDGET_US[this.kind])))
    if (n !== this.session.decide_every()) this.session.set_decide_every(n)
  }

  private frame(now: number) {
    const dt = Math.min(0.1, (now - this.last) / 1000)
    this.last = now
    this.fx.t += dt
    this.fx.thrustGlow *= Math.exp(-dt * 7)
    this.acc += dt

    if (!this.session.done()) {
      if (this.kind === 'rescue') {
        if (this.mode === 'human') {
          const key = this.queue.shift()
          if (key) {
            this.stepHuman(key, now)
            this.fx.tween = 0
          }
          this.acc = 0
        } else if (this.acc >= 1 / RESCUE_AI_RATE) {
          this.acc = 0
          this.stepAi(now)
          this.fx.tween = 0
        }
        this.fx.tween = Math.min(1, this.fx.tween + dt / 0.16)
      } else {
        let n = Math.floor(this.acc * 60)
        this.acc -= n / 60
        if (n > 4) n = 4 // drop the backlog after a stall instead of fast-forwarding
        for (let i = 0; i < n && !this.session.done(); i++) {
          if (this.mode === 'ai') this.stepAi(now)
          else if (this.kind === 'flappy') this.stepHuman(this.queue.shift() ?? '', now)
          else this.stepHuman(this.heldKey, now)
        }
      }
    } else {
      this.acc = 0
      const auto = this.opts.autoResetMs ?? 900
      if (this.mode === 'ai' && now - this.doneAt > auto) this.reset()
    }

    this.render()

    if (now - this.hudAt > 66) {
      this.hudAt = now
      const t = JSON.parse(this.session.timing())
      this.hud.medianUs = t.median_us
      this.hud.meanUs = t.mean_us
      this.hud.decisionsPerS = this.rate.rate(now)
      this.hud.decideEvery = this.session.decide_every()
      this.hud.score = this.session.score()
      this.opts.onHud?.(this.hud)
    }
  }
}
