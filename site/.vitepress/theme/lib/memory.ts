// The pure logic of the memory demo (MemoryDemo.vue, the Memory page and the Space of
// s6-sleep3-nano): the seeds the browser plays, the tally of the asks by delay bucket, a 95%
// interval for a share, the words of a `recall` frame, and the published test numbers the
// browser's tally is shown next to. No DOM, no wasm: tested by tests/memory.test.ts.
//
// The episodes themselves come from the wasm `RecallProbe` (crates/sky-web/src/recall.rs), which
// plays the evaluation's generator (`sky_games::make_game("recall", seed)`) and lets the game
// score each ask.

/** The delay buckets of `recall` (frames between the show of a door and its ask, inclusive),
 * `sky_games::recall::BUCKETS`. */
export const BUCKETS: readonly [number, number][] = [
  [1, 4],
  [5, 8],
  [9, 16],
  [17, 32],
]

export const BUCKET_LABELS = BUCKETS.map(([lo, hi]) => `${lo}-${hi}`)

/** The answers of the question, in the order of the model's probabilities
 * (`sky_games::recall::QUESTIONS`). */
export const KEYS = ['wait', 'red', 'green', 'blue', 'yellow'] as const
export type Key = (typeof KEYS)[number]

/** The first seed the browser plays: the validation split of `sky_games::gen::split_for_seed`
 * (≡ 18 mod 20), which no training sequence comes from, far from every seed the experiment read
 * (dev2 from 2,000,000,000 and the test from 5,000,000,000, both ≡ 19 mod 20). */
export const BROWSER_FROM = 7_000_000_018

/** Episodes of the batch button. */
export const BATCH_EPISODES = 200

/** The `n` seeds ≡ 18 (mod 20) from `from` upward. */
export function validationSeeds(n: number, from = BROWSER_FROM): number[] {
  const first = from + ((((18 - (from % 20)) % 20) + 20) % 20)
  return Array.from({ length: Math.max(0, n) }, (_, k) => first + 20 * k)
}

/** One scored ask, as `RecallProbe.runEpisode` returns it. */
export interface Ask {
  frame: number
  delay: number
  bucket: number
  correct: boolean
}

/** Right answers and asks, per delay bucket, and the decision times. */
export class Tally {
  right = [0, 0, 0, 0]
  asked = [0, 0, 0, 0]
  episodes = 0
  decisions = 0
  micros = 0
  private perDecision: number[] = []

  /** Adds one episode: its asks, its number of decisions and their summed time (µs). */
  addEpisode(asks: Ask[], decisions: number, micros: number): void {
    for (const a of asks) {
      if (a.bucket < 0 || a.bucket >= BUCKETS.length) throw new Error(`bucket ${a.bucket}`)
      this.asked[a.bucket] += 1
      if (a.correct) this.right[a.bucket] += 1
    }
    this.episodes += 1
    this.decisions += decisions
    this.micros += micros
    if (decisions > 0) this.perDecision.push(micros / decisions)
  }

  get totalAsked(): number {
    return this.asked.reduce((x, y) => x + y, 0)
  }

  get totalRight(): number {
    return this.right.reduce((x, y) => x + y, 0)
  }

  /** Accuracy on the asks of bucket `b`, or over every ask when `b` is omitted; null with no ask. */
  accuracy(b?: number): number | null {
    const [r, n] = b === undefined ? [this.totalRight, this.totalAsked] : [this.right[b], this.asked[b]]
    return n ? r / n : null
  }

  /** The 95% interval of that accuracy (Wilson). */
  interval(b?: number): [number, number] | null {
    const [r, n] = b === undefined ? [this.totalRight, this.totalAsked] : [this.right[b], this.asked[b]]
    return wilson(r, n)
  }

  /** The median over episodes of the mean time per decision (µs); null before any episode. */
  get medianMicros(): number | null {
    if (!this.perDecision.length) return null
    const s = [...this.perDecision].sort((x, y) => x - y)
    return s[Math.floor((s.length - 1) / 2)]
  }
}

/** The Wilson 95% interval of `right` successes in `n` trials; null when `n` is 0. */
export function wilson(right: number, n: number, z = 1.96): [number, number] | null {
  if (!n) return null
  const p = right / n
  const z2 = z * z
  const centre = (p + z2 / (2 * n)) / (1 + z2 / n)
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / (1 + z2 / n)
  return [Math.max(0, centre - half), Math.min(1, centre + half)]
}

/** A share as the browser's table writes it: three decimals. */
export function formatShare(x: number | null): string {
  return x === null ? '—' : x.toFixed(3)
}

/** An interval as `0.512–0.611`. */
export function formatInterval(iv: [number, number] | null): string {
  return iv ? `${iv[0].toFixed(3)}–${iv[1].toFixed(3)}` : ''
}

/** What a frame of the episode shows (`RecallProbe.start`'s `frames`). */
export interface Frame {
  frame: number
  state: string
  kind: 'show' | 'ask' | 'idle'
  door?: string
  colour?: string
  answer: Key
  delay?: number
  bucket?: number
}

/** The distractor number of a state text (`... noise <n>`), or null. */
export function noiseOf(state: string): number | null {
  const m = /\bnoise (\d+)\b/.exec(state)
  return m ? Number(m[1]) : null
}

/** One line for a frame: what appears on it. */
export function frameLine(f: Frame): string {
  if (f.kind === 'show') return `door ${f.door} is given ${f.colour}`
  if (f.kind === 'ask') return `which colour was door ${f.door} given?`
  const n = noiseOf(f.state)
  return n === null ? 'nothing to remember' : `nothing to remember (a distractor number, ${n})`
}

/** For an ask: when its door was shown, and its bucket. */
export function askLine(f: Frame): string {
  if (f.kind !== 'ask' || f.delay === undefined || f.bucket === undefined) return ''
  const frames = f.delay === 1 ? 'frame' : 'frames'
  return `door ${f.door} was shown ${f.delay} ${frames} earlier (delay bucket ${BUCKET_LABELS[f.bucket]})`
}

/** The score line of the episode being played: its total once, at the end ("4 of 8 questions
 * right"), and before it how many questions were asked so far and how many the episode holds. */
export function episodeLine(right: number, asked: number, planned: number, finished: boolean): string {
  const q = (n: number) => (n === 1 ? 'question' : 'questions')
  if (!finished && asked === 0) return `This episode asks ${planned} ${q(planned)}`
  return finished
    ? `This episode: ${right} of ${asked} ${q(asked)} right`
    : `This episode so far: ${right} of ${asked} ${q(asked)} right (it asks ${planned} in all)`
}

/** The index of the largest probability (the first on ties, as the probe plays it). */
export function argmax(p: number[]): number {
  let b = 0
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i
  return b
}

/** A reading of the test, written exactly as its result file writes it (strings, so that no
 * rounding is added): overall and per delay bucket. */
export interface Reading {
  overall: string
  buckets: [string, string, string, string]
}

/** The test reading of the published files, seed 0, from `runs/s6-sleep3/test/sleep-s0/recall.json`
 * and `runs/s6-sleep3/test/control-s0/recall.json` (`score` and `buckets`: 1,000 episodes from seed
 * 5,000,000,000, ≡ 19 mod 20, read by `sky-train s2-probe` on the checkpoint each file was exported
 * from). The memory zeroed was not read on the test seeds. */
export const TEST_SEED0: { sleep: Record<Passes, Reading>; control: Reading } = {
  sleep: {
    1: { overall: '0.4765', buckets: ['0.591', '0.5375', '0.438', '0.3395'] },
    2: { overall: '0.484875', buckets: ['0.6025', '0.569', '0.459', '0.309'] },
    4: { overall: '0.486875', buckets: ['0.6245', '0.589', '0.4535', '0.2805'] },
  },
  control: { overall: '0.245125', buckets: ['0.241', '0.2405', '0.2515', '0.2475'] },
}

/** The same reading pooled over the three training seeds, from `runs/s6-sleep3/REPORT.md` (the
 * numbers the gate reads: H0 at N = 1, H1 at N = 4 against N = 1). */
export const TEST_POOLED: { sleep: Record<Passes, Reading>; control: Reading } = {
  sleep: {
    1: { overall: '0.4892', buckets: ['0.6147', '0.5590', '0.4530', '0.3302'] },
    2: { overall: '0.4953', buckets: ['0.6213', '0.5823', '0.4693', '0.3083'] },
    4: { overall: '0.4922', buckets: ['0.6290', '0.5910', '0.4620', '0.2867'] },
  },
  control: { overall: '0.2440', buckets: ['0.2420', '0.2398', '0.2518', '0.2423'] },
}

/** The numbers of sleep passes the demo offers (the ones the test read). */
export const PASSES = [1, 2, 4] as const
export type Passes = (typeof PASSES)[number]

/** The arms of the batch, in the order of the table. */
export type Arm = 'on' | 'zeroed' | 'control'

export const ARM_LABEL: Record<Arm, string> = {
  on: 'memory on',
  zeroed: 'memory zeroed',
  control: 'control file (no memory)',
}
