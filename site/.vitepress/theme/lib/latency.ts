// What the live latency readouts of a game decision say: the home page counters (LiveStats) and the
// Play page's HUD (PlayGame). No import, so that `node --test` loads it without VitePress.
//
// A decision is timed in WASM with `performance.now()`, whose step depends on the browser: Chrome
// counts in steps of 100 µs on a page that is not cross-origin isolated, Firefox and Safari in steps
// of 1,000 µs (`clockStepMicros` in wasm.ts measures it). s3-pico decides in about 250 µs, so under
// a 1,000 µs step most of its decisions measure 0 µs, and so does their median. A median of 0 is not
// a time, so the readout then says what the measurements do support:
//   - the mean of the measured decisions, labelled as a mean, once there are enough of them. A
//     decision does not start in step with the clock's ticks, so one that takes d µs measures one
//     step with a probability of about d / step and 0 otherwise: the measured times are right on
//     average, and the mean of many is an estimate of the true time (shown to two significant
//     digits);
//   - before that, or if every measured time is 0, the bound the median gives: at least half the
//     decisions measured under one clock step.

/** Measured decisions needed before the mean is shown in place of a median under the clock step. */
export const MIN_MEAN_DECISIONS = 30

/** The session's rolling timing (`GameSession.timing()`): over the last 120 decisions at most. */
export interface DecisionTiming {
  medianUs: number
  meanUs: number
  /** How many decisions the median and the mean are over. */
  n: number
}

export interface LatencyReadout {
  /** `median`: the median as measured; `mean`: the median is under the clock step, the mean is
   * shown; `bound`: the median is under the clock step, and only that bound is shown; `none`:
   * nothing measured yet. */
  kind: 'none' | 'median' | 'mean' | 'bound'
  /** µs per decision: `2,427`, `≈ 270`, `< 1,000`, `—`. */
  us: string
  /** Decisions per second that time allows: `412`, `≈ 3,700`, `> 1,000`, `—`. */
  capacity: string
  /** What the number is, for `mean` and `bound` (empty otherwise): `mean of 120 decisions: this
   * browser's clock counts in steps of 1,000 µs`. */
  detail: string
}

function int(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}

/** Two significant digits: 266.7 → 270, 3,750 → 3,800. */
function approx(n: number): string {
  if (!(n > 0)) return '0'
  const scale = 10 ** Math.max(0, Math.floor(Math.log10(n)) - 1)
  return int(Math.round(n / scale) * scale)
}

/** The readout of a decision time, given the step of the browser clock in µs (0 when unknown). */
export function decisionLatency(t: DecisionTiming, stepUs: number): LatencyReadout {
  if (!(t.n > 0)) return { kind: 'none', us: '—', capacity: '—', detail: '' }
  const coarse = stepUs > 1 && t.medianUs < stepUs
  if (!coarse && t.medianUs > 0) {
    return { kind: 'median', us: int(t.medianUs), capacity: int(1e6 / t.medianUs), detail: '' }
  }
  const clock = `this browser's clock counts in steps of ${int(stepUs)} µs`
  if (coarse && t.n >= MIN_MEAN_DECISIONS && t.meanUs > 0) {
    return {
      kind: 'mean',
      us: `≈ ${approx(t.meanUs)}`,
      capacity: `≈ ${approx(1e6 / t.meanUs)}`,
      detail: `mean of ${int(t.n)} decisions: ${clock}`,
    }
  }
  if (coarse) {
    return {
      kind: 'bound',
      us: `< ${int(stepUs)}`,
      capacity: `> ${int(1e6 / stepUs)}`,
      detail: `median under one clock step: ${clock}`,
    }
  }
  // a median of 0 with a clock whose step is unknown: no true statement to make. The step is
  // unknown only while the clock has not been seen to tick (a clock rounded to 100 ms, say, under
  // which every decision measures 0); the callers measure it again with each new decision.
  return { kind: 'none', us: '—', capacity: '—', detail: '' }
}
