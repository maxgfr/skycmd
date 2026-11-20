// What the live latency readouts of a game decision say: the home page counters (LiveStats) and the
// Play page's HUD (PlayGame). No import, so that `node --test` loads it without VitePress.
//
// A decision is timed in WASM with `performance.now()`, whose step depends on the browser: Chrome
// counts in steps of 100 µs on a page that is not cross-origin isolated, Firefox and Safari in steps
// of 1,000 µs (`clockStepMicros` in wasm.ts measures it). Every measured time is a whole number of
// steps. s3-pico decides in about 250 µs, so under a 1,000 µs step most of its decisions measure
// 0 µs and the others 1,000 µs, and their median is one or the other: neither is its time. A median
// of a few steps is not a measurement either, so under 10 steps (`COARSE_MEDIAN_STEPS`) the readout
// says what the measurements do support:
//   - the mean of the measured decisions, labelled as a mean, once there are enough of them. A
//     decision does not start in step with the clock's ticks, so one that takes d µs measures
//     floor(d / step) or one more step, the latter with a probability of about the fraction left
//     over: the measured times are right on average, and the mean of many is an estimate of the
//     true time (shown to two significant digits);
//   - before that, the median labelled as coarse when it is one step or more, and the bound it gives
//     when it is under one step: at least half the decisions measured under one clock step.

/** Measured decisions needed before the mean is shown in place of a median of a few clock steps. */
export const MIN_MEAN_DECISIONS = 30

/** Below this many clock steps the median is not shown as a measurement. A median of k steps can be
 * off by up to one step from the true time, so its error can exceed 10% under 10 steps: 1,000 µs
 * read for a decision of about 430 µs under a 1,000 µs step. From 10 steps up, the error is at most
 * 10% and the median is shown as measured. */
export const COARSE_MEDIAN_STEPS = 10

/** The session's rolling timing (`GameSession.timing()`): over the last 120 decisions at most. */
export interface DecisionTiming {
  medianUs: number
  meanUs: number
  /** How many decisions the median and the mean are over. */
  n: number
}

export interface LatencyReadout {
  /** `median`: the median as measured; `mean`: the median is a few clock steps, the mean is shown;
   * `coarse`: the median is a few clock steps (one or more), shown as approximate until there are
   * enough decisions for the mean; `bound`: the median is under one clock step, and only that bound
   * is shown; `none`: nothing measured yet. */
  kind: 'none' | 'median' | 'mean' | 'coarse' | 'bound'
  /** µs per decision: `2,427`, `≈ 270`, `≈ 1,000`, `< 1,000`, `—`. */
  us: string
  /** Decisions per second that time allows: `412`, `≈ 3,700`, `≈ 1,000`, `> 1,000`, `—`. */
  capacity: string
  /** What the number is, for `mean`, `coarse` and `bound` (empty otherwise): `mean of 120
   * decisions: this browser's clock counts in steps of 1,000 µs`. */
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
  const known = stepUs > 1
  if (!known || t.medianUs >= COARSE_MEDIAN_STEPS * stepUs) {
    if (t.medianUs > 0) {
      return { kind: 'median', us: int(t.medianUs), capacity: int(1e6 / t.medianUs), detail: '' }
    }
    // a median of 0 with a clock whose step is unknown: no true statement to make. The step is
    // unknown only while the clock has not been seen to tick (a clock rounded to 100 ms, say,
    // under which every decision measures 0); the callers measure it again with each new decision.
    return { kind: 'none', us: '—', capacity: '—', detail: '' }
  }
  const clock = `this browser's clock counts in steps of ${int(stepUs)} µs`
  if (t.n >= MIN_MEAN_DECISIONS && t.meanUs > 0) {
    return {
      kind: 'mean',
      us: `≈ ${approx(t.meanUs)}`,
      capacity: `≈ ${approx(1e6 / t.meanUs)}`,
      detail: `mean of ${int(t.n)} decisions: ${clock}`,
    }
  }
  if (t.medianUs >= stepUs) {
    return {
      kind: 'coarse',
      us: `≈ ${int(t.medianUs)}`,
      capacity: `≈ ${approx(1e6 / t.medianUs)}`,
      detail: `median of ${int(t.n)} decisions: ${clock}`,
    }
  }
  return {
    kind: 'bound',
    us: `< ${int(stepUs)}`,
    capacity: `> ${int(1e6 / stepUs)}`,
    detail: `median under one clock step: ${clock}`,
  }
}
