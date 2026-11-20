// The step of the browser clock (`performance.now`), measured by reading it. No import, so that
// `node --test` loads it without VitePress; wasm.ts binds it to `performance.now` as
// `clockStepMicros`.
//
// Browsers coarsen the clock: Chromium counts in steps of 100 µs on a page that is not
// cross-origin isolated (5 µs when it is), Firefox and Safari in steps of 1,000 µs, and a browser
// that resists fingerprinting may round it to 16.7 ms or 100 ms. A fast machine reads the clock
// tens of thousands of times within one step of 1,000 µs (about 30,000 in Firefox and 45,000 in
// WebKit on an Apple laptop), so a fixed number of reads can see no tick at all. The measurement
// therefore reads the clock until it has seen it tick, under a cap on the number of reads.
//
// What is measured: the smallest positive difference between two successive distinct readings,
// over the first few ticks. Firefox jitters its coarse clock in time, not in value: the instant at
// which the reading moves to the next step is drawn at random within each step, but the readings
// stay on the grid of steps and each tick moves them by exactly one step. A difference between two
// distinct readings is therefore a whole number of steps even when the ticks are irregular, and
// the smallest over a few ticks is the step. The first difference counts too: the readings, not
// the instants, are compared.

/** Positive differences to collect before the measurement stops. */
export const CLOCK_TICKS = 4
/** Once one tick is seen, the measurement also stops when the clock says this much has passed. */
export const CLOCK_BUDGET_MS = 5
/** Hard cap on reads per measurement. A read of `performance.now` costs 20 to 70 ns in current
 * engines (measured: about 23 ns in WebKit, 34 ns in Firefox, 70 ns in Chromium), so the cap ends
 * a measurement that sees no tick within about 10 to 35 ms, and it still leaves room for several
 * ticks of a 1,000 µs clock on the fastest of them. */
export const CLOCK_MAX_READS = 500_000
/** After a measurement that saw no tick, the next one waits at least this long (by the clock). */
export const CLOCK_RETRY_MS = 1000
/** Measurements that saw no tick before the step is settled as unknown (0) for the session. */
export const CLOCK_MAX_ATTEMPTS = 10

/** One measurement of the step of `now` (a clock in ms, like `performance.now`), in µs.
 * 0 when no tick was seen within the cap (unknown); 1 for a clock finer than 1 µs. */
export function measureClockStep(now: () => number, maxReads = CLOCK_MAX_READS): number {
  const start = now()
  let prev = start
  let step = Infinity
  let ticks = 0
  for (let reads = 1; reads < maxReads; reads++) {
    const t = now()
    if (t !== prev) {
      if (t > prev) {
        step = Math.min(step, t - prev)
        ticks++
        if (ticks >= CLOCK_TICKS || t - start >= CLOCK_BUDGET_MS) break
      }
      prev = t
    }
  }
  if (!Number.isFinite(step)) return 0
  return Math.max(1, Math.round(step * 1000))
}

/** The session's step of `now`, in µs, measured on the first call and kept once a tick is seen.
 * A measurement that sees no tick returns 0 and is not kept: a later call measures again, at most
 * once per `CLOCK_RETRY_MS` and `CLOCK_MAX_ATTEMPTS` times in all, so that a clock too coarse to
 * tick within the cap does not cost the page a measurement on every call. */
export function clockStepOf(now: () => number, maxReads = CLOCK_MAX_READS): () => number {
  let step: number | null = null
  let attempts = 0
  let lastAttempt = -Infinity
  return () => {
    if (step !== null) return step
    const t = now()
    if (t - lastAttempt < CLOCK_RETRY_MS) return 0
    lastAttempt = t
    attempts++
    const s = measureClockStep(now, maxReads)
    if (s > 0 || attempts >= CLOCK_MAX_ATTEMPTS) step = s
    return s
  }
}
