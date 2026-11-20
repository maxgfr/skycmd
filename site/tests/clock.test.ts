// Tests of the browser clock step measurement (clock.ts) with fake clocks.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  CLOCK_MAX_ATTEMPTS,
  CLOCK_MAX_READS,
  CLOCK_RETRY_MS,
  clockStepOf,
  measureClockStep,
} from '../.vitepress/theme/lib/clock.ts'

/** A deterministic number in [0, 1) for bucket `k`. */
function frac(k: number, seed: number): number {
  const x = Math.sin(k * 12.9898 + seed * 78.233) * 43758.5453
  return x - Math.floor(x)
}

interface FakeClock {
  /** The reading in ms, as `performance.now` gives it; each read takes `readNs` of true time. */
  now: () => number
  /** True time in ns. */
  t: number
  reads: number
}

/** A clock that counts in steps of `stepUs`, read in `readNs` per read, from true time `startNs`.
 * With `jitterSeed`, the reading moves to the next step at an instant drawn within each step
 * (Firefox's jittered reduced-precision timer): the readings stay on the grid. */
function fakeClock(stepUs: number, readNs: number, startNs = 0, jitterSeed?: number): FakeClock {
  const stepNs = stepUs * 1000
  const c: FakeClock = {
    t: startNs,
    reads: 0,
    now: () => {
      const t = c.t
      c.t += readNs
      c.reads++
      let k = Math.floor(t / stepNs)
      if (jitterSeed !== undefined && t >= k * stepNs + frac(k, jitterSeed) * stepNs) k++
      return (k * stepNs) / 1e6
    },
  }
  return c
}

test('a 1,000 µs step read very fast (more than 20,000 reads per tick) is measured', () => {
  // 20 ns per read: 50,000 reads per tick, where 20,000 reads saw no tick
  for (const startNs of [0, 1, 400_000, 999_999]) {
    const c = fakeClock(1000, 20, startNs)
    assert.equal(measureClockStep(c.now), 1000)
    // the reads stop after a few ticks: well within 5 ms of true time
    assert.ok(c.t - startNs <= 5_000_000, `took ${(c.t - startNs) / 1e6} ms`)
  }
})

test('a 100 µs step (Chromium, not cross-origin isolated) is measured', () => {
  for (const readNs of [20, 70, 500]) {
    const c = fakeClock(100, readNs, 37_000)
    assert.equal(measureClockStep(c.now), 100)
  }
})

test('a 5 µs step (Chromium, cross-origin isolated) is measured', () => {
  for (const readNs of [20, 70]) {
    const c = fakeClock(5, readNs, 1_234_567)
    assert.equal(measureClockStep(c.now), 5)
  }
})

test('a clock finer than 1 µs reads as 1, not as unknown', () => {
  const c = fakeClock(0.01, 30)
  assert.equal(measureClockStep(c.now), 1)
})

test('a jittered 1,000 µs clock (Firefox) is measured as 1,000 µs', () => {
  for (let seed = 1; seed <= 50; seed++) {
    const startNs = Math.floor(frac(seed, 7) * 10_000_000)
    const c = fakeClock(1000, 20, startNs, seed)
    assert.equal(measureClockStep(c.now), 1000, `seed ${seed}`)
    assert.ok(c.t - startNs <= 8_000_000, `seed ${seed} took ${(c.t - startNs) / 1e6} ms`)
  }
})

test('a coarse clock that ticks once is measured from that tick (16.7 ms under fingerprinting resistance)', () => {
  const stepUs = 100_000 / 6 // 16,666.7 µs
  const c = fakeClock(stepUs, 20, 16_000_000) // 0.67 ms before a tick
  assert.equal(measureClockStep(c.now), 16_667)
})

test('a clock that never ticks within the cap returns 0 after the cap, and is not kept', () => {
  // 100 ms steps (fingerprinting resistance), 20 ns per read: no tick within the cap
  const c = fakeClock(100_000, 20)
  assert.equal(measureClockStep(c.now), 0)
  assert.equal(c.reads, CLOCK_MAX_READS)
  // the cap bounds the time spent: 10 ms at 20 ns per read, 35 ms at 70 ns
  assert.ok((CLOCK_MAX_READS * 70) / 1e6 <= 35)

  const c2 = fakeClock(100_000, 20)
  const step = clockStepOf(c2.now)
  assert.equal(step(), 0)
  // a call within CLOCK_RETRY_MS does not measure again: one read for the time
  const reads = c2.reads
  assert.equal(step(), 0)
  assert.equal(c2.reads, reads + 1)
  // later, close before a tick, a call measures again and the step is kept from then on
  c2.t = 1_099_000_000 // 1 ms before the tick at 1,100 ms
  assert.equal(step(), 100_000)
  const after = c2.reads
  assert.equal(step(), 100_000)
  assert.equal(c2.reads, after, 'a known step is not measured again')
})

test('a clock that never ticks is settled as unknown after the last attempt', () => {
  let reads = 0
  let t = 0
  // a frozen clock, moved forward between calls by more than the retry delay
  const step = clockStepOf(() => {
    reads++
    return t
  }, 1000)
  for (let i = 0; i < CLOCK_MAX_ATTEMPTS; i++) {
    assert.equal(step(), 0)
    t += CLOCK_RETRY_MS
  }
  const before = reads
  assert.equal(step(), 0)
  assert.equal(reads, before, 'no further measurement')
})

test('a known step is measured once per session', () => {
  const c = fakeClock(1000, 20, 500_000)
  const step = clockStepOf(c.now)
  assert.equal(step(), 1000)
  const reads = c.reads
  assert.equal(step(), 1000)
  assert.equal(c.reads, reads)
})
