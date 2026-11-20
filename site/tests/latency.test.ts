// Tests of the live latency readouts (home page counters, Play HUD) under coarse browser clocks.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { COARSE_MEDIAN_STEPS, decisionLatency, MIN_MEAN_DECISIONS } from '../.vitepress/theme/lib/latency.ts'

test('nothing measured yet reads as a dash', () => {
  const r = decisionLatency({ medianUs: 0, meanUs: 0, n: 0 }, 1000)
  assert.deepEqual(r, { kind: 'none', us: '—', capacity: '—', detail: '' })
})

test('a median of 10 clock steps or more is shown as measured', () => {
  // Chrome: 100 µs steps, a model at about 2,400 µs keeps its median
  const r = decisionLatency({ medianUs: 2400, meanUs: 2450, n: 120 }, 100)
  assert.equal(r.kind, 'median')
  assert.equal(r.us, '2,400')
  assert.equal(r.capacity, '417')
  assert.equal(r.detail, '')
  // a precise clock (step unknown or 1 µs) is read the same way
  assert.equal(decisionLatency({ medianUs: 2427, meanUs: 2500, n: 5 }, 0).us, '2,427')
  assert.equal(decisionLatency({ medianUs: 250, meanUs: 260, n: 120 }, 1).kind, 'median')
})

test('a median of exactly 10 clock steps is shown as measured, one step less is not', () => {
  assert.equal(COARSE_MEDIAN_STEPS, 10)
  const at = decisionLatency({ medianUs: 10_000, meanUs: 10_400, n: 120 }, 1000)
  assert.equal(at.kind, 'median')
  assert.equal(at.us, '10,000')
  assert.equal(at.capacity, '100')
  const under = decisionLatency({ medianUs: 9000, meanUs: 9400, n: 120 }, 1000)
  assert.equal(under.kind, 'mean')
  assert.equal(under.us, '≈ 9,400')
})

test('a median of 3 steps of 100 µs shows the mean, labelled as a mean (Chrome, s3-pico)', () => {
  const r = decisionLatency({ medianUs: 300, meanUs: 266.7, n: 120 }, 100)
  assert.equal(r.kind, 'mean')
  assert.equal(r.us, '≈ 270')
  assert.equal(r.capacity, '≈ 3,700')
  assert.equal(r.detail, "mean of 120 decisions: this browser's clock counts in steps of 100 µs")
})

test('a median of 0 under a 1,000 µs step shows the mean, labelled as a mean', () => {
  // Firefox and WebKit: 1,000 µs steps; 32 of 120 decisions measured one step, the others 0
  const mean = (32 * 1000) / 120
  const r = decisionLatency({ medianUs: 0, meanUs: mean, n: 120 }, 1000)
  assert.equal(r.kind, 'mean')
  assert.equal(r.us, '≈ 270')
  assert.equal(r.capacity, '≈ 3,700')
  assert.equal(r.detail, "mean of 120 decisions: this browser's clock counts in steps of 1,000 µs")
  assert.ok(!r.us.includes('—') && !r.capacity.includes('—'))
})

test('a median of 0 under a 1,000 µs step, before enough decisions, states the bound', () => {
  const r = decisionLatency({ medianUs: 0, meanUs: 500, n: MIN_MEAN_DECISIONS - 1 }, 1000)
  assert.equal(r.kind, 'bound')
  assert.equal(r.us, '< 1,000')
  assert.equal(r.capacity, '> 1,000')
  assert.equal(r.detail, "median under one clock step: this browser's clock counts in steps of 1,000 µs")
})

test('every decision measured 0 under a 1,000 µs step states the bound, never 0 µs', () => {
  const r = decisionLatency({ medianUs: 0, meanUs: 0, n: 120 }, 1000)
  assert.equal(r.kind, 'bound')
  assert.equal(r.us, '< 1,000')
  assert.equal(r.capacity, '> 1,000')
})

test('a median of exactly one 1,000 µs step shows the mean, not 1,000 µs (WebKit, s3-pico)', () => {
  const r = decisionLatency({ medianUs: 1000, meanUs: 430, n: 120 }, 1000)
  assert.equal(r.kind, 'mean')
  assert.equal(r.us, '≈ 430')
  assert.equal(r.capacity, '≈ 2,300')
  assert.equal(r.detail, "mean of 120 decisions: this browser's clock counts in steps of 1,000 µs")
})

test('a median of a few steps, before enough decisions, is labelled as coarse', () => {
  const r = decisionLatency({ medianUs: 1000, meanUs: 430, n: MIN_MEAN_DECISIONS - 1 }, 1000)
  assert.equal(r.kind, 'coarse')
  assert.equal(r.us, '≈ 1,000')
  assert.equal(r.capacity, '≈ 1,000')
  assert.equal(r.detail, "median of 29 decisions: this browser's clock counts in steps of 1,000 µs")
  const c = decisionLatency({ medianUs: 300, meanUs: 266.7, n: 12 }, 100)
  assert.equal(c.kind, 'coarse')
  assert.equal(c.us, '≈ 300')
  assert.equal(c.capacity, '≈ 3,300')
  // under one step, the bound stays
  assert.equal(decisionLatency({ medianUs: 0, meanUs: 40, n: 12 }, 100).kind, 'bound')
  // from 10 steps up, the median as measured, however few the decisions
  assert.equal(decisionLatency({ medianUs: 1000, meanUs: 1100, n: 3 }, 100).kind, 'median')
})
