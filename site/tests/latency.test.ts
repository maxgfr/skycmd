// Tests of the live latency readouts (home page counters, Play HUD) under coarse browser clocks.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { decisionLatency, MIN_MEAN_DECISIONS } from '../.vitepress/theme/lib/latency.ts'

test('nothing measured yet reads as a dash', () => {
  const r = decisionLatency({ medianUs: 0, meanUs: 0, n: 0 }, 1000)
  assert.deepEqual(r, { kind: 'none', us: '—', capacity: '—', detail: '' })
})

test('a median above the clock step is shown as measured', () => {
  // Chrome: 100 µs steps, s3-pico at about 250 µs
  const r = decisionLatency({ medianUs: 300, meanUs: 266.7, n: 120 }, 100)
  assert.equal(r.kind, 'median')
  assert.equal(r.us, '300')
  assert.equal(r.capacity, '3,333')
  assert.equal(r.detail, '')
  // a precise clock (step unknown or 1 µs) is read the same way
  assert.equal(decisionLatency({ medianUs: 2427, meanUs: 2500, n: 5 }, 0).us, '2,427')
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

test('a median of one step or more is the measured median, even under a coarse clock', () => {
  const r = decisionLatency({ medianUs: 1000, meanUs: 1400, n: 120 }, 1000)
  assert.equal(r.kind, 'median')
  assert.equal(r.us, '1,000')
  assert.equal(r.capacity, '1,000')
})
