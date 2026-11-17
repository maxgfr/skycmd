// Tests of the memory demo's pure logic (lib/memory.ts): the seeds the browser plays, the tally by
// delay bucket and its interval, the words of a frame, and the published numbers it shows, checked
// against the result files they are copied from (runs/s6-sleep3/). Pure node: no DOM, no wasm.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  BATCH_EPISODES,
  BROWSER_FROM,
  BUCKETS,
  BUCKET_LABELS,
  KEYS,
  PASSES,
  TEST_POOLED,
  TEST_SEED0,
  Tally,
  argmax,
  askLine,
  formatInterval,
  formatShare,
  frameLine,
  noiseOf,
  validationSeeds,
  wilson,
  type Frame,
} from '../.vitepress/theme/lib/memory.ts'

const here = dirname(fileURLToPath(import.meta.url))
const repo = join(here, '..', '..')
const runs = join(repo, 'runs', 's6-sleep3')

test('the browser plays seeds of the validation split, away from the dev2 and test seeds', () => {
  assert.equal(BROWSER_FROM % 20, 18)
  const s = validationSeeds(BATCH_EPISODES)
  assert.equal(s.length, 200)
  assert.equal(s[0], BROWSER_FROM)
  for (let i = 0; i < s.length; i++) {
    assert.equal(s[i] % 20, 18)
    if (i) assert.equal(s[i] - s[i - 1], 20)
    assert.ok(Number.isSafeInteger(s[i]))
  }
  // dev2 starts at 2,000,000,000 and the test at 5,000,000,000, both ≡ 19 (mod 20).
  assert.ok(s.every((x) => x % 20 !== 19))
  assert.deepEqual(validationSeeds(3, 7_000_000_000), [7_000_000_018, 7_000_000_038, 7_000_000_058])
  assert.deepEqual(validationSeeds(2, 19), [38, 58])
  assert.deepEqual(validationSeeds(0), [])
})

test('the tally counts the asks of each delay bucket', () => {
  const t = new Tally()
  t.addEpisode(
    [
      { frame: 3, delay: 2, bucket: 0, correct: true },
      { frame: 9, delay: 6, bucket: 1, correct: false },
      { frame: 40, delay: 20, bucket: 3, correct: true },
      { frame: 41, delay: 30, bucket: 3, correct: false },
    ],
    64,
    6400,
  )
  t.addEpisode([{ frame: 5, delay: 4, bucket: 0, correct: false }], 64, 3200)
  assert.deepEqual(t.asked, [2, 1, 0, 2])
  assert.deepEqual(t.right, [1, 0, 0, 1])
  assert.equal(t.accuracy(0), 0.5)
  assert.equal(t.accuracy(2), null)
  assert.equal(t.accuracy(), 2 / 5)
  assert.equal(t.totalAsked, 5)
  assert.equal(t.episodes, 2)
  assert.equal(t.decisions, 128)
  assert.equal(t.medianMicros, 50)
  assert.equal(t.interval(2), null)
  const iv = t.interval()!
  assert.ok(iv[0] < 0.4 && 0.4 < iv[1])
  assert.throws(() => t.addEpisode([{ frame: 0, delay: 1, bucket: 4, correct: true }], 1, 1))
  assert.equal(new Tally().medianMicros, null)
})

test('the Wilson interval', () => {
  const [lo, hi] = wilson(200, 400)!
  assert.ok(Math.abs(lo - 0.4512) < 1e-3 && Math.abs(hi - 0.5488) < 1e-3, `${lo} ${hi}`)
  assert.deepEqual(wilson(0, 0), null)
  const [a, b] = wilson(0, 10)!
  assert.equal(a, 0)
  assert.ok(b > 0.25 && b < 0.35)
  assert.equal(wilson(10, 10)![1], 1)
  assert.equal(formatShare(0.61475), '0.615')
  assert.equal(formatShare(null), '—')
  assert.equal(formatInterval([0.4512, 0.5488]), '0.451–0.549')
  assert.equal(formatInterval(null), '')
})

test('the words of a frame', () => {
  const show: Frame = { frame: 2, state: 'show a colour red ask none noise 46', kind: 'show', door: 'a', colour: 'red', answer: 'wait' }
  const ask: Frame = { frame: 9, state: 'show none colour none ask a noise 3', kind: 'ask', door: 'a', answer: 'red', delay: 7, bucket: 1 }
  const idle: Frame = { frame: 0, state: 'show none colour none ask none noise 12', kind: 'idle', answer: 'wait' }
  assert.equal(frameLine(show), 'door a is given red')
  assert.equal(frameLine(ask), 'which colour was door a given?')
  assert.equal(frameLine(idle), 'nothing to remember (a distractor number, 12)')
  assert.equal(askLine(ask), 'door a was shown 7 frames earlier (delay bucket 5-8)')
  assert.equal(askLine({ ...ask, delay: 1, bucket: 0 }), 'door a was shown 1 frame earlier (delay bucket 1-4)')
  assert.equal(askLine(show), '')
  assert.equal(noiseOf(idle.state), 12)
  assert.equal(noiseOf('ask b'), null)
  assert.equal(argmax([0.1, 0.4, 0.4, 0.1]), 1)
  assert.deepEqual([...KEYS], ['wait', 'red', 'green', 'blue', 'yellow'])
  assert.deepEqual(BUCKET_LABELS, ['1-4', '5-8', '9-16', '17-32'])
  assert.deepEqual(BUCKETS[3], [17, 32])
})

/** The readings of `s2-probe` in a result file: `[{passes, memory, score, buckets}]`. */
function probe(file: string): { passes: number; memory: boolean; score: number; buckets: number[] }[] {
  return JSON.parse(readFileSync(join(runs, file), 'utf8'))
}

test('the test numbers of the published files are the ones of their result files', () => {
  const sleep = probe('test/sleep-s0/recall.json')
  for (const n of PASSES) {
    const r = sleep.find((x) => x.passes === n)!
    assert.ok(r && r.memory, `N = ${n}`)
    assert.equal(Number(TEST_SEED0.sleep[n].overall), r.score, `N = ${n} overall`)
    assert.deepEqual(TEST_SEED0.sleep[n].buckets.map(Number), r.buckets, `N = ${n} buckets`)
  }
  const [c] = probe('test/control-s0/recall.json')
  assert.equal(c.memory, false)
  assert.equal(Number(TEST_SEED0.control.overall), c.score)
  assert.deepEqual(TEST_SEED0.control.buckets.map(Number), c.buckets)
})

test('the pooled test numbers are the ones of REPORT.md', () => {
  const report = readFileSync(join(runs, 'REPORT.md'), 'utf8')
  const row = (label: string) => {
    const line = report.split('\n').find((l) => l.startsWith(`| ${label} |`) && l.split('|').length > 7)
    assert.ok(line, label)
    return line!.split('|').slice(2, 6).map((x) => x.trim())
  }
  assert.deepEqual(row('control'), TEST_POOLED.control.buckets)
  for (const n of PASSES) assert.deepEqual(row(`N = ${n}`), TEST_POOLED.sleep[n].buckets)
  const probes = report.split('\n').find((l) => l.startsWith('| recall |'))!.split('|').map((x) => x.trim())
  assert.deepEqual(probes.slice(2, 6), [
    TEST_POOLED.control.overall,
    TEST_POOLED.sleep[1].overall,
    TEST_POOLED.sleep[2].overall,
    TEST_POOLED.sleep[4].overall,
  ])
})
