// Tests of lib/size.ts: one unit for file sizes on the site, the decimal one of the repository's
// documents (KB = 1,000 bytes, MB = 1,000,000 bytes), so a demo's facts line, the home page counter
// and the Models catalogue print the same size for the same file.
//
//   cd site && npm test
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { formatSize, sizeParts } from '../.vitepress/theme/lib/size.ts'

const here = dirname(fileURLToPath(import.meta.url))
const site = join(here, '..')
// The catalogue's sizes are checked against formatSize in catalogue.test.ts.

test('decimal units: whole KB under 1 MB, one decimal of MB above', () => {
  assert.equal(formatSize(631_936), '632 KB') // s6-ear2-nano: not 617 (631,936 / 1,024)
  assert.equal(formatSize(251_104), '251 KB') // s3-pico: not 245
  assert.equal(formatSize(142_912), '143 KB')
  assert.equal(formatSize(2_418_912), '2.4 MB')
  assert.equal(formatSize(3_692_224), '3.7 MB')
  assert.equal(formatSize(20_684_640), '20.7 MB')
  assert.equal(formatSize(21_329_304), '21.3 MB')
  assert.equal(formatSize(0), '0 KB')
  assert.equal(formatSize(499), '0 KB')
  assert.equal(formatSize(999_499), '999 KB')
  assert.equal(formatSize(999_500), '1.0 MB') // never "1,000 KB"
  assert.equal(formatSize(1_000_000), '1.0 MB')
  assert.equal(formatSize(1_234_567_890), '1,234.6 MB')
  assert.deepEqual(sizeParts(631_936), { value: '632', unit: 'KB' })
  assert.deepEqual(sizeParts(20_684_640), { value: '20.7', unit: 'MB' })
})

/** Every source file of the theme and of the Spaces. */
function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n)
    if (statSync(p).isDirectory()) return n === 'node_modules' || n === 'dist' || n === 'cache' ? [] : sources(p)
    return /\.(ts|vue)$/.test(n) ? [p] : []
  })
}

test('no component formats a byte count in binary units', () => {
  const files = [...sources(join(site, '.vitepress/theme')), ...sources(join(site, 'space'))]
  assert.ok(files.length > 20)
  for (const f of files) {
    const text = readFileSync(f, 'utf8')
    assert.doesNotMatch(text, /\/\s*1024\b/, `${f} divides by 1,024: use formatSize (lib/size.ts)`)
    assert.doesNotMatch(text, /\bformatKB\b/, `${f} uses formatKB: use formatSize (lib/size.ts)`)
    assert.doesNotMatch(text, /\b(KiB|MiB)\b/, `${f}: the site writes decimal units`)
  }
})
