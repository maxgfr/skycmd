// Tests of the Models catalogue (lib/catalogue.ts) against the files it describes:
// the manifest, the Hugging Face cards and the pages of this site.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { CATALOGUE, GROUPS, STATUS, isExternal } from '../.vitepress/theme/lib/catalogue.ts'

const here = dirname(fileURLToPath(import.meta.url))
const site = join(here, '..')
const repo = join(site, '..')
const manifest: { name: string; bytes: number }[] = JSON.parse(
  readFileSync(join(site, 'public/models/manifest.json'), 'utf8'),
)

/** "715 KB", "2.4 MB": the decimal units the site writes, rounded as written. */
function sizeOf(bytes: number, written: string): string {
  const [value, unit] = written.split(' ')
  const digits = value.includes('.') ? value.split('.')[1].length : 0
  const scaled = unit === 'MB' ? bytes / 1e6 : bytes / 1e3
  return `${scaled.toFixed(digits)} ${unit}`
}

test('every model of the manifest has a card', () => {
  const covered = new Set(CATALOGUE.flatMap((e) => e.manifest ?? []))
  for (const m of manifest) assert.ok(covered.has(m.name), `${m.name} has no entry in lib/catalogue.ts`)
})

test('every size matches the bytes of the manifest', () => {
  for (const e of CATALOGUE) {
    if (!e.manifest) {
      assert.equal(e.size, undefined, `${e.id}: a size with no manifest entry`)
      continue
    }
    const m = manifest.find((x) => x.name === e.manifest![0])
    assert.ok(m, `${e.id}: ${e.manifest[0]} is not in the manifest`)
    assert.ok(e.size, `${e.id}: no size`)
    assert.equal(sizeOf(m.bytes, e.size!), e.size, `${e.id}: ${m.bytes} bytes`)
  }
})

test('ids are unique and every entry has a known group and status', () => {
  const ids = CATALOGUE.map((e) => e.id)
  assert.equal(new Set(ids).size, ids.length)
  for (const e of CATALOGUE) {
    assert.ok(e.group in GROUPS, e.id)
    assert.ok(e.status in STATUS, e.id)
    assert.ok(e.purpose && e.brings && e.source.length, e.id)
  }
})

test('the groups hold the statuses they promise', () => {
  for (const e of CATALOGUE) {
    if (e.group === 'works') assert.ok(['passed', 'second-look', 'ungated'].includes(e.status), e.id)
    if (e.status === 'passed' || e.status === 'second-look') assert.equal(e.group, 'works', e.id)
    if (e.group === 'progress') {
      assert.equal(e.status, 'progress', e.id)
      assert.equal(e.size, undefined, e.id)
      assert.equal(e.try, undefined, `${e.id}: nothing to try before a result`)
    }
    if (e.status === 'failed') assert.equal(e.group, 'results', e.id)
  }
})

test('every "Get it" link has a model card in hf/cards/', () => {
  for (const e of CATALOGUE) {
    if (!e.get) continue
    const name = e.get.replace('https://huggingface.co/maxgfr/', '')
    assert.notEqual(name, e.get, `${e.id}: not a maxgfr model page`)
    assert.ok(existsSync(join(repo, 'hf/cards', `${name}.md`)), `${e.id}: no hf/cards/${name}.md`)
  }
})

test('every page linked exists', () => {
  for (const e of CATALOGUE) {
    for (const l of [e.try, e.read]) {
      if (!l) continue
      if (isExternal(l.link)) {
        const docs = 'https://github.com/maxgfr/skycmd/blob/main/'
        if (l.link.startsWith(docs)) assert.ok(existsSync(join(repo, l.link.slice(docs.length))), l.link)
        continue
      }
      const path = l.link.split('#')[0]
      const file = path.endsWith('/') ? join(site, path, 'index.md') : join(site, `${path}.md`)
      assert.ok(existsSync(file), `${e.id}: ${l.link}`)
    }
  }
})

test('the home page strip shows three or four working models, each with something to try', () => {
  const home = CATALOGUE.filter((e) => e.home)
  assert.ok(home.length >= 3 && home.length <= 4)
  for (const e of home) {
    assert.equal(e.group, 'works', e.id)
    assert.ok(e.try, e.id)
  }
})
