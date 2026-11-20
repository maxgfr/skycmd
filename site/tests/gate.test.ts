// Tests of the verdict notice (lib/gate.ts, rendered by components/GateNotice.vue) and of the
// manifest's gates against the Models catalogue: a failed model's Space and demos say it failed, a
// passed model's say it passed, with a space between the lead, the note and the link.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import { gateNotice, gateNoticeText, type GateFields, type GateNotice } from '../.vitepress/theme/lib/gate.ts'
import { CATALOGUE, type Status } from '../.vitepress/theme/lib/catalogue.ts'

const here = dirname(fileURLToPath(import.meta.url))
const site = join(here, '..')
const theme = join(site, '.vitepress/theme')
type Entry = GateFields & { name: string }
const manifest: Entry[] = JSON.parse(readFileSync(join(site, 'public/models/manifest.json'), 'utf8'))
const forecast: { models: Entry[] } = JSON.parse(readFileSync(join(site, 'public/models/forecast.json'), 'utf8'))

const v1 = {
  gate: 'failed',
  gate_note: 'It missed its calibration guard: ECE 0.0821 against 0.08 at full depth (0.547 on the typed-decisions bench, where the gate asked for at least 0.45)',
  gate_url: 'https://example.org/result',
} as const

/** The text of GateNotice.vue's template rendered by Vue's own compiler, tags removed. */
async function rendered(notice: GateNotice): Promise<string> {
  const sfc = readFileSync(join(theme, 'components/GateNotice.vue'), 'utf8')
  const start = sfc.indexOf('<template>') + '<template>'.length
  const template = sfc.slice(start, sfc.lastIndexOf('</template>', sfc.indexOf('<style')))
  const html = await renderToString(createSSRApp({ data: () => ({ notice }), template }))
  const text = html.replace(/<!--[^]*?-->/g, '').replace(/<[^>]+>/g, '')
  const entities: Record<string, string> = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'" }
  return text.replace(/&(amp|lt|gt|quot|#39);/g, (x) => entities[x])
}

test('the three parts of the notice are separated by a space', () => {
  assert.equal(
    gateNoticeText(gateNotice(v1)!),
    'Failed its pre-registered gate: published as a result, not for use. It missed its calibration guard: ECE 0.0821 against 0.08 at full depth (0.547 on the typed-decisions bench, where the gate asked for at least 0.45). The result.',
  )
  assert.equal(
    gateNoticeText(gateNotice(v1, 's6-micro-v1')!),
    's6-micro-v1 failed its pre-registered gate: published as a result, not for use. It missed its calibration guard: ECE 0.0821 against 0.08 at full depth (0.547 on the typed-decisions bench, where the gate asked for at least 0.45). The result.',
  )
  assert.equal(
    gateNoticeText(gateNotice({ ...v1, gate: 'passed', gate_note: 'A note' }, 'm', 'It starts unticked.')!),
    'm passed its pre-registered gate. A note. It starts unticked. The result.',
  )
  assert.equal(gateNoticeText(gateNotice({ gate: 'passed' })!), 'Passed its pre-registered gate.')
  assert.equal(gateNoticeText(gateNotice({ gate: 'failed', gate_note: 'A note', gate_url: null })!).endsWith('use. A note.'), true)
})

test('the lead says how the gate was fixed: pre-registered by default, else fixed before training', () => {
  const fixed = { ...v1, gate_kind: 'fixed-before-training' } as const
  assert.equal(gateNotice(fixed)!.lead, 'Failed its gate, fixed before training: published as a result, not for use.')
  assert.equal(gateNotice(fixed, 's4-micro')!.lead, 's4-micro failed its gate, fixed before training: published as a result, not for use.')
  assert.equal(gateNotice({ ...fixed, gate: 'passed' })!.lead, 'Passed its gate, fixed before training.')
  assert.equal(gateNotice({ ...v1, gate_kind: 'pre-registered' })!.lead, gateNotice(v1)!.lead)
  assert.equal(gateNotice({ ...v1, gate: 'passed' })!.lead, 'Passed its pre-registered gate.')
  const s4 = manifest.find((e) => e.name === 's4-micro')!
  assert.equal(
    gateNoticeText(gateNotice(s4)!),
    `Failed its gate, fixed before training: published as a result, not for use. ${s4.gate_note}. The result.`,
  )
  // a gate fixed before training is named so only where no pre-registration document exists
  for (const e of manifest) if (e.gate_kind) assert.equal(e.gate_kind, 'fixed-before-training', e.name)
  assert.deepEqual(manifest.filter((e) => e.gate_kind).map((e) => e.name), ['s4-micro'])
})

test('a model without a gate gets no notice', () => {
  assert.equal(gateNotice(null), null)
  assert.equal(gateNotice({}), null)
  assert.equal(gateNotice({ gate: null, gate_note: null, gate_url: null }), null)
})

test('GateNotice.vue renders the text with its spaces, for every model with a gate', async () => {
  const gated = [...manifest, ...forecast.models].filter((e) => e.gate)
  assert.ok(gated.length > 0)
  for (const e of gated) {
    for (const n of [gateNotice(e)!, gateNotice(e, e.name)!, gateNotice(e, e.name, 'It starts unticked; tick it to see how it forecasts.')!]) {
      const text = await rendered(n)
      assert.equal(text, gateNoticeText(n), e.name)
      assert.doesNotMatch(text, /[.:)][A-Z]/, `${e.name}: two sentences run together`)
    }
  }
})

test('every demo and the Space app render a gate through GateNotice, never by hand', () => {
  const files = [
    ...readdirSync(join(theme, 'components'))
      .filter((f) => f.endsWith('.vue') && f !== 'GateNotice.vue')
      .map((f) => join(theme, 'components', f)),
    ...readdirSync(join(site, 'space'))
      .filter((f) => f.endsWith('.vue'))
      .map((f) => join(site, 'space', f)),
  ]
  for (const f of files) {
    const template = readFileSync(f, 'utf8').split('<template>').slice(1).join('<template>')
    assert.doesNotMatch(template, /gate_note\s*}}/, `${f}: renders gate_note itself; use lib/gate.ts and GateNotice`)
  }
})

const FAILED: Status[] = ['failed']
const PASSED: Status[] = ['passed', 'second-look', 'bench-only']

test('the manifest gives each model the verdict of its catalogue card', () => {
  for (const c of CATALOGUE) {
    for (const name of c.manifest ?? []) {
      const m = manifest.find((e) => e.name === name)
      assert.ok(m, `${name} is not in the manifest`)
      if (FAILED.includes(c.status)) {
        assert.equal(m.gate, 'failed', `${name}: the catalogue says it failed (${c.status})`)
      } else if (PASSED.includes(c.status)) {
        assert.equal(m.gate, 'passed', `${name}: the catalogue says it passed (${c.status})`)
      } else {
        assert.equal(m.gate ?? null, null, `${name}: the catalogue says it has no gate (${c.status})`)
        continue
      }
      assert.ok(m.gate_note && !m.gate_note.endsWith('.'), `${name}: a gate note, without its final period`)
      assert.match(m.gate_url ?? '', /^https:\/\//, `${name}: a link to the result`)
    }
  }
})
