// Tests of the Space app (site/space/): the demo of one model (SpaceView.vue) and the global Space
// (HubView.vue, hub.ts), on the repository's manifest, hf/spaces.json and Models catalogue. The
// templates are rendered by Vue's own compiler with the demo components stubbed, each stub
// printing the props it was given.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import { CATALOGUE, GROUPS, STATUS } from '../.vitepress/theme/lib/catalogue.ts'
import { gateNotice } from '../.vitepress/theme/lib/gate.ts'
import { scopeFilter, setManifestScope } from '../.vitepress/theme/lib/scope.ts'
import { formatSize } from '../.vitepress/theme/lib/size.ts'
import { hashOf, hubGroups, hubModels, spaceOfHash, type HubConfig, type HubSpace, type SpaceConfig } from '../space/hub.ts'
import { DECISION, PROOFS } from '../space/sections.ts'

const here = dirname(fileURLToPath(import.meta.url))
const site = join(here, '..')
const repo = join(site, '..')
type Manifest = { name: string; bytes: number; task?: string; gate?: 'passed' | 'failed'; gate_kind?: 'fixed-before-training'; gate_note?: string; gate_url?: string }
type SpaceEntry = { model?: string; space_name?: string; kind: string; extra_models?: string[]; space?: boolean; tagline: string; notes?: string[]; proofs?: boolean; game?: string }
const manifest: Manifest[] = JSON.parse(readFileSync(join(site, 'public/models/manifest.json'), 'utf8'))
const entries: SpaceEntry[] = JSON.parse(readFileSync(join(repo, 'hf/spaces.json'), 'utf8'))
const SITE = 'https://maxgfr.github.io/skycmd'

/** What scripts/hf-publish.py puts in the global Space's `space.json` for one model's Space
 * (`space_config`, and the models and address of that Space). */
function hubSpace(s: SpaceEntry): HubSpace {
  const m = s.model!
  const e = manifest.find((x) => x.name === m)!
  const config: SpaceConfig = {
    model: m,
    title: m,
    tagline: s.tagline,
    kind: s.kind as SpaceConfig['kind'],
    proofs: s.proofs ?? false,
    model_repo: `https://huggingface.co/maxgfr/${m}`,
    notes: s.notes ?? [],
    game: s.game ?? null,
    gate: e.gate ?? null,
    gate_kind: e.gate_kind ?? null,
    gate_note: e.gate_note ?? null,
    gate_url: e.gate_url ?? null,
  }
  return { config, models: [m, ...(s.extra_models ?? [])], space_url: `https://huggingface.co/spaces/maxgfr/${m}` }
}
const spaces = entries.filter((s) => s.model && s.space !== false).map(hubSpace)
const hubEntry = entries.find((s) => s.kind === 'hub')!

function template(file: string): string {
  const sfc = readFileSync(join(site, file), 'utf8')
  const start = sfc.indexOf('<template>') + '<template>'.length
  const style = sfc.indexOf('<style')
  return sfc.slice(start, sfc.lastIndexOf('</template>', style < 0 ? sfc.length : style))
}

/** A stub of a demo component: a div that names it and prints its props. */
const stub = (name: string) => ({
  inheritAttrs: false,
  template: `<div data-stub="${name}" :data-props="JSON.stringify($attrs)"></div>`,
})
const DEMOS = ['EarDemo', 'ForecastView', 'LiveDecide', 'LiveProof', 'MemoryDemo', 'PlayGame', 'RouterDemo', 'SecretsVerifyDemo', 'TurnDemo']
const GateNotice = { props: ['notice'], template: template('.vitepress/theme/components/GateNotice.vue') }
const StatusBadge = { props: ['status'], data: () => ({ STATUS }), computed: { meta(this: { status: string }) { return STATUS[this.status as keyof typeof STATUS] } }, template: template('.vitepress/theme/components/StatusBadge.vue') }

/** SpaceView.vue as a component: its template, with its script's bindings. */
const SpaceView = {
  props: ['cfg'],
  data: () => ({ general: false, SITE, DECISION, PROOFS }),
  computed: { verdict(this: { cfg: SpaceConfig }) { return gateNotice(this.cfg) } },
  template: template('space/SpaceView.vue'),
  components: { ...Object.fromEntries(DEMOS.map((n) => [n, stub(n)])), GateNotice },
}

const strip = (html: string) => html.replace(/<!--[^]*?-->/g, '')
const text = (html: string) => strip(html).replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim()
const unescape = (s: string) =>
  s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')

async function renderSpace(cfg: SpaceConfig): Promise<string> {
  return strip(await renderToString(createSSRApp({ data: () => ({ cfg }), template: '<SpaceView :cfg="cfg" />', components: { SpaceView } })))
}

async function renderHub(open: HubSpace | null): Promise<string> {
  const hub: HubConfig = { kind: 'hub', title: 'skycmd', tagline: hubEntry.tagline, notes: hubEntry.notes, spaces }
  const SpaceStub = { props: ['cfg'], template: '<div data-stub="SpaceView" :data-model="cfg.model" :data-kind="cfg.kind"></div>' }
  return strip(
    await renderToString(
      createSSRApp({
        data: () => ({ cfg: hub, groups: hubGroups(spaces, CATALOGUE, GROUPS), open, back: null, SITE, hashOf }),
        template: template('space/HubView.vue'),
        components: { SpaceView: SpaceStub, StatusBadge },
      }),
    ),
  )
}

const of = (m: string) => spaces.find((s) => s.config.model === m)!

test('the global Space lists exactly the models of the manifest, each in its catalogue group', () => {
  const groups = hubGroups(spaces, CATALOGUE, GROUPS)
  const listed = groups.flatMap((g) => g.items.map((i) => i.name))
  assert.equal(new Set(listed).size, listed.length, 'a model listed twice')
  assert.deepEqual([...listed].sort(), manifest.map((m) => m.name).sort())
  assert.deepEqual(hubModels(spaces).sort(), manifest.map((m) => m.name).sort())
  for (const g of groups) {
    assert.equal(g.title, GROUPS[g.id].title)
    for (const it of g.items) {
      const cards = CATALOGUE.filter((c) => c.manifest?.includes(it.name))
      assert.equal(cards.length, 1, `${it.name}: one catalogue card`)
      assert.equal(it.card, cards[0])
      assert.equal(g.id, cards[0].group, `${it.name}: in the group of its card`)
      assert.equal(it.space.config.model, it.name)
      // the size on the card is the size of this model's file
      assert.equal(it.card.size, formatSize(manifest.find((m) => m.name === it.name)!.bytes), it.name)
    }
  }
  // the groups in the catalogue's order, the models in the catalogue's order
  assert.deepEqual(groups.map((g) => g.id), (Object.keys(GROUPS) as (keyof typeof GROUPS)[]).filter((id) => CATALOGUE.some((c) => c.group === id)))
  const order = CATALOGUE.flatMap((c) => c.manifest ?? [])
  for (const g of groups) {
    const names = g.items.map((i) => i.name)
    assert.deepEqual(names, order.filter((n) => names.includes(n)))
  }
})

test('the hub entry of hf/spaces.json has no model and counts nothing by hand', () => {
  assert.equal(entries.filter((s) => s.kind === 'hub').length, 1)
  assert.equal(hubEntry.model, undefined)
  assert.equal(hubEntry.space_name, 'skycmd')
  const notes = hubEntry.notes!.join(' ')
  // the counts come from the manifest when the Space is assembled (scripts/hf-publish.py)
  assert.doesNotMatch(notes, /\d/, 'a number typed in the hub notes')
  for (const field of ['{models}', '{passed}', '{failed}', '{ungated}']) assert.ok(notes.includes(field), field)
})

test('a hash opens the demo of its model, anything else the list', () => {
  assert.equal(spaceOfHash('#s6-route', spaces), of('s6-route'))
  assert.equal(spaceOfHash('#s6-route-nano', spaces), of('s6-route-nano'))
  assert.equal(spaceOfHash('s4-micro', spaces), of('s4-micro'))
  assert.equal(spaceOfHash('#s6%2Droute', spaces), of('s6-route'))
  for (const h of ['', '#', '#nope', '#%E0%A4%A', '#s6-route/']) assert.equal(spaceOfHash(h, spaces), null, h)
  for (const s of spaces) assert.equal(spaceOfHash(hashOf(s), spaces), s)
})

test('the list shows every model with its name, size, badge and purpose, each linking to its demo', async () => {
  const html = await renderHub(null)
  assert.doesNotMatch(html, /data-stub="SpaceView"/)
  for (const g of hubGroups(spaces, CATALOGUE, GROUPS)) {
    assert.ok(html.includes(`>${g.title}</h2>`), g.title)
    for (const it of g.items) {
      assert.ok(html.includes(`href="#${it.name}">${it.name}</a>`), it.name)
      assert.ok(unescape(html).includes(it.card.purpose), `${it.name}: purpose`)
    }
  }
  for (const s of Object.values(STATUS)) if (CATALOGUE.some((c) => STATUS[c.status] === s && c.group !== 'progress')) assert.ok(html.includes(`<span>${s.label}</span>`), s.label)
})

test('picking a model mounts the demo of its own Space, with that Space', async () => {
  for (const m of ['s6-route', 's6-micro-v1', 's2-nano-doom', 's4-micro']) {
    const html = await renderHub(of(m))
    assert.match(html, new RegExp(`data-stub="SpaceView" data-model="${m}" data-kind="${of(m).config.kind}"`), m)
    assert.ok(html.includes('<a href="#">Back to all models</a>'), m)
    assert.ok(html.includes(`<a href="https://huggingface.co/spaces/maxgfr/${m}">This demo in its own Space</a>`), m)
    assert.doesNotMatch(html, /hub-grid/, `${m}: the list is not shown under a demo`)
  }
})

test('a passed and a failed model: the verdict first, then the tagline, the links, the notes and the demo', async () => {
  for (const m of ['s6-route', 's6-micro-v1']) {
    const s = of(m)
    const html = await renderSpace(s.config)
    const notice = gateNotice(s.config)!
    assert.equal(notice.tone, m === 's6-route' ? 'passed' : 'failed')
    // the verdict notice comes right after the title
    assert.ok(html.startsWith(`<header><h1>${m}</h1><p class="gate-notice gate-${notice.tone} space-verdict"><strong>${notice.lead}</strong>`), m)
    const t = unescape(text(html))
    const at = (x: string) => {
      const i = t.indexOf(x)
      assert.ok(i >= 0, `${m}: ${x.slice(0, 60)}`)
      return i
    }
    const order = [notice.lead, s.config.tagline, 'Model card · Every demo · Project site · Results · Technical report · Source (Rust)', ...(s.config.notes ?? []), 'Everything runs in your browser']
    const where = order.map(at)
    assert.deepEqual([...where].sort((a, b) => a - b), where, `${m}: order`)
    assert.ok(html.includes(`href="https://huggingface.co/maxgfr/${m}">Model card</a>`))
  }
  assert.match(await renderSpace(of('s6-route').config), /data-stub="RouterDemo" data-props="\{&quot;prefer&quot;:&quot;s6-route&quot;\}"/)
  assert.match(await renderSpace(of('s6-micro-v1').config), /data-stub="PlayGame"/)
  // a model without a gate gets no notice
  assert.doesNotMatch(await renderSpace(of('s3-pico').config), /gate-notice/)
})

test('the Space of s6-route7-nano opens with its failure and runs the router demo on its own two files', async () => {
  const s = of('s6-route7-nano')
  assert.equal(gateNotice(s.config)!.tone, 'failed')
  assert.deepEqual(s.models, ['s6-route7-nano'])
  assert.match(await renderSpace(s.config), /data-stub="RouterDemo" data-props="\{&quot;prefer&quot;:&quot;s6-route7-nano&quot;\}"/)
})

test('the end-of-turn Spaces run the turn demo on their own file: s6-ear2-nano passed, s6-ear3-nano failed', async () => {
  for (const [m, tone] of [['s6-ear2-nano', 'passed'], ['s6-ear3-nano', 'failed']] as const) {
    const s = of(m)
    assert.equal(s.config.kind, 'turn', m)
    assert.equal(gateNotice(s.config)!.tone, tone, m)
    assert.deepEqual(s.models, [m])
    const html = await renderSpace(s.config)
    assert.match(html, new RegExp(`data-stub="TurnDemo" data-props="\\{&quot;prefer&quot;:&quot;${m}&quot;\\}"`), m)
  }
  // the failed file's Space points to the file to use; the passed one's does not name v3
  assert.match(unescape(text(await renderSpace(of('s6-ear3-nano').config))), /The detector to use is s6-ear2-nano/)
  assert.doesNotMatch(await renderSpace(of('s6-ear2-nano').config), /s6-ear3-nano/)
})

test('the scope restricts the manifest readers to one Space, never a reference forecaster', () => {
  assert.equal(scopeFilter(), null)
  const universe = hubModels(spaces)
  setManifestScope(of('s6-cast-pico').models, universe)
  const ok = scopeFilter()!
  assert.equal(ok('s6-cast-pico'), true)
  assert.equal(ok('s6-cast2-pico'), false)
  assert.equal(ok('TinyCast'), true)
  setManifestScope(of('s6-route-nano').models, universe)
  // a reader keeps the scope it was called with
  assert.equal(ok('s6-route'), false)
  assert.equal(scopeFilter()!('s6-route'), true)
  assert.equal(scopeFilter()!('s6-route-nano'), true)
  assert.equal(scopeFilter()!('s6-nano'), false)
  setManifestScope(null)
  assert.equal(scopeFilter(), null)
})
