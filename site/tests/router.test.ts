// Tests of the Router page (s6-route): the tool-list helpers of lib/router.ts against the rules of
// `sky_data::route`, and the call the page makes (`ToolRouter.route` with the manifest's two files)
// on its presets.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
//
// The routing tests drive the browser module (public/pkg, built by scripts/build-wasm.sh) with the
// two model files of the manifest entry; they are skipped without the module. They check that the
// page gets an answer it can draw, not the router's accuracy (runs/s6-nano3/eval-fresh/results.md).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import {
  MAX_TOOLS,
  ROUTER_PRESETS,
  cutDescription,
  maskNames,
  parseTools,
  renderTools,
  tauAskName,
} from '../.vitepress/theme/lib/router.ts'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../public/pkg')
const models = join(here, '../public/models')

test('descriptions are cut as sky_data::route::cut_description cuts them', () => {
  // the cases of the Rust unit test
  assert.equal(cutDescription('  Get the  weather.  It is free. '), 'Get the weather.')
  assert.equal(cutDescription('Version 2.5 of the API'), 'Version 2.5 of the API')
  assert.equal(cutDescription('What? No'), 'What?')
  const long = Array.from({ length: 40 }, (_, i) => `w${i}`).join(' ')
  assert.equal(cutDescription(long).split(' ').length, 30)
})

test('a tool list keeps at most 7 tools, renders name: description, and flags cuts and duplicates', () => {
  const lines = Array.from({ length: 9 }, (_, i) => `t${i}: Does thing ${i}. More text.`).join('\n')
  const l = parseTools(`${lines}\n\n`)
  assert.equal(l.tools.length, MAX_TOOLS)
  assert.equal(l.dropped, 2)
  assert.ok(l.tools.every((t) => t.cut))
  assert.deepEqual(renderTools(l.tools.slice(0, 1)), ['t0: Does thing 0.'])
  assert.deepEqual(renderTools(parseTools('lonely').tools), ['lonely'])
  assert.deepEqual(parseTools('a: x.\na: y.').duplicates, ['a'])
})

test('masking renames every tool tool_<k>, distinct k in 1..99, descriptions kept, reproducibly', () => {
  const tools = parseTools(ROUTER_PRESETS[2].tools).tools
  const m = maskNames(tools)
  assert.equal(new Set(m.map((t) => t.name)).size, tools.length)
  for (const [i, t] of m.entries()) {
    const k = Number(t.name.slice('tool_'.length))
    assert.ok(t.name.startsWith('tool_') && k >= 1 && k <= 99)
    assert.equal(t.desc, tools[i].desc)
  }
  assert.deepEqual(maskNames(tools), m)
})

test('the presets are well formed: at most 7 tools, every intended tool listed, one none request each', () => {
  for (const p of ROUTER_PRESETS) {
    const names = parseTools(p.tools).tools.map((t) => t.name)
    assert.ok(names.length >= 2 && names.length <= MAX_TOOLS, p.id)
    assert.equal(parseTools(p.tools).dropped, 0)
    for (const r of p.requests) assert.ok(r.intended === 'none' || names.includes(r.intended), `${p.id}: ${r.intended}`)
    assert.equal(p.requests.filter((r) => r.intended === 'none').length, 1, p.id)
  }
})

const manifest = JSON.parse(readFileSync(join(models, 'manifest.json'), 'utf8')) as {
  name: string
  task?: string
  file?: string
  files?: { role: string; file: string; bytes: number }[]
}[]
const entry = manifest.find((m) => m.task === 'router')
const hasPkg = existsSync(join(pkg, 'sky_web.js')) && existsSync(join(pkg, 'sky_web_bg.wasm'))
const hasFiles = !!entry?.files?.every((f) => existsSync(join(models, f.file)))

test('the router entry has two files and no single file, so the single-file pickers skip it', () => {
  assert.ok(entry, 'a task: router entry')
  assert.equal(entry.file, undefined)
  assert.deepEqual(entry.files?.map((f) => f.role).sort(), ['embedding', 'model'])
  for (const f of entry.files ?? []) {
    if (existsSync(join(models, f.file))) assert.equal(readFileSync(join(models, f.file)).length, f.bytes, f.file)
  }
})

test('ToolRouter routes every preset request as the page calls it', { skip: !hasPkg || !hasFiles }, async () => {
  const mod = await import(join(pkg, 'sky_web.js'))
  await mod.default({ module_or_path: readFileSync(join(pkg, 'sky_web_bg.wasm')) })
  const file = (role: string) => readFileSync(join(models, entry!.files!.find((f) => f.role === role)!.file))
  const router = new mod.ToolRouter(new Uint8Array(file('model')), new Uint8Array(file('embedding')))
  const tau = router.tauAsk()
  assert.ok(Math.abs(tau - 0.671047) < 1e-6, `tau_ask3 ${tau}`)
  for (const p of ROUTER_PRESETS) {
    for (const masked of [false, true]) {
      const base = parseTools(p.tools).tools
      const tools = renderTools(masked ? maskNames(base) : base)
      for (const [j, r] of p.requests.entries()) {
        for (const fast of [false, true]) {
          const a = JSON.parse(router.route(r.text, JSON.stringify(tools), fast))
          assert.equal(a.probs.length, tools.length + 1)
          assert.ok(Math.abs(a.probs.reduce((s: number, x: number) => s + x, 0) - 1) < 1e-6)
          assert.equal(a.model.length, tools.length + 1)
          assert.equal(a.embedding.length, tools.length + 1)
          assert.ok(['tool', 'none', 'ask'].includes(a.decision))
          assert.equal(a.ask, a.top < tau)
          assert.equal(a.decision === 'none', !a.ask && a.index === tools.length)
          assert.ok(a.micros > 0)
          // the tool list is encoded on its first request and cached on the next ones
          if (j > 0) assert.ok(a.static_hit && a.tools_hit, `${p.id} request ${j} not cached`)
        }
      }
    }
  }
  router.free()
})

/** RouterDemo.vue's template, rendered with the bindings of one answer (no engine, no browser). */
async function renderDemo(name: string, decision: 'tool' | 'none' | 'ask'): Promise<string> {
  const sfc = readFileSync(join(here, '../.vitepress/theme/components/RouterDemo.vue'), 'utf8')
  const start = sfc.indexOf('<template>') + '<template>'.length
  const tpl = sfc.slice(start, sfc.lastIndexOf('</template>', sfc.indexOf('<style')))
  const answer = { decision, answer: 'set_thermostat', top: 0.69, tau_ask3: 0.762097, fast: false, layers_used: null, micros_model: 1, micros_embedding: 1 }
  const demo = {
    template: tpl,
    data: () => ({
      ROUTER_PRESETS, MAX_TOOLS, presetIdx: 0, preset: null, toolsLabel: 'Tools', toolsText: '', request: '',
      parsed: { tools: [], dropped: 0, duplicates: [] }, cutCount: 0, fast: false, masked: false, maskLabel: 'Mask',
      modelOnly: false, entry: { name }, status: '', error: '', answer, modelAnswer: null, rows: [], intended: null,
      shownAnswer: null, modelFile: null, modelSize: '', pairSize: '3.5 MB', cachedMedian: null, cachedUs: [],
      encodedUs: null, clockStep: 0, tau: 0.762097,
    }),
    computed: { tauName(this: { entry: { name: string } }) { return tauAskName(this.entry.name) } },
    methods: {
      pct: (p: number) => `${(p * 100).toFixed(1)}%`, formatMicros: String, formatInt: String, engineLabel: () => '',
      timeOf: () => 1, cachedOf: () => false, pickPreset() {}, pickRequest() {},
    },
  }
  return (await renderToString(createSSRApp(demo))).replace(/<!--[^]*?-->/g, '').replace(/<[^>]+>/g, '').replace(/\s+/g, ' ')
}

test('the demo names the ask threshold of the router shown: τ_ask7 for s6-route7-nano, τ_ask3 for s6-route', async () => {
  for (const decision of ['ask', 'none', 'tool'] as const) {
    const v7 = await renderDemo('s6-route7-nano', decision)
    assert.match(v7, /τ_ask7 = 76\.2%/, decision)
    assert.match(v7, /below τ_ask7 = 0\.762097 \(fixed on validation\)/)
    assert.doesNotMatch(v7, /τ_ask3/, decision)
    const v3 = await renderDemo('s6-route', decision)
    assert.match(v3, /τ_ask3 = 76\.2%/, decision)
    assert.doesNotMatch(v3, /τ_ask7/, decision)
  }
  assert.match(await renderDemo('s6-route7-nano', 'ask'), /Ask the user\.\s*The top option, set_thermostat, has 69\.0%, below τ_ask7 = 76\.2%\./)
})
