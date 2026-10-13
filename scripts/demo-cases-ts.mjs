// Writes site/.vitepress/theme/lib/demo-cases.ts from eval/s5/demo-cases.jsonl, so that the /live
// presets are exactly the cases the s5 models are evaluated on.
//
//   node scripts/demo-cases-ts.mjs
import { readFileSync, writeFileSync } from 'node:fs'

const root = new URL('..', import.meta.url)
const cases = readFileSync(new URL('eval/s5/demo-cases.jsonl', root), 'utf8')
  .split('\n')
  .filter((l) => l.trim())
  .map((l) => {
    const r = JSON.parse(l)
    const gold = r.target.indexOf(Math.max(...r.target))
    return { src: r.src, t: r.t, ins: r.ins, opts: r.opts_kv, state: r.state, gold, lang: r.lang ?? 'en' }
  })

const out = [
  '// Generated from eval/s5/demo-cases.jsonl, the cases every s5 model is evaluated on: the /live',
  '// presets ask exactly these questions, with these options and states. Regenerate it when that file',
  '// changes (node scripts/demo-cases-ts.mjs).',
  '',
  'export interface DemoCase {',
  '  src: string',
  "  t: 'choice' | 'score' | 'noul'",
  '  ins: string',
  '  /** [key, definition] */',
  '  opts: [string, string][]',
  '  state: string',
  '  /** Index of the expected option. */',
  '  gold: number',
  "  lang: 'en' | 'fr'",
  '}',
  '',
  `export const DEMO_CASES: DemoCase[] = ${JSON.stringify(cases, null, 2)}`,
  '',
].join('\n')
writeFileSync(new URL('site/.vitepress/theme/lib/demo-cases.ts', root), out)
console.log(`${cases.length} cases`)
