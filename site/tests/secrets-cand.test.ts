// Tests of the secrets verifier demo's simplified proposer (lib/secrets-cand.ts): each form it
// reads, the 8-character rule, UTF-16 columns, no candidate on prose, the fields the engine
// requires, and the presets. The last test drives the browser module (public/pkg, built by
// scripts/build-wasm.sh) with the manifest's model and word list, as the page does; it is skipped
// without the module. It checks that the engine accepts the records, not the model's accuracy.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  MIN_VALUE_CHARS,
  RULE_ID,
  SECRETS_PRESETS,
  isCredentialName,
  proposeCandidates,
  splitLines,
} from '../.vitepress/theme/lib/secrets-cand.ts'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../public/pkg')
const models = join(here, '../public/models')

/** The value of the only candidate of a one-line text. */
function only(text: string) {
  const c = proposeCandidates(text)
  assert.equal(c.length, 1, `${text}: ${c.length} candidates`)
  return c[0]
}

test('each form: NAME=value, NAME: value, "name": "value", name = "value"', () => {
  const env = only('DB_PASSWORD=example-not-a-real-password-0000')
  assert.equal(env.name, 'DB_PASSWORD')
  assert.equal(env.value, 'example-not-a-real-password-0000')

  const yaml = only('  auth_token: placeholder-value-0000')
  assert.equal(yaml.name, 'auth_token')
  assert.equal(yaml.value, 'placeholder-value-0000')

  const json = only('  "password": "changeme-example-only",')
  assert.equal(json.name, 'password')
  assert.equal(json.value, 'changeme-example-only')

  const code = only('SECRET_KEY = "fake-example-secret-key-for-docs-0000"')
  assert.equal(code.name, 'SECRET_KEY')
  assert.equal(code.value, 'fake-example-secret-key-for-docs-0000')

  const single = only("api_token = 'FAKE_EXAMPLE_TOKEN_1234567890'")
  assert.equal(single.value, 'FAKE_EXAMPLE_TOKEN_1234567890')
})

test('the columns point at the value in the line', () => {
  for (const line of [
    'DB_PASSWORD=example-not-a-real-password-0000',
    '  "password": "changeme-example-only",',
    'SECRET_KEY = "fake-example-secret-key-for-docs-0000"',
    'export API_KEY=your_api_key_here # comment',
  ]) {
    const c = only(line)
    assert.equal(line.slice(c.start, c.end), c.value, line)
    assert.equal(c.lineText, line)
    assert.equal(c.lineFrom, 0)
  }
})

test('a name proposes only with a credential word', () => {
  for (const n of ['DB_PASSWORD', 'passwd', 'clientSecret', 'AUTH', 'api-key', 'apiKey', 'API_KEY', 'credentials', 'private_key', 'privateKey', 'x.token'])
    assert.ok(isCredentialName(n), n)
  for (const n of ['DB_HOST', 'username', 'api', 'key', 'private', 'timeout', 'service']) assert.ok(!isCredentialName(n), n)
  assert.deepEqual(proposeCandidates('DB_HOST=example-host-name-0000\nUSERNAME=example_user_0000'), [])
})

test(`values shorter than ${MIN_VALUE_CHARS} characters are skipped`, () => {
  assert.deepEqual(proposeCandidates('PASSWORD=1234567'), [])
  assert.equal(proposeCandidates('PASSWORD=12345678').length, 1)
  assert.deepEqual(proposeCandidates('"token": ""'), [])
  // characters, not UTF-16 units: seven characters, one of them outside the BMP
  assert.deepEqual(proposeCandidates('PASSWORD=abcdef\u{1F600}'), [])
})

test('the columns are UTF-16 units on a line with non-ASCII characters', () => {
  const line = 'mot_de_passe_é = "déjà-vu-exemple-0000" # ✓ \u{1F600}'
  // the name has no credential word in English: the line proposes nothing
  assert.deepEqual(proposeCandidates(line), [])
  const line2 = '# café \u{1F600} PASSWORD="exemple-déjà-0000"'
  const c = only(line2)
  assert.equal(c.value, 'exemple-déjà-0000')
  // '# café ' is 7 UTF-16 units, the emoji 2 and the space 1: the name starts at 10
  assert.equal(line2.indexOf('PASSWORD'), 10)
  assert.equal(c.start, 10 + 'PASSWORD="'.length)
  assert.equal(c.end, c.start + 'exemple-déjà-0000'.length)
  assert.equal(line2.slice(c.start, c.end), c.value)
})

test('no candidate on plain prose', () => {
  const prose = [
    'The password policy asks for twelve characters or more.',
    'Rotate every token after an incident: see the runbook.',
    'Contact the security team at the usual address.',
  ].join('\n')
  assert.deepEqual(proposeCandidates(prose), [])
})

test('the record has the fields the engine requires, with the lines around the value', () => {
  const text = 'first line\nAPI_KEY=your_api_key_here\r\nlast line'
  assert.deepEqual(splitLines(text), ['first line', 'API_KEY=your_api_key_here', 'last line'])
  const [c] = proposeCandidates(text, ' .env ')
  assert.equal(c.rule, RULE_ID)
  assert.equal(typeof c.value, 'string')
  assert.equal(c.file, '.env')
  assert.equal(c.line, 2)
  assert.equal(c.endLine, 2)
  assert.equal(c.prevLine, 'first line')
  assert.equal(c.nextLine, 'last line')
  for (const k of ['start', 'end', 'lineFrom', 'line', 'endLine'] as const) assert.ok(Number.isInteger(c[k]) && c[k] >= 0, k)
  assert.deepEqual(c.filters, { entropy: false, placeholder: false, stopword: false, allowlist: false })
  // no file, no neighbours: the fields are absent, not empty
  const [d] = proposeCandidates('API_KEY=your_api_key_here', '')
  assert.equal('file' in d, false)
  assert.equal('prevLine' in d, false)
  assert.equal('nextLine' in d, false)
})

test('every preset proposes candidates, and every preset value is built from placeholder words', () => {
  for (const p of SECRETS_PRESETS) {
    const c = proposeCandidates(p.text, p.file)
    assert.ok(c.length >= 2, `${p.id}: ${c.length}`)
    for (const r of c) {
      assert.match(r.value, /example|fake|changeme|your_|os\.environ/i, `${p.id}: ${r.value}`)
      assert.equal(r.file, p.file)
    }
  }
})

const manifest = JSON.parse(readFileSync(join(models, 'manifest.json'), 'utf8')) as {
  name: string
  task?: string
  file?: string
  dictionary?: string
}[]
const entry = manifest.find((m) => m.task === 'secrets-verify')
const hasPkg = existsSync(join(pkg, 'sky_web.js')) && existsSync(join(pkg, 'sky_web_bg.wasm'))
const hasFiles = !!entry?.file && !!entry.dictionary && existsSync(join(models, entry.file)) && existsSync(join(models, entry.dictionary))

test('the verifier entry names its word list', () => {
  assert.ok(entry, 'a task: secrets-verify entry')
  assert.equal(entry.dictionary, 's6-secrets-verify-dictionary.txt')
  if (existsSync(join(models, entry.dictionary!))) assert.equal(readFileSync(join(models, entry.dictionary!)).length, 1706770)
})

test('SecretVerifier judges every preset candidate as the page sends it', { skip: !hasPkg || !hasFiles }, async () => {
  const mod = await import(join(pkg, 'sky_web.js'))
  await mod.default({ module_or_path: readFileSync(join(pkg, 'sky_web_bg.wasm')) })
  const v = new mod.SecretVerifier(
    new Uint8Array(readFileSync(join(models, entry!.file!))),
    new Uint8Array(readFileSync(join(models, entry!.dictionary!))),
  )
  const tau = JSON.parse(v.thresholds())
  assert.ok(Math.abs(tau.tau_f1 - 0.2598) < 1e-4, `tau_f1 ${tau.tau_f1}`)
  assert.ok(Math.abs(tau.tau_f05 - 0.468437) < 1e-6, `tau_f05 ${tau.tau_f05}`)
  assert.ok(Math.abs(tau.tau_f2 - 0.0780795) < 1e-7, `tau_f2 ${tau.tau_f2}`)
  for (const p of SECRETS_PRESETS) {
    const cands = proposeCandidates(p.text, p.file)
    const answers = JSON.parse(v.judgeMany(JSON.stringify(cands)))
    assert.equal(answers.length, cands.length)
    for (const [i, a] of answers.entries()) {
      assert.ok(a.p_real >= 0 && a.p_real <= 1, `${p.id} ${i}: ${a.p_real}`)
      assert.equal(typeof a.truncated, 'boolean')
      const state = v.state(JSON.stringify(cands[i]))
      assert.ok(state.startsWith(`rule ${RULE_ID}\n`), state)
      assert.ok(state.includes(`\nvalue ${cands[i].value}\n`), state)
    }
  }
  v.free()
})
