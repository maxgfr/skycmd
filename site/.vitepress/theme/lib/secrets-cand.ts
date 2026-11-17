// A SIMPLIFIED candidate proposer for the secrets verifier demo (s6-secrets-verify-pico): it turns
// a pasted text into candidate records in the shape the engine accepts (`sg-cand-1`, read by
// crates/sky-web/src/verify.rs). It is NOT secretgate's rule engine: the real generator is
// secretgate's (`scripts/rule-candidates.mjs`, frozen as sg-cand-1), with many rules and the
// default engine's filters. This one has a single rule, so that the model's judgement can be seen
// in a browser: a value of at least 8 characters assigned to a name that contains a credential
// word, in the forms `NAME=value`, `NAME: value`, `"name": "value"` and `name = "value"`.
// Pure functions, so that tests/secrets-cand.test.ts checks them without a browser.

/** The rule id of every candidate this proposer makes (a rule id of the generator). */
export const RULE_ID = 'generic-keyword'

/** A value shorter than this (in characters) is not proposed. */
export const MIN_VALUE_CHARS = 8

/** The credential words: a name proposes its value when it contains one of them. */
export const CREDENTIAL_WORDS = [
  'password',
  'passwd',
  'secret',
  'token',
  'api key',
  'apikey',
  'auth',
  'credential',
  'private key',
] as const

/** The filters of secretgate's default engine, which this proposer does not run: every flag is
 * sent as `false` (the engine's default). */
export interface CandidateFilters {
  entropy: boolean
  placeholder: boolean
  stopword: boolean
  allowlist: boolean
}

/** One candidate record, in the fields of `sg-cand-1` that the engine reads. Columns are in
 * UTF-16 code units (JavaScript string indices) of the full line; lines count from 1. */
export interface CandidateRecord {
  rule: string
  value: string
  /** The file's path relative to its repository root (absent: no path). */
  file?: string
  name: string
  lineText: string
  start: number
  end: number
  lineFrom: number
  line: number
  endLine: number
  prevLine?: string
  nextLine?: string
  filters: CandidateFilters
}

/** Whether a variable or key name contains a credential word: compared in lower case, with `_`,
 * `-`, `.` and spaces read as one space (`DB_PASSWORD`, `api-key`) and also with them removed
 * (`apiKey`, `privateKey`). */
export function isCredentialName(name: string): boolean {
  const spaced = name.toLowerCase().replace(/[\s_.-]+/g, ' ')
  const compact = spaced.replace(/ /g, '')
  return CREDENTIAL_WORDS.some((w) => spaced.includes(w) || compact.includes(w.replace(/ /g, '')))
}

// A name (optionally in double or single quotes), `=` or `:`, then a value in double quotes, in
// single quotes, or bare (up to a space, a quote, `,`, `;`, a bracket or a parenthesis). The
// name must not follow a word character, so that `x.password` or `my-token` start at their own
// first character. The `d` flag gives the value's columns.
const ASSIGNMENT =
  /(?<![\w.\-"'])(["']?)([A-Za-z_][\w.\-]*)\1[ \t]*[:=][ \t]*(?:"([^"\n]*)"|'([^'\n]*)'|([^\s"'`,;()[\]{}]+))/dg

/** The lines of a text, without their line endings (`\n` or `\r\n`). */
export function splitLines(text: string): string[] {
  return text.split('\n').map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l))
}

/** The candidates of a text, in reading order. `file`: the file's path relative to its
 * repository root (empty or absent: no path). */
export function proposeCandidates(text: string, file?: string): CandidateRecord[] {
  const lines = splitLines(text)
  const out: CandidateRecord[] = []
  for (let i = 0; i < lines.length; i++) {
    const lineText = lines[i]
    for (const m of lineText.matchAll(ASSIGNMENT)) {
      const name = m[2]
      if (!isCredentialName(name)) continue
      const group = m[3] !== undefined ? 3 : m[4] !== undefined ? 4 : 5
      const value = m[group]!
      if (Array.from(value).length < MIN_VALUE_CHARS) continue
      const [start, end] = m.indices![group]!
      const rec: CandidateRecord = {
        rule: RULE_ID,
        value,
        name,
        lineText,
        start,
        end,
        lineFrom: 0,
        line: i + 1,
        endLine: i + 1,
        filters: { entropy: false, placeholder: false, stopword: false, allowlist: false },
      }
      if (file && file.trim()) rec.file = file.trim()
      if (i > 0) rec.prevLine = lines[i - 1]
      if (i + 1 < lines.length) rec.nextLine = lines[i + 1]
      out.push(rec)
    }
  }
  return out
}

export interface SecretsPreset {
  id: string
  label: string
  file: string
  text: string
}

/** Presets written for this demo. Every value is fake, built from placeholder words (example,
 * fake, changeme, your ... here) and digits; none follows a real provider's key format, and none
 * comes from a dataset. */
export const SECRETS_PRESETS: SecretsPreset[] = [
  {
    id: 'env',
    label: '.env file',
    file: '.env',
    text: [
      '# local development settings (example values only)',
      'APP_NAME=example-shop',
      'DB_HOST=localhost',
      'DB_USER=example_user',
      'DB_PASSWORD=example-not-a-real-password-0000',
      'API_KEY=your_api_key_here',
      'SESSION_SECRET=FAKE0EXAMPLE1234FAKE5678EXAMPLE90',
    ].join('\n'),
  },
  {
    id: 'json',
    label: 'JSON config',
    file: 'config/settings.json',
    text: [
      '{',
      '  "service": "example-api",',
      '  "timeout": 30,',
      '  "auth_token": "EXAMPLE-0000-FAKE-1111-EXAMPLE-2222",',
      '  "password": "changeme-example-only"',
      '}',
    ].join('\n'),
  },
  {
    id: 'python',
    label: 'Python settings',
    file: 'app/settings.py',
    text: [
      'import os',
      '',
      'DEBUG = True',
      'SECRET_KEY = "fake-example-secret-key-for-docs-0000"',
      'DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD")',
      "api_token = 'FAKE_EXAMPLE_TOKEN_1234567890'",
    ].join('\n'),
  },
]
