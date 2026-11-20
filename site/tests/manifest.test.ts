// Tests of public/models/manifest.json against the committed model files: every entry names its
// files, and each file's size and sha256 are those of the file on disk, so a page, a card or the
// technical report can point to the manifest for the hash of what was measured.
//
//   cd site && npm test
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const models = join(here, '..', 'public', 'models')

interface File {
  file: string
  bytes: number
  sha256?: string
}
interface Entry extends Partial<File> {
  name: string
  files?: File[]
}

const manifest: Entry[] = JSON.parse(readFileSync(join(models, 'manifest.json'), 'utf8'))

/** The files of an entry: its one `file`, or each of its `files` (two-file entries). */
function filesOf(e: Entry): File[] {
  if (e.files) return e.files
  return e.file ? [{ file: e.file, bytes: e.bytes!, sha256: e.sha256 }] : []
}

test('every entry names at least one file', () => {
  for (const e of manifest) assert.ok(filesOf(e).length > 0, `${e.name}: no file and no files`)
})

test('every file of every entry has a sha256 equal to the committed file', () => {
  for (const e of manifest) {
    for (const f of filesOf(e)) {
      const path = join(models, f.file)
      assert.ok(existsSync(path), `${e.name}: ${f.file} is not in public/models/`)
      assert.match(f.sha256 ?? '', /^[0-9a-f]{64}$/, `${e.name}: ${f.file} has no sha256`)
      const bytes = readFileSync(path)
      assert.equal(bytes.length, f.bytes, `${e.name}: ${f.file} is not ${f.bytes} bytes`)
      const sha = createHash('sha256').update(bytes).digest('hex')
      assert.equal(sha, f.sha256, `${e.name}: the sha256 of ${f.file} is not the manifest's`)
    }
  }
})
