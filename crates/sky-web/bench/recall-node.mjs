// The `recall` memory probe in Node, outside the browser: loads the browser module (site/public/pkg,
// built by scripts/build-wasm.sh) and plays recall episodes with `RecallProbe` (crates/sky-web/src/
// recall.rs: the evaluation's episode generator and scoring, the model at its last exit), as the
// Memory page does. Prints one JSON line per model and memory mode: accuracy on the ask frames,
// overall and per delay bucket (1-4, 5-8, 9-16, 17-32 frames), with the asks counted, and the
// median time per decision.
//
//   node crates/sky-web/bench/recall-node.mjs [--episodes 200] [--from 7000000018] [--passes 1]
//        [--zeroed] [--relaxed] model.gguf...
//
// The episode seeds are the first ones ≡ 18 (mod 20) from `--from` upward (the validation split of
// `sky_games::gen::split_for_seed`, never trained on); the pre-registered test read 1,000 episodes
// from 5,000,000,000 (≡ 19 mod 20) with the training checkpoint, so these numbers differ from it
// by sampling noise and by the engine. `--zeroed` also plays each memory model with its memory
// reset before every frame.
import { readFileSync } from 'node:fs'
import { dirname, join, basename } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../../../site/public/pkg')

const args = process.argv.slice(2)
let episodes = 200
let from = 7_000_000_018
let passes = 1
let zeroed = false
let relaxed = false
const models = []
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--episodes') episodes = Number(args[++i])
  else if (a === '--from') from = Number(args[++i])
  else if (a === '--passes') passes = Number(args[++i])
  else if (a === '--zeroed') zeroed = true
  else if (a === '--relaxed') relaxed = true
  else models.push(a)
}
if (!models.length) {
  console.error('usage: recall-node.mjs [--episodes N] [--from SEED] [--passes N] [--zeroed] [--relaxed] model.gguf...')
  process.exit(2)
}

const mod = await import(join(pkg, 'sky_web.js'))
await mod.default({ module_or_path: readFileSync(join(pkg, relaxed ? 'sky_web_relaxed_bg.wasm' : 'sky_web_bg.wasm')) })

/** The first `n` seeds ≡ 18 (mod 20) from `start` upward. */
function seeds(start, n) {
  const first = start + ((18 - (start % 20)) + 20) % 20
  return Array.from({ length: n }, (_, k) => first + 20 * k)
}

function play(probe, label, list) {
  const right = [0, 0, 0, 0]
  const asked = [0, 0, 0, 0]
  const times = []
  for (const seed of list) {
    const r = JSON.parse(probe.runEpisode(seed))
    for (const a of r.asks) {
      asked[a.bucket] += 1
      right[a.bucket] += a.correct ? 1 : 0
    }
    times.push(r.micros / r.decisions)
  }
  const all = asked.reduce((x, y) => x + y, 0)
  times.sort((x, y) => x - y)
  return {
    ...label,
    episodes: list.length,
    from: list[0],
    asks: all,
    accuracy: all ? right.reduce((x, y) => x + y, 0) / all : null,
    buckets: asked.map((n, b) => (n ? right[b] / n : null)),
    bucket_asks: asked,
    median_us_per_decision: times[Math.floor((times.length - 1) / 2)],
  }
}

const list = seeds(from, episodes)
for (const path of models) {
  const probe = new mod.RecallProbe(new Uint8Array(readFileSync(path)))
  const model = basename(path)
  if (probe.hasMemory()) {
    probe.setSleepPasses(passes)
    probe.setMemory('on')
    console.log(JSON.stringify(play(probe, { model, memory: 'on', passes }, list)))
    if (zeroed) {
      probe.setMemory('zeroed')
      console.log(JSON.stringify(play(probe, { model, memory: 'zeroed', passes }, list)))
    }
  } else {
    console.log(JSON.stringify(play(probe, { model, memory: 'none' }, list)))
  }
  probe.free()
}
