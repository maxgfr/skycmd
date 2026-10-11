// WASM product kernels in Node (sky-infer built with the `wasm-bench` feature):
//
//   RUSTFLAGS="-C target-feature=+simd128" wasm-pack build crates/sky-infer --target nodejs \
//     --release --out-dir <pkg> --no-typescript -- --features wasm-bench
//   node crates/sky-infer/bench/kernels-node.mjs --pkg <pkg>
//
// Prints a markdown table: µs per product of f32 panels, int8 x ternary (relaxed-simd builds
// only), lookup tables with one / two activation passes, and the packed add/sub kernel, on the
// matrix shapes of the s3 ladder with a 9-token state (and 2 option markers).
import { resolve } from 'node:path'
import { createRequire } from 'node:module'

const args = process.argv.slice(2)
const pkg = args[args.indexOf('--pkg') + 1]
if (!pkg || args.indexOf('--pkg') < 0) {
  console.error('usage: kernels-node.mjs --pkg <wasm-pack nodejs dir built with --features wasm-bench>')
  process.exit(2)
}
const { kernelBench } = createRequire(import.meta.url)(resolve(pkg, 'sky_infer.js'))
const shapes = [
  ['s3-pico', 288, 96, 9], ['s3-pico', 384, 96, 9], ['s3-pico', 96, 192, 9], ['s3-pico', 192, 96, 2],
  ['s3-nano', 384, 128, 9], ['s3-nano', 512, 128, 9], ['s3-nano', 128, 256, 9], ['s3-nano', 256, 128, 2],
]
const kinds = ['f32', 'int8', 'lut1', 'lut2', 'packed']
// each call also builds the matrix: the time of a call without products is subtracted
const time = (kind, n, k, rows) => {
  const iters = Math.max(400, Math.round(4e7 / (n * k * rows)))
  const run = (m) => {
    let best = Infinity
    for (let r = 0; r < 9; r++) {
      const t0 = performance.now()
      kernelBench(kind, n, k, rows, m)
      best = Math.min(best, (performance.now() - t0) * 1000)
    }
    return best
  }
  run(iters) // warm-up
  return (run(iters) - run(0)) / iters
}
console.log(`| model | shape (n x k) | rows | ${kinds.join(' (µs) | ')} (µs) |`)
console.log(`|---|---|---:|${kinds.map(() => '---:').join('|')}|`)
for (const [name, n, k, rows] of shapes) {
  const cells = kinds.map((kind) => time(kind, n, k, rows).toFixed(2))
  console.log(`| ${name} | ${n} x ${k} | ${rows} | ${cells.join(' | ')} |`)
}
