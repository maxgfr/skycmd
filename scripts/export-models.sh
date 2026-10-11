#!/usr/bin/env bash
# Convert every trained model in models/<name>/ (Laya layout, written by `sky-train export`) to
# site/public/models/<name>.gguf, then write site/public/models/manifest.json and
# site/public/bench.json for the website.
#
# Encoder matrices are stored in f16 and the decision head in f32 (the sky-convert defaults):
# the golden tests of sky-infer measure 1e-3 probability error for f16 files against 3e-2 for
# Q8_0 files, and the browser engine computes in f32 anyway, so Q8_0 would only save a few
# kilobytes on these tiny models at a visible accuracy cost.
#
# bench.json joins, per model: params and file size (from the conversion), held-out metrics
# (from runs/<name>/report.md, `all` / `games` rows) and the native latency of `sky-bench`.
# Values that are not available yet are written as null and shown as "TBD" by the site.
#
# Usage: scripts/export-models.sh [--type f16|q8_0|f32]
set -euo pipefail
cd "$(dirname "$0")/.."

ftype=f16
if [[ "${1:-}" == "--type" ]]; then ftype=${2:?--type needs a value}; fi

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-web-tools}
cargo build -q --release -p sky-infer --bin sky-convert --bin sky-bench
bin=$CARGO_TARGET_DIR/release

out=site/public/models
mkdir -p "$out"
rows=()
shopt -s nullglob
for dir in models/*/; do
  name=$(basename "$dir")
  [[ -f "$dir/model.safetensors" ]] || { echo "skip $dir (no model.safetensors)"; continue; }
  gguf=$out/$name.gguf
  log=$("$bin/sky-convert" "$dir" "$gguf" --type "$ftype" --name "skycmd-$name")
  echo "$log"
  params=$(sed -nE 's/.* ([0-9]+) parameters.*/\1/p' <<<"$log")
  bytes=$(wc -c <"$gguf" | tr -d ' ')
  # Native latency (median / p95 per decision) on a flappy-like state.
  bench=$("$bin/sky-bench" "$gguf" --iters 300 2>&1 || true)
  med=$(sed -nE 's/.*median[^0-9]*([0-9.]+).*/\1/p' <<<"$bench" | head -1)
  rows+=("$name|$params|$bytes|${med:-}")
done

if [[ ${#rows[@]} -eq 0 ]]; then
  echo "no trained model under models/: the site falls back to the oracle policy"
fi

printf '%s\n' "${rows[@]+"${rows[@]}"}" | node --input-type=module -e '
import { readFileSync, writeFileSync, existsSync } from "node:fs";
const rows = readFileSync(0, "utf8").split("\n").filter(Boolean).map((l) => {
  const [name, params, bytes, nativeUs] = l.split("|");
  return { name, params: Number(params), bytes: Number(bytes), nativeUs: nativeUs ? Number(nativeUs) : null };
});
const order = { "s1-pico": 0, "s1-nano": 1, "s1-micro": 2 };
rows.sort((a, b) => (order[a.name] ?? 9) - (order[b.name] ?? 9) || a.name.localeCompare(b.name));

// Parse the `all` / `games` / per-game rows of the first tables of runs/<name>/report.md.
function report(name) {
  const path = `runs/${name}/report.md`;
  if (!existsSync(path)) return {};
  const metrics = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const m = line.match(/^\|\s*([\w:]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|/);
    if (!m || metrics[m[1]]) continue; // keep the first (calibrated) occurrence
    metrics[m[1]] = { n: +m[2], accuracy: +m[3], nll: +m[4], brier: +m[5], ece: +m[6] };
  }
  return metrics;
}

const manifest = rows.map(({ name, params, bytes }) => ({ name, file: `${name}.gguf`, params, bytes }));
writeFileSync("site/public/models/manifest.json", JSON.stringify(manifest, null, 2) + "\n");

const bench = {
  generated: new Date().toISOString().slice(0, 10),
  note: "accuracy/NLL/ECE: held-out test split (runs/<model>/report.md); native_us: sky-bench median per decision on the build machine; null = not measured yet",
  models: rows.map(({ name, params, bytes, nativeUs }) => {
    const r = report(name);
    const pick = (k, f) => (r[k] ? r[k][f] : null);
    return {
      name, params, bytes,
      accuracy: pick("all", "accuracy"),
      games_accuracy: pick("games", "accuracy"),
      flappy_accuracy: pick("game:flappy", "accuracy"),
      rescue_accuracy: pick("game:rescue", "accuracy"),
      dodge_accuracy: pick("game:dodge", "accuracy"),
      nll: pick("all", "nll"),
      ece: pick("all", "ece"),
      native_us: nativeUs,
      wasm_us: null,
    };
  }),
};
writeFileSync("site/public/bench.json", JSON.stringify(bench, null, 2) + "\n");
console.log(`wrote site/public/models/manifest.json (${manifest.length} models) and site/public/bench.json`);
'
