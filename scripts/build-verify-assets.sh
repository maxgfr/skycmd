#!/usr/bin/env bash
# The engine assets of s6-secrets-verify-pico for third parties, written to target/verify-assets/
# (not committed; uploaded under engine/ of the model's Hugging Face repository):
#   sky_web_bg.wasm           SIMD128 build of crates/sky-web (scripts/build-wasm.sh, release)
#   sky_web_relaxed_bg.wasm   the same with relaxed SIMD
#   sky_web.js                the wasm-bindgen glue shared by both (optional: verify-host.mjs
#                             needs no glue)
#   verify-host.mjs           crates/sky-web/js/verify-host.mjs, the glue-free host
#   dictionary.txt            the word list of the state's word feature (the build's)
#   s6-secrets-verify-pico.gguf   the model, as evaluated
#   MANIFEST.json             bytes and sha256 of each, the commit and the toolchain
#
# Usage: scripts/build-verify-assets.sh DICTIONARY
#   DICTIONARY: the build's dictionary.txt (<data>/s6-secrets2/build/dictionary.txt); the script
#   refuses any other file, as it refuses another model file, and a working tree with changes
#   (the manifest names the commit the files are built from).
# The paths of the build machine are remapped out of the modules (panic locations). After the
# build, crates/sky-web/bench/verify-node.mjs checks both modules through the host and the glue.
set -euo pipefail
cd "$(dirname "$0")/.."

dict=${1:?usage: scripts/build-verify-assets.sh path/to/dictionary.txt}
model=site/public/models/s6-secrets-verify-pico.gguf
dict_sha=83f3f32b0df5fc92418dac25b8d1317ec537c0d4c6405961c9c06f31159c21e9
model_sha=01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
[[ "$(sha "$dict")" == "$dict_sha" ]] || { echo "$dict: not the build's dictionary.txt (sha256 $dict_sha)" >&2; exit 1; }
[[ "$(sha "$model")" == "$model_sha" ]] || { echo "$model: not the evaluated model (sha256 $model_sha)" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "the working tree has changes: commit them first" >&2; exit 1; }
commit=$(git rev-parse HEAD)

out=$PWD/target/verify-assets
rm -rf "$out"
mkdir -p "$out"
# a target dir of its own: the remapped flags would rebuild the site's every time
CARGO_TARGET_DIR=$PWD/target/verify-assets-build \
  RUSTFLAGS="--remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$HOME/.rustup=/rustup --remap-path-prefix=$PWD=/skycmd" \
  SKY_WASM_OUT=$out scripts/build-wasm.sh
cp crates/sky-web/js/verify-host.mjs "$out/"
cp "$dict" "$out/dictionary.txt"
cp "$model" "$out/"

node crates/sky-web/bench/verify-node.mjs --assets "$out" --model "$out/s6-secrets-verify-pico.gguf" --glue

python3 -I - "$out" "$commit" "$(rustc --version)" "$(wasm-pack --version)" "$(wasm-opt --version 2>/dev/null || echo unknown)" <<'PY'
import hashlib, json, os, sys
out, commit, rustc, wasm_pack, wasm_opt = sys.argv[1:]
what = {
    "sky_web_bg.wasm": "the engine, SIMD128 (every runtime with WebAssembly SIMD)",
    "sky_web_relaxed_bg.wasm": "the same engine with relaxed SIMD (Node 21+, Chrome 114+): same answers within 1e-5",
    "sky_web.js": "the wasm-bindgen glue of both modules (ES module, --target web); not needed with verify-host.mjs",
    "verify-host.mjs": "a host without the glue: instantiates either module, allows its five imports only",
    "dictionary.txt": "the word list of the state's `word` feature: the entries of /usr/share/dict/words (macOS) written in a to z, 4 to 12 letters",
    "s6-secrets-verify-pico.gguf": "the model as evaluated (full depth, the three operating thresholds in its metadata)",
}
files = {}
for name in what:
    with open(os.path.join(out, name), "rb") as f:
        data = f.read()
    files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "what": what[name]}
manifest = {
    "model": "s6-secrets-verify-pico",
    "source": "https://github.com/maxgfr/skycmd",
    "commit": commit,
    "toolchain": {"rustc": rustc, "wasm-pack": wasm_pack, "wasm-opt": wasm_opt},
    "contract": "crates/sky-web/src/verify.rs (the candidate JSON) and crates/sky-web/js/verify-host.mjs, at the commit",
    "files": files,
}
with open(os.path.join(out, "MANIFEST.json"), "w") as f:
    f.write(json.dumps(manifest, indent=2) + "\n")
PY
ls -l "$out"
