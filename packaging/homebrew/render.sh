#!/usr/bin/env sh
# Fill packaging/homebrew/skycmd.rb from a release's SHA256SUMS and print the formula.
# Usage: sh packaging/homebrew/render.sh <version> <SHA256SUMS>
#   e.g. sh packaging/homebrew/render.sh 0.1.0 SHA256SUMS > Formula/skycmd.rb
set -eu
version=${1:?version, e.g. 0.1.0}
sums=${2:?path to SHA256SUMS}
template="$(dirname "$0")/skycmd.rb"

script="s/@VERSION@/$version/g"
for asset in \
  "skycmd-$version-aarch64-apple-darwin.tar.gz" \
  "skycmd-$version-x86_64-apple-darwin.tar.gz" \
  "skycmd-$version-aarch64-unknown-linux-gnu.tar.gz" \
  "skycmd-$version-x86_64-unknown-linux-gnu.tar.gz" \
  "s1-pico.gguf"; do
  sha=$(awk -v f="$asset" '$2 == f || $2 == "*" f { print $1 }' "$sums")
  [ -n "$sha" ] || { echo "render.sh: $asset is not in $sums" >&2; exit 1; }
  key=${asset#"skycmd-$version-"}
  key=${key%.tar.gz}
  script="$script;s/@SHA256_$key@/$sha/g"
done

# drop the template header (everything before `class`)
sed -e "$script" "$template" | sed -n '/^class /,$p'
