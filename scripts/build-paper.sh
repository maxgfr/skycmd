#!/usr/bin/env bash
# Build the PDF of the technical report from its LaTeX source.
#
#   scripts/build-paper.sh            # target/paper/skycmd.pdf, copied to site/public/paper/
#   scripts/build-paper.sh --arxiv    # the same, and target/paper/skycmd-arxiv.tar.gz
#
# The source is paper/skycmd.tex, with paper/references.bib and paper/figure-1.tex; the figure is
# written first by scripts/paper-figures.py from the numbers that also draw the SVG of the web
# page. Tectonic (https://tectonic-typesetting.github.io) compiles it and runs BibTeX itself. The
# PDF is copied to site/public/paper/skycmd-technical-report.pdf, the path every link uses.
#
# --arxiv also packs what arXiv needs to compile the report with pdfLaTeX: the .tex, the .bib, the
# .bbl that BibTeX wrote (arXiv does not run BibTeX) and the figure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/target/paper"
PDF="$ROOT/site/public/paper/skycmd-technical-report.pdf"

arxiv=0
for arg in "$@"; do
  case "$arg" in
    --arxiv) arxiv=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (expected --arxiv)" >&2; exit 2 ;;
  esac
done

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found: install it (brew install tectonic, or see its site)" >&2
  exit 1
fi

python3 -I "$ROOT/scripts/paper-figures.py"

mkdir -p "$OUT"
# The same source gives the same PDF: its dates are those of the last commit of the source.
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
  SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --format=%ct -- paper 2>/dev/null || true)"
  export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(date +%s)}"
fi
tectonic --keep-intermediates --keep-logs --outdir "$OUT" "$ROOT/paper/skycmd.tex"

# Fail on what a reader would see as a defect: an undefined reference or citation, a missing
# glyph, or a line that runs into the margin by more than 10pt.
log="$OUT/skycmd.log"
if grep -E "undefined|Missing character" "$log" >/dev/null; then
  grep -E "undefined|Missing character" "$log" >&2
  echo "the log of $log reports undefined references or missing characters" >&2
  exit 1
fi
wide="$(grep -E '^Overfull \\hbox \(([1-9][0-9]+(\.[0-9]+)?)pt too wide' "$log" || true)"
if [ -n "$wide" ]; then
  echo "$wide" >&2
  echo "lines wider than the text by more than 10pt, see $log" >&2
  exit 1
fi

cp "$OUT/skycmd.pdf" "$PDF"
echo "wrote ${PDF#"$ROOT"/}"

if [ "$arxiv" -eq 1 ]; then
  stage="$OUT/arxiv"
  rm -rf "$stage"
  mkdir -p "$stage"
  cp "$ROOT/paper/skycmd.tex" "$ROOT/paper/references.bib" "$ROOT/paper/figure-1.tex" \
    "$OUT/skycmd.bbl" "$stage/"
  tar -czf "$OUT/skycmd-arxiv.tar.gz" -C "$stage" skycmd.tex skycmd.bbl references.bib figure-1.tex
  echo "wrote ${OUT#"$ROOT"/}/skycmd-arxiv.tar.gz"
fi
