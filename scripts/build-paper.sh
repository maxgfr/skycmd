#!/usr/bin/env bash
# Build the PDF of a technical report from its LaTeX source.
#
#   scripts/build-paper.sh                     # target/paper/skycmd.pdf, copied to site/public/paper/
#   scripts/build-paper.sh --arxiv             # the same, and target/paper/skycmd-arxiv.tar.gz
#   scripts/build-paper.sh --methods [--arxiv] # the methods paper instead (paper/methods/)
#
# The first report's source is paper/skycmd.tex, with paper/references.bib and paper/figure-1.tex;
# the figure is written first by scripts/paper-figures.py from the numbers that also draw the SVG
# of the web page. Its PDF is copied to site/public/paper/skycmd-technical-report.pdf.
#
# --methods builds the methods paper, paper/methods/skycmd-methods.tex with
# paper/methods/references.bib and paper/methods/figure-1.tex, written first by
# scripts/paper-methods-figure.py from the coordinates that also draw the web page's SVG, into
# site/public/paper/skycmd-methods.pdf. Tectonic (https://tectonic-typesetting.github.io)
# compiles either one and runs BibTeX itself; the paths every link uses are those above.
#
# --arxiv also packs what arXiv needs to compile the report with pdfLaTeX: the .tex, the .bib, the
# .bbl that BibTeX wrote (arXiv does not run BibTeX) and the figure, if it has its own file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/target/paper"

arxiv=0
methods=0
for arg in "$@"; do
  case "$arg" in
    --arxiv) arxiv=1 ;;
    --methods) methods=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (expected --arxiv or --methods)" >&2; exit 2 ;;
  esac
done

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found: install it (brew install tectonic, or see its site)" >&2
  exit 1
fi

if [ "$methods" -eq 1 ]; then
  SRC_DIR="$ROOT/paper/methods"
  NAME="skycmd-methods"
  PDF="$ROOT/site/public/paper/skycmd-methods.pdf"
  SOURCES=(paper/methods)
  EXTRA=(figure-1.tex)
  python3 -I "$ROOT/scripts/paper-methods-figure.py"
else
  SRC_DIR="$ROOT/paper"
  NAME="skycmd"
  PDF="$ROOT/site/public/paper/skycmd-technical-report.pdf"
  # The first report's own files, so that a commit to paper/methods/ leaves its PDF unchanged.
  SOURCES=(paper/skycmd.tex paper/references.bib paper/figure-1.tex)
  EXTRA=(figure-1.tex)
  python3 -I "$ROOT/scripts/paper-figures.py"
fi

mkdir -p "$OUT"
# The same source gives the same PDF: its dates are those of the last commit of the source.
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
  SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --format=%ct -- "${SOURCES[@]}" 2>/dev/null || true)"
  export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(date +%s)}"
fi
tectonic --keep-intermediates --keep-logs --outdir "$OUT" "$SRC_DIR/$NAME.tex"

# Fail on what a reader would see as a defect: an undefined reference or citation, a missing
# glyph, or a line that runs into the margin by more than 10pt.
log="$OUT/$NAME.log"
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

cp "$OUT/$NAME.pdf" "$PDF"
echo "wrote ${PDF#"$ROOT"/}"

if [ "$arxiv" -eq 1 ]; then
  stage="$OUT/arxiv-$NAME"
  rm -rf "$stage"
  mkdir -p "$stage"
  cp "$SRC_DIR/$NAME.tex" "$SRC_DIR/references.bib" "$OUT/$NAME.bbl" "$stage/"
  for f in "${EXTRA[@]}"; do cp "$SRC_DIR/$f" "$stage/"; done
  tar -czf "$OUT/$NAME-arxiv.tar.gz" -C "$stage" "$NAME.tex" "$NAME.bbl" references.bib "${EXTRA[@]}"
  echo "wrote ${OUT#"$ROOT"/}/$NAME-arxiv.tar.gz"
fi
