#!/usr/bin/env python3
"""Write the figure of the technical report (site/paper/index.md) as an SVG file.

    python3 -I scripts/paper-figures.py            # writes site/public/paper/figure-1.svg

Standard library only. Every number below is copied from a published page or result file, named
in the comment next to it; nothing is computed except the positions of the marks. The output is
deterministic: running the script again gives the committed file byte for byte.

The SVG is included inline in the report page (VitePress `<!--@include: -->`), so its text uses
`currentColor` and follows the site's light and dark themes; opened on its own, it reads as dark
ink on a light background. It holds no style element, which Vue drops from a page template: the
dark-theme step of the orange (#ea580c instead of #c2410c) is in the theme's custom.css
(`.paper-fig1-b`). The two series colours were checked with the dataviz palette validator for the
light surface (#0284c7, #c2410c) and the dark one (#0284c7, #ea580c); the series also differ by
shape (a ring and a diamond), so colour is never the only cue.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "site", "public", "paper", "figure-1.svg")

# One row per line: (name, metric, own label, own value, other label, other value,
# reference label, reference value). Sources:
ROWS = [
    # site/results/index.md, section "s6-route7-nano": "its validation read 0.8802"; the table
    # "T5 (2,000 fresh requests)": the pair 0.5610, BM25 0.5675 (runs/s6-route7/gate.json).
    ("s6-route7-nano", "accuracy", "validation", 0.8802, "fresh tools of a new family", 0.5610,
     "BM25", 0.5675),
    # site/results/index.md, section "s6-route": T2-tools 0.8868 (runs/s6-nano3/eval-fresh/results.md);
    # section "s6-route7-nano", row "s6-route (v3), reported": T5-tools 0.5530; BM25 on T5 0.5675.
    ("s6-route", "accuracy", "held-out tools", 0.8868, "a new tool family (reported)", 0.5530,
     "BM25", 0.5675),
    # site/results/index.md, section "s6-ear2-nano": AUC 0.9600 on the 31,527 test clips of Smart
    # Turn v3.2; "What the pass does not say": eot-bench AUC 0.7299 against Smart Turn's 0.7838.
    ("s6-ear2-nano", "AUC", "Smart Turn's test clips", 0.9600, "eot-bench, real speech", 0.7299,
     "Smart Turn", 0.7838),
    # BENCH.md, section "s6-secrets-verify-pico": validation line F1 0.8603 (the rehearsal, read
    # before the test); site/results/index.md: test line F1 0.8561, best rule scanner 0.7332
    # (runs/s6-secrets2/eval-test/results.md).
    ("s6-secrets-verify-pico", "line F1", "validation repositories", 0.8603, "test repositories",
     0.8561, "best rule scanner", 0.7332),
]

W = 480
LEFT, RIGHT = 12, W - 16          # plot range in px
LO, HI = 0.4, 1.0                 # axis range (dot plot: position, not length)
ROW_H = 96
TOP = 74
BLUE = "#0284c7"
ORANGE_LIGHT = "#c2410c"  # its dark-theme step, #ea580c, is in site/.vitepress/theme/custom.css


def x(v):
    return LEFT + (v - LO) / (HI - LO) * (RIGHT - LEFT)


def f(v):
    """A coordinate with one decimal and no trailing zero."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def diamond(cx, cy, r):
    return f"M{f(cx)},{f(cy - r)}L{f(cx + r)},{f(cy)}L{f(cx)},{f(cy + r)}L{f(cx - r)},{f(cy)}Z"


def build():
    height = TOP + ROW_H * len(ROWS) + 44
    desc = ["Dot plot, one row per line, on an axis from 0.4 to 1."]
    for name, metric, own_l, own, oth_l, oth, ref_l, ref in ROWS:
        desc.append(f"{name}, {metric}: {own_l} {own:.4f}; {oth_l} {oth:.4f}; {ref_l} on the "
                    f"other data {ref:.4f}.")
    desc.append("On other data the reading falls far below the line's own for the two routers and "
                "the end-of-turn detector, and stays level with it for the secrets verifier.")
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" '
               f'height="{height}" role="img" aria-labelledby="paper-fig1-title paper-fig1-desc" '
               'fill="currentColor" font-family="ui-sans-serif, system-ui, -apple-system, '
               "'Segoe UI', Roboto, Arial, sans-serif\" font-size=\"14\" "
               'style="display:block;width:100%;max-width:520px;height:auto;margin:0 auto">')
    out.append('<title id="paper-fig1-title">Readings of the same file on its own distribution '
               "and on other data</title>")
    out.append(f'<desc id="paper-fig1-desc">{esc(" ".join(desc))}</desc>')
    # Legend.
    out.append('<g font-size="13">')
    out.append(f'<circle cx="8" cy="12" r="5.5" fill="none" stroke="{BLUE}" stroke-width="3"/>'
               '<text x="20" y="17">its own distribution</text>')
    out.append(f'<path class="paper-fig1-b" fill="{ORANGE_LIGHT}" d="{diamond(171, 12, 7)}"/>'
               '<text x="184" y="17">other data</text>')
    out.append('<path d="M271,4V20" stroke="currentColor" stroke-width="2"/>'
               '<text x="279" y="17">reference, on the other data</text>')
    out.append("</g>")
    out.append('<text x="0" y="44" font-size="12" fill-opacity=".75">Each row has its own metric; '
               "higher is better on all four. The axis starts at 0.4.</text>")
    for i, (name, metric, own_l, own, oth_l, oth, ref_l, ref) in enumerate(ROWS):
        y0 = TOP + i * ROW_H
        cy = y0 + 58
        out.append("<g>")
        out.append(f'<text x="0" y="{y0 + 12}" font-weight="650">{esc(name)}'
                   f'<tspan font-weight="400" fill-opacity=".75"> · {esc(metric)}</tspan></text>')
        out.append(f'<text x="0" y="{y0 + 30}" font-size="12" fill-opacity=".8">{esc(own_l)} '
                   f"{own:.4f} → {esc(oth_l)} {oth:.4f}</text>")
        out.append(f'<text x="0" y="{y0 + 45}" font-size="12" fill-opacity=".8">{esc(ref_l)} on '
                   f"the other data: {ref:.4f}</text>")
        out.append(f'<path d="M{LEFT},{cy}H{RIGHT}" stroke="currentColor" stroke-opacity=".18" '
                   'stroke-width="2"/>')
        lo, hi = sorted((x(own), x(oth)))
        out.append(f'<path d="M{f(lo)},{cy}H{f(hi)}" stroke="currentColor" stroke-opacity=".45" '
                   'stroke-width="2"/>')
        out.append(f'<g><title>{esc(ref_l)}, {esc(oth_l)}: {ref:.4f}</title><path '
                   f'd="M{f(x(ref))},{cy - 10}V{cy + 10}" stroke="currentColor" stroke-width="2"/></g>')
        out.append(f'<g><title>{esc(name)}, {esc(oth_l)}: {oth:.4f}</title><path '
                   f'class="paper-fig1-b" fill="{ORANGE_LIGHT}" d="{diamond(x(oth), cy, 7)}"/></g>')
        out.append(f'<g><title>{esc(name)}, {esc(own_l)}: {own:.4f}</title><circle '
                   f'cx="{f(x(own))}" cy="{cy}" r="5.5" fill="none" stroke="{BLUE}" stroke-width="3"/></g>')
        out.append("</g>")
    # Axis.
    ya = TOP + ROW_H * len(ROWS) + 6
    out.append(f'<g stroke="currentColor" stroke-opacity=".5"><path d="M{LEFT},{ya}H{RIGHT}"/>')
    ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for t in ticks:
        out.append(f'<path d="M{f(x(t))},{ya}V{ya + 5}"/>')
    out.append("</g>")
    out.append('<g font-size="12" fill-opacity=".8" text-anchor="middle">')
    for t in ticks:
        label = "1" if t == 1.0 else f"{t:.1f}"
        out.append(f'<text x="{f(x(t))}" y="{ya + 20}">{label}</text>')
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) > 1:
        sys.exit("usage: python3 -I scripts/paper-figures.py")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build())
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
