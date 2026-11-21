#!/usr/bin/env python3
"""Write the two figures of the technical report as SVG files and as TikZ pictures.

    python3 -I scripts/paper-figures.py
    # writes site/public/paper/figure-1.svg and figure-2.svg, paper/figure-1.tex and figure-2.tex

Figure 1 is the gate dashboard, drawn from GATES (one condition of each gate of Table 1); Figure 2
is the same files read on their own distribution and on other data, drawn from ROWS (Table 4). The
SVGs are the figures of the web page (site/paper/index.md); each TikZ picture is the same figure,
drawn from the same list, for the LaTeX source of the PDF (paper/skycmd.tex, built by
scripts/build-paper.sh).

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
OUT = os.path.join(ROOT, "site", "public", "paper", "figure-2.svg")
TEX_OUT = os.path.join(ROOT, "paper", "figure-2.tex")
GATES_OUT = os.path.join(ROOT, "site", "public", "paper", "figure-1.svg")
GATES_TEX_OUT = os.path.join(ROOT, "paper", "figure-1.tex")

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


# The TikZ picture: the same layout as the SVG, in centimetres, for a page of the PDF.
TEX_W = 12.0      # plot range, 0.4 to 1 on the axis
TEX_ROW_H = 1.9
TEX_TOP = -1.25   # top of the first row, under the legend and the note


def tx(v):
    return f"{(v - LO) / (HI - LO) * TEX_W:.3f}"


TEX_SPECIAL = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#",
               "_": "\\_", "{": "\\{", "}": "\\}"}


def tex_esc(s):
    return "".join(TEX_SPECIAL.get(c, c) for c in s)


def tex_diamond(cx, cy, r=0.14):
    return (f"({cx},{cy + r:.3f}) -- ({float(cx) + r:.3f},{cy:.3f}) -- ({cx},{cy - r:.3f}) -- "
            f"({float(cx) - r:.3f},{cy:.3f}) -- cycle")


def build_tex():
    out = []
    out.append("% Figure 2 of the technical report, written by scripts/paper-figures.py from the same")
    out.append("% ROWS as site/public/paper/figure-2.svg. Do not edit: run the script again.")
    out.append(f"\\definecolor{{paperfigblue}}{{HTML}}{{{BLUE[1:].upper()}}}")
    out.append(f"\\definecolor{{paperfigorange}}{{HTML}}{{{ORANGE_LIGHT[1:].upper()}}}")
    out.append("\\begin{tikzpicture}[x=1cm,y=1cm,font=\\small]")
    # Legend.
    out.append("\\draw[paperfigblue,line width=1.2pt] (0.11,0) circle (0.11);")
    out.append("\\node[anchor=west] at (0.3,0) {its own distribution};")
    out.append(f"\\fill[paperfigorange] {tex_diamond('4.000', 0.0)};")
    out.append("\\node[anchor=west] at (4.2,0) {other data};")
    out.append("\\draw[line width=0.8pt] (6.5,-0.2) -- (6.5,0.2);")
    out.append("\\node[anchor=west] at (6.6,0) {reference, on the other data};")
    out.append("\\node[anchor=west,inner sep=0pt,text=black!75,font=\\footnotesize] at (0,-0.6) "
               "{Each row has its own metric; higher is better on all four. The axis starts at 0.4.};")
    for i, (name, metric, own_l, own, oth_l, oth, ref_l, ref) in enumerate(ROWS):
        y0 = TEX_TOP - i * TEX_ROW_H
        cy = y0 - 1.35
        out.append(f"\\node[anchor=north west,inner sep=0pt] at (0,{y0:.3f}) "
                   f"{{\\textbf{{{tex_esc(name)}}}\\textcolor{{black!75}}{{~$\\cdot$~{tex_esc(metric)}}}}};")
        out.append(f"\\node[anchor=north west,inner sep=0pt,text=black!80,font=\\footnotesize] at "
                   f"(0,{y0 - 0.45:.3f}) {{{tex_esc(own_l)} {own:.4f} $\\rightarrow$ {tex_esc(oth_l)} "
                   f"{oth:.4f}}};")
        out.append(f"\\node[anchor=north west,inner sep=0pt,text=black!80,font=\\footnotesize] at "
                   f"(0,{y0 - 0.8:.3f}) {{{tex_esc(ref_l)} on the other data: {ref:.4f}}};")
        out.append(f"\\draw[black!18,line width=0.8pt] (0,{cy:.3f}) -- ({TEX_W:.3f},{cy:.3f});")
        lo, hi = sorted((own, oth))
        out.append(f"\\draw[black!45,line width=0.8pt] ({tx(lo)},{cy:.3f}) -- ({tx(hi)},{cy:.3f});")
        out.append(f"\\draw[line width=0.8pt] ({tx(ref)},{cy - 0.2:.3f}) -- ({tx(ref)},{cy + 0.2:.3f});")
        out.append(f"\\fill[paperfigorange] {tex_diamond(tx(oth), cy)};")
        out.append(f"\\draw[paperfigblue,line width=1.2pt] ({tx(own)},{cy:.3f}) circle (0.11);")
    # Axis.
    ya = TEX_TOP - TEX_ROW_H * len(ROWS) - 0.05
    out.append(f"\\draw[black!50] (0,{ya:.3f}) -- ({TEX_W:.3f},{ya:.3f});")
    for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        label = "1" if t == 1.0 else f"{t:.1f}"
        out.append(f"\\draw[black!50] ({tx(t)},{ya:.3f}) -- ({tx(t)},{ya - 0.12:.3f}) "
                   f"node[below,text=black!80,font=\\footnotesize] {{{label}}};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


# Figure 1, the gate dashboard: one condition of each gate of Table 1 of the report (for a failure,
# the first condition it missed; for a pass, its first condition). Each entry: (name, condition,
# reading as published, interval or None, bar, "high" or "low" (the better side), passed, second look, the bar
# as written). Every number is copied from Table 1 of site/paper/index.md, which copies
# site/results/index.md and the result files it names.
GATES = [
    ("s6-nano", "typed_decisions_bench", "0.480", None, 0.398, "high", True, False, "> 0.398"),
    ("s6-micro v2", "typed_decisions_bench", "0.553", None, 0.45, "high", True, True, "≥ 0.45"),
    ("s6-micro-v1", "ECE", "0.0821", ("0.0746", "0.0900"), 0.08, "low", False, False, "≤ 0.08"),
    ("s4-micro", "typed_decisions_bench", "0.343", None, 0.55, "high", False, False, "> 0.55"),
    ("s6-secrets-pico", "secret exposed, secret_bench", "0.9990", None, 0.977, "high", True, False,
     "> 0.977"),
    ("s6-secrets-verify-pico", "line F1 over the best rule scanner", "0.1229", ("0.0413", "0.3613"), 0.15,
     "high", False, False, "≥ +0.15"),
    ("s6-ear-nano", "end of turn", "0.5681", None, 0.9263, "high", False, False, "≥ 0.9263"),
    ("s6-ear2-nano", "accuracy, Smart Turn's test", "0.8993", ("0.8962", "0.9027"), 0.85, "high", True,
     False, "≥ 0.85"),
    ("s6-ear3-nano", "AUC change on eot-bench (E1)", "-0.0252", ("-0.0336", "-0.0167"), 0.02, "high", False,
     False, "≥ +0.02"),
    ("s6-cast-pico", "GIFT-Eval nGMASE", "0.9471", None, 0.838, "low", False, False, "≤ 0.838"),
    ("s6-cast2-pico", "GIFT-Eval nGMASE (G1)", "0.8847", None, 0.838, "low", False, False, "≤ 0.838"),
    ("s6-route-nano", "accuracy minus BM25 (G1)", "-0.0105", ("-0.0265", "0.0055"), 0.0, "high", False,
     False, "> 0"),
    ("s6-route", "accuracy minus BM25", "0.0698", ("0.0455", "0.0952"), 0.0, "high", True, True,
     "lower bound > 0"),
    ("s6-route7-nano", "accuracy minus BM25 on T5 (G1)", "-0.0065", ("-0.0235", "0.0120"), 0.0, "high",
     False, False, "lower bound > 0"),
    ("s6-sleep3-nano", "sleep gain on recall (H1)", "0.0030", ("-0.0010", "0.0068"), 0.0, "high", False,
     False, "lower bound > 0"),
]
NICE = [1, 2, 2.5, 5]


def half_width(g):
    """The smallest half-width of 1, 2, 2.5 or 5 times a power of ten that holds the row."""
    _, _, val, ci, bar = g[:5]
    pts = [float(val)] + ([float(c) for c in ci] if ci else [])
    d = max(abs(p - bar) for p in pts) * 1.15
    k = 1e-6
    while True:
        for n in NICE:
            if n * k >= d - 1e-12:
                return n * k
        k *= 10


def num(v, plus=False):
    """A number as published (a string), with a minus sign and, for a difference, a plus sign."""
    if v.startswith("-"):
        return "−" + v[1:]
    return ("+" if plus and float(v) > 0 else "") + v


def hw_label(h):
    return "±" + f"{h:.4f}".rstrip("0").rstrip(".")


def gate_pos(g, u):
    """The position of u on the row's own axis, from -1 (the worse side) to +1 (the better side)."""
    bar, better = g[4], g[5]
    t = (float(u) - bar) / half_width(g)
    return t if better == "high" else -t


def gate_reading(g):
    name, cond, val, ci, bar, better, ok, second, bar_s = g
    plus = bar == 0.0 or "+" in bar_s
    s = f"{cond}: {num(val, plus)}"
    if ci:
        s += f" [{num(ci[0], plus)}, {num(ci[1], plus)}]"
    return s + f", bar {bar_s}"


def build_gates():
    W, PX0, PX1 = 640, 330, 586
    ROW = 36
    top = 70
    height = top + ROW * len(GATES) + 8
    cx = (PX0 + PX1) / 2

    def X(t):
        return cx + t * (PX1 - PX0) / 2

    desc = ["Gate dashboard: one row per gated file, with one condition of its gate. Each row has its "
            "own scale centred on the bar, and right is better."]
    for g in GATES:
        desc.append(f"{g[0]}: {gate_reading(g)}; {'passed' if g[6] else 'failed'}"
                    f"{', a second look' if g[7] else ''}.")
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" '
               f'height="{height}" role="img" aria-labelledby="paper-gates-title paper-gates-desc" '
               'fill="currentColor" font-family="ui-sans-serif, system-ui, -apple-system, '
               "'Segoe UI', Roboto, Arial, sans-serif\" font-size=\"13\" "
               'style="display:block;width:100%;max-width:680px;height:auto;margin:0 auto">')
    out.append('<title id="paper-gates-title">Every gate, as read: one condition of each gated file '
               "against its bar</title>")
    out.append(f'<desc id="paper-gates-desc">{esc(" ".join(desc))}</desc>')
    out.append('<g font-size="12">')
    out.append(f'<circle cx="8" cy="12" r="6" fill="{BLUE}"/><text x="20" y="16">passed</text>')
    out.append(f'<path class="paper-fig1-b" fill="{ORANGE_LIGHT}" d="{diamond(88, 12, 7)}"/>'
               '<text x="100" y="16">failed</text>')
    out.append('<circle cx="166" cy="12" r="9.5" fill="none" stroke="currentColor" stroke-width="1.5"/>'
               '<text x="182" y="16">second look</text>')
    out.append('<path d="M281,3V21" stroke="currentColor" stroke-width="2"/><text x="289" y="16">bar</text>')
    out.append('<path d="M325,12H351" stroke="currentColor" stroke-opacity=".55" stroke-width="2"/>'
               '<text x="358" y="16">published interval</text>')
    out.append("</g>")
    out.append('<text x="0" y="42" font-size="12" fill-opacity=".75">Each row has its own scale, centred '
               "on its bar, with the half-width at its right; right is better on every row.</text>")
    out.append(f'<g font-size="11" fill-opacity=".75" text-anchor="middle"><text x="{f(X(-1))}" '
               f'y="60">worse</text><text x="{f(cx)}" y="60">bar</text><text x="{f(X(1))}" '
               'y="60">better</text></g>')
    for i, g in enumerate(GATES):
        name, cond, val, ci, bar, better, ok, second, bar_s = g
        y0 = top + i * ROW
        cy = y0 + 17
        out.append("<g>")
        out.append(f'<text x="0" y="{y0 + 13}" font-weight="650">{esc(name)}<tspan font-weight="400" '
                   f'fill-opacity=".75"> · {"passed" if ok else "failed"}'
                   f'{", second look" if second else ""}</tspan></text>')
        out.append(f'<text x="0" y="{y0 + 28}" font-size="11.5" fill-opacity=".8">'
                   f"{esc(gate_reading(g))}</text>")
        out.append(f'<path d="M{PX0},{cy}H{PX1}" stroke="currentColor" stroke-opacity=".18" '
                   'stroke-width="2"/>')
        out.append(f'<path d="M{f(cx)},{cy - 10}V{cy + 10}" stroke="currentColor" stroke-width="2"/>')
        if ci:
            a, b = sorted((X(gate_pos(g, ci[0])), X(gate_pos(g, ci[1]))))
            out.append(f'<path d="M{f(a)},{cy}H{f(b)}" stroke="currentColor" stroke-opacity=".55" '
                       'stroke-width="2"/>')
        px = X(gate_pos(g, val))
        tip = f"<title>{esc(name)}: {esc(gate_reading(g))}</title>"
        if ok:
            out.append(f'<g>{tip}<circle cx="{f(px)}" cy="{cy}" r="6" fill="{BLUE}"/></g>')
        else:
            out.append(f'<g>{tip}<path class="paper-fig1-b" fill="{ORANGE_LIGHT}" '
                       f'd="{diamond(px, cy, 7)}"/></g>')
        if second:
            out.append(f'<circle cx="{f(px)}" cy="{cy}" r="9.5" fill="none" stroke="currentColor" '
                       'stroke-width="1.5"/>')
        out.append(f'<text x="{PX1 + 8}" y="{cy + 4}" font-size="11" fill-opacity=".75">'
                   f"{esc(hw_label(half_width(g)))}</text>")
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def build_gates_tex():
    PX0, PX1, ROW = 8.9, 14.9, 0.66
    cx = (PX0 + PX1) / 2

    def X(t):
        return f"{cx + t * (PX1 - PX0) / 2:.3f}"

    def tt(s):
        for a, b in (("−", "$-$"), ("±", "$\\pm$"), ("≥", "$\\geq$"), ("≤", "$\\leq$"),
                     (">", "\\textgreater{}")):
            s = s.replace(a, b)
        return s

    out = []
    out.append("% Figure 1 of the technical report, the gate dashboard, written by")
    out.append("% scripts/paper-figures.py from the same GATES as site/public/paper/figure-1.svg.")
    out.append("% Do not edit: run the script again.")
    out.append(f"\\definecolor{{paperfigblue}}{{HTML}}{{{BLUE[1:].upper()}}}")
    out.append(f"\\definecolor{{paperfigorange}}{{HTML}}{{{ORANGE_LIGHT[1:].upper()}}}")
    out.append("\\begin{tikzpicture}[x=1cm,y=1cm,font=\\footnotesize]")
    out.append("\\fill[paperfigblue] (0.12,0) circle (0.12);")
    out.append("\\node[anchor=west] at (0.3,0) {passed};")
    out.append(f"\\fill[paperfigorange] {tex_diamond('1.900', 0.0)};")
    out.append("\\node[anchor=west] at (2.1,0) {failed};")
    out.append("\\draw[line width=0.6pt] (3.6,0) circle (0.19);")
    out.append("\\node[anchor=west] at (3.85,0) {second look};")
    out.append("\\draw[line width=0.8pt] (6.1,-0.2) -- (6.1,0.2);")
    out.append("\\node[anchor=west] at (6.2,0) {bar};")
    out.append("\\draw[black!55,line width=0.8pt] (7.2,0) -- (7.8,0);")
    out.append("\\node[anchor=west] at (7.9,0) {published interval};")
    out.append("\\node[anchor=west,inner sep=0pt,text=black!75,font=\\scriptsize] at (0,-0.5) {Each row "
               "has its own scale, centred on its bar, with the half-width at its right; right is better "
               "on every row.};")
    for t, lab in ((-1, "worse"), (0, "bar"), (1, "better")):
        out.append(f"\\node[text=black!75,font=\\scriptsize] at ({X(t)},-0.95) {{{lab}}};")
    for i, g in enumerate(GATES):
        name, cond, val, ci, bar, better, ok, second, bar_s = g
        cy = -1.5 - i * ROW
        verdict = ("passed" if ok else "failed") + (", second look" if second else "")
        out.append(f"\\node[anchor=west,inner sep=0pt] at (0,{cy + 0.12:.3f}) "
                   f"{{\\textbf{{{tex_esc(name)}}}\\textcolor{{black!75}}{{~$\\cdot$~{verdict}}}}};")
        out.append(f"\\node[anchor=west,inner sep=0pt,text=black!80,font=\\scriptsize] at "
                   f"(0,{cy - 0.17:.3f}) {{{tt(tex_esc(gate_reading(g)))}}};")
        out.append(f"\\draw[black!18,line width=0.8pt] ({PX0},{cy:.3f}) -- ({PX1},{cy:.3f});")
        out.append(f"\\draw[line width=0.8pt] ({cx:.3f},{cy - 0.2:.3f}) -- ({cx:.3f},{cy + 0.2:.3f});")
        if ci:
            a, b = sorted((float(X(gate_pos(g, ci[0]))), float(X(gate_pos(g, ci[1])))))
            out.append(f"\\draw[black!55,line width=0.8pt] ({a:.3f},{cy:.3f}) -- ({b:.3f},{cy:.3f});")
        px = X(gate_pos(g, val))
        if ok:
            out.append(f"\\fill[paperfigblue] ({px},{cy:.3f}) circle (0.12);")
        else:
            out.append(f"\\fill[paperfigorange] {tex_diamond(px, cy)};")
        if second:
            out.append(f"\\draw[line width=0.6pt] ({px},{cy:.3f}) circle (0.19);")
        out.append(f"\\node[anchor=west,text=black!75,font=\\scriptsize] at ({PX1 + 0.05:.3f},{cy:.3f}) "
                   f"{{{tt(hw_label(half_width(g)))}}};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def main():
    if len(sys.argv) > 1:
        sys.exit("usage: python3 -I scripts/paper-figures.py")
    write(GATES_OUT, build_gates())
    write(GATES_TEX_OUT, build_gates_tex())
    write(OUT, build())
    write(TEX_OUT, build_tex())


if __name__ == "__main__":
    main()
