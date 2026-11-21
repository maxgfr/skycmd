#!/usr/bin/env python3
"""Write the architecture figure of the methods paper as an SVG file and as a TikZ picture.

    python3 -I scripts/paper-methods-figure.py
    # writes site/public/paper/methods/figure-1.svg and paper/methods/figure-1.tex

The SVG is the figure of the web page (site/paper/methods/index.md, included inline); the TikZ
picture is the same figure, drawn from the same BOXES and ARROWS, for the LaTeX source of the PDF
(paper/methods/skycmd-methods.tex, built by scripts/build-paper.sh --methods). Both are drawn from
one list of coordinates in centimetres, so the two stay the same drawing.

Standard library only; no number is drawn, only the graph of the `skycmd-s2` decision model as
crates/sky-infer/src/s2/model.rs runs it. The output is deterministic: running the script again
gives the committed files byte for byte. The SVG uses currentColor, so it follows the site's light
and dark themes, and holds no style element (Vue drops one from a page template).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG_OUT = os.path.join(ROOT, "site", "public", "paper", "methods", "figure-1.svg")
TEX_OUT = os.path.join(ROOT, "paper", "methods", "figure-1.tex")

PX = 48.0  # SVG pixels per centimetre
C1, C2, C3 = 2.3, 7.45, 12.6  # column centres, cm
Y1, Y2, Y3, Y4 = 8.5, 6.2, 3.9, 1.6  # row centres, cm
BW, BH = 4.3, 1.5  # box size, cm
TOP, BOTTOM = 9.75, 0.55  # drawing extent, cm
LEFT, RIGHT = 0.0, 14.9

# Each box: key, centre x, centre y, shaded (computed once per question), title (svg, tex), and
# its lines as (svg, tex) pairs.
BOXES = [
    ("in_q", C1, Y1, True, ("Question and options", "Question and options"), [
        ("[CLS] {t} question: {ins} [SEP]", r"\texttt{[CLS]} \{t\} question: \{ins\} \texttt{[SEP]}"),
        ("[MASK] opt₁ … [MASK] optₙ", r"\texttt{[MASK]} opt$_1$ \ldots{} \texttt{[MASK]} opt$_n$"),
    ]),
    ("static", C2, Y1, True, ("Static encoder", "Static encoder"), [
        ("Q sees Q; each Oᵢ sees Q and Oᵢ", r"$Q$ sees $Q$; each $O_i$ sees $Q$ and $O_i$"),
        ("every Oᵢ starts at position |Q|", r"every $O_i$ starts at position $|Q|$"),
    ]),
    ("cache", C3, Y1, True, ("Cache, once per question", "Cache, once per question"), [
        ("markers mᵢ + type embedding", r"markers $m_i$ + type embedding"),
        ("question tokens as keys, values", r"question tokens as keys, values"),
    ]),
    ("in_s", C1, Y2, False, ("State, every frame", "State, every frame"), [
        ("[CLS] vy -3 dx 35 … [SEP]", r"\texttt{[CLS] vy -3 dx 35} \ldots{} \texttt{[SEP]}"),
        ("one token per key and value", r"one token per key and value"),
    ]),
    ("loop", C2, Y2, False, ("Shared state blocks", "Shared state blocks"), [
        ("B blocks, applied up to L times", r"$B$ blocks, applied up to $L$ times"),
        ("pass l adds loop_embd[l]", r"pass $l$ adds \texttt{loop\_embd}$[l]$"),
    ]),
    ("exit", C2, Y3, False, ("Exit after each pass", "Exit after each pass"), [
        ("Hₑ = LNₑ(hidden states)", r"$H_e = \mathrm{LN}_e(\text{hidden states})$"),
        ("exit e counts the block passes", r"exit $e$ counts the block passes"),
    ]),
    ("fusion", C3, Y3, False, ("Fusion head, shared by exits", "Fusion head, shared by exits"), [
        ("markers attend to [Hₑ; question]", r"markers attend to $[H_e;\,\text{question}]$"),
        ("never to each other", r"never to each other"),
    ]),
    ("score", C3, Y4, False, ("Scorer and temperature", "Scorer and temperature"), [
        ("one logit sᵢ per option", r"one logit $s_i$ per option"),
        ("p = softmax(s / T[e, t, bucket])", r"$p = \mathrm{softmax}(s / T_{e,t,b})$"),
    ]),
    ("test", C2, Y4, False, ("Exit test", "Exit test"), [
        ("max p ≥ τₜ, or last exit?", r"$\max_i p_i \geq \tau_t$, or last exit?"),
    ]),
    ("answer", C1, Y4, False, ("Answer", "Answer"), [
        ("p over the options,", r"$p$ over the options,"),
        ("the exit and the passes used", r"the exit and the passes used"),
    ]),
]

# Each arrow: points (cm), and an optional label (svg, tex) with its position and anchor.
ARROWS = [
    ([(C1 + BW / 2, Y1), (C2 - BW / 2, Y1)], None),
    ([(C2 + BW / 2, Y1), (C3 - BW / 2, Y1)], None),
    ([(C3, Y1 - BH / 2), (C3, Y3 + BH / 2)], (("cached", "cached"), (C3 + 0.12, (Y1 + Y3) / 2), "start")),
    ([(C1 + BW / 2, Y2), (C2 - BW / 2, Y2)], None),
    ([(C2, Y2 - BH / 2), (C2, Y3 + BH / 2)], (("pass l", r"pass $l$"), (C2 + 0.12, (Y2 + Y3) / 2), "start")),
    ([(C2 + BW / 2, Y3), (C3 - BW / 2, Y3)], (("Hₑ", r"$H_e$"), ((C2 + C3) / 2, Y3 + 0.18), "middle")),
    ([(C3, Y3 - BH / 2), (C3, Y4 + BH / 2)], None),
    ([(C3 - BW / 2, Y4), (C2 + BW / 2, Y4)], None),
    ([(C2 - BW / 2, Y4 - 0.3), (C1 + BW / 2, Y4 - 0.3)], (("yes", "yes"), ((C1 + C2) / 2, Y4 - 0.48), "middle-below")),
    ([(C2 - BW / 2, Y4 + 0.3), ((C1 + C2) / 2, Y4 + 0.3), ((C1 + C2) / 2, Y2 - 0.45), (C2 - BW / 2, Y2 - 0.45)],
     (("no: next pass", "no: next pass"), ((C1 + C2) / 2 - 0.12, (Y2 + Y4) / 2), "end")),
]

NOTE = ("Linear layers ternary, w ∈ {−1, 0, +1}·γ (2 bits each); embeddings int8; norms f32.",
        r"Linear layers ternary, $w \in \{-1, 0, +1\}\cdot\gamma$ (2 bits each); embeddings int8; norms f32.")
LEGEND = ("Shaded: computed once per question and cached. White: every frame.",
          r"Shaded: computed once per question and cached. White: every frame.")


def sx(x):
    return (x - LEFT) * PX


def sy(y):
    return (TOP - y) * PX


def f(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg():
    w, h = sx(RIGHT), sy(BOTTOM) + 34
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f(w)} {f(h)}" width="{f(w)}" '
           f'height="{f(h)}" role="img" aria-labelledby="paper-m-fig1-title paper-m-fig1-desc" '
           'fill="currentColor" font-family="ui-sans-serif, system-ui, -apple-system, '
           "'Segoe UI', Roboto, Arial, sans-serif\" font-size=\"12.5\" "
           'style="display:block;width:100%;max-width:720px;height:auto;margin:0 auto">']
    out.append('<title id="paper-m-fig1-title">The skycmd-s2 decision graph: two streams, a cache, '
               "a looped state block with an exit after each pass, and a shared fusion head</title>")
    desc = ["The question and its options are encoded once by the static encoder, in which each "
            "option sees the question and itself only, and cached as option markers and question "
            "tokens. Every frame, the state is embedded with one token per key and value and passed "
            "through B shared blocks, up to L times. After each pass an exit normalises the hidden "
            "states; the fusion head, shared by all exits, lets each option marker attend to those "
            "states and the cached question tokens, never to the other markers; the scorer gives one "
            "logit per option, divided by a temperature per exit, type and bucket. If the top "
            "probability reaches the threshold of the question's type, or the exit is the last, the "
            "answer is returned; otherwise the next pass runs."]
    out.append(f'<desc id="paper-m-fig1-desc">{esc(" ".join(desc))}</desc>')
    out.append('<defs><marker id="paper-m-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               '<path d="M0,0L10,5L0,10Z" fill="currentColor"/></marker></defs>')
    for key, cx, cy, shaded, title, lines in BOXES:
        x0, y0 = sx(cx - BW / 2), sy(cy + BH / 2)
        fill = 'fill="currentColor" fill-opacity=".08"' if shaded else 'fill="none"'
        out.append(f'<g><rect x="{f(x0)}" y="{f(y0)}" width="{f(BW * PX)}" height="{f(BH * PX)}" '
                   f'rx="5" {fill} stroke="currentColor" stroke-opacity=".7" stroke-width="1.2"/>')
        n = 1 + len(lines)
        line_h = 14.5
        first = sy(cy) - (n - 1) * line_h / 2 + 4.5
        out.append(f'<text x="{f(sx(cx))}" y="{f(first)}" text-anchor="middle" '
                   f'font-weight="650">{esc(title[0])}</text>')
        for i, (s, _) in enumerate(lines):
            out.append(f'<text x="{f(sx(cx))}" y="{f(first + (i + 1) * line_h)}" '
                       f'text-anchor="middle" font-size="10.5" fill-opacity=".85">{esc(s)}</text>')
        out.append("</g>")
    for pts, label in ARROWS:
        d = "M" + "L".join(f"{f(sx(x))},{f(sy(y))}" for x, y in pts)
        out.append(f'<path d="{d}" fill="none" stroke="currentColor" stroke-width="1.4" '
                   'marker-end="url(#paper-m-arrow)"/>')
        if label:
            (s, _), (lx, ly), anchor = label
            a = {"start": "start", "end": "end"}.get(anchor, "middle")
            dy = 4 if anchor in ("start", "end") else (12 if anchor == "middle-below" else -3)
            out.append(f'<text x="{f(sx(lx))}" y="{f(sy(ly) + dy)}" text-anchor="{a}" '
                       f'font-size="10.5" font-style="italic">{esc(s)}</text>')
    out.append(f'<text x="{f(sx(LEFT))}" y="{f(sy(BOTTOM) + 12)}" font-size="11.5" '
               f'fill-opacity=".8">{esc(LEGEND[0])}</text>')
    out.append(f'<text x="{f(sx(LEFT))}" y="{f(sy(BOTTOM) + 28)}" font-size="11.5" '
               f'fill-opacity=".8">{esc(NOTE[0])}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def tex():
    out = ["% The architecture figure of the methods paper, written by scripts/paper-methods-figure.py",
           "% from the same coordinates as site/public/paper/methods/figure-1.svg. Do not edit by hand.",
           r"\begin{tikzpicture}[x=1cm,y=1cm,>=stealth,font=\footnotesize,"
           r"box/.style={draw,rounded corners=2pt,minimum width=%.2fcm,minimum height=%.2fcm,"
           r"align=center,inner sep=2pt}]" % (BW, BH)]
    for key, cx, cy, shaded, title, lines in BOXES:
        body = r"\textbf{%s}" % title[1] + "".join(r"\\{\scriptsize %s}" % t for _, t in lines)
        style = "box,fill=black!7" if shaded else "box"
        out.append(r"\node[%s] (%s) at (%.2f,%.2f) {%s};" % (style, key, cx, cy, body))
    for pts, label in ARROWS:
        path = " -- ".join("(%.3f,%.3f)" % p for p in pts)
        out.append(r"\draw[->,thick] %s;" % path)
        if label:
            (_, t), (lx, ly), anchor = label
            a = {"start": "east", "end": "west"}
            if anchor in a:
                where = "anchor=" + {"start": "west", "end": "east"}[anchor]
            elif anchor == "middle-below":
                where = "anchor=north"
            else:
                where = "anchor=south"
            out.append(r"\node[%s,font=\scriptsize\itshape] at (%.3f,%.3f) {%s};"
                       % (where, lx, ly, t))
    out.append(r"\node[anchor=south west,font=\scriptsize,align=left] at (%.2f,%.2f) {%s\\%s};"
               % (LEFT, BOTTOM - 0.95, LEGEND[1], NOTE[1]))
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(os.path.dirname(SVG_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with open(SVG_OUT, "w", encoding="utf-8") as fh:
        fh.write(svg())
    with open(TEX_OUT, "w", encoding="utf-8") as fh:
        fh.write(tex())
    print(f"wrote {os.path.relpath(SVG_OUT, ROOT)} and {os.path.relpath(TEX_OUT, ROOT)}")


if __name__ == "__main__":
    main()
