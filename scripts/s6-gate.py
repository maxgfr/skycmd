#!/usr/bin/env python3
"""Publication gate and results table of s6-nano / s6-micro (docs/s6-gen-prereg.md §6).

Reads what `scripts/train-s6.sh eval` writes in runs/s6-<size>/eval:
  eval-<name>-full.json, eval-<name>-crc.json   sky-eval of the final ternary GGUF at full depth
                                                (tau = 1) and at its CRC thresholds (if kept)
  ref-*.json                                    sky-eval of the references (s3-pico, s4-micro,
                                                s5-lite), full depth, same records
  indist-<name>.json                            Banking77 (trained on: reported, never counted)
  arena-baseline.json, <model>-arena.json, <model>-crc-arena.json   arena of s3-pico and of the
                                                model at full depth and at its CRC thresholds
  truncation.json, wasm.json                    truncated shares, WASM latency and parity

The gate of a size holds at a depth setting when
  - typed_decisions_bench accuracy > 0.398 (nano, the majority class) or >= 0.45 (micro);
  - ECE over every language test source (`all`) <= 0.08;
  - no game's mean arena score more than 5% below s3-pico.
The model ships at its CRC thresholds when the gate holds there and at full depth, at full depth
when it holds only there, and is not published otherwise.

Usage: s6-gate.py runs/s6-micro/eval --size micro --model models/s6-micro/skycmd-s6-micro-g3-ternary
                  [--out runs/s6-micro/eval/results.md]
Exit status 0 when the gate passes, 1 otherwise.
"""
import argparse
import glob
import json
import os
import sys

BENCH = "typed_decisions_bench"
GATES = {"nano": (0.398, False), "micro": (0.45, True)}  # (threshold, inclusive)
TARGET_MICRO = 0.50
MAX_ECE = 0.08
ARENA_LOSS = 0.05
PRETRAIN_TOKENS_PER_STEP = 64 * 256  # scripts/train-s6.sh: --batch 64 --seq 256
# Published typed_decisions_bench accuracies (the tutorial's chapters 11 and 22, the s5 report).
PUBLISHED = {"s3-pico": 0.382, "s4-micro": 0.343, "oscar-1-17m": 0.6775, "s5-lite": 0.696}
ORDER = [BENCH, "zs_massive_en", "zs_massive_fr", "when2call", "notam:deel_subject", "notam:notam_multi", "all"]


def load(path):
    with open(path) as f:
        return json.load(f)


def maybe(path):
    return load(path) if os.path.exists(path) else None


def check(ev, arena, base, size):
    """The failed conditions of one depth setting (empty: the gate holds)."""
    gate, inclusive = GATES[size]
    bad = []
    acc = ev["sources"].get(BENCH, {}).get("accuracy")
    if acc is None:
        bad.append(f"no {BENCH} evaluation")
    elif not (acc >= gate if inclusive else acc > gate):
        bad.append(f"{BENCH} {acc:.4f} {'<' if inclusive else '<='} {gate}")
    ece = ev["sources"]["all"]["ece"]
    if ece > MAX_ECE:
        bad.append(f"ECE {ece:.4f} > {MAX_ECE}")
    if arena is None or base is None:
        bad.append("no arena result")
        return bad
    for g, v in base.items():
        if not (isinstance(v, dict) and "model" in v):
            continue
        a = v["model"]["mean_score"]
        b = arena.get(g, {}).get("model", {}).get("mean_score")
        if b is None:
            bad.append(f"{g}: no arena score")
        elif b < a - ARENA_LOSS * abs(a) - 1e-9:
            bad.append(f"{g} arena {b:.2f} < s3-pico {a:.2f} - 5%")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--size", choices=sorted(GATES), required=True)
    ap.add_argument("--model", required=True, help="final GGUF path without .gguf")
    ap.add_argument("--pretrain-log", help="s2-pretrain log.jsonl (tokens seen, batch 64 x 256)")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = a.eval_dir
    name = f"s6-{a.size}"

    full = maybe(os.path.join(d, f"eval-{name}-full.json"))
    if full is None:
        print(f"gate: no {d}/eval-{name}-full.json")
        return 1
    crc = maybe(os.path.join(d, f"eval-{name}-crc.json"))
    base = maybe(os.path.join(d, "arena-baseline.json"))
    arena_full = maybe(a.model + "-arena.json")
    arena_crc = maybe(a.model + "-crc-arena.json")

    rows = [(f"{name} (τ = 1)", full)]
    if crc:
        rows.append((f"{name} (CRC)", crc))
    for p in sorted(glob.glob(os.path.join(d, "ref-*.json"))):
        rows.append((os.path.basename(p)[len("ref-") : -len(".json")] + " (τ = 1)", load(p)))
    srcs = [s for s in ORDER if any(s in r["sources"] for _, r in rows)]
    lines = [f"# {name}: results (docs/s6-gen-prereg.md)", ""]
    lines.append("| model | params | " + " | ".join(srcs) + " |")
    lines.append("|---|---:|" + "---:|" * len(srcs))
    for n, r in rows:
        cells = [f"{r['sources'][s]['accuracy']:.3f}" if s in r["sources"] else "—" for s in srcs]
        lines.append(f"| {n} | {r['n_params']:,} | " + " | ".join(cells) + " |")
    lines.append("")
    pub = ", ".join(f"{k} {v}" for k, v in PUBLISHED.items())
    lines.append(f"Published {BENCH} accuracies: {pub}. Majority class: 0.398.")
    lines.append("")
    lines.append("| model | ECE (all) | mean block passes | early-exit rate | median µs / decision (native) |")
    lines.append("|---|---:|---:|---:|---:|")
    for n, r in rows:
        x = r["sources"]["all"]
        lines.append(
            f"| {n} | {x['ece']:.4f} | {x['mean_layers']:.2f} | {x['early_exit_rate']:.3f} | {x['median_us']:.0f} |"
        )
    indist = maybe(os.path.join(d, f"indist-{name}.json"))
    if indist:
        for s, x in indist["sources"].items():
            if s != "all":
                lines.append("")
                lines.append(f"In distribution (trained on, not zero-shot): {s} {x['accuracy']:.3f} (n {x['n']}).")

    if base:
        games = [g for g, v in base.items() if isinstance(v, dict) and "model" in v]
        lines += ["", "| arena mean score | " + " | ".join(games) + " |", "|---|" + "---:|" * len(games)]
        for n, r in [("s3-pico", base), (f"{name} (τ = 1)", arena_full), (f"{name} (CRC)", arena_crc)]:
            if r:
                cells = [f"{r[g]['model']['mean_score']:.2f}" if g in r else "—" for g in games]
                lines.append(f"| {n} | " + " | ".join(cells) + " |")

    trunc = maybe(os.path.join(d, "truncation.json"))
    if trunc:
        t = trunc["all"]
        lines.append("")
        lines.append(
            f"Truncated (first 500 records per source, n {t['n']}): state {t['state']:.1%}, "
            f"head {t['head']:.1%}, either {t['any']:.1%}."
        )
        if BENCH in trunc:
            b = trunc[BENCH]
            lines.append(f"On {BENCH}: state {b['state']:.1%}, head {b['head']:.1%}.")
    if a.pretrain_log and os.path.exists(a.pretrain_log):
        with open(a.pretrain_log) as f:
            steps = [json.loads(l) for l in f if l.strip()]
        last = next((e for e in reversed(steps) if "loss" in e and "epoch" in e), None)
        if last:
            tokens = last["step"] * PRETRAIN_TOKENS_PER_STEP
            lines.append(
                f"Pretraining: {last['step']:,} steps, {tokens / 1e9:.2f}B tokens "
                f"({tokens / full['n_params']:.0f} per parameter), epoch {last['epoch']}, "
                f"MLM loss {last['loss']:.3f}."
            )
    wasm = maybe(os.path.join(d, "wasm.json"))
    if wasm:
        lines.append(
            f"WASM ({wasm['n']} {BENCH} decisions): median {wasm['median_wasm_us']:.0f} µs, "
            f"max |p_wasm - p_native| {wasm['max_abs_diff']:.2e}, argmax agree {wasm['argmax_agree']}."
        )

    lines.append("")
    bad_full = check(full, arena_full, base, a.size)
    bad_crc = check(crc, arena_crc, base, a.size) if crc else ["CRC thresholds not kept"]
    for n, bad in [("τ = 1", bad_full), ("CRC", bad_crc)]:
        lines.append(f"- gate at {n}: " + ("holds" if not bad else "fails (" + "; ".join(bad) + ")"))
    if not bad_full and not bad_crc:
        verdict, ok = "PASS: publish at the CRC thresholds", True
    elif not bad_full:
        verdict, ok = "PASS: publish at full depth", True
    else:
        verdict, ok = "FAIL: not published", False
    if a.size == "micro":
        acc = full["sources"].get(BENCH, {}).get("accuracy", 0.0)
        lines.append(f"- announced goal {BENCH} >= {TARGET_MICRO}: {'met' if acc >= TARGET_MICRO else 'not met'} ({acc:.4f})")
    lines.append(f"- gate: {verdict}")
    md = "\n".join(lines) + "\n"
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
