#!/usr/bin/env python3
"""s6-route v6 (docs/s6-route6-prereg.md 3, as v5 and v4 3.2): β and τ_ask6 of the pair, read on
`v5_tools` only, by v3's rules (scripts/s6-route3-fit.py, imported as it is):

  beta      p(i) ∝ (p_model(i) · p_embedding(i))^beta; the value of the grid 0.01 .. 4.00 that
            minimises the NLL on v5_tools
  tau_ask6  v2's rule on the fused probabilities: the smallest top probability at which the
            accuracy among answered requests is >= 0.95

Inputs are validation dumps of v5_tools only: the model part at full depth and at its first exit
(sky-eval --dump-probs), and potion-cos-4bit with v2's fitted values (scripts/route-baselines.py
score --split v5val). No test or fresh file is read, and nothing is chosen on a blind reading.

  s6-route6-fit.py --model <v5_tools.dump.jsonl> --exit0 <v5_tools.exit0.jsonl> \\
      --partner <eval>/v5val/potion-cos-4bit/v5_tools.jsonl --out runs/s6-route6/fit.json
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('s6_route3_fit', os.path.join(HERE, 's6-route3-fit.py'))
fit3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit3)


def gold_rows(rows):
    """sky-eval's dumps carry `target`; the baselines' carry `gold`: both to `gold`."""
    out = []
    for r in rows:
        g = r['gold'] if 'gold' in r else r['target'].index(1.0)
        out.append({'probs': r['probs'], 'gold': g})
    return out


def fit(model, exit0, partner):
    model, exit0, partner = gold_rows(model), gold_rows(exit0), gold_rows(partner)
    golds = [r['gold'] for r in model]
    for name, rows in (('exit0', exit0), (fit3.PARTNER, partner)):
        if [r['gold'] for r in rows] != golds:
            raise SystemExit('%s: not the same slates in the same order as the model\'s dump' % name)
    grid = [i / 100 for i in range(1, 401)]
    beta = min(grid, key=lambda b: fit3.nll(fit3.fuse(model, partner, b)))
    fused = fit3.fuse(model, partner, beta)
    fast = fit3.fuse(exit0, partner, beta)
    return {
        'about': 'docs/s6-route6-prereg.md 3: beta and tau_ask6 read on v5_tools only (v3\'s rules)',
        'slates': len(model), 'partner': fit3.PARTNER, 'beta': beta,
        'nll_at_beta': fit3.nll(fused), 'tau_ask6': fit3.tau_ask(fused),
        'system': {'accuracy': fit3.accuracy(fused), 'ece': fit3.ece(fused)},
        'fast_mode': {'accuracy': fit3.accuracy(fast), 'ece': fit3.ece(fast)},
        'alone': {'model': fit3.accuracy(model), 'model_exit0': fit3.accuracy(exit0),
                  fit3.PARTNER: fit3.accuracy(partner)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k in ('--model', '--exit0', '--partner', '--out'):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    for p in (a.model, a.exit0, a.partner):
        if '/fresh/' in os.path.abspath(p) or 'blind' in os.path.basename(p):
            sys.exit('refusing %s: the fit reads the ordinary validation dumps of v5_tools only' % p)
    out = fit(fit3.load(a.model), fit3.load(a.exit0), fit3.load(a.partner))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    print(json.dumps({k: out[k] for k in ('beta', 'tau_ask6')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
