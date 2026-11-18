#!/usr/bin/env python3
"""s6-route v7 (docs/s6-route7-prereg.md 8, v6 8 G4 and 14 item 8): the Python reference of the
pair under D4, and G4's JSON from the engine's WASM bench. Standard library only.

  reference  one JSON line per slate ({uid, gold, probs}, in slate order): the fused
             probabilities p(i) ∝ (p_model(i) · p_embedding(i))^beta of v3's rule
             (scripts/s6-route3-fit.py fuse) at the beta of v7's fit (runs/s6-route7/fit.json).
             The model dump is sky-eval's on the model copy rendered by D4 (`score <s> rank <r>; `
             in each tool option: `sky-data route render --render scores-in-options`); the
             embedding dump is potion-cos-4bit's on the plain slate file
             (scripts/route-baselines.py). The three files must hold the same slates in the same
             order (uid and gold), and every tool option of the model copy must carry D4's prefix.

      s6-route7-reference.py reference --slates <t5_tools.jsonl> --rendered <t5_tools.model.jsonl> \\
          --model <model dump> --embedding <potion-cos-4bit dump> --fit runs/s6-route7/fit.json \\
          --out <t5_tools.fused.jsonl>

  g4         G4's JSON in the fields of v6's amendment 2 point 5, from the one JSON line of
             crates/sky-web/bench/route3-node.mjs (run with --render scores-in-options and --n 200):
             max_abs_diff, argmax_agree, n, median_ms (the bench's median_cached_us / 1000: the
             request after the first on the same tool list, the embedding part's tool list
             cached and the model part encoding the options each time, option (i) of v4 3.4 and
             8), bytes (the model and embedding files as shipped), with the bench's other fields
             kept under `bench`.

      s6-route7-reference.py g4 --bench <wasm7.json> --out runs/s6-route7/g4.json
"""
import argparse
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('s6_route3_fit', os.path.join(HERE, 's6-route3-fit.py'))
fit3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit3)

D4 = re.compile(r'^score -?\d+\.\d\d rank \d+; ')
G4_FIELDS = ('max_abs_diff', 'argmax_agree', 'n', 'median_ms', 'bytes')


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_of(r):
    return r['gold'] if 'gold' in r else r['target'].index(1.0)


def reference(slates, rendered, model, embedding, beta):
    n = len(slates)
    for name, rows in (('the model copy', rendered), ('the model dump', model), ('the embedding dump', embedding)):
        if len(rows) != n:
            raise SystemExit('%s: %d lines for %d slates' % (name, len(rows), n))
    for i, (s, r, m, e) in enumerate(zip(slates, rendered, model, embedding)):
        uids = {s['uid'], r['uid'], m.get('uid', s['uid']), e.get('uid', s['uid'])}
        if len(uids) != 1 or len({gold_of(s), gold_of(r), gold_of(m), gold_of(e)}) != 1:
            raise SystemExit('line %d: not the same slate in the four files (%s)' % (i + 1, sorted(uids)))
        tools = r['opts'][:-1]
        if not tools or not all(D4.match(o) for o in tools):
            raise SystemExit('line %d (%s): the model copy is not rendered by D4 (score <s> rank <r>; )' % (i + 1, s['uid']))
        if len(m['probs']) != len(s['opts']) or len(e['probs']) != len(s['opts']):
            raise SystemExit('line %d (%s): the dumps do not have one probability per option' % (i + 1, s['uid']))
    rows = lambda d: [{'probs': x['probs'], 'gold': gold_of(x)} for x in d]
    fused = fit3.fuse(rows(model), rows(embedding), beta)
    return [{'uid': s['uid'], 'gold': x['gold'], 'probs': x['probs']} for s, x in zip(slates, fused)]


def g4(bench):
    for k in ('n', 'max_abs_diff', 'argmax_agree', 'median_cached_us', 'bytes_model', 'bytes_embedding'):
        if bench.get(k) is None:
            raise SystemExit('the bench JSON has no %s (route3-node.mjs with a reference file, not --uncached)' % k)
    if bench.get('render') != 'scores-in-options' or bench.get('fast'):
        raise SystemExit('the bench did not route by D4 at full depth (render %r, fast %r)'
                         % (bench.get('render'), bench.get('fast')))
    return {
        'about': "s6-route v7 (docs/s6-route7-prereg.md 8, v6 8 G4 and amendment 2 point 5): the engine's WASM "
                 "bench (crates/sky-web/bench/route3-node.mjs --render scores-in-options) against the Python "
                 "reference of the pair under D4; median_ms is the bench's median_cached_us / 1000",
        'max_abs_diff': bench['max_abs_diff'], 'argmax_agree': bench['argmax_agree'], 'n': bench['n'],
        'median_ms': bench['median_cached_us'] / 1000.0,
        'bytes': bench['bytes_model'] + bench['bytes_embedding'],
        'bench': bench,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('reference')
    for k in ('--slates', '--rendered', '--model', '--embedding', '--fit', '--out'):
        r.add_argument(k, required=True)
    g = sub.add_parser('g4')
    g.add_argument('--bench', required=True)
    g.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'reference':
        with open(a.fit, encoding='utf-8') as f:
            beta = json.load(f)['beta']
        out = reference(load(a.slates), load(a.rendered), load(a.model), load(a.embedding), beta)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            for x in out:
                f.write(json.dumps(x) + '\n')
        print(json.dumps({'slates': len(out), 'beta': beta,
                          'accuracy': sum(fit3.argmax(x['probs']) == x['gold'] for x in out) / max(len(out), 1)}))
    else:
        with open(a.bench, encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]
        if len(lines) != 1:
            raise SystemExit('%s: the bench prints one JSON line, found %d' % (a.bench, len(lines)))
        out = g4(json.loads(lines[0]))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
            f.write('\n')
        print(json.dumps({k: out[k] for k in G4_FIELDS}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
