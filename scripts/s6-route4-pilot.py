#!/usr/bin/env python3
"""The readings of s6-route v4's pilot and its rule (docs/s6-route4-prereg.md, section 5).

Standard library only; run with `python3 -I`. Validation files only: no test or fresh file is
read (a path under a fresh folder is refused).

  s6-route4-pilot.py read --ckpt <dir> --val <v4 build>/val
      the readings of one kept checkpoint from the dumps scripts/train-route4.sh writes in <dir>:
      `<name>.dump.jsonl` (sky-eval --dump-probs, the model part, full depth) for every name of
      MODEL_FILES, and `pair_<name>.jsonl` (sky-route fuse --out, the pair at the embedding file's
      beta, 0.81) for PAIR_FILES; writes <dir>/readings.json
  s6-route4-pilot.py summary --pilot <pilot dir> --bar <potion-cos on v4_tools - 0.15> --out <summary.json>
      every checkpoint's readings (<pilot>/ckpt-*/readings.json, and <pilot>/last), and the pilot
      rule at the last checkpoint: abort (1) when the model part's v4_tools accuracy is below the
      bar written in the amendment, abort (2) when the rewritten distractor is the model part's
      answer on more than 0.30 of the tool-gold slates of v4_tools_masked_rw; nothing else is
      chosen. Exit 0; the summary's `abort` says what the rule gave.
"""
import argparse
import glob
import json
import os
import re
import sys

MODEL_FILES = ['v4_tools', 'v4_tools_masked', 'v4_tools_rw', 'v4_tools_masked_rw', 'r_tools_renamed',
               'r_tools_natural', 'v_tools']
PAIR_FILES = ['v4_tools', 'v4_tools_masked']
PROBE_ABORT = 0.30


def load(path):
    if '/fresh/' in os.path.abspath(path):
        sys.exit('refusing a fresh file: the pilot reads validation only (%s)' % path)
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def gold_of(r):
    return r['target'].index(1.0)


def reading(slates, probs, follow=False):
    if len(slates) != len(probs):
        sys.exit('a dump of %d lines for %d slates' % (len(probs), len(slates)))
    n = len(slates)
    right = none = tg = tg_right = fol = 0
    for r, d in zip(slates, probs):
        p = d['probs']
        if len(p) != len(r['opts']):
            sys.exit('a dump line with another number of options (%s)' % r.get('uid'))
        a, g = argmax(p), gold_of(r)
        right += a == g
        none += a == len(p) - 1
        if g != len(p) - 1:
            tg += 1
            tg_right += a == g
            if follow:
                fol += a == r['follow']
    out = {'n': n, 'accuracy': right / max(n, 1), 'none answers': none / max(n, 1),
           'tool-gold slates': tg, 'tool-gold accuracy': tg_right / max(tg, 1)}
    if follow:
        out['answer = the rewritten distractor (tool-gold slates)'] = fol / max(tg, 1)
    return out


def read(ckpt, val):
    out = {'checkpoint': os.path.basename(os.path.normpath(ckpt)), 'model': {}, 'pair': {}}
    for name in MODEL_FILES:
        dump = os.path.join(ckpt, name + '.dump.jsonl')
        if not os.path.exists(dump):
            continue
        s = load(os.path.join(val, name + '.model.jsonl'))
        out['model'][name] = reading(s, load(dump), follow='rw' in name)
    for name in PAIR_FILES:
        dump = os.path.join(ckpt, 'pair_%s.jsonl' % name)
        if os.path.exists(dump):
            out['pair'][name] = reading(load(os.path.join(val, name + '.jsonl')), load(dump))
    m = out['model']
    if 'r_tools_renamed' in m and 'r_tools_natural' in m:
        out['renamed - natural (model)'] = m['r_tools_renamed']['accuracy'] - m['r_tools_natural']['accuracy']
    with open(os.path.join(ckpt, 'readings.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


def minutes(name):
    m = re.fullmatch(r'ckpt-(\d+(?:\.\d+)?)', name)
    return float(m.group(1)) if m else float('inf')


def summary(pilot, bar, out):
    rows = []
    for p in glob.glob(os.path.join(pilot, '*', 'readings.json')):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        sys.exit('no readings under %s' % pilot)
    rows.sort(key=lambda r: minutes(r['checkpoint']))
    last = rows[-1]
    m = last['model']
    acc = m['v4_tools']['accuracy']
    probe = m['v4_tools_masked_rw']['answer = the rewritten distractor (tool-gold slates)']
    abort = []
    if acc < bar:
        abort.append('1: the model part on v4_tools %.4f < %.4f' % (acc, bar))
    if probe > PROBE_ABORT:
        abort.append('2: the rewritten distractor answered on %.4f > %.2f of v4_tools_masked_rw' % (probe, PROBE_ABORT))
    s = {'about': 's6-route v4: the pilot (docs/s6-route4-prereg.md 5), validation only',
         'bar (rule 1, from the amendment)': bar, 'probe bar (rule 2)': PROBE_ABORT,
         'last checkpoint': last['checkpoint'], 'model v4_tools': acc,
         'rewritten distractor answered on v4_tools_masked_rw': probe,
         'abort': bool(abort), 'why': abort, 'checkpoints': rows}
    with open(out, 'w') as f:
        json.dump(s, f, indent=1)
    print(json.dumps({k: s[k] for k in ('last checkpoint', 'model v4_tools',
                                        'rewritten distractor answered on v4_tools_masked_rw', 'abort', 'why')}))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('read')
    r.add_argument('--ckpt', required=True)
    r.add_argument('--val', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--pilot', required=True)
    s.add_argument('--bar', type=float, required=True)
    s.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'read':
        read(a.ckpt, a.val)
    else:
        summary(a.pilot, a.bar, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
