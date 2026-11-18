#!/usr/bin/env python3
"""The readings of s6-route v6's pilot and its rule (docs/s6-route6-prereg.md, section 5): v5's
(scripts/s6-route5-pilot.py, imported as it is) and the blind readings.

Standard library only; run with `python3 -I`. Validation files only (a fresh path is refused).

  s6-route6-pilot.py read --ckpt <dir> --val <build6>/val
      v5's readings of one kept checkpoint, then the blind ones (scripts/s6-route6-blind.py) of
      the model part and of the pair on v5_tools and v5_tools_masked: B0 from
      `<name>.blind.dump.jsonl` (sky-eval on <val>/<name>.blind.model.jsonl) and
      `pair_<name>.blind.jsonl` (sky-route fuse on <val>/<name>.blind.jsonl), B1 from
      `<name>.blind-bare.dump.jsonl` (sky-eval on <val>/<name>.blind-bare.model.jsonl); written
      into <dir>/readings.json under `blind`
  s6-route6-pilot.py summary --pilot <pilot dir> --bar <potion-cos on v5_tools - 0.15> --out <summary.json>
      the pilot rule at the last checkpoint: v5's rules 1 and 2, and
        3. abort if the model part's blind accuracy (B0) on the tool-right slates of v5_tools is
           above 1/7 + 0.05 = 0.1929;
        4. the same on v5_tools_masked.
      Nothing else is chosen. Exit 0; the summary's `abort` says what the rule gave.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, f):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p5 = _load('route5_pilot', 's6-route5-pilot.py')
b6 = _load('route6_blind', 's6-route6-blind.py')
BLIND_FILES = ['v5_tools', 'v5_tools_masked']
BLIND_BAR = 1 / 7 + 0.05
load = p5.load


def blind_reading(slates_path, dump_path, field='probs'):
    s, _ = b6.read(load(slates_path), load(dump_path), field)
    return {k: s.get(k) for k in ('tool-right slates', 'blind accuracy', 'chance c', 'blind accuracy - c',
                                   'answer with none allowed: accuracy', 'none share', 'blind none gap',
                                   'share asked', 'intervals, slates resampled')}


def read(ckpt, val):
    out = p5.read(ckpt, val)
    blind = {}
    for name in BLIND_FILES:
        for reading, suffix in (('B0', '.blind'), ('B1', '.blind-bare')):
            dump = os.path.join(ckpt, name + suffix + '.dump.jsonl')
            if os.path.exists(dump):
                blind['model %s %s' % (reading, name)] = blind_reading(
                    os.path.join(val, name + suffix + '.model.jsonl'), dump)
        pair = os.path.join(ckpt, 'pair_%s.blind.jsonl' % name)
        if os.path.exists(pair):
            blind['pair B0 %s' % name] = blind_reading(os.path.join(val, name + '.blind.jsonl'), pair)
    out['blind'] = blind
    with open(os.path.join(ckpt, 'readings.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


def summary(pilot, bar, out):
    s = p5.summary(pilot, bar, out)
    last = s['checkpoints'][-1]
    why = list(s['why'])
    blind = last.get('blind', {})
    vals = {}
    for rule, name in (('3', 'v5_tools'), ('4', 'v5_tools_masked')):
        r = blind.get('model B0 %s' % name)
        if r is None or r.get('blind accuracy') is None:
            sys.exit('no blind reading of the model part on %s at the last checkpoint' % name)
        a = r['blind accuracy']
        vals[name] = a
        if a > BLIND_BAR:
            why.append('%s: the model part\'s blind accuracy (B0) on %s %.4f > %.4f' % (rule, name, a, BLIND_BAR))
    s.update({'about': 's6-route v6: the pilot (docs/s6-route6-prereg.md 5), validation only',
              'blind bar (rules 3 and 4)': BLIND_BAR,
              'model B0 blind accuracy': vals, 'abort': bool(why), 'why': why})
    with open(out, 'w') as f:
        json.dump(s, f, indent=1)
    print(json.dumps({k: s[k] for k in ('last checkpoint', 'model v5_tools', 'model B0 blind accuracy', 'abort', 'why')}))
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
