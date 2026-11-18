#!/usr/bin/env python3
"""The readings of s6-route v5's pilot and its rule (docs/s6-route5-prereg.md, section 5): v4's
(scripts/s6-route4-pilot.py, imported as it is) with v5's names.

Standard library only; run with `python3 -I`. Validation files only: no test or fresh file is
read (a path under a fresh folder is refused).

  s6-route5-pilot.py read --ckpt <dir> --val <v5 build>/val
      the readings of one kept checkpoint (scripts/train-route5.sh writes the dumps):
      `<name>.dump.jsonl` (sky-eval --dump-probs, the model part, full depth) for every name of
      MODEL_FILES (v5_tools, v5_tools_masked, the two probe copies, the renamed and natural
      validation records; v4_tools and v_tools reported), and `pair_<name>.jsonl` (sky-route fuse,
      the pair at the embedding file's beta, 0.81) for PAIR_FILES; writes <dir>/readings.json
  s6-route5-pilot.py summary --pilot <pilot dir> --bar <potion-cos on v5_tools - 0.15> --out <summary.json>
      every checkpoint's readings, and the pilot rule at the last checkpoint: abort (1) when the
      model part's v5_tools accuracy is below the bar the amendment of 4.5 writes, abort (2) when
      the rewritten distractor is the model part's answer on more than 0.30 of the tool-gold slates
      of v5_tools_masked_rw; nothing else is chosen. Exit 0; the summary's `abort` says what the
      rule gave.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route4_pilot', os.path.join(HERE, 's6-route4-pilot.py'))
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

MODEL_FILES = ['v5_tools', 'v5_tools_masked', 'v5_tools_rw', 'v5_tools_masked_rw', 'r_tools_renamed',
               'r_tools_natural', 'v4_tools', 'v_tools']
PAIR_FILES = ['v5_tools', 'v5_tools_masked']
PROBE_ABORT = p4.PROBE_ABORT
load = p4.load


def read(ckpt, val):
    """v4's reading of one checkpoint, on v5's files."""
    p4.MODEL_FILES, p4.PAIR_FILES = MODEL_FILES, PAIR_FILES
    return p4.read(ckpt, val)


def summary(pilot, bar, out):
    rows = []
    for p in glob.glob(os.path.join(pilot, '*', 'readings.json')):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        sys.exit('no readings under %s' % pilot)
    rows.sort(key=lambda r: p4.minutes(r['checkpoint']))
    last = rows[-1]
    m = last['model']
    acc = m['v5_tools']['accuracy']
    probe = m['v5_tools_masked_rw']['answer = the rewritten distractor (tool-gold slates)']
    abort = []
    if acc < bar:
        abort.append('1: the model part on v5_tools %.4f < %.4f' % (acc, bar))
    if probe > PROBE_ABORT:
        abort.append('2: the rewritten distractor answered on %.4f > %.2f of v5_tools_masked_rw' % (probe, PROBE_ABORT))
    s = {'about': 's6-route v5: the pilot (docs/s6-route5-prereg.md 5), validation only',
         'bar (rule 1, from the amendment of 4.5)': bar, 'probe bar (rule 2)': PROBE_ABORT,
         'last checkpoint': last['checkpoint'], 'model v5_tools': acc,
         'rewritten distractor answered on v5_tools_masked_rw': probe,
         'reported: model v4_tools': m.get('v4_tools', {}).get('accuracy'),
         'reported: model v_tools': m.get('v_tools', {}).get('accuracy'),
         'abort': bool(abort), 'why': abort, 'checkpoints': rows}
    with open(out, 'w') as f:
        json.dump(s, f, indent=1)
    print(json.dumps({k: s[k] for k in ('last checkpoint', 'model v5_tools',
                                        'rewritten distractor answered on v5_tools_masked_rw', 'abort', 'why')}))
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
