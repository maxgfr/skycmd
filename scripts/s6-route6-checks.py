#!/usr/bin/env python3
"""The data checks of s6-route v6 after the When2Call fix (docs/s6-route6-prereg.md 4.4, 4.8 and
amendment 1), no model.

Standard library only; run with `python3 -I`. v5's checks (scripts/s6-route5-checks.py, C1 to C6
and N1 to N6 with v5's tolerances) are imported as they are and run on the rebuilt build; every
one is REPORTED, and none stops v6, except the one stop rule of 4.8:

    if, in any N6 family judged by v5's floor (at least 200 tool-gold slates), the accuracy of N6
    or of its inverse exceeds the family's none share by more than 0.10, v6 stops before
    training, and the table is published.

  s6-route6-checks.py checks --build <b6> --requests <v2>/stage/requests.jsonl --tools <v2>/stage/tools.jsonl \\
      --pools <b6>/stage/pools5.json --plan-counts <b6>/counts/v5-plan.json --val <b6>/val/v5_tools.jsonl ... \\
      [--v5-report runs/s6-route5/checks.json] --out runs/s6-route6/checks.json [--md runs/s6-route6/checks.md]
      exit 0 when the stop rule does not fire (whatever the other checks show), 2 when it does

  s6-route6-checks.py spread --build <b6> --v5-build <b5> --plan-counts <b5>/counts/v5-plan.json \\
      --out runs/s6-route6/spread.json [--md runs/s6-route6/spread.md]
      the spread table of 4.8 and amendment 1 (reported): for the When2Call group, the share of
      slates whose options spread over several L0 cells (spread above 0), by kind (natural, hard,
      renamed), by none-right or tool-right and by split, after the fix with v5's beside it; and
      the slates by level, with the L6 share, by the same splits. Exit 0.
"""
import argparse
import collections
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route5_checks', os.path.join(HERE, 's6-route5-checks.py'))
c5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c5)
c4 = c5.c4

STOP_EXCESS = 0.10
GROUP = 'when2call'
KINDS = ('natural', 'hard', 'renamed')


def stop_rule(families):
    """4.8: the judged N6 families where N6 or its inverse exceeds the none share by more than
    0.10."""
    out = []
    for name, f in sorted(families.items()):
        if not f.get('judged') or 'none share (chance)' not in f:
            continue
        for r in f['rules']:
            excess = r['accuracy'] - f['none share (chance)']
            if r['answered none'] and excess > STOP_EXCESS + 1e-12:
                out.append({'family': name, 'rule': r['rule'], 'accuracy': r['accuracy'],
                            'none share': f['none share (chance)'], 'excess': excess})
    return out


def largest_excess(families):
    rows = []
    for name, f in families.items():
        for r in f.get('rules', []):
            if r['answered none']:
                rows.append((r['accuracy'] - f['none share (chance)'], name, r['rule'], f.get('judged')))
    rows.sort(reverse=True)
    return [{'excess': e, 'family': n, 'rule': r, 'judged': j} for e, n, r, j in rows[:5]]


def checks(a):
    rep = c5.run(a.build, a.requests, a.tools, a.pools, a.plan_counts, a.val, a.fresh, a.fresh_manifest)
    v5_verdict = rep['verdict']
    stops = stop_rule(rep['N6 families'])
    v5_beside = None
    if a.v5_report and os.path.exists(a.v5_report):
        with open(a.v5_report, encoding='utf-8') as f:
            old = json.load(f)
        v5_beside = {'N6 families': {k: {'none share': x.get('none share (chance)'),
                                         'rules': [(r['rule'], r['accuracy']) for r in x.get('rules', [])]}
                                     for k, x in old.get('N6 families', {}).items() if k.startswith('records ' + GROUP)},
                     'verdict': {k: (len(x) if isinstance(x, list) else x) for k, x in old.get('verdict', {}).items()}}
    rep['v5 verdict, reported (none stops v6)'] = v5_verdict
    rep['v5 beside (When2Call families)'] = v5_beside
    rep['verdict'] = {
        'stop rule of 4.8 (N6 or its inverse more than 0.10 above the none share in a judged family)': stops,
        'the largest N6 excesses': largest_excess(rep['N6 families']),
        'v6 stops before training': bool(stops),
        'exit': 2 if stops else 0,
        'meaning': [('The stop rule of 4.8 fires: after the fix a none cue of more than +0.10 above the none share '
                     'remains in at least one judged family. v6 stops before training, and the table is published '
                     'with the spread table (11).')] if stops else
                   [('The stop rule of 4.8 does not fire: every check is reported, none stops v6, and the model\'s '
                     'blind reading decides (G6).')],
    }
    return rep


# ---------------------------------------------------------------------------------- the spread

def spread_table(build, classes):
    """For the When2Call group's records: by split, kind and none-right or tool-right, the
    slates, those whose spread is above 0, and the slates by level."""
    rows = {}
    rd = os.path.join(build, 'records', GROUP)
    for split in ('train', 'val'):
        p = os.path.join(rd, split + '.jsonl')
        if not os.path.exists(p):
            continue
        for r in c4.load(p):
            key = '%s %s %s' % (split, r['kind'], 'none-right' if c4.is_none(r) else 'tool-right')
            e = rows.setdefault(key, {'slates': 0, 'spread above 0': 0, 'slates by level': [0] * 7})
            e['slates'] += 1
            s = c5.spread(r, classes)
            e['spread above 0'] += bool(s and s > 0)
            lv = r.get('cell_level')
            if lv is not None:
                e['slates by level'][lv] += 1
    for e in rows.values():
        e['share spread above 0'] = e['spread above 0'] / max(e['slates'], 1)
        e['share at L6'] = e['slates by level'][6] / max(e['slates'], 1)
    return dict(sorted(rows.items()))


def spread(a):
    with open(a.plan_counts, encoding='utf-8') as f:
        classes = json.load(f)['classes of F2 and F3']
    after = spread_table(a.build, classes)
    before = spread_table(a.v5_build, classes) if a.v5_build else {}
    return {'about': 's6-route v6: the spread table of docs/s6-route6-prereg.md 4.8 and amendment 1 (reported)',
            'group': GROUP, 'after the fix': after, "v5's build": before}


def markdown(rep):
    if 'after the fix' in rep:
        out = ['# s6-route v6: the spread table of When2Call (docs/s6-route6-prereg.md 4.8, amendment 1)', '',
               '| split, kind, answer | slates | spread above 0 | L6 | v5: slates | v5: spread above 0 | v5: L6 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        b = rep["v5's build"]
        for k, e in rep['after the fix'].items():
            o = b.get(k, {})
            out.append('| %s | %d | %.4f | %.4f | %s | %s | %s |' % (
                k, e['slates'], e['share spread above 0'], e['share at L6'], o.get('slates', ''),
                '%.4f' % o['share spread above 0'] if o else '', '%.4f' % o['share at L6'] if o else ''))
        out += ['', '## Slates by level (L0 ... L6)', '']
        for k, e in rep['after the fix'].items():
            out.append('- %s: %s (v5: %s)' % (k, ' '.join(map(str, e['slates by level'])),
                                             ' '.join(map(str, b.get(k, {}).get('slates by level', [])))))
        return '\n'.join(out) + '\n'
    v = rep['verdict']
    out = ['# s6-route v6: the data checks after the fix (docs/s6-route6-prereg.md 4.4, 4.8)', '',
           'exit %d; the stop rule fires in %d families' % (v['exit'], len(v['stop rule of 4.8 (N6 or its inverse more than 0.10 above the none share in a judged family)'])), '']
    out += v['meaning'] + ['', '## The largest N6 excesses', '', '| family | rule | excess | judged |', '|---|---|---:|---|']
    out += ['| %s | %s | %+.4f | %s |' % (x['family'], x['rule'], x['excess'], x['judged']) for x in v['the largest N6 excesses']]
    out += ['', "## v5's checks, reported (none stops v6)", '', c5.markdown(dict(rep, verdict=rep['v5 verdict, reported (none stops v6)']))]
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('checks')
    for k in ('--build', '--requests', '--tools', '--pools', '--plan-counts', '--fresh-manifest', '--v5-report'):
        c.add_argument(k)
    c.add_argument('--val', nargs='*', default=[])
    c.add_argument('--fresh', nargs='*', default=[])
    c.add_argument('--out', required=True)
    c.add_argument('--md')
    s = sub.add_parser('spread')
    s.add_argument('--build', required=True)
    s.add_argument('--v5-build')
    s.add_argument('--plan-counts', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--md')
    a = ap.parse_args(argv)
    rep = checks(a) if a.cmd == 'checks' else spread(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(rep, f, indent=1, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    if a.md:
        with open(a.md, 'w') as f:
            f.write(markdown(rep))
    if a.cmd == 'spread':
        print(json.dumps({k: (e['slates'], round(e['share spread above 0'], 4)) for k, e in rep['after the fix'].items()}))
        return 0
    v = rep['verdict']
    print(json.dumps({'stop rule fires': v['v6 stops before training'], 'exit': v['exit']}))
    return v['exit']


if __name__ == '__main__':
    sys.exit(main())
