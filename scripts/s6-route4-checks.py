#!/usr/bin/env python3
"""The data checks of s6-route v4 before any training (docs/s6-route4-prereg.md 4.4), no model.

Standard library only; run with `python3 -I`. Reads slate files, writes one JSON report and one
Markdown table, and exits:

  0  every judged check holds;
  1  C1 or C3 fails: the build has a defect (a technical failure, fixed before anything else);
  2  a rule of C2 fails: the build stops. The report says whether the one remedy of 4.4 is open
     (every failing rule among R1 to R5, R8, R9, R12, R13 and their inverses) or v4 stops.

  s6-route4-checks.py --build <v4 build> --requests <v2 build>/stage/requests.jsonl \\
      [--val <v4 build>/val/v4_tools.jsonl ...] [--fresh <t4>/fresh/t4_tools.jsonl ...] \\
      [--fresh-manifest <t4>/fresh/FRESH4.json] --out <report.json> [--md <table.md>]

Cells: every (group, kind, split) of `<build>/records/<group>/<split>.jsonl`, and each file of
--val and --fresh. A cell with fewer than 200 slates whose answer is a tool is reported and not
judged. The --fresh files (the fresh test, 6.2 item 5) are always reported and never judged: a
rule above its tolerance there is reported with the gate and does not change the file.

- C1, the rewrite is independent of the answer (amendment 1 of 18/11/2026, which replaced the
  frozen bars 0.01 and 0.03 before any data was built): in every judged cell, each of the two
  rates P(rewritten | right tool) (over its n right tools) and P(rewritten | other tool option)
  (over its n other tool options) is within 3 standard errors of a fair draw of 0.5,
  |rate - 0.5| <= 3 sqrt(0.25 / n); and over the tool-gold slates of every record cell pooled,
  both rates are within 0.01 of 0.5. The difference of the two rates, its spread and z are
  reported per cell (the builder's own flag `rewritten`; cells without it are not read for C1).
- C2, request-blind rules at chance: each rule picks one tool option from the option texts alone
  (the score and rank of D4 are taken off first); a tie counts 1/|tie| when the right tool is in
  it; chance c is the mean of 1/(tool options); a rule fails when its accuracy on the n
  tool-gold slates of a cell exceeds c + 0.02 + 2 sqrt(c (1 - c) / n). R1 to R13 of 4.4, each with
  its inverse, and every other rule of scripts/s6-route4-blind.py (marked B), each with its
  inverse; blind's rules that are R1 to R13 word for word are not counted twice, and its
  name-style rules (five classes: Pascal apart from other) are tagged R8 and R9.
- C3, none: in each group and split of the records, none is right on 0.20 of the renamed slates
  (to the nearest slate: the build draws round(0.2 n)) and on 0.20 of the other slates (within one
  slate); in each --val file on 0.20 (to the nearest slate). The share of none by number of options
  is written for every cell, with a chi-square of its independence (reported, not judged: 4.4 C3).
- C4, the surface table: for the right option against the others, by cell, the means of every
  feature of C2 and the share of options from each source (reported, not judged).
- C5, no fresh tool in training: enforced by `sky-data route toucan`, which stops when a tool it
  would show is within v2's cluster rule of a training or validation tool; --fresh-manifest
  reports its count.
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route4_blind', os.path.join(HERE, 's6-route4-blind.py'))
blind = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blind)

MIN_JUDGED = 200
C1_SE = 3.0
C1_POOLED = 0.01
C2_MARGIN = 0.02
NONE_SHARE = 0.2
SCORE = re.compile(r'^score -?[0-9.]+ rank [0-9]+; ')
# 4.4: the remedy is open when every failing rule is one of these (or its inverse)
REMEDY_RULES = {'R1', 'R2', 'R3', 'R4', 'R5', 'R8', 'R9', 'R12', 'R13'}


def plain(o):
    return SCORE.sub('', o, count=1)


def style4(n):
    """R8's four classes (4.4): snake_case, camelCase, spaced words, other."""
    s = blind.name_style(n)
    return s if s in ('snake', 'camel', 'spaced') else 'other'


def features(r, srcmap):
    out = []
    flags = r.get('rewritten')
    for i, (o, t) in enumerate(zip(r['opts'][:-1], r['tools'])):
        f = blind.feats(plain(o), srcmap.get(t, '?'))
        f['style4'] = style4(blind.split_opt(plain(o))[0])
        f['rewritten'] = bool(flags[i]) if flags is not None else None
        f['position'] = i + 1
        out.append(f)
    return out


def neg(rule):
    return lambda F: [-x for x in rule(F)]


def rule_set(with_flags):
    """(label, R tag, rule): R1 to R13 of 4.4 with their inverses, then blind's rules (B)."""
    B = blind.rules()
    R = []

    def both(tag, name, inv_name, f):
        R.append((name, tag, f))
        R.append((inv_name, tag, neg(f)))
    both('R1', 'R1 ends with a period', 'R1 inverse: no final period', B['period'])
    both('R2', 'R2 first word a capitalised word in -s', 'R2 inverse', B['third'])
    both('R3', "R3 the diagnosis's rule (R1, then R2)", 'R3 inverse', B['diagnosis style (period, then -s)'])
    both('R4', 'R4 most words', 'R4 inverse: fewest words', B['most words'])
    both('R5', 'R5 most characters', 'R5 inverse: fewest characters', B['most characters'])
    if with_flags:
        both('R6', 'R6 the option rewritten', 'R6 inverse: an option not rewritten',
             lambda F: [int(bool(x['rewritten'])) for x in F])
    for p in range(7):
        both('R7', 'R7 position %d' % (p + 1), 'R7 inverse: not position %d' % (p + 1),
             lambda F, p=p: [int(i == p) for i in range(len(F))])
    both('R8', "R8 name style (4 classes), the slate's minority", 'R8 inverse: the majority',
         lambda F: blind.minority([x['style4'] for x in F]))
    both('R9', "R9 (name style, R1), the slate's minority", 'R9 inverse: the majority',
         lambda F: blind.minority([(x['style4'], x['period']) for x in F]))
    both('R10', 'R10 a digit in the name', 'R10 inverse', B['digits'])
    both('R11', 'R11 the shortest name', 'R11 inverse: the longest name', B['fewest name length'])
    both('R12', 'R12 the fewest schema-like words', 'R12 inverse: the most', B['fewest schema-like words'])
    both('R13', 'R13 an imperative first word', 'R13 inverse', B['imperative'])
    # every other rule of scripts/s6-route4-blind.py: the ones above are not counted twice; its
    # name-style rules (five classes, Pascal apart from other) are R8 and R9 with one more class
    for name, f in B.items():
        if name in SAME_AS_R or name.startswith('position '):
            continue
        tag = BLIND_TAG.get(name, 'B')
        R.append(('B: ' + name, tag, f))
        if name in BLIND_WITHOUT_INVERSE:
            R.append(('B: inverse of ' + name, tag, neg(f)))
    return R


# blind's rules that are R1 to R5 and R10 to R13 (or their inverses) word for word
SAME_AS_R = {'period', 'not period', 'third', 'not third', 'imperative', 'not imperative', 'digits',
             'not digits', 'diagnosis style (period, then -s)', 'inverse of the diagnosis style',
             'most words', 'fewest words', 'most characters', 'fewest characters',
             'most schema-like words', 'fewest schema-like words', 'most name length',
             'fewest name length'}
BLIND_TAG = {'name style, minority': 'R8', 'name style, majority': 'R8',
             '(name style, period), minority': 'R9'}
# blind's rules whose inverse blind does not list
BLIND_WITHOUT_INVERSE = {'(name style, period), minority'}


def gold_of(r):
    return r['target'].index(1.0)


def is_none(r):
    return gold_of(r) == len(r['opts']) - 1


def chi2_independence(table):
    """Chi-square statistic, degrees of freedom and p-value of a {k: (n, none)} table."""
    rows = [(n - m, m) for n, m in table.values() if n > 0]
    if len(rows) < 2:
        return 0.0, 0, 1.0
    N = sum(a + b for a, b in rows)
    ca = sum(a for a, _ in rows)
    cb = N - ca
    if ca == 0 or cb == 0:
        return 0.0, 0, 1.0
    x2 = 0.0
    for a, b in rows:
        n = a + b
        for obs, col in ((a, ca), (b, cb)):
            e = n * col / N
            x2 += (obs - e) ** 2 / e
    df = len(rows) - 1
    return x2, df, chi2_sf(x2, df)


def chi2_sf(x, k):
    """The upper tail of a chi-square distribution (regularised gamma, series and fraction)."""
    if x <= 0:
        return 1.0
    a, z = k / 2.0, x / 2.0
    if z < a + 1:
        s, t, n = 0.0, 1.0 / a, a
        while abs(t) > 1e-14 * abs(s + t):
            s += t
            n += 1
            t *= z / n
        s += t
        return max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return min(1.0, h * math.exp(-z + a * math.log(z) - math.lgamma(a)))


def cell_report(name, recs, srcmap, judged=True):
    tg = [r for r in recs if not is_none(r)]
    n = len(tg)
    with_flags = bool(recs) and all(r.get('rewritten') is not None for r in recs)
    out = {'slates': len(recs), 'tool-gold slates': n, 'none-gold slates': len(recs) - n,
           'judged': judged and n >= MIN_JUDGED}
    # C3: none by number of options
    by_k = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        e = by_k[len(r['opts'])]
        e[0] += 1
        e[1] += is_none(r)
    x2, df, p = chi2_independence({k: tuple(v) for k, v in by_k.items()})
    out['C3 none by number of options'] = {
        'table': {str(k): {'slates': v[0], 'none': v[1], 'share': v[1] / v[0]} for k, v in sorted(by_k.items())},
        'chi2': x2, 'df': df, 'p': p, 'departs beyond noise (p < 0.001; reported, not judged)': p < 0.001}
    if n == 0:
        out['C2'] = []
        return out
    F = [features(r, srcmap) for r in tg]
    golds = [gold_of(r) for r in tg]
    c = sum(1 / len(f) for f in F) / n
    tol = c + C2_MARGIN + 2 * math.sqrt(c * (1 - c) / n)
    rules = []
    for label, tag, rule in rule_set(with_flags):
        acc = 0.0
        for f, g in zip(F, golds):
            v = rule(f)
            m = max(v)
            best = [i for i, x in enumerate(v) if x == m]
            if g in best:
                acc += 1 / len(best)
        acc /= n
        rules.append({'rule': label, 'tag': tag, 'accuracy': acc, 'chance': c, 'tolerance': tol,
                      'fails': out['judged'] and acc > tol, 'above tolerance': acc > tol})
    rules.sort(key=lambda x: -x['accuracy'])
    out['C2'] = rules
    # C1
    if with_flags:
        kg = sum(bool(r['rewritten'][g]) for r, g in zip(tg, golds))
        oth = [bool(r['rewritten'][i]) for r, g in zip(tg, golds) for i in range(len(r['tools'])) if i != g]
        no_, ko = len(oth), sum(oth)
        rg, ro = kg / n, ko / max(no_, 1)
        tg_, to_ = C1_SE * math.sqrt(0.25 / n), C1_SE * math.sqrt(0.25 / max(no_, 1))
        # the spread of the difference if every flag were an independent draw of p 0.5 (D1)
        sd = 0.5 * math.sqrt(1 / n + 1 / max(no_, 1))
        out['C1'] = {'P(rewritten | right tool)': rg, 'right tools': n, 'its bar (3 SE)': tg_,
                     'P(rewritten | other tool option)': ro, 'other tool options': no_, 'its bar (3 SE) ': to_,
                     'counts (right rewritten, others rewritten)': [kg, ko],
                     'difference (reported)': rg - ro,
                     'sd of the difference under D1 (reported)': sd,
                     'z of the difference (reported)': (rg - ro) / sd,
                     'fails': out['judged'] and (abs(rg - 0.5) > tg_ or (no_ > 0 and abs(ro - 0.5) > to_))}
    # C4: the surface table
    keys = ['period', 'third', 'imperative', 'words', 'chars', 'schema', 'digits', 'namelen', 'position']
    if with_flags:
        keys.append('rewritten')
    right = collections.Counter()
    other = collections.Counter()
    no = 0
    srcs_r, srcs_o = collections.Counter(), collections.Counter()
    sty_r, sty_o = collections.Counter(), collections.Counter()
    for f, g in zip(F, golds):
        for i, x in enumerate(f):
            tgt, sc, st = (right, srcs_r, sty_r) if i == g else (other, srcs_o, sty_o)
            for k in keys:
                tgt[k] += float(x[k])
            sc[x['src']] += 1
            st[x['style4']] += 1
            st['5:' + x['style']] += 1
            no += i != g
    out['C4 surface (right option against the others; reported)'] = {
        'right': {k: right[k] / n for k in keys}, 'others': {k: other[k] / max(no, 1) for k in keys},
        'name style, right': {k: v / n for k, v in sorted(sty_r.items())},
        'name style, others': {k: v / max(no, 1) for k, v in sorted(sty_o.items())},
        'source, right': {k: v / n for k, v in sorted(srcs_r.items())},
        'source, others': {k: v / max(no, 1) for k, v in sorted(srcs_o.items())}}
    return out


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def tool_sources(path):
    if not path:
        return {}
    src = collections.defaultdict(set)
    for r in load(path):
        for t in r.get('tools', []) + ([r['gold']] if r.get('gold') else []) + r.get('exclude', []):
            src[t].add(r['source'])
    return {t: '+'.join(sorted(s)) for t, s in src.items()}


def none_share_checks(cells, records):
    """C3 judged: per (group, split) of the records, the renamed share and the other share; per
    --val file, the share."""
    out = []
    by = collections.defaultdict(lambda: {'renamed': [0, 0], 'other': [0, 0]})
    for (group, kind, split), recs in records.items():
        e = by[(group, split)]['renamed' if kind == 'renamed' else 'other']
        e[0] += len(recs)
        e[1] += sum(is_none(r) for r in recs)
    for (group, split), d in sorted(by.items()):
        for part, (n, m) in d.items():
            if n == 0:
                continue
            ok = m == round(NONE_SHARE * n) if part == 'renamed' else abs(m - NONE_SHARE * n) <= 1.0
            out.append({'cell': '%s %s %s' % (group, split, part), 'slates': n, 'none': m,
                        'share': m / n, 'fails': not ok})
    for name, c in cells.items():
        if c.get('kind') == 'val':
            n, m = c['slates'], c['none-gold slates']
            out.append({'cell': name, 'slates': n, 'none': m, 'share': m / max(n, 1),
                        'fails': n > 0 and m != round(NONE_SHARE * n)})
    return out


def run(build, requests, val, fresh, fresh_manifest):
    srcmap = tool_sources(requests)
    cells = {}
    records = {}
    if build:
        rd = os.path.join(build, 'records')
        for g in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
            for f in sorted(os.listdir(os.path.join(rd, g))):
                if not f.endswith('.jsonl'):
                    continue
                split = f[:-len('.jsonl')]
                by = collections.defaultdict(list)
                for r in load(os.path.join(rd, g, f)):
                    by[r['kind']].append(r)
                for kind, recs in sorted(by.items()):
                    records[(g, kind, split)] = recs
                    c = cell_report('%s %s %s' % (g, kind, split), recs, srcmap)
                    c['kind'] = 'records'
                    cells['records %s %s %s' % (g, kind, split)] = c
    for p in val or []:
        c = cell_report(p, load(p), srcmap)
        c['kind'] = 'val'
        cells['val ' + os.path.basename(p)] = c
    for p in fresh or []:
        c = cell_report(p, load(p), srcmap, judged=False)
        c['kind'] = 'fresh (reported, never judged)'
        cells['fresh ' + os.path.basename(p)] = c
    c3 = none_share_checks(cells, records)
    c1_fail = [k for k, c in cells.items() if c.get('C1', {}).get('fails')]
    # C1 pooled: the tool-gold slates of every record cell, both rates within 0.01 of 0.5
    pk = [0, 0, 0, 0]
    for k, c in cells.items():
        x = c.get('C1')
        if x and c.get('kind') == 'records':
            pk[0] += x['counts (right rewritten, others rewritten)'][0]
            pk[1] += x['right tools']
            pk[2] += x['counts (right rewritten, others rewritten)'][1]
            pk[3] += x['other tool options']
    c1_pooled = None
    if pk[1]:
        pr, po = pk[0] / pk[1], pk[2] / max(pk[3], 1)
        c1_pooled = {'P(rewritten | right tool)': pr, 'right tools': pk[1],
                     'P(rewritten | other tool option)': po, 'other tool options': pk[3],
                     'bar': C1_POOLED, 'fails': abs(pr - 0.5) > C1_POOLED or abs(po - 0.5) > C1_POOLED}
        if c1_pooled['fails']:
            c1_fail.append('pooled')
    c3_fail = [x['cell'] for x in c3 if x['fails']]
    c2_fail = [(k, r['rule'], r['tag']) for k, c in cells.items() for r in c['C2'] if r['fails']]
    remedy_open = bool(c2_fail) and all(tag in REMEDY_RULES for _, _, tag in c2_fail)
    c5 = None
    if fresh_manifest and os.path.exists(fresh_manifest):
        m = json.load(open(fresh_manifest))
        c5 = {'Toucan tools within v2\'s cluster rule (never shown)':
              m.get('counts', {}).get("Toucan tools within v2's cluster rule (never shown)"),
              'enforced by': 'sky-data route toucan (stops on a violation)'}
    verdict = {
        'C1 fails': c1_fail, 'C3 fails': c3_fail,
        'C2 fails': [{'cell': k, 'rule': r, 'tag': t} for k, r, t in c2_fail],
        'defect (C1 or C3)': bool(c1_fail or c3_fail),
        'C2 stops the build': bool(c2_fail),
        'the remedy of 4.4 is open': remedy_open,
        'exit': 1 if (c1_fail or c3_fail) else (2 if c2_fail else 0),
    }
    return {'about': 's6-route v4: the data checks of docs/s6-route4-prereg.md 4.4 (no model)',
            'rules': {'C1': '|rate - 0.5| <= 3 sqrt(0.25 / n) per cell and rate; pooled |rate - 0.5| <= 0.01 (amendment 1)',
                      'C2': 'c + 0.02 + 2 sqrt(c(1-c)/n)', 'cells judged from': MIN_JUDGED},
            'cells': cells, 'C1 pooled': c1_pooled, 'C3': c3, 'C5': c5, 'verdict': verdict}


def markdown(rep, top=5):
    out = ['# s6-route v4: the data checks (docs/s6-route4-prereg.md 4.4)', '']
    v = rep['verdict']
    out += ['exit %d; C1 fails %d; C3 fails %d; C2 fails %d; remedy open: %s' % (
        v['exit'], len(v['C1 fails']), len(v['C3 fails']), len(v['C2 fails']), v['the remedy of 4.4 is open']), '']
    out += ['| cell | slates | tool-gold | judged | C1 diff | best rule | accuracy | tolerance | failing rules |',
            '|---|---:|---:|---|---:|---|---:|---:|---:|']
    for k, c in rep['cells'].items():
        best = c['C2'][0] if c['C2'] else None
        c1 = c.get('C1', {}).get('difference (reported)')
        out.append('| %s | %d | %d | %s | %s | %s | %s | %s | %d |' % (
            k, c['slates'], c['tool-gold slates'], 'yes' if c['judged'] else 'no',
            '%+.4f' % c1 if c1 is not None else '', best['rule'] if best else '',
            '%.4f' % best['accuracy'] if best else '', '%.4f' % best['tolerance'] if best else '',
            sum(r['fails'] for r in c['C2'])))
    out += ['', '## C3', '', '| cell | slates | none | share | fails |', '|---|---:|---:|---:|---|']
    out += ['| %s | %d | %d | %.4f | %s |' % (x['cell'], x['slates'], x['none'], x['share'], x['fails'])
            for x in rep['C3']]
    for k, c in rep['cells'].items():
        out += ['', '### %s: the %d highest rules' % (k, top), '', '| rule | accuracy | chance | tolerance |',
                '|---|---:|---:|---:|']
        out += ['| %s | %.4f | %.4f | %.4f |' % (r['rule'], r['accuracy'], r['chance'], r['tolerance'])
                for r in c['C2'][:top]]
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--build')
    ap.add_argument('--requests')
    ap.add_argument('--val', nargs='*', default=[])
    ap.add_argument('--fresh', nargs='*', default=[])
    ap.add_argument('--fresh-manifest')
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args(argv)
    rep = run(a.build, a.requests, a.val, a.fresh, a.fresh_manifest)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(rep, f, indent=1)
    if a.md:
        with open(a.md, 'w') as f:
            f.write(markdown(rep))
    v = rep['verdict']
    print(json.dumps({k: (len(x) if isinstance(x, list) else x) for k, x in v.items()}))
    return v['exit']


if __name__ == '__main__':
    sys.exit(main())
