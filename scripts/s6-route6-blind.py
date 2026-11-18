#!/usr/bin/env python3
"""The statistics of the request-blind reading of s6-route v6 (docs/s6-route6-prereg.md 4.3, 8,
9 and 9.1). Standard library only; run with `python3 -I`. No system is run here: it reads the
dumps that sky-eval (`--dump-probs`) and sky-route fuse (`--out`) wrote on the blind copies of
`sky-data route blind`.

  s6-route6-blind.py read --slates <slate file> --dump <dump> [--field probs|model|embedding]
                          [--servers <servers.jsonl>] [--unit server|cluster] [--out <json>]
      for one system on one blind file: the blind pick (the tool option of highest probability
      among the k tool options, none left out, ties 1/|tie|), the blind accuracy on the tool-right
      slates and its chance c (the mean of 1/k), the answer with none allowed (argmax over every
      option), the none share, the blind none gap (the share of none answers among none-right
      slates minus that among tool-right slates), the share asked (the pair's `ask`), the share of
      blind picks that are the slate's fewest-words option (study 7), the blind accuracy split by
      the number of the right tool's server-mates shown (the options of role `exempt`: 0, 1 to 2,
      3 or more; and none against some); 95% percentile intervals of 1,000 bootstrap resamples
      (seed 0) of the slates and, with --servers, of the servers (or clusters)
  s6-route6-blind.py support --records <files...> [--out <json>]
      4.3, the support of B0: for each group and kind, the share of the tool-right records in
      which every tool option scores 0.00 (the options' prefixes only)
"""
import argparse
import collections
import json
import random
import re
import sys

RESAMPLES = 1000
SEED = 0
SCORE = re.compile(r'^score (-?[0-9.]+) rank [0-9]+; ')


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_of(r):
    return r['target'].index(1.0)


def plain(o):
    return SCORE.sub('', o, count=1)


def words(o):
    _, _, d = plain(o).partition(': ')
    return len(d.split())


def blind_pick(probs, k):
    """The share of the right answer each tool option takes: the options of highest probability
    among the k tool options, 1/|tie| each."""
    tools = probs[:k]
    m = max(tools)
    best = [i for i, x in enumerate(tools) if x == m]
    return best


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def rows_of(slates, dump, field='probs'):
    if len(slates) != len(dump):
        sys.exit('a dump of %d lines for %d slates' % (len(dump), len(slates)))
    rows = []
    for r, d in zip(slates, dump):
        p = d[field]
        k = len(r['opts']) - 1
        if len(p) != k + 1:
            sys.exit('a dump line with another number of options (%s)' % r.get('uid'))
        g = gold_of(r)
        best = blind_pick(p, k)
        roles = r.get('roles') or []
        fewest = min(range(k), key=lambda i: (words(r['opts'][i]), i)) if k else None
        rows.append({'uid': r.get('uid'), 'k': k, 'none right': g == k,
                     'blind right': (1 / len(best)) if g in best else 0.0,
                     'answer none': argmax(p) == k, 'answer right': argmax(p) == g,
                     'asked': bool(d.get('ask')) if 'ask' in d else None,
                     'fewest words picked': (1 / len(best)) if fewest in best else 0.0,
                     'server-mates': sum(x == 'exempt' for x in roles)})
    return rows


def stats(rows):
    tr = [x for x in rows if not x['none right']]
    nr = [x for x in rows if x['none right']]
    out = {'slates': len(rows), 'tool-right slates': len(tr), 'none-right slates': len(nr)}
    if tr:
        out['blind accuracy'] = sum(x['blind right'] for x in tr) / len(tr)
        out['chance c'] = sum(1 / x['k'] for x in tr) / len(tr)
        out['blind accuracy - c'] = out['blind accuracy'] - out['chance c']
        out['fewest-words share of the blind picks'] = sum(x['fewest words picked'] for x in tr) / len(tr)
        out['answer with none allowed: accuracy on the tool-right slates'] = sum(x['answer right'] for x in tr) / len(tr)
    out['answer with none allowed: accuracy'] = sum(x['answer right'] for x in rows) / max(len(rows), 1)
    out['none share'] = sum(x['answer none'] for x in rows) / max(len(rows), 1)
    if tr and nr:
        out['blind none gap'] = (sum(x['answer none'] for x in nr) / len(nr)
                                 - sum(x['answer none'] for x in tr) / len(tr))
    asked = [x['asked'] for x in rows if x['asked'] is not None]
    if asked:
        out['share asked'] = sum(asked) / len(asked)
        out['share none or asked'] = sum(x['answer none'] or bool(x['asked']) for x in rows) / len(rows)
    return out


def interval(values, rows, groups=None, seed=SEED, n=RESAMPLES):
    """The 95% percentile interval of a statistic `values(rows)` over resamples of the slates or,
    with groups (a list of lists of rows), of the groups."""
    rng = random.Random(seed)
    units = groups if groups is not None else [[x] for x in rows]
    vals = []
    for _ in range(n):
        sample = [x for _ in range(len(units)) for x in units[rng.randrange(len(units))]]
        v = values(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def acc(rows):
    tr = [x for x in rows if not x['none right']]
    return sum(x['blind right'] for x in tr) / len(tr) if tr else None


def gap(rows):
    tr = [x for x in rows if not x['none right']]
    nr = [x for x in rows if x['none right']]
    if not tr or not nr:
        return None
    return sum(x['answer none'] for x in nr) / len(nr) - sum(x['answer none'] for x in tr) / len(tr)


def read(slates, dump, field='probs', servers=None, unit='server'):
    rows = rows_of(slates, dump, field)
    out = stats(rows)
    out['intervals, slates resampled'] = {'blind accuracy': interval(acc, rows), 'blind none gap': interval(gap, rows)}
    if servers is not None:
        by = collections.defaultdict(list)
        for x in rows:
            by[servers.get(x['uid'], '?')].append(x)
        groups = [by[k] for k in sorted(by)]
        out['intervals, %ss resampled' % unit] = {'blind accuracy': interval(acc, rows, groups),
                                                  'blind none gap': interval(gap, rows, groups), unit + 's': len(groups)}
    split = {}
    for name, pick in (('no server-mate shown', lambda m: m == 0), ('server-mates shown', lambda m: m > 0),
                       ('0', lambda m: m == 0), ('1 to 2', lambda m: 1 <= m <= 2), ('3 or more', lambda m: m >= 3)):
        sub = [x for x in rows if pick(x['server-mates'])]
        tr = [x for x in sub if not x['none right']]
        if tr:
            split[name] = {'tool-right slates': len(tr), 'blind accuracy': acc(sub),
                           'chance c': sum(1 / x['k'] for x in tr) / len(tr)}
    out['by server-mates shown (tool-right slates)'] = split
    return out, rows


def support(paths):
    """4.3: per group and kind, the share of tool-right records whose tool options all score 0.00."""
    c = collections.defaultdict(lambda: [0, 0])
    for p in paths:
        for r in load(p):
            if gold_of(r) == len(r['opts']) - 1:
                continue
            key = '%s %s %s' % (r.get('src', '?').split(':')[0], r.get('kind'), r.get('split'))
            e = c[key]
            e[0] += 1
            ms = [SCORE.match(o) for o in r['opts'][:-1]]
            e[1] += all(m is not None and float(m.group(1)) == 0.0 for m in ms)
    return {k: {'tool-right records': n, 'all scores 0.00': z, 'share': z / n} for k, (n, z) in sorted(c.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('read')
    r.add_argument('--slates', required=True)
    r.add_argument('--dump', required=True)
    r.add_argument('--field', default='probs', choices=('probs', 'model', 'embedding'))
    r.add_argument('--servers')
    r.add_argument('--unit', default='server', choices=('server', 'cluster'))
    r.add_argument('--out')
    s = sub.add_parser('support')
    s.add_argument('--records', nargs='+', required=True)
    s.add_argument('--out')
    a = ap.parse_args(argv)
    if a.cmd == 'read':
        servers = None
        if a.servers:
            servers = {x['uid']: x.get(a.unit) for x in load(a.servers)}
        out, _ = read(load(a.slates), load(a.dump), a.field, servers, a.unit)
    else:
        out = support(a.records)
    s_ = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s_ + '\n')
    print(s_)
    return 0


if __name__ == '__main__':
    sys.exit(main())
