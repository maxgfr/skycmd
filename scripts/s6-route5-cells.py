#!/usr/bin/env python3
"""s6-route v5: the feasibility count of the surface cell (docs/s6-route5-prereg.md, amendment 1).

It builds no slate, reads no request and calls no model. Standard library only; run with
`python3 -I scripts/s6-route5-cells.py [--data DIR] [--out FILE]`. The data folder is read,
never written, and only these files of it are opened:

  build/stage/tools.jsonl                       every tool's name and cut description (v2)
  build/teacher/describe-intent.items.jsonl     the intents: id, source, name
  build/teacher/describe-intent.jsonl           the intents' descriptions (v2's teacher)
  build/teacher/rewrite-description.items.jsonl v2's rewrite items: a tool id and its split
  build/teacher/rewrite-description.jsonl       v2's rewrites
  build4/teacher/rewrite-description.items.jsonl v4's rewrite items
  build4/teacher/rewrite-description.jsonl      v4's rewrites
  build/counts/split.json                       v2's requests per group, split and source (a count)

The pools: every tool of v2's and v4's rewrite items, which v4's plan counted as "the tools of the
pools and intents" (27,950). An intent is a tool whose name and description are those of an
intent of the intent files; its pool is its source. The other tools are split by their item's
split: the training pool and the validation pool (When2Call's validation pool is both). No test
or fresh tool has a rewrite item, so none is read.

For each pool, each tool is taken as the anchor in each of its two texts (the original and the
rewrite, each with weight 1/2, as D1 shows them), and the other tools of the pool are counted as
candidates at each level L0 to L6 of section 4.3, each text of a candidate weighing 1/2 (D1's
draw). The level of an anchor is the lowest level whose expected candidates reach the minimum
(16, 8, 4, and every minimum from 4 to 16; for the intents also the slate's drawn distractors
plus 3, by kind, the sources weighted by v2's request counts). Anchors weigh alike: how often a
tool is a gold needs the requests, which are not read. The kind's distractor filter (the gold's
cluster, the request's other tools) is not applied: the counts are an upper bound on the
candidates. The stage tool table also lists test tools; only the 27,950 pool tools are kept and
no test tool enters a count. Only aggregates are printed.
"""
import argparse
import collections
import json
import os
import re

SCHEMA = re.compile(r'\b(api|endpoint|parameters?|json|ids?|specified|given|provided|based on|'
                    r'information|details|data|list)\b', re.I)
DESC_WORDS = 30
MINIMA = (16, 8, 4)


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def cut_description(d):
    """crates/sky-data/src/route/mod.rs `cut_description`: the first sentence, then 30 words."""
    d = ' '.join(d.split())
    end = len(d)
    for i, c in enumerate(d):
        if c in '.!?' and (i + 1 == len(d) or d[i + 1] == ' '):
            end = i + 1
            break
    return ' '.join([w for w in d[:end].split(' ') if w][:DESC_WORDS])


def name_style(n):
    """scripts/s6-route4-blind.py `name_style` (five classes)."""
    if ' ' in n:
        return 'spaced'
    if '_' in n or '-' in n:
        return 'snake'
    if re.fullmatch(r'[a-z][a-z0-9]*([A-Z][a-z0-9]*)+', n):
        return 'camel'
    if re.fullmatch(r'([A-Z][a-z0-9]+){2,}', n):
        return 'pascal'
    return 'other'


def first_form(d):
    w = d.split(' ')[0] if d else ''
    if w[:1].isupper() and w.endswith('s') and not w.endswith('ss'):
        return 'third'
    if w[:1].isupper() and not w.endswith('s') and w.isalpha():
        return 'imperative'
    return 'other'


def quintile(sorted_vals, v):
    """crates/sky-data/src/route/v4.rs `surface::quintile`: 5 x the share strictly below, floor."""
    import bisect
    if not sorted_vals:
        return 0
    return min(4, 5 * bisect.bisect_left(sorted_vals, v) // len(sorted_vals))


def raw(name, desc):
    return (name_style(name), len(name), len(desc.split()), desc.endswith('.'), first_form(desc),
            min(2, len(SCHEMA.findall(desc))))


def level_match(a, b, level):
    """Whether candidate key b matches anchor key a at a level (section 4.3)."""
    f1, f2, f3, f4, f5, f6 = a
    g1, g2, g3, g4, g5, g6 = b
    if level == 6:
        return True
    if level == 0:
        return a == b
    if f4 != g4 or abs(f3 - g3) > 1:
        return False
    if level <= 4 and abs(f2 - g2) > 1:
        return False
    if level <= 3 and f1 != g1:
        return False
    if level <= 2 and f5 != g5:
        return False
    if level <= 1 and f6 != g6:
        return False
    return True


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def level_dist(anchors, m):
    """The weighted share of anchors at each level L0 to L6 for a minimum of m candidates."""
    dist = [0.0] * 7
    tot = sum(w for w, _ in anchors)
    for w, ns in anchors:
        lev = next((L for L in range(6) if ns[L] >= m), 6)
        dist[lev] += w / tot
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ROUTE_DATA') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'skycmd-data', 's6-route'),
        help='the data folder (default: ROUTE_DATA, else <repo>/../skycmd-data/s6-route)')
    ap.add_argument('--out')
    a = ap.parse_args()
    D = a.data.rstrip('/') + '/'
    tools = {t['id']: t for t in load(D + 'build/stage/tools.jsonl')}
    items = load(D + 'build/teacher/rewrite-description.items.jsonl') + \
        load(D + 'build4/teacher/rewrite-description.items.jsonl')
    rw = {}
    for p in ('build/teacher/rewrite-description.jsonl', 'build4/teacher/rewrite-description.jsonl'):
        for r in load(D + p):
            rw[r['id']] = cut_description(r['description'])
    idesc = {d['id']: d['description'] for d in load(D + 'build/teacher/describe-intent.jsonl')}
    intent_of = {}
    for d in load(D + 'build/teacher/describe-intent.items.jsonl'):
        intent_of[(d['name'], cut_description(idesc[d['id']]))] = d['source']

    pools = collections.defaultdict(list)
    for it in items:
        t = tools[it['id']]
        src = intent_of.get((t['name'], t['desc']))
        if src:
            pools['intents ' + src].append(it['id'])
        else:
            pools['tools ' + it['split']].append(it['id'])
    pools = {k: sorted(set(v)) for k, v in pools.items()}
    every = sorted({t for v in pools.values() for t in v})
    texts = {t: ((tools[t]['name'], tools[t]['desc']), (tools[t]['name'], rw.get(t, tools[t]['desc'])))
             for t in every}
    names = sorted(len(texts[t][i][0]) for t in every for i in (0, 1))
    words = sorted(len(texts[t][i][1].split()) for t in every for i in (0, 1))

    def key(name, desc):
        s, nl, wl, per, ff, sc = raw(name, desc)
        return (s, quintile(names, nl), quintile(words, wl), per, ff, sc)

    keys = {t: (key(*texts[t][0]), key(*texts[t][1])) for t in every}
    groups = [('tools train (and When2Call train)', pools['tools train'], (0, 1)),
              ('tools val', pools['tools val'], (0, 1)),
              ('tools val, originals only (v5_tools)', pools['tools val'], (0,)),
              ('When2Call val (training and validation pools)',
               sorted(set(pools['tools train']) | set(pools['tools val'])), (0, 1))]
    for k in sorted(p for p in pools if p.startswith('intents ')):
        groups.append((k + ' (train and val)', pools[k], (0, 1)))

    report = {'about': __doc__.split('\n')[0], 'pools': {k: len(v) for k, v in pools.items()},
              'groups': {}}
    anchors_of = {}
    for label, pool, sides in groups:
        w = 1.0 / len(sides)
        weight = collections.Counter()
        for t in pool:
            for i in sides:
                weight[keys[t][i]] += w
        cells = sorted(weight)
        # for each anchor (tool, text), the expected candidates at L0 to L5 (L6: the whole pool)
        anchors = []
        cache = {}
        for t in pool:
            own = tuple(keys[t][j] for j in sides)
            for i in sides:
                ak = keys[t][i]
                if (ak, own) not in cache:
                    ns = []
                    for L in range(6):
                        n = sum(weight[c] for c in cells if level_match(ak, c, L))
                        n -= sum(w for o in own if level_match(ak, o, L))
                        ns.append(n)
                    cache[(ak, own)] = ns
                anchors.append((w, cache[(ak, own)]))
        anchors_of[label] = anchors
        out = {'tools': len(pool),
               'non-empty cells, originals': len({keys[t][0] for t in pool}),
               'non-empty cells, rewrites': len({keys[t][1] for t in pool})}
        for m in MINIMA:
            dist = level_dist(anchors, m)
            out['minimum %d' % m] = {'L%d' % L: round(dist[L], 4) for L in range(7)}
            out['minimum %d' % m]['L5 or L6'] = round(dist[5] + dist[6], 4)
            out['minimum %d' % m]['by L4'] = round(sum(dist[:5]), 4)
        out['L5 or L6 by minimum'] = {m: round(sum(level_dist(anchors, m)[5:]), 4)
                                      for m in range(4, 17)}
        report['groups'][label] = out
    # the intents' cells pool both sources, weighted by v2's requests per source and split
    # (build/counts/split.json, an aggregate count); the minimum is the slate's drawn
    # distractors plus 3, by kind (amendment 1)
    req = load_json(D + 'build/counts/split.json')['requests by group, split and source']
    kinds = {'natural or hard': [(k, 1 / 6) for k in range(1, 7)],
             'removed': [(k, 1 / 7) for k in range(1, 8)],
             'renamed': [(6, 0.8), (7, 0.2)]}
    plus3 = {}
    for split in ('train', 'val'):
        wsrc = req['intents ' + split]
        tot = sum(wsrc.values())
        for kind, ks in kinds.items():
            share = 0.0
            for src, n in wsrc.items():
                anchors = anchors_of['intents %s (train and val)' % src]
                share += n / tot * sum(p * sum(level_dist(anchors, k + 3)[5:]) for k, p in ks)
            plus3['intents %s %s' % (kind, split)] = round(share, 4)
    for src in req['intents train']:
        anchors = anchors_of['intents %s (train and val)' % src]
        for kind, ks in kinds.items():
            plus3['%s only, %s' % (src, kind)] = round(
                sum(p * sum(level_dist(anchors, k + 3)[5:]) for k, p in ks), 4)
    report['intents, minimum = drawn distractors + 3: share at L5 or L6'] = plus3
    s = json.dumps(report, indent=1)
    print(s)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(s + '\n')


if __name__ == '__main__':
    main()
