#!/usr/bin/env python3
"""Request-blind rules on the test-like slates of s6-route (docs/s6-route4-study.md, section 7).

Standard library only; run with `python3 -I`. The data folder (DATA, default
/Users/maxime/Documents/Code/skycmd-data/s6-route) is read, never written. No model is run.

  s6-route4-blind.py rules      every rule on build/val/v_tools.jsonl (validation), then on the
                                published test files build/test/t_tools.jsonl (T-tools of v2) and
                                build/fresh/f_tools.jsonl (T2-tools of v3); these were read for the
                                published results, and here only the features of their option texts
                                and their targets are counted, as any file property; no system's
                                output on them is read
  s6-route4-blind.py val-split  on v_tools only: the published dumps of the model part (v2's file),
                                of BM25 and of potion-cos-4bit, fused at v3's beta 0.81, split by
                                whether the best rule is right

A rule sees the option texts of a slate (and, for the S rules, the source of each option, which no
system sees), never the request. It picks one tool option; a tie counts as 1/|tie| when the right
tool is in it. Chance is 1/(number of tool options). Intervals are 95%, normal, on the per-slate
difference between the rule and chance.
"""
import collections
import json
import math
import os
import re
import sys

DATA = os.environ.get('DATA', '/Users/maxime/Documents/Code/skycmd-data/s6-route').rstrip('/') + '/'
EV = DATA + 'ev' + 'al/val/'
FILES = [('v_tools (validation)', 'build/val/v_tools.jsonl'),
         ('t_tools (T-tools, read for v2)', 'build/test/t_tools.jsonl'),
         ('f_tools (T2-tools, read for v3)', 'build/fresh/f_tools.jsonl')]
SCHEMA = re.compile(r'\b(api|endpoint|parameters?|json|ids?|specified|given|provided|based on|'
                    r'information|details|data|list)\b', re.I)


def load(path):
    with open(path if path.startswith('/') else DATA + path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def split_opt(o):
    n, _, d = o.partition(': ')
    return n, d


def name_style(n):
    if ' ' in n:
        return 'spaced'
    if '_' in n or '-' in n:
        return 'snake'
    if re.fullmatch(r'[a-z][a-z0-9]*([A-Z][a-z0-9]*)+', n):
        return 'camel'
    if re.fullmatch(r'([A-Z][a-z0-9]+){2,}', n):
        return 'pascal'
    return 'other'


def first_word(d):
    return d.split(' ')[0] if d else ''


def third(d):
    w = first_word(d)
    return w[:1].isupper() and w.endswith('s') and not w.endswith('ss')


def imperative(d):
    w = first_word(d)
    return w[:1].isupper() and not w.endswith('s') and w.isalpha()


def feats(o, src):
    n, d = split_opt(o)
    return {'period': d.endswith('.'), 'third': third(d), 'imperative': imperative(d),
            'words': len(d.split()), 'chars': len(d), 'schema': len(SCHEMA.findall(d)),
            'style': name_style(n), 'digits': bool(re.search(r'\d', n)), 'namelen': len(n),
            'src': src}


def argbest(vals):
    m = max(vals)
    return [i for i, v in enumerate(vals) if v == m]


def minority(keys):
    c = collections.Counter(keys)
    return [-c[k] for k in keys]


def majority(keys):
    c = collections.Counter(keys)
    return [c[k] for k in keys]


def rules():
    R = {}
    for f in ('period', 'third', 'imperative', 'digits'):
        R['%s' % f] = lambda F, f=f: [int(x[f]) for x in F]
        R['not %s' % f] = lambda F, f=f: [-int(x[f]) for x in F]
    R['diagnosis style (period, then -s)'] = lambda F: [2 * x['period'] + x['third'] for x in F]
    R['inverse of the diagnosis style'] = lambda F: [-(2 * x['period'] + x['third']) for x in F]
    for f, lab in (('words', 'words'), ('chars', 'characters'), ('schema', 'schema-like words'),
                   ('namelen', 'name length')):
        R['most %s' % lab] = lambda F, f=f: [x[f] for x in F]
        R['fewest %s' % lab] = lambda F, f=f: [-x[f] for x in F]
    for p in range(7):
        R['position %d' % (p + 1)] = lambda F, p=p: [int(i == p) for i in range(len(F))]
    R['name style, minority'] = lambda F: minority([x['style'] for x in F])
    R['name style, majority'] = lambda F: majority([x['style'] for x in F])
    R['(name style, period), minority'] = lambda F: minority([(x['style'], x['period']) for x in F])
    R['(name style, period, first word form), minority'] = lambda F: minority(
        [(x['style'], x['period'], x['third'], x['imperative']) for x in F])
    R['(name style, period, first word form), majority'] = lambda F: majority(
        [(x['style'], x['period'], x['third'], x['imperative']) for x in F])
    R['S: source, minority (sources not visible to any system)'] = lambda F: minority([x['src'] for x in F])
    R['S: source, majority (sources not visible to any system)'] = lambda F: majority([x['src'] for x in F])
    return R


def tool_sources():
    src = collections.defaultdict(set)
    for r in load('build/stage/requests.jsonl'):
        for t in r.get('tools', []) + ([r['gold']] if r.get('gold') else []) + r.get('exclude', []):
            src[t].add(r['source'])
    return {t: '+'.join(sorted(s)) for t, s in src.items()}


def gold_of(r):
    return r['target'].index(1.0)


def score(rule, r, srcmap):
    F = [feats(o, srcmap.get(t, '?')) for o, t in zip(r['opts'][:-1], r['tools'])]
    best = argbest(rule(F))
    g = gold_of(r)
    return (1 / len(best) if g in best else 0.0), best


def ci(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    v = sum((d - m) ** 2 for d in diffs) / (n - 1)
    h = 1.96 * math.sqrt(v / n)
    return m, m - h, m + h


def table(name, recs, srcmap, R):
    tg = [r for r in recs if gold_of(r) < len(r['opts']) - 1]
    ng = [r for r in recs if gold_of(r) == len(r['opts']) - 1]
    out = []
    for lab, rule in R.items():
        acc = [score(rule, r, srcmap)[0] for r in tg]
        ch = [1 / (len(r['opts']) - 1) for r in tg]
        m, lo, hi = ci([a - c for a, c in zip(acc, ch)])
        out.append((m, lab, sum(acc) / len(acc), sum(ch) / len(ch), lo, hi))
    out.sort(reverse=True)
    print('\n## %s: %d slates, %d with a tool right, %d with none right' % (name, len(recs), len(tg), len(ng)))
    print('| rule | accuracy | chance | rule - chance [95%] |')
    print('|---|---:|---:|---:|')
    for m, lab, a, c, lo, hi in out:
        print('| %s | %.4f | %.4f | %+.4f [%+.4f, %+.4f] |' % (lab, a, c, m, lo, hi))
    return out, tg, ng


def standout(rule, r, srcmap):
    """Whether the rule singles out one option (a unique best)."""
    return len(score(rule, r, srcmap)[1]) == 1


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'rules'
    srcmap = tool_sources()
    R = rules()
    if cmd == 'rules':
        best_text = None
        for name, path in FILES:
            out, tg, ng = table(name, load(path), srcmap, R)
            text = [o for o in out if not o[1].startswith('S:')]
            b = text[0][1] if best_text is None else best_text
            best_text = best_text or b
            rule = R[b]
            su_t = sum(standout(rule, r, srcmap) for r in tg) / len(tg)
            su_n = sum(standout(rule, r, srcmap) for r in ng) / len(ng)
            right_unique = sum(1 for r in tg if score(rule, r, srcmap)[0] == 1.0) / len(tg)
            print('\nbest text rule on validation: "%s". Here: it singles out one option on %.4f of the '
                  'tool-right slates (and that option is right on %.4f of all of them), and on %.4f of the '
                  'none-right slates (n %d).' % (b, su_t, right_unique, su_n, len(ng)))
        return
    if cmd == 'val-split':
        recs = load('build/val/v_tools.jsonl')
        model = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 's6-nano2', 'v_tools.dump.jsonl'))
        bm = load(EV + 'bm25/v_tools.jsonl')
        emb = load(EV + 'potion-cos-4bit/v_tools.jsonl')
        for d in (model, bm, emb):
            assert [x['gold'] for x in d] == [gold_of(r) for r in recs]
        rule = R[sys.argv[2]]

        def am(p):
            return max(range(len(p)), key=lambda i: p[i])

        def fused(a, b):
            l = [0.81 * (math.log(max(x, 1e-12)) + math.log(max(y, 1e-12))) for x, y in zip(a, b)]
            return am(l)
        groups = collections.defaultdict(list)
        for r, m, b, e in zip(recs, model, bm, emb):
            g = gold_of(r)
            if g == len(r['opts']) - 1:
                key = 'none right'
            else:
                key = 'rule right' if score(rule, r, srcmap)[0] == 1.0 else 'rule not right'
            groups[key].append((int(am(m['probs']) == g), int(am(b['probs']) == g),
                                int(fused(m['probs'], e['probs']) == g), int(am(e['probs']) == g)))
        print('| v_tools slates | n | model part | BM25 | pair (beta 0.81) | embedding | model - BM25 [95%] | pair - BM25 [95%] |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|')
        for k in ('rule right', 'rule not right', 'none right'):
            v = groups[k]
            n = len(v)
            mm = [x[0] - x[1] for x in v]
            pp = [x[2] - x[1] for x in v]
            a, al, ah = ci(mm)
            c, cl, chh = ci(pp)
            print('| %s | %d | %.4f | %.4f | %.4f | %.4f | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] |' % (
                k, n, sum(x[0] for x in v) / n, sum(x[1] for x in v) / n, sum(x[2] for x in v) / n,
                sum(x[3] for x in v) / n, a, al, ah, c, cl, chh))
        return
    sys.exit(__doc__)


if __name__ == '__main__':
    main()
