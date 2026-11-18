#!/usr/bin/env python3
"""The gate of s6-route v6 (docs/s6-route6-prereg.md 8: G1 to G5 as v5 and v4 8, and G6), from
the dumps. Standard library only; run with `python3 -I`.

  s6-route6-gate.py --root <eval>/t5 --slates <t5>/fresh --test-files runs/s6-route6/test-files.json \\
      --open runs/s6-route6/FRESH-OPEN --t2 <eval>/fresh/pair/f_tools.jsonl --g4 runs/s6-route6/g4.json \\
      [--chance <c of the counts amendment>] --out runs/s6-route6/gate.json [--md gate.md]

Refuses to read a T5 file before FRESH-OPEN exists and while a file's sha256 is not the one of
test-files.json. The dumps under --root:

  pair/<file>.jsonl     sky-route fuse --out on the slate file: the pair (`probs`, `answer`,
                        `ask`), its model part (`model`) and its embedding part (`embedding`,
                        potion-cos-4bit)
  bm25/<file>.jsonl     scripts/route-baselines.py (`probs`, `gold`)

for the files t5_tools, t5_tools_masked, t5_tools_rw, t5_tools_masked_rw and t5_tools.blind (B0,
sky-route fuse on the blind copy). The servers are <slates>/t5_tools.servers.jsonl.

- G1: the pair - BM25 and the pair - potion-cos-4bit each with the lower bound of its 95% slate
  bootstrap above 0, and the pair above its model part (accuracies of the argmax).
- G2 (a): on the tool-gold slates of t5_tools_rw and t5_tools_masked_rw, the share whose answer is
  the rewritten distractor (`follow`) <= 1/7 + 0.05 = 0.1929, and that share minus the share whose
  answer is the same option in the unchanged slate <= +0.03; (b) on t5_tools_masked, the pair >=
  potion-cos-4bit - 0.02; (c) on t5_tools_masked, none answered on 0.10 to 0.30 of the slates.
- G3: ECE (15 bins) of the pair on t5_tools <= 0.05.
- G4: from --g4 (the engine's bench): max |dp| <= 1e-3 with the same answer on >= 199 of 200
  slates, a median <= 60 ms, the two files <= 4,000,000 bytes.
- G5: the pair on T2-tools >= 0.8668.
- G6, on the tool-right slates of t5_tools in B0, c the chance (the mean of 1/k, or --chance):
  (a) the model part's blind accuracy <= c + 0.05 and the upper bound of its 95% bootstrap with
  servers resampled <= c + 0.08; (b) the same for the pair; (c) for each, the blind none gap
  <= 0.05 and its upper bound (servers) <= 0.15. Reported split by whether a server-mate of the
  right tool is shown; when G6 fails and holds on the slates that show none, the report says in
  its first sentence that it fails only on the slates that show server-mates, and G6 is still
  failed.
- The headline: the pair - BM25 and the pair - potion-cos-4bit with the lower bound of their
  bootstrap with servers resampled above 0.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route6_blind', os.path.join(HERE, 's6-route6-blind.py'))
blind = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blind)

RESAMPLES = 1000
G2A = 1 / 7 + 0.05
G2A_PAIRED = 0.03
G2B = 0.02
G2C = (0.10, 0.30)
G3_ECE = 0.05
G4_DP, G4_AGREE, G4_N, G4_MS, G4_BYTES = 1e-3, 199, 200, 60.0, 4_000_000
G5_T2 = 0.8668
G6_ACC, G6_UPPER = 0.05, 0.08
G6_GAP, G6_GAP_UPPER = 0.05, 0.15
FILES = ('t5_tools', 't5_tools_masked', 't5_tools_rw', 't5_tools_masked_rw', 't5_tools.blind')


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def argmax(p):
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best


def gold_of(r):
    return r['target'].index(1.0)


def correct(slates, dump, field='probs'):
    return [argmax(d[field]) == gold_of(r) for r, d in zip(slates, dump)]


def mean(v):
    return sum(v) / len(v) if v else float('nan')


def boot_diff(a, b, groups=None, seed=0):
    """The paired difference mean(a) - mean(b) and its 95% percentile interval, resampling the
    slates or, with groups (lists of indices), the groups."""
    rng = random.Random(seed)
    units = groups if groups is not None else [[i] for i in range(len(a))]
    diffs = []
    for _ in range(RESAMPLES):
        idx = [i for _ in units for i in units[rng.randrange(len(units))]]
        diffs.append(mean([a[i] for i in idx]) - mean([b[i] for i in idx]))
    diffs.sort()
    return {'diff': mean(a) - mean(b), 'lo': diffs[int(0.025 * (RESAMPLES - 1))],
            'hi': diffs[int(0.975 * (RESAMPLES - 1))]}


def server_groups(slates, servers):
    by = {}
    for i, r in enumerate(slates):
        by.setdefault(servers.get(r['uid'], '?'), []).append(i)
    return [by[k] for k in sorted(by)]


def ece(slates, dump, bins=15):
    cells = [[0, 0.0, 0] for _ in range(bins)]
    for r, d in zip(slates, dump):
        k = argmax(d['probs'])
        c = d['probs'][k]
        b = min(bins - 1, int(c * bins))
        cells[b][0] += 1
        cells[b][1] += c
        cells[b][2] += int(k == gold_of(r))
    n = len(slates)
    return sum(m / n * abs(conf / m - ok / m) for m, conf, ok in cells if m)


def g6_part(slates, dump, field, servers, c):
    rows = blind.rows_of(slates, dump, field)
    out = blind.stats(rows)
    groups = None
    if servers is not None:
        by = {}
        for x in rows:
            by.setdefault(servers.get(x['uid'], '?'), []).append(x)
        groups = [by[k] for k in sorted(by)]
    up_acc = blind.interval(blind.acc, rows, groups)
    up_gap = blind.interval(blind.gap, rows, groups)
    a, g = out.get('blind accuracy'), out.get('blind none gap')
    conds = {
        'blind accuracy <= c + 0.05': a is not None and a <= c + G6_ACC,
        'its upper bound (servers) <= c + 0.08': up_acc is not None and up_acc[1] <= c + G6_UPPER,
        'blind none gap <= 0.05': g is not None and g <= G6_GAP,
        'its upper bound (servers) <= 0.15': up_gap is not None and up_gap[1] <= G6_GAP_UPPER,
    }
    return {'blind accuracy': a, 'chance c': c, 'interval of the blind accuracy (servers)': up_acc,
            'blind none gap': g, 'interval of the blind none gap (servers)': up_gap,
            'none share': out.get('none share'), 'share none or asked': out.get('share none or asked'),
            'conditions': conds,
            'accuracy holds': conds['blind accuracy <= c + 0.05'] and conds['its upper bound (servers) <= c + 0.08'],
            'gap holds': conds['blind none gap <= 0.05'] and conds['its upper bound (servers) <= 0.15']}


def g6(slates, dump, servers, c):
    """G6 on all slates, and on the slates that show no server-mate of the right tool (with the
    none-right slates, which show none of a right tool)."""
    def run(sl, du):
        model = g6_part(sl, du, 'model', servers, c)
        pair = g6_part(sl, du, 'probs', servers, c)
        return {'(a) the model part': model, '(b) the pair': pair,
                '(c) none without the request': {'model part': model['gap holds'], 'pair': pair['gap holds']},
                'holds': model['accuracy holds'] and pair['accuracy holds'] and model['gap holds'] and pair['gap holds']}
    allr = run(slates, dump)
    keep = [i for i, r in enumerate(slates) if not any(x == 'exempt' for x in r.get('roles') or [])]
    sub = run([slates[i] for i in keep], [dump[i] for i in keep]) if keep else None
    allr['without the slates that show server-mates'] = sub
    allr['fails only on the slates that show server-mates'] = (not allr['holds']) and bool(sub and sub['holds'])
    return allr


def gate(root, slates_dir, t2, g4, servers, chance=None):
    sl = {f: load(os.path.join(slates_dir, f + '.jsonl')) for f in FILES}
    pair = {f: load(os.path.join(root, 'pair', f + '.jsonl')) for f in FILES}
    bm25 = load(os.path.join(root, 'bm25', 't5_tools.jsonl'))
    for f in FILES:
        if len(sl[f]) != len(pair[f]):
            raise SystemExit('pair/%s: %d lines for %d slates' % (f, len(pair[f]), len(sl[f])))
    t = sl['t5_tools']
    pr = correct(t, pair['t5_tools'])
    mo = correct(t, pair['t5_tools'], 'model')
    em = correct(t, pair['t5_tools'], 'embedding')
    bm = [argmax(d['probs']) == gold_of(r) for r, d in zip(t, bm25)]
    groups = server_groups(t, servers)
    out = {}
    d_bm, d_em = boot_diff(pr, bm), boot_diff(pr, em)
    out['G1'] = {'pair': mean(pr), 'model part': mean(mo), 'potion-cos-4bit': mean(em), 'bm25': mean(bm),
                 'pair - bm25': d_bm, 'pair - potion-cos-4bit': d_em,
                 'holds': d_bm['lo'] > 0 and d_em['lo'] > 0 and mean(pr) > mean(mo)}
    g2a = {}
    for rw, base in (('t5_tools_rw', 't5_tools'), ('t5_tools_masked_rw', 't5_tools_masked')):
        by = {r['uid']: d for r, d in zip(sl[base], pair[base])}
        f_rw, f_base = [], []
        for r, d in zip(sl[rw], pair[rw]):
            g = gold_of(r)
            if g == len(r['opts']) - 1:
                continue
            fo = r['follow']
            f_rw.append(argmax(d['probs']) == fo)
            f_base.append(argmax(by[r['uid']]['probs']) == fo)
        share, paired = mean(f_rw), mean(f_rw) - mean(f_base)
        g2a[rw] = {'tool-gold slates': len(f_rw), 'answer = the rewritten distractor': share,
                   'minus the same option in the unchanged slate': paired,
                   'holds': share <= G2A and paired <= G2A_PAIRED}
    m = sl['t5_tools_masked']
    pm = mean(correct(m, pair['t5_tools_masked']))
    em_m = mean(correct(m, pair['t5_tools_masked'], 'embedding'))
    none_m = mean([argmax(d['probs']) == len(r['opts']) - 1 for r, d in zip(m, pair['t5_tools_masked'])])
    out['G2'] = {'(a)': g2a, '(b)': {'pair': pm, 'potion-cos-4bit': em_m, 'holds': pm >= em_m - G2B},
                 '(c)': {'none share': none_m, 'holds': G2C[0] <= none_m <= G2C[1]}}
    out['G2']['holds'] = all(x['holds'] for x in g2a.values()) and out['G2']['(b)']['holds'] and out['G2']['(c)']['holds']
    e = ece(t, pair['t5_tools'])
    out['G3'] = {'ece': e, 'holds': e <= G3_ECE}
    out['G4'] = {**g4, 'holds': (g4['max_abs_diff'] <= G4_DP and g4['n'] >= G4_N and g4['argmax_agree'] >= G4_AGREE
                                 and g4['median_ms'] <= G4_MS and g4['bytes'] <= G4_BYTES)}
    t2s, t2d = t2
    a5 = mean(correct(t2s, t2d))
    out['G5'] = {'pair on T2-tools': a5, 'holds': a5 >= G5_T2}
    bl = sl['t5_tools.blind']
    c = chance if chance is not None else mean([1 / (len(r['opts']) - 1) for r in bl if gold_of(r) < len(r['opts']) - 1])
    out['G6'] = g6(bl, pair['t5_tools.blind'], servers, c)
    out['headline'] = {'pair - bm25 (servers)': boot_diff(pr, bm, groups),
                       'pair - potion-cos-4bit (servers)': boot_diff(pr, em, groups)}
    out['headline']['holds'] = (out['headline']['pair - bm25 (servers)']['lo'] > 0
                                and out['headline']['pair - potion-cos-4bit (servers)']['lo'] > 0)
    passed = all(out[g]['holds'] for g in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6'))
    sentences = []
    if out['G6']['fails only on the slates that show server-mates']:
        sentences.append('G6 fails only on the slates that show server-mates of the right tool; the gate is '
                         'still failed, since T5\'s construction was frozen with the gate.')
    sentences.append('The gate %s (%s).' % ('passes' if passed else 'fails', ', '.join(
        '%s %s' % (g, 'holds' if out[g]['holds'] else 'fails') for g in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6'))))
    out['verdict'] = {'passed': passed, 'report': sentences}
    return out


def guard(slates_dir, test_files, open_file):
    if not os.path.exists(open_file):
        raise SystemExit('no %s: the fresh files are read after FRESH-OPEN only' % open_file)
    with open(test_files, encoding='utf-8') as f:
        want = json.load(f)['files']
    for name, h in want.items():
        p = os.path.join(slates_dir, name)
        if os.path.exists(p) and sha256(p) != h:
            raise SystemExit('%s: not the file of %s' % (p, test_files))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k in ('--root', '--slates', '--test-files', '--open', '--t2', '--t2-slates', '--g4', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--chance', type=float)
    ap.add_argument('--md')
    a = ap.parse_args(argv)
    guard(a.slates, a.test_files, a.open)
    servers = {x['uid']: x['server'] for x in load(os.path.join(a.slates, 't5_tools.servers.jsonl'))}
    with open(a.g4, encoding='utf-8') as f:
        g4 = json.load(f)
    out = gate(a.root, a.slates, (load(a.t2_slates), load(a.t2)), g4, servers, a.chance)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    if a.md:
        with open(a.md, 'w') as f:
            f.write('\n\n'.join(out['verdict']['report']) + '\n')
    print('\n'.join(out['verdict']['report']))
    return 0 if out['verdict']['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
