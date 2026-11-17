#!/usr/bin/env python3
"""The tables of docs/s6-route4-diagnosis.md (why s6-route-nano fails when tool names are masked).

Standard library only; run with `python3 -I`. The data folder (DATA, default
/Users/maxime/Documents/Code/skycmd-data/s6-route) is read, never written.

  s6-route4-diag.py kinds                  record kinds of the training and validation files: share,
                                           none-gold count, how often the gold is BM25 rank 1
  s6-route4-diag.py style                  surface features of the descriptions in training renamed
                                           slates (the teacher's rewrite against the originals), and a
                                           rule that never reads the request, by kind
  s6-route4-diag.py make OUT               the validation files of the counterfactual readings (§2 of
                                           the diagnosis) into OUT; OUT must not be under DATA
  s6-route4-diag.py read OUT NAME...       accuracy and its parts from OUT/NAME.jsonl and the
                                           sky-eval dump OUT/NAME.probs.jsonl (--dump-probs)
  s6-route4-diag.py val-dumps              the same parts, and the style of the picks, from the
                                           published validation dumps of the model (rehearsal)
  s6-route4-diag.py published              what the model answered in the published per-slate
                                           outputs of the names-masked readings (T-tools of v2,
                                           T2-tools of v3): none answers and the style of the picks;
                                           no score is computed

Only validation files are scored. The test and fresh files are read for their text only by
`published`, next to the outputs the published readings already wrote.

The model is run on the files of `make` by sky-eval (full depth):
  sky-eval site/public/models/s6-route-nano.gguf --records OUT/NAME.jsonl --split val \
      --max 100000 --exit-threshold 1.0 --dump-probs OUT/NAME.probs.jsonl --json OUT/NAME.eval.json
sky-eval reads each state as written, so a file can carry a BM25 line other than the engine's.
The pair (model and embedding, fused) is run by sky-route, which writes the BM25 line itself from
the options (so only files whose line is the engine's are read that way):
  sky-route fuse --model site/public/models/s6-route-nano.gguf \
      --embed site/public/models/s6-route-embed.gguf --slates OUT/NAME.jsonl --out OUT/fuse_NAME.jsonl
"""
import collections
import json
import math
import os
import random
import re
import sys

DATA = os.environ.get('DATA', '/Users/maxime/Documents/Code/skycmd-data/s6-route').rstrip('/') + '/'
NONE = 'none of these tools fits the request'


def load(path):
    with open(path if path.startswith('/') else DATA + path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---- a twin of sky_infer::route (crates/sky-infer/src/route.rs); `make` checks it on the
# published model copies of the validation slates
STOP = set('a an and are as at be but by for if in into is it no not of on or such that the their '
           'then there these they this to was will with'.split())


def tokens(t):
    return [w for w in re.split(r'[^\w]', t.lower()) if len(w) >= 2 and w not in STOP]


def bm25(q, docs):
    T = [tokens(d) for d in docs]
    n = len(docs)
    avg = sum(map(len, T)) / n
    out = []
    for d in T:
        s = 0.0
        for t in tokens(q):
            df = sum(1 for o in T if t in o)
            if df == 0:
                continue
            tf = d.count(t)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = 1.5 * (1 - 0.75 + 0.75 * len(d) / avg) if avg > 0 else 1.5
            s += idf * tf / (tf + norm)
        out.append(s)
    return out


def ranks(s):
    o = sorted(range(len(s)), key=lambda i: (-s[i], i))
    r = [0] * len(s)
    for k, i in enumerate(o):
        r[i] = k + 1
    return r


def name_of(o):
    return o.split(': ', 1)[0] if ': ' in o else o


def desc_of(o):
    return o.split(': ', 1)[1] if ': ' in o else ''


def request_of(state):
    return state.split('\nBM25: ')[0][len('User: '):]


def state(req, tools):
    s = bm25(req, tools)
    r = ranks(s)
    line = '; '.join('%s score %.2f rank %d' % (name_of(o), x, k) for o, x, k in zip(tools, s, r))
    return 'User: ' + req.strip() + '\nBM25: ' + line


def cut(d):
    """crates/sky-data/src/route/mod.rs cut_description."""
    d = ' '.join(d.split())
    end = len(d)
    for i, c in enumerate(d):
        if c in '.!?' and (i + 1 == len(d) or d[i + 1] == ' '):
            end = i + 1
            break
    return ' '.join([w for w in d[:end].split(' ') if w][:30])


# ---- surface features of a description (a proxy of the teacher's style, not the model's own cue)
def third(d):
    w = d.split(' ')[0] if d else ''
    return w[:1].isupper() and w.endswith('s') and not w.endswith('ss')


def feats(d):
    return {'period': d.endswith('.'), 'third': third(d), 'both': d.endswith('.') and third(d)}


def gold_of(r):
    return r['target'].index(1.0)


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def number_taken(text):
    """Whether crates/sky-infer/src/num.rs extract() turns some number of `text` into [NUM]
    (a match glued to a preceding ASCII letter is kept as text)."""
    b = text.encode()
    i = 0
    while i < len(b):
        if not (48 <= b[i] <= 57 or b[i] == 45):
            i += 1
            continue
        j = i + (b[i] == 45)
        s = j
        while j < len(b) and 48 <= b[j] <= 57:
            j += 1
        if j == s:
            i += 1
            continue
        if j + 1 < len(b) and b[j] == 46 and 48 <= b[j + 1] <= 57:
            j += 2
            while j < len(b) and 48 <= b[j] <= 57:
                j += 1
        if not (i > 0 and (65 <= b[i - 1] <= 90 or 97 <= b[i - 1] <= 122)):
            return True
        i = j
    return False


# ------------------------------------------------------------------------------------- kinds
def kinds():
    assert number_taken('tool_75') and number_taken('search 2') and not number_taken('glm-5')
    n = m = 0
    names, taken = set(), set()
    for r in load('build/records/tools/train.jsonl'):
        if r['kind'] == 'renamed':
            continue
        for o in r['opts'][:-1]:
            nm = name_of(o)
            n += 1
            names.add(nm)
            if number_taken(nm):
                m += 1
                taken.add(nm)
    print('tools/train, not renamed: %d tool options, %d whose name has a number taken by the extraction '
          '(%d of %d distinct names)' % (n, m, len(taken), len(names)))
    for grp in ['tools', 'when2call', 'intents']:
        for split in ['train', 'val']:
            recs = load('build/records/%s/%s.jsonl' % (grp, split))
            C = collections.defaultdict(collections.Counter)
            for r in recs:
                k = r['kind']
                c = C[k]
                c['n'] += 1
                g = gold_of(r)
                if g == len(r['opts']) - 1:
                    c['none gold'] += 1
                    continue
                line = r['state'].split('\nBM25: ', 1)[1]
                ent = re.findall(r' score (-?\d+\.\d\d) rank (\d+)(?:; |$)', line)
                assert len(ent) == len(r['opts']) - 1, r['uid']
                s = [float(x) for x, _ in ent]
                rk = [int(y) for _, y in ent]
                c['tool gold'] += 1
                c['gold is BM25 rank 1'] += rk[g] == 1
                c['every score 0'] += max(s) == 0
            tot = len(recs)
            print('%s/%s: %d records' % (grp, split, tot))
            for k, c in sorted(C.items()):
                tg = max(c['tool gold'], 1)
                print('  %-8s %6d (share %.4f)  none gold %6d  gold BM25 rank 1 %.4f  every score 0 %.4f'
                      % (k, c['n'], c['n'] / tot, c['none gold'], c['gold is BM25 rank 1'] / tg,
                         c['every score 0'] / tg))


# ------------------------------------------------------------------------------------- style
def style():
    rng = random.Random(0)

    def rule(opts):
        s = [2 * desc_of(o).endswith('.') + third(desc_of(o)) for o in opts[:-1]]
        m = max(s)
        return rng.choice([i for i, x in enumerate(s) if x == m])

    for grp in ['tools', 'intents', 'when2call']:
        F = collections.defaultdict(collections.Counter)
        R = collections.defaultdict(lambda: [0, 0, 0.0])
        for r in load('build/records/%s/train.jsonl' % grp):
            g = gold_of(r)
            if g == len(r['opts']) - 1:
                continue
            a = R[r['kind']]
            a[0] += rule(r['opts']) == g
            a[1] += 1
            a[2] += 1 / (len(r['opts']) - 1)
            if r['kind'] == 'renamed':
                for i, o in enumerate(r['opts'][:-1]):
                    k = 'gold (rewrite)' if i == g else 'other (original)'
                    F[k]['n'] += 1
                    for f, v in feats(desc_of(o)).items():
                        F[k][f] += v
        print('%s/train, renamed slates:' % grp)
        for k, c in F.items():
            print('  %-17s n %6d  ends with a period %.3f  first word in -s %.3f'
                  % (k, c['n'], c['period'] / c['n'], c['third'] / c['n']))
        print('  request-blind rule (period, then first word in -s; ties at random), slates with a tool gold:')
        for k, (a, n, ch) in sorted(R.items()):
            print('    %-8s %.3f   chance among the tools %.3f   (n %d)' % (k, a / n, ch / n, n))
    a = n = 0
    for r in load('build/val/v_tools_masked.model.jsonl'):
        g = gold_of(r)
        if g == len(r['opts']) - 1:
            continue
        a += rule(r['opts']) == g
        n += 1
    print('v_tools_masked, slates with a tool gold: rule %.3f (n %d), chance 1/7 = 0.143' % (a / n, n))


# -------------------------------------------------------------------------------------- make
def write(out, name, recs):
    with open(os.path.join(out, name + '.jsonl'), 'w') as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('%-34s %d' % (name, len(recs)))


def make(out):
    out = os.path.abspath(out)
    if (out + '/').startswith(DATA):
        sys.exit('refusing to write under the data folder')
    os.makedirs(out, exist_ok=True)
    tools = {t['id']: t for t in load('build/stage/tools.jsonl')}
    rew = {t['id']: cut(t['description']) for t in load('build/teacher/rewrite-description.jsonl')}
    m = load('build/val/v_tools_masked.model.jsonl')
    u = load('build/val/v_tools.model.jsonl')
    bad = sum(1 for r in m + u if state(request_of(r['state']), r['opts'][:-1]) != r['state'])
    if bad:
        sys.exit('the BM25 twin differs from the published model copies on %d slates' % bad)
    tv = load('build/records/tools/val.jsonl')
    ren = [r for r in tv if r['kind'] == 'renamed']
    nat = [r for r in tv if r['kind'] == 'natural']
    for r in ren:  # the renamed record is the twin of build.rs: rewrite on the gold, originals elsewhere
        g = gold_of(r)
        for i, t in enumerate(r['tools']):
            want = rew.get(t, tools[t]['desc']) if i == g else tools[t]['desc']
            if desc_of(r['opts'][i]) != want:
                sys.exit('a renamed record is not built as expected: ' + r['uid'])

    def with_opts(r, opts, st=None):
        r = dict(r)
        r['opts'] = opts
        r['state'] = st if st is not None else state(request_of(r['state']), opts[:-1])
        return r

    def opt(n, d):
        return n + ': ' + d if d else n

    def rename(r, f):
        return with_opts(r, [opt(f(name_of(o), i), desc_of(o)) for i, o in enumerate(r['opts'][:-1])]
                         + [r['opts'][-1]])

    def no_line(r):
        return with_opts(r, list(r['opts']), 'User: ' + request_of(r['state']).strip())

    write(out, 'val_renamed', ren)
    write(out, 'val_natural', nat)
    x = []
    for r in ren:
        o = list(r['opts'])
        g = gold_of(r)
        o[g] = opt(name_of(o[g]), tools[r['tools'][g]]['desc'])
        x.append(with_opts(r, o))
    write(out, 'renamed_gold_original', x)
    x = []
    for r in ren:
        req = request_of(r['state'])
        tl = r['opts'][:-1]
        s = bm25(req, tl)
        k = ranks(s)
        n = len(tl)
        line = '; '.join('%s score %.2f rank %d' % (name_of(tl[i]), s[(i + 1) % n], k[(i + 1) % n])
                         for i in range(n))
        x.append(with_opts(r, list(r['opts']), 'User: ' + req.strip() + '\nBM25: ' + line))
    write(out, 'renamed_line_shifted', x)
    write(out, 'renamed_no_line', [no_line(r) for r in ren])
    x = []
    for r in m:
        g = gold_of(r)
        if g < len(r['opts']) - 1 and r['tools'][g] in rew:
            o = list(r['opts'])
            o[g] = opt(name_of(o[g]), rew[r['tools'][g]])
            x.append(with_opts(r, o))
    write(out, 'masked_gold_rewritten', x)
    rng = random.Random(0)
    x = []
    for r in m:
        g = gold_of(r)
        cand = [i for i in range(len(r['opts']) - 1) if i != g and r['tools'][i] in rew]
        if not cand:
            continue
        d = rng.choice(cand)
        o = list(r['opts'])
        o[d] = opt(name_of(o[d]), rew[r['tools'][d]])
        y = with_opts(r, o)
        y['follow'] = d
        x.append(y)
    write(out, 'masked_distractor_rewritten', x)
    write(out, 'masked_no_line', [no_line(r) for r in m])
    write(out, 'unmasked_no_line', [no_line(r) for r in u])
    write(out, 'masked_glued', [rename(r, lambda n, i: n.replace('tool_', 'tool')) for r in m])
    write(out, 'masked_letters', [rename(r, lambda n, i: 'tool_' + 'abcdefg'[i]) for r in m])
    words = ('amber birch cedar delta ember fjord grove harbor indigo juniper kestrel lagoon maple '
             'nectar opal pebble quartz raven sierra tundra umber violet willow yarrow zephyr').split()

    def wordy(r):
        w = random.Random(r['uid']).sample(words, len(r['opts']) - 1)
        return rename(r, lambda n, i: w[i])
    write(out, 'masked_words', [wordy(r) for r in m])
    rng = random.Random(1)

    def km(r):
        ks = rng.sample(range(1, 100), len(r['opts']) - 1)
        return rename(r, lambda n, i: 'tool_%d' % ks[i])
    write(out, 'natural_masked', [km(r) for r in nat])


# -------------------------------------------------------------------------------------- read
def parts(recs, probs):
    assert len(recs) == len(probs)
    n = len(recs)
    C = collections.Counter()
    conf = 0.0
    tg = 0
    for r, d in zip(recs, probs):
        p = d['probs']
        assert d['target'] == r['target']
        a = argmax(p)
        k = len(p)
        g = gold_of(r)
        r1 = ranks(bm25(request_of(r['state']), r['opts'][:-1])).index(1)
        conf += p[a]
        C['right'] += a == g
        C['none answers'] += a == k - 1
        if g != k - 1:
            tg += 1
            C['tool gold right'] += a == g
            C['gold rank 1'] += r1 == g
        C['pick rank 1'] += a == r1
        if 'follow' in r:
            C['follow'] += a == r['follow']
    out = {'n': n, 'accuracy': C['right'] / n, 'mean top p': conf / n, 'none answers': C['none answers'] / n,
           'tool-gold accuracy': C['tool gold right'] / max(tg, 1), 'pick = BM25 rank 1': C['pick rank 1'] / n,
           'gold = BM25 rank 1 (tool gold)': C['gold rank 1'] / max(tg, 1)}
    if any('follow' in r for r in recs):
        out['pick = the rewritten distractor'] = C['follow'] / n
    return out


def show(name, o):
    print('%-30s ' % name + '  '.join('%s %s' % (k, v if isinstance(v, int) else '%.4f' % v)
                                      for k, v in o.items()))


def read(out, names):
    for nm in names:
        show(nm, parts(load(os.path.join(os.path.abspath(out), nm + '.jsonl')),
                       load(os.path.join(os.path.abspath(out), nm + '.probs.jsonl'))))


def pick_style(slates, probs):
    by = {d['uid']: d for d in probs}
    F = collections.defaultdict(collections.Counter)
    for r in slates:
        p = by[r['uid']]['probs']
        a = argmax(p)
        g = gold_of(r)
        k = len(p)
        rows = [('option', o) for o in r['opts'][:-1]]
        if g != k - 1:
            rows.append(('gold', r['opts'][g]))
        if a != k - 1:
            rows.append(('pick', r['opts'][a]))
        for key, o in rows:
            F[key]['n'] += 1
            for f, v in feats(desc_of(o)).items():
                F[key][f] += v
    for key in ['option', 'gold', 'pick']:
        c = F[key]
        print('  %-6s n %5d  ends with a period %.3f  first word in -s %.3f'
              % (key, c['n'], c['period'] / c['n'], c['third'] / c['n']))
    src = collections.defaultdict(lambda: [0, 0])
    sty = collections.defaultdict(lambda: [0, 0])
    for r in slates:
        p = by[r['uid']]['probs']
        g = gold_of(r)
        if g == len(p) - 1:
            continue
        ok = argmax(p) == g
        s = src[r['request'].split(':')[0]]
        s[0] += ok
        s[1] += 1
        t = sty[feats(desc_of(r['opts'][g]))['both']]
        t[0] += ok
        t[1] += 1
    print('  tool-gold accuracy by source:', ', '.join('%s %d/%d' % (k, a, n) for k, (a, n) in sorted(src.items())))
    print('  tool-gold accuracy, gold ends with a period and starts with a word in -s: %d/%d; otherwise %d/%d'
          % (sty[True][0], sty[True][1], sty[False][0], sty[False][1]))


def val_dumps():
    for f, d in [('v_tools', 'v_tools'), ('v_tools_masked', 'v_tools_masked')]:
        s = load('build/val/%s.model.jsonl' % f)
        p = load('eval/fresh/rehearsal/model/%s.jsonl' % d)
        by = {x['uid']: x for x in p}
        p = [by[r['uid']] for r in s]
        show(f, parts(s, p))
        pos = collections.Counter()
        rk = collections.Counter()
        for r, d in zip(s, p):
            a = argmax(d['probs'])
            if a == len(d['probs']) - 1:
                continue
            pos[a] += 1
            rk[ranks(bm25(request_of(r['state']), r['opts'][:-1]))[a]] += 1
        print('  picks by position 0..6:', [pos[i] for i in range(7)],
              ' picks by BM25 rank 1..7:', [rk[i] for i in range(1, 8)])
        pick_style(s, p)


def published():
    for slates, dump in [('build/test/t_tools_masked.jsonl', 'eval/test/model/t_tools_masked.jsonl'),
                         ('build/fresh/f_tools_masked.jsonl', 'eval/fresh/model/f_tools_masked.jsonl')]:
        S = load(slates)
        P = load(dump)
        assert len(S) == len(P) and all(a['uid'] == b['uid'] for a, b in zip(S, P))
        none = sum(1 for d in P if argmax(d['probs']) == len(d['probs']) - 1)
        conf = sum(max(d['probs']) for d in P) / len(P)
        print('%s (outputs %s): %d slates, %s options each, none answers %d, mean top probability %.4f'
              % (slates, dump, len(S), sorted({len(d['probs']) for d in P}), none, conf))
        picks = [desc_of(s['opts'][argmax(d['probs'])]) for s, d in zip(S, P)
                 if argmax(d['probs']) != len(d['probs']) - 1]
        opts = [desc_of(o) for s in S for o in s['opts'][:-1]]
        print('  picked option ends with a period %.4f, every tool option %.4f'
              % (sum(x.endswith('.') for x in picks) / len(picks), sum(x.endswith('.') for x in opts) / len(opts)))


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == 'kinds':
        kinds()
    elif cmd == 'style':
        style()
    elif cmd == 'make' and len(a) == 2:
        make(a[1])
    elif cmd == 'read' and len(a) >= 3:
        read(a[1], a[2:])
    elif cmd == 'val-dumps':
        val_dumps()
    elif cmd == 'published':
        published()
    else:
        sys.exit(__doc__)
