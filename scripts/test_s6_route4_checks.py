"""Tests of scripts/s6-route4-checks.py on invented slates: planted cues are caught, cue-free slates
pass. Run: python3 -I -m unittest scripts/test_s6_route4_checks.py (from the repository root)."""
import importlib.util
import json
import os
import random
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route4_checks', os.path.join(HERE, 's6-route4-checks.py'))
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)

NONE = 'none of these tools fits the request'
WORDS = 'alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike'.split()
STYLES = [lambda i: 'tool_name_%d' % i, lambda i: 'toolName%d' % i, lambda i: 'Tool Name %d' % i,
          lambda i: 'toolname']


def option(rng, i, words=None, period=None, style=None, first=None):
    w = words if words is not None else rng.randint(3, 12)
    d = [rng.choice(WORDS) for _ in range(w)]
    d[0] = first or rng.choice(['Gets', 'Get', 'Returns', 'Return', 'Lists', 'Find'])
    d = ' '.join(d)
    if period if period is not None else rng.random() < 0.5:
        d += '.'
    name = (style if style is not None else rng.choice(STYLES))(i)
    return '%s: %s' % (name, d)


def slate(rng, uid, k=7, none=False, gold=None, opts=None, flags=None, d4=False):
    opts = opts or [option(rng, i) for i in range(k)]
    if d4:
        opts = ['score %.2f rank %d; %s' % (rng.random(), i + 1, o) for i, o in enumerate(opts)]
    g = len(opts) if none else (gold if gold is not None else rng.randrange(len(opts)))
    target = [0.0] * (len(opts) + 1)
    target[g] = 1.0
    r = {'uid': uid, 'opts': opts + [NONE], 'target': target, 'tools': ['t%s-%d' % (uid, i) for i in range(len(opts))],
         'kind': 'test', 'request': uid}
    if flags is not None:
        r['rewritten'] = flags
    return r


def balanced(j, g, k=7):
    """Rewrite flags with P(rewritten | right) = P(rewritten | other) = 0.5 exactly: the right
    tool's flag alternates over the slates, the other options alternate within the slate."""
    f, t = [], j % 2 == 1
    for i in range(k):
        if i == g:
            f.append(j % 2 == 0)
        else:
            f.append(t)
            t = not t
    return f


def clean(rng, n, k=7, flags=False, d4=False):
    out = []
    for j in range(n):
        none = j % 5 == 0
        g = rng.randrange(k)
        f = balanced(j // 5 * 4 + j % 5, g, k) if flags else None
        out.append(slate(rng, 'c%d' % j, k=k, none=none, gold=g, flags=f, d4=d4))
    return out


def report(recs, judged=True):
    return checks.cell_report('cell', recs, {}, judged=judged)


def failing(rep):
    return [r['rule'] for r in rep['C2'] if r['fails']]


class Checks(unittest.TestCase):
    def test_cue_free_slates_pass(self):
        rng = random.Random(1)
        rep = report(clean(rng, 3000, flags=True, d4=True))
        self.assertTrue(rep['judged'])
        self.assertEqual(failing(rep), [])
        self.assertFalse(rep['C1']['fails'])
        self.assertAlmostEqual(rep['C2'][0]['chance'], 1 / 7)

    def test_shortest_option_is_right(self):
        rng = random.Random(2)
        recs = []
        for j in range(1500):
            opts = [option(rng, i, words=rng.randint(6, 14)) for i in range(7)]
            g = rng.randrange(7)
            opts[g] = option(rng, g, words=rng.randint(3, 5))
            recs.append(slate(rng, 's%d' % j, opts=opts, gold=g))
        f = failing(report(recs))
        self.assertIn('R4 inverse: fewest words', f)

    def test_rewritten_option_is_right(self):
        rng = random.Random(3)
        recs = []
        for j in range(1500):
            g = rng.randrange(7)
            flags = [i == g or rng.random() < 0.3 for i in range(7)]
            recs.append(slate(rng, 'r%d' % j, gold=g, flags=flags))
        rep = report(recs)
        self.assertIn('R6 the option rewritten', failing(rep))
        self.assertTrue(rep['C1']['fails'])

    def test_fewer_options_when_none_is_right(self):
        rng = random.Random(4)
        recs = []
        for j in range(3000):
            none = j % 5 == 0
            recs.append(slate(rng, 'n%d' % j, k=rng.randint(1, 4) if none else rng.randint(4, 7), none=none))
        c3 = report(recs)['C3 none by number of options']
        self.assertTrue(c3['departs beyond noise (p < 0.001; reported, not judged)'])
        self.assertLess(c3['p'], 1e-6)
        # flat by construction (D5): no departure
        c3 = report(clean(random.Random(5), 3000))['C3 none by number of options']
        self.assertFalse(c3['departs beyond noise (p < 0.001; reported, not judged)'])

    def test_source_style_cue(self):
        # the right tool from a glaive-like source (snake names, imperative, no period) among a
        # fill of toolace-like options (spaced names, third person, a period)
        rng = random.Random(6)
        recs = []
        for j in range(1500):
            opts = [option(rng, i, period=True, style=STYLES[2], first='Returns') for i in range(7)]
            g = rng.randrange(7)
            opts[g] = option(rng, g, period=False, style=STYLES[0], first='Get')
            recs.append(slate(rng, 'g%d' % j, opts=opts, gold=g))
        f = failing(report(recs))
        for want in ('R1 inverse: no final period', "R8 name style (4 classes), the slate's minority",
                     'R13 an imperative first word'):
            self.assertIn(want, f)

    def test_c1_on_random_draws(self):
        # D1 draws each flag independently with p 0.5: the rule of amendment 1 (3 standard
        # errors of a fair draw per rate) fails a cell by chance on about 0.5% of draws; the
        # frozen bar of 0.01 failed about half of the cells of ~1,000 slates
        rng = random.Random(10)
        fails = frozen = 0
        for rep_ in range(40):
            recs = [slate(rng, 'd%d' % j, gold=rng.randrange(7), flags=[rng.random() < 0.5 for _ in range(7)])
                    for j in range(1100)]
            c1 = report(recs)['C1']
            fails += c1['fails']
            frozen += abs(c1['difference (reported)']) > 0.01
            self.assertAlmostEqual(c1['its bar (3 SE)'], 3 * (0.25 / c1['right tools']) ** 0.5)
        self.assertLessEqual(fails, 2)
        self.assertGreater(frozen, 10)

    def test_c1_pooled(self):
        # the pooled rates off by more than 0.01: a defect
        rng = random.Random(11)
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, 'records', 'tools')
            os.makedirs(rd)
            recs = []
            for kind in ('natural', 'hard', 'renamed'):
                for j in range(1000):
                    g = rng.randrange(7)
                    f = [rng.random() < 0.525 for _ in range(7)]
                    r = slate(rng, '%s%d' % (kind, j), gold=g, flags=f, none=j % 5 == 0)
                    r['kind'] = kind
                    recs.append(r)
            with open(os.path.join(rd, 'val.jsonl'), 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in recs)
            out = os.path.join(d, 'c.json')
            checks.main(['--build', d, '--out', out])
            with open(out) as f:
                rep = json.load(f)
            self.assertTrue(rep['C1 pooled']['fails'])
            self.assertIn('pooled', rep['verdict']['C1 fails'])

    def test_small_cells_and_fresh_files_are_not_judged(self):
        rng = random.Random(7)
        recs = []
        for j in range(150):
            opts = [option(rng, i, words=10) for i in range(7)]
            opts[0] = option(rng, 0, words=3)
            recs.append(slate(rng, 'x%d' % j, opts=opts, gold=0))
        rep = report(recs)
        self.assertFalse(rep['judged'])
        self.assertEqual(failing(rep), [])
        self.assertTrue(any(r['above tolerance'] for r in rep['C2']))
        self.assertEqual(failing(report(recs * 20, judged=False)), [])

    def test_the_tolerance(self):
        rng = random.Random(8)
        rep = report(clean(rng, 1000))
        r = rep['C2'][0]
        c, n = r['chance'], rep['tool-gold slates']
        self.assertAlmostEqual(r['tolerance'], c + 0.02 + 2 * (c * (1 - c) / n) ** 0.5)

    def test_a_build_and_the_exit_status(self):
        rng = random.Random(9)
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, 'records', 'tools')
            os.makedirs(rd)
            recs = []
            for j in range(1500):
                for kind in ('natural', 'renamed'):
                    none = j % 5 == 0
                    g = rng.randrange(7)
                    r = slate(rng, '%s%d' % (kind, j), none=none, gold=g, flags=balanced(j, g), d4=True)
                    r['kind'] = kind
                    recs.append(r)
            with open(os.path.join(rd, 'train.jsonl'), 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in recs)
            vp = os.path.join(d, 'v4_tools.jsonl')
            with open(vp, 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in clean(rng, 1000))
            out = os.path.join(d, 'checks.json')
            self.assertEqual(checks.main(['--build', d, '--val', vp, '--out', out, '--md', out + '.md']), 0)
            with open(out) as f:
                rep = json.load(f)
            self.assertEqual(rep['verdict']['exit'], 0)
            self.assertIn('records tools renamed train', rep['cells'])
            # a renamed share off by one slate is a defect (exit 1)
            recs = [r for r in recs if not (r['kind'] == 'renamed' and r['uid'] == 'renamed0')]
            with open(os.path.join(rd, 'train.jsonl'), 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in recs)
            self.assertEqual(checks.main(['--build', d, '--out', out]), 1)
            # a surface cue in a cell: exit 2, the remedy open
            os.remove(os.path.join(rd, 'train.jsonl'))
            cue = []
            for j in range(1500):
                opts = [option(rng, i, words=rng.randint(6, 14)) for i in range(7)]
                g = rng.randrange(7)
                opts[g] = option(rng, g, words=3)
                r = slate(rng, 'q%d' % j, opts=opts, gold=g, none=j % 5 == 0)
                r['kind'] = 'natural'
                cue.append(r)
            with open(os.path.join(rd, 'train.jsonl'), 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in cue)
            self.assertEqual(checks.main(['--build', d, '--out', out]), 2)
            with open(out) as f:
                v = json.load(f)['verdict']
            self.assertTrue(v['C2 stops the build'])
            self.assertEqual({x['tag'] for x in v['C2 fails']}, {'R4', 'R5'})
            self.assertTrue(v['the remedy of 4.4 is open'])
            # a cue of another kind (the rewritten option, R6): the remedy is closed
            for r in cue:
                g = r['target'].index(1.0)
                r['rewritten'] = [i == g for i in range(7)]
                r['opts'] = [o for o in clean(rng, 1)[0]['opts']]
            with open(os.path.join(rd, 'train.jsonl'), 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in cue)
            checks.main(['--build', d, '--out', out])
            with open(out) as f:
                v = json.load(f)['verdict']
            self.assertIn('R6', {x['tag'] for x in v['C2 fails']})
            self.assertFalse(v['the remedy of 4.4 is open'])

    def test_scores_are_taken_off(self):
        self.assertEqual(checks.plain('score 1.39 rank 1; lights_on: Turns.'), 'lights_on: Turns.')
        self.assertEqual(checks.plain('score_tool: Scores.'), 'score_tool: Scores.')
        self.assertEqual(checks.style4('GetStockPrice'), 'other')
        self.assertEqual(checks.style4('get stock'), 'spaced')


if __name__ == '__main__':
    unittest.main()
