#!/usr/bin/env python3
"""Tests of scripts/s6-route5-checks.py on invented slates: each planted cue of N1 to N5 is caught,
cue-free slates pass, C6 reads the same bar at plan time and after the build, the exit statuses.
Run: python3 -I scripts/test_s6_route5_checks.py (from the repository root)."""
import collections
import importlib.util
import json
import os
import random
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route5_checks', os.path.join(HERE, 's6-route5-checks.py'))
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)

NONE = 'none of these tools fits the request'
WORDS = 'alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike'.split()
STYLES = [lambda i: 'tool_name_%d' % i, lambda i: 'toolName%d' % i, lambda i: 'Tool Name %d' % i,
          lambda i: 'toolname']
FIRST = ['Gets', 'Get', 'Returns', 'Return', 'Lists', 'Find']
# every word of WORDS and every first word in 1,000 descriptions of the training pool: class 9
DF = collections.Counter({w.lower(): 1000 for w in WORDS + FIRST})


def letters(rng, n=8):
    return ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n))


def option(rng, i, words=None, period=None, style=None, first=None, body=None):
    w = words if words is not None else rng.randint(3, 12)
    d = body if body is not None else [rng.choice(WORDS) for _ in range(w)]
    d = [first or rng.choice(FIRST)] + list(d[1:])
    d = ' '.join(d)
    if period if period is not None else rng.random() < 0.5:
        d += '.'
    name = (style if style is not None else rng.choice(STYLES))(i)
    return '%s: %s' % (name, d)


def slate(rng, uid, opts, gold, none=False, level=0, roles=None, kind='natural', flags=None):
    g = len(opts) if none else gold
    target = [0.0] * (len(opts) + 1)
    target[g] = 1.0
    if roles is None:
        roles = [checks.RIGHT if i == g else checks.DRAWN for i in range(len(opts))]
    r = {'uid': uid, 'opts': ['score %.2f rank %d; %s' % (rng.random(), i + 1, o) for i, o in enumerate(opts)] + [NONE],
         'target': target, 'tools': ['t%s-%d' % (uid, i) for i in range(len(opts))], 'kind': kind, 'request': uid,
         'cell_level': level, 'roles': roles}
    if flags is not None:
        r['rewritten'] = flags
    return r


def clean(rng, n, k=7, level=0):
    out = []
    for j in range(n):
        opts = [option(rng, i) for i in range(k)]
        out.append(slate(rng, 'c%d' % j, opts, rng.randrange(k), none=j % 5 == 0, level=level))
    return out


def planted(rng, n, make):
    """Slates whose right option carries a cue: make(rng, i, is_gold) gives option i."""
    out = []
    for j in range(n):
        g = rng.randrange(7)
        opts = [make(rng, i, i == g) for i in range(7)]
        out.append(slate(rng, 'p%d' % j, opts, g))
    return out


def report(recs, judged=True, df=DF):
    return checks.cell_report5('cell', recs, {}, df, judged=judged)


def failing_n(rep):
    return [r['rule'] for r in rep['N'] if r['fails']]


def failing_all(rep):
    return [r['rule'] for r in rep['C2'] + rep['N'] if r['fails']]


def counts(drawn_by_level, slates=None, tool_gold=1000):
    d = sum(drawn_by_level)
    return {'slates': slates or d, 'tool-gold slates': tool_gold, 'drawn slates': d,
            'slates by level': list(drawn_by_level), 'drawn slates by level': list(drawn_by_level)}


class NRules(unittest.TestCase):
    def test_cue_free_slates_pass(self):
        rep = report(clean(random.Random(1), 3000))
        self.assertTrue(rep['judged'])
        self.assertEqual(failing_all(rep), [])
        self.assertEqual(len(rep['N']), 10)
        self.assertFalse(rep['C6']['fails'])

    def test_n1_the_most_digits(self):
        def make(rng, i, gold):
            o = option(rng, i, words=6)
            return o + (' v2 build 3' if gold else '')
        f = failing_n(report(planted(random.Random(2), 1500, make)))
        self.assertIn('N1 the highest share of digits in the description', f)

    def test_n2_the_rarest_word(self):
        def make(rng, i, gold):
            body = [rng.choice(WORDS) for _ in range(6)]
            if gold:
                body[3] = 'zyxqv'  # in no description of the pool: class 0
            return option(rng, i, body=body)
        f = failing_n(report(planted(random.Random(3), 1500, make)))
        self.assertIn('N2 the rarest word (lowest frequency class)', f)
        # the classes: floor(log2 df), a word absent or in one description class 0
        self.assertEqual([checks.word_class({'a': n}, 'a') for n in (0, 1, 2, 3, 4, 1000)], [0, 0, 1, 1, 2, 9])

    def test_n3_the_first_name(self):
        def make(rng, i, gold):
            return option(rng, i, style=(lambda k: 'aardvark_%d' % k) if gold else (lambda k: 'zulu_tool_%d' % k))
        f = failing_n(report(planted(random.Random(4), 1500, make)))
        self.assertIn('N3 the first name in alphabetical order', f)
        self.assertEqual(checks.n3_rank([{'n3 name': ('b', 'B')}, {'n3 name': ('a', 'a')},
                                         {'n3 name': ('b', 'B')}]), [-1, 0, -1])

    def test_n4_capitals(self):
        def make(rng, i, gold):
            body = [rng.choice(WORDS) for _ in range(6)]
            if gold:
                body = [w.capitalize() for w in body]
            return option(rng, i, body=body)
        f = failing_n(report(planted(random.Random(5), 1500, make)))
        self.assertIn('N4 the highest share of capitalised words after the first', f)

    def test_n5_a_slate_built_around_the_gold(self):
        # the right option's words are shared, a few with each distractor; the distractors share
        # nothing with each other: the right option is the slate's centre
        def make_slate(rng, j):
            g = rng.randrange(7)
            core = [letters(rng) for _ in range(12)]
            opts = []
            for i in range(7):
                if i == g:
                    body = ['Gets'] + core
                else:
                    own = [letters(rng) for _ in range(6)]
                    body = ['Gets'] + rng.sample(core, 3) + own
                opts.append(option(rng, i, body=body))
            return slate(rng, 'n%d' % j, opts, g)
        rng = random.Random(6)
        rep = report([make_slate(rng, j) for j in range(1500)], df=collections.Counter())
        self.assertIn('N5 centrality: the highest mean word overlap with the other options', failing_n(rep))
        self.assertAlmostEqual(checks.jaccard({'a', 'b'}, {'b', 'c'}), 1 / 3)

    def test_n_rules_read_the_text_without_d4s_score(self):
        f = checks.n_feats(checks.c4.plain('score 9.99 rank 1; get_x: Gets 2 Data Now'), DF)
        self.assertEqual(f['n3 name'], ('get_x', 'get_x'))
        self.assertAlmostEqual(f['n1 digits'], 1 / len('Gets 2 Data Now'))
        self.assertAlmostEqual(f['n4 capitals'], 2 / 3)
        self.assertEqual(f['n5 words'], {'gets', 'data', 'now'})

    def test_small_cells_and_fresh_files_are_reported_not_judged(self):
        def make(rng, i, gold):
            return option(rng, i, words=6) + (' v2 build 3' if gold else '')
        recs = planted(random.Random(7), 150, make)
        rep = report(recs)
        self.assertFalse(rep['judged'])
        self.assertEqual(failing_n(rep), [])
        self.assertTrue(any(r['above tolerance'] for r in rep['N']))
        rep = report(recs * 20, judged=False)
        self.assertEqual(failing_n(rep), [])
        self.assertFalse(rep['C6']['fails'])


class Restrictions(unittest.TestCase):
    def test_exempt_and_drawn_options_reported_apart(self):
        # the cue is in the exempt options only: the right option is shorter than them, as long
        # as the drawn ones
        rng = random.Random(8)
        recs = []
        for j in range(1500):
            g = rng.randrange(7)
            roles, opts = [], []
            for i in range(7):
                ex = i != g and i % 2 == 0
                roles.append(checks.RIGHT if i == g else (checks.EXEMPT if ex else checks.DRAWN))
                opts.append(option(rng, i, words=14 if ex else 4))
            recs.append(slate(rng, 'r%d' % j, opts, g, roles=roles))
        rep = report(recs)
        rx = rep['restrictions (reported)']
        ex = rx['the right option with the exempt options only']
        dr = rx['the right option with the drawn options only']
        self.assertEqual(ex['slates'], 1500)
        self.assertIn('R4 inverse: fewest words', ex['above tolerance'])
        self.assertNotIn('R4 inverse: fewest words', dr['above tolerance'])
        self.assertIn('L0', rep['by level (reported)'])
        self.assertGreater(rep['C4 by role (reported)'][checks.EXEMPT]['words'], 10)
        self.assertLess(rep['C4 by role (reported)'][checks.DRAWN]['words'], 6)

    def test_positions_in_a_restriction_are_the_slates(self):
        rules = dict((r[0], r[2]) for r in checks.by_position(checks.c4.rule_set(False)))
        F = [{'position': 3}, {'position': 6}]
        self.assertEqual(rules['R7 position 6'](F), [0, 1])
        self.assertEqual(rules['R7 inverse: not position 6'](F), [0, -1])

    def test_distractor_counts(self):
        rng = random.Random(9)
        recs = []
        for j in range(100):
            opts = [option(rng, i) for i in range(3)]
            r = slate(rng, 'd%d' % j, opts, 0)
            r['tools'] = ['gold%d' % j, 'same', 'x%d' % j]
            recs.append(r)
        d = checks.distractors(recs)[checks.DRAWN]
        self.assertEqual((d['options'], d['distinct tools']), (200, 101))
        self.assertAlmostEqual(d['largest share of one tool'], 0.5)


class C6(unittest.TestCase):
    def test_a_pass_and_a_fail_at_the_bar(self):
        at = checks.c6_of(counts([900, 0, 0, 0, 0, 50, 50]))
        self.assertAlmostEqual(at['share of the drawn slates at L5 or L6'], 0.10)
        self.assertFalse(at['fails'])
        above = checks.c6_of(counts([899, 0, 0, 0, 0, 50, 51]))
        self.assertTrue(above['fails'])
        self.assertAlmostEqual(above['share of the drawn slates at L6 (outside the cell)'], 0.051)
        # a cell of fewer than 200 tool-gold slates is reported, not judged
        self.assertFalse(checks.c6_of(counts([10, 0, 0, 0, 0, 0, 90], tool_gold=150))['fails'])
        # slates without a drawn distractor do not count
        c = counts([900, 0, 0, 0, 0, 50, 50], slates=2000)
        c['slates by level'][6] += 1000
        self.assertFalse(checks.c6_of(c)['fails'])

    def test_plan_time_and_built_records_read_the_same_bar(self):
        rng = random.Random(10)
        for l5, l6, fails in ((50, 50, False), (50, 51, True)):
            recs = []
            for lv, n in ((0, 1000 - l5 - l6), (5, l5), (6, l6)):
                for j in range(n):
                    opts = [option(rng, i) for i in range(4)]
                    recs.append(slate(rng, 'l%d-%d' % (lv, j), opts, 0, level=lv))
            # slates with no drawn distractor (the right tool and the request's own tools) are not
            # drawn slates
            for j in range(300):
                opts = [option(rng, i) for i in range(2)]
                recs.append(slate(rng, 'o%d' % j, opts, 0, level=6, roles=[checks.RIGHT, checks.EXEMPT]))
            built = checks.level_counts(recs)
            self.assertEqual(built['drawn slates'], 1000)
            self.assertEqual(checks.c6_of(built)['fails'], fails)
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, 'v5-plan.json')
                with open(p, 'w') as f:
                    json.dump({'C6 cells': [{'cell': 'records tools hard train', 'counts': built}]}, f)
                out = os.path.join(d, 'c6.json')
                self.assertEqual(checks.main(['--plan', p, '--out', out, '--md', out + '.md']), 2 if fails else 0)
                with open(out) as f:
                    rep = json.load(f)
                self.assertEqual(rep['bar'], checks.C6_BAR)
                self.assertEqual(rep['cells']['records tools hard train']['fails'], fails)
                if fails:
                    self.assertIn('before any teacher call', rep['verdict']['meaning'][0])

    def test_the_cell_consistency(self):
        classes = {'names': [0] * 12 + [1] * 10 + [2] * 10, 'words': [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]}
        k = checks.key5('score 1.00 rank 1; get_x: Gets the data.', classes)
        self.assertEqual(k, ('snake', 0, 1, True, 0, 1))
        self.assertTrue(checks.matches(k, ('camel', 1, 2, True, 1, 0), 4))
        self.assertFalse(checks.matches(k, ('camel', 1, 2, True, 1, 0), 3))
        self.assertTrue(checks.matches(k, ('camel', 4, 0, True, 2, 2), 5))
        self.assertFalse(checks.matches(k, ('snake', 0, 1, False, 0, 1), 5))


class Build(unittest.TestCase):
    def write(self, d, recs, group='tools', split='train'):
        rd = os.path.join(d, 'records', group)
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, split + '.jsonl'), 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in recs)

    def test_the_exit_statuses(self):
        rng = random.Random(11)
        with tempfile.TemporaryDirectory() as d:
            recs = []
            for j in range(1500):
                for kind in ('natural', 'renamed'):
                    g = rng.randrange(7)
                    opts = [option(rng, i) for i in range(7)]
                    flags = [(i + j) % 2 == 0 for i in range(7)]
                    flags[g] = j % 2 == 0
                    r = slate(rng, '%s%d' % (kind, j), opts, g, none=j % 5 == 0, kind=kind, flags=flags)
                    recs.append(r)
            self.write(d, recs)
            out = os.path.join(d, 'checks.json')
            self.assertEqual(checks.main(['--build', d, '--out', out, '--md', out + '.md']), 0)
            with open(out) as f:
                v = json.load(f)['verdict']
            self.assertEqual(v['meaning'], [checks.MEANING['pass']])
            # C6: more than 10% of a judged cell's drawn slates at L5 or L6: exit 2
            for r in recs[:400]:
                r['cell_level'] = 6
            self.write(d, recs)
            self.assertEqual(checks.main(['--build', d, '--out', out]), 2)
            with open(out) as f:
                v = json.load(f)['verdict']
            self.assertTrue(v['C6 fails'] and not v['C2 fails'] and not v['N fails'])
            self.assertIn(checks.MEANING['C6'], v['meaning'])
            for r in recs[:400]:
                r['cell_level'] = 0
            # an N rule (N1): exit 2, no remedy
            cue = []
            for j in range(1500):
                g = rng.randrange(7)
                opts = [option(rng, i, words=6) + (' v2 3' if i == g else '') for i in range(7)]
                cue.append(slate(rng, 'q%d' % j, opts, g, none=j % 5 == 0))
            self.write(d, cue)
            self.assertEqual(checks.main(['--build', d, '--out', out]), 2)
            with open(out) as f:
                v = json.load(f)['verdict']
            self.assertIn('N1', {x['tag'] for x in v['N fails']})
            self.assertIn(checks.MEANING['N'], v['meaning'])
            # a defect (C3: the renamed share off by one slate): exit 1 first
            self.write(d, [r for r in recs if r['uid'] != 'renamed0'])
            self.assertEqual(checks.main(['--build', d, '--out', out]), 1)

    def test_n2_reads_the_groups_training_pool(self):
        with tempfile.TemporaryDirectory() as d:
            tp, pp = os.path.join(d, 'tools.jsonl'), os.path.join(d, 'pools5.json')
            with open(tp, 'w') as f:
                for i, desc in enumerate(['Gets the weather.', 'Gets the news.', 'Sets the alarm.']):
                    f.write(json.dumps({'id': 't%d' % i, 'name': 'n%d' % i, 'desc': desc}) + '\n')
            with open(pp, 'w') as f:
                json.dump({'about': 'x', 'tools': ['t0', 't1', 't2'], 'intents': ['t2']}, f)
            dfs = checks.frequency_classes(tp, pp)
            self.assertEqual(dfs['tools']['the'], 3)
            self.assertEqual(dfs['tools']['gets'], 2)
            self.assertEqual(dfs['intents']['gets'], 0)


if __name__ == '__main__':
    unittest.main()
