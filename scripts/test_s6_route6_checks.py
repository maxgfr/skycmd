#!/usr/bin/env python3
"""Tests of scripts/s6-route6-checks.py: the stop rule of docs/s6-route6-prereg.md 4.8 on both
sides of 0.10, v5's other failures reported without stopping v6, the spread table.
python3 -I scripts/test_s6_route6_checks.py"""
import importlib.util
import json
import os
import random
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, f):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


checks = load('route6_checks', 's6-route6-checks.py')
t5 = load('test_route5_checks', 'test_s6_route5_checks.py')


def family(rng, spread_none, n_none=200, n=1000, n_spread=400, cue_words=False):
    """n When2Call natural slates: n_spread of them spread over two L0 cells (one option without
    its final period), the others over one; n_none none-right, spread_none of them among the
    spread slates."""
    recs = []
    spread_ids = set(range(n_spread))
    none_ids = set(range(spread_none)) | set(range(n_spread, n_spread + n_none - spread_none))
    for j in range(n):
        opts = [t5.option(rng, i, words=5, period=True, style=t5.STYLES[0], first='Gets') for i in range(5)]
        if j in spread_ids:
            opts[3] = t5.option(rng, 3, words=5, period=False, style=t5.STYLES[0], first='Gets')
        if cue_words and j not in none_ids:
            g = rng.randrange(5)
            opts[g] = t5.option(rng, g, period=True, style=t5.STYLES[0], first='Gets',
                                body=['Gets', 'alpha', 'v2', '3', 'bravo'])
        else:
            g = rng.randrange(5)
        r = t5.slate(rng, 'w%d' % j, opts, g, none=j in none_ids)
        r.update({'uid': 'when2call:%d/natural' % j, 'request': 'when2call:%d' % j})
        recs.append(r)
    return recs


class StopRule(unittest.TestCase):
    def run_build(self, recs):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, 'records', 'when2call')
            os.makedirs(rd)
            with open(os.path.join(rd, 'train.jsonl'), 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in recs)
            out = os.path.join(d, 'checks.json')
            st = checks.main(['checks', '--build', d, '--plan-counts', t5.write_classes(d), '--out', out, '--md', out + '.md'])
            with open(out) as f:
                return st, json.load(f)

    def test_at_0_10_v6_goes_on(self):
        # 120 of the 400 slates answered none are none-right: 0.30 = 0.20 + 0.10 exactly
        st, rep = self.run_build(family(random.Random(1), 120))
        fam = rep['N6 families']['records when2call train other']
        self.assertAlmostEqual(fam['rules'][0]['accuracy'], 0.30)
        self.assertAlmostEqual(fam['none share (chance)'], 0.20)
        self.assertEqual(st, 0)
        self.assertFalse(rep['verdict']['v6 stops before training'])
        # N6 itself is above its own tolerance: reported by v5's checks, without stopping v6
        self.assertIn('N6', {x['tag'] for x in rep['v5 verdict, reported (none stops v6)']['N fails']})

    def test_just_above_0_10_v6_stops(self):
        st, rep = self.run_build(family(random.Random(2), 124))
        self.assertEqual(st, 2)
        s = rep['verdict']['stop rule of 4.8 (N6 or its inverse more than 0.10 above the none share in a judged family)']
        self.assertEqual([x['family'] for x in s], ['records when2call train other'])
        self.assertAlmostEqual(s[0]['excess'], 0.11)
        self.assertIn('stops before training', rep['verdict']['meaning'][0])

    def test_a_family_below_the_floor_never_stops(self):
        st, rep = self.run_build(family(random.Random(3), 30, n_none=40, n=180, n_spread=60))
        self.assertFalse(rep['N6 families']['records when2call train other']['judged'])
        self.assertEqual(st, 0)

    def test_other_failures_are_reported_only(self):
        # a planted N1 cue, the N6 family at chance: exit 0
        st, rep = self.run_build(family(random.Random(4), 80, cue_words=True))
        self.assertEqual(st, 0)
        self.assertIn('N1', {x['tag'] for x in rep['v5 verdict, reported (none stops v6)']['N fails']})


class Spread(unittest.TestCase):
    def test_the_table_after_the_fix_and_v5s(self):
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as d:
            builds = {}
            for name, spread_none in (('b6', 0), ('b5', 200)):
                b = os.path.join(d, name)
                os.makedirs(os.path.join(b, 'records', 'when2call'))
                recs = family(rng, spread_none, n=500, n_spread=200)
                for j, r in enumerate(recs):
                    r['kind'] = 'hard' if j % 2 else 'natural'
                    r['cell_level'] = 6 if (j < 200 and name == 'b5') else 0
                with open(os.path.join(b, 'records', 'when2call', 'train.jsonl'), 'w') as f:
                    f.writelines(json.dumps(r) + '\n' for r in recs)
                builds[name] = b
            out = os.path.join(d, 'spread.json')
            pc = t5.write_classes(d)
            self.assertEqual(checks.main(['spread', '--build', builds['b6'], '--v5-build', builds['b5'],
                                          '--plan-counts', pc, '--out', out, '--md', out + '.md']), 0)
            with open(out) as f:
                rep = json.load(f)
            after, before = rep['after the fix'], rep["v5's build"]
            self.assertEqual(sum(e['slates'] for e in after.values()), 500)
            # v5's planted none-right slates all spread; after the fix none does
            self.assertAlmostEqual(before['train natural none-right']['share spread above 0'], 1.0)
            self.assertEqual(after['train natural none-right']['spread above 0'], 0)
            self.assertAlmostEqual(before['train hard none-right']['share at L6'], 1.0)
            with open(out + '.md') as f:
                self.assertIn('| train hard none-right |', f.read())


if __name__ == '__main__':
    unittest.main()
