#!/usr/bin/env python3
"""Tests of scripts/s6-route6-blind.py on invented dumps: a uniform system scores c exactly; a
system that prefers the right tool among the tools while answering none everywhere is caught by
the pick among the tool options; the blind none gap on an invented split; the server bootstrap;
the support of B0. python3 -I scripts/test_s6_route6_blind.py"""
import importlib.util
import os
import random
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route6_blind', os.path.join(HERE, 's6-route6-blind.py'))
blind = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blind)
NONE = 'none of these tools fits the request'


def slate(j, k, g, server_mates=0):
    opts = ['score 0.00 rank %d; tool_%d: Gets %s.' % (i + 1, i, ' '.join(['w'] * (i + 2))) for i in range(k)]
    t = [0.0] * (k + 1)
    t[g] = 1.0
    roles = ['right' if i == g else ('exempt' if 0 < (i - g) % k <= server_mates else 'drawn') for i in range(k)]
    return {'uid': 'u%d' % j, 'opts': opts + [NONE], 'target': t, 'roles': roles, 'kind': 'test', 'src': 't5-tools'}


def dump(slates, f):
    return [{'probs': f(r), 'target': r['target']} for r in slates]


def files(n=1000, seed=1):
    rng = random.Random(seed)
    out = []
    for j in range(n):
        k = 7 if j % 3 else 4
        none = j % 5 == 0
        out.append(slate(j, k, k if none else rng.randrange(k), server_mates=j % 4))
    return out


class Blind(unittest.TestCase):
    def test_a_uniform_system_scores_c_exactly(self):
        s = files()
        out, _ = blind.read(s, dump(s, lambda r: [1.0 / len(r['opts'])] * len(r['opts'])))
        self.assertAlmostEqual(out['blind accuracy'], out['chance c'])
        tr = [r for r in s if r['target'][-1] != 1.0]
        self.assertAlmostEqual(out['chance c'], sum(1 / (len(r['opts']) - 1) for r in tr) / len(tr))
        self.assertAlmostEqual(out['blind accuracy - c'], 0.0)

    def test_a_preference_hidden_behind_none_answers_is_caught(self):
        # none takes most of the mass, the right tool the most of the rest
        s = files()

        def f(r):
            k = len(r['opts']) - 1
            g = r['target'].index(1.0)
            p = [0.01] * k + [0.9]
            if g < k:
                p[g] = 0.05
            return p
        out, _ = blind.read(s, dump(s, f))
        self.assertEqual(out['none share'], 1.0)
        self.assertEqual(out['answer with none allowed: accuracy on the tool-right slates'], 0.0)
        self.assertEqual(out['blind accuracy'], 1.0)
        self.assertGreater(out['blind accuracy'], out['chance c'] + 0.05)
        self.assertAlmostEqual(out['blind none gap'], 0.0)

    def test_the_blind_none_gap(self):
        s = files()

        def f(r):
            k = len(r['opts']) - 1
            none = r['target'][-1] == 1.0
            # none answered on every none-right slate and on one tool-right slate in four
            say = none or int(r['uid'][1:]) % 4 == 0
            return [0.1] * k + [0.9 if say else 0.0]
        out, rows = blind.read(s, dump(s, f))
        tr = [x for x in rows if not x['none right']]
        want = 1.0 - sum(x['answer none'] for x in tr) / len(tr)
        self.assertAlmostEqual(out['blind none gap'], want)
        lo, hi = out['intervals, slates resampled']['blind none gap']
        self.assertLess(lo, want)
        self.assertGreater(hi, want)

    def test_servers_resampled_and_the_server_mates(self):
        s = files(400)
        servers = {r['uid']: 'srv%d' % (int(r['uid'][1:]) % 20) for r in s}
        out, _ = blind.read(s, dump(s, lambda r: [1.0] * len(r['opts'])), servers=servers)
        self.assertEqual(out['intervals, servers resampled']['servers'], 20)
        lo, hi = out['intervals, servers resampled']['blind accuracy']
        self.assertLessEqual(lo, out['blind accuracy'])
        self.assertGreaterEqual(hi, out['blind accuracy'])
        split = out['by server-mates shown (tool-right slates)']
        self.assertEqual(split['0']['tool-right slates'] + split['server-mates shown']['tool-right slates'],
                         out['tool-right slates'])
        self.assertIn('3 or more', split)
        # every tie: the fewest-words option (the first) shares the pick
        self.assertAlmostEqual(out['fewest-words share of the blind picks'], out['chance c'])

    def test_the_support_of_b0(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'train.jsonl')
            a = slate(0, 3, 1)
            b = slate(1, 3, 0)
            b['opts'][1] = 'score 1.25 rank 1; tool_1: Gets w.'
            c = slate(2, 3, 3)
            for r in (a, b, c):
                r['src'], r['split'] = 'tools:glaive', 'train'
            with open(p, 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in (a, b, c))
            out = blind.support([p])
            self.assertEqual(out['tools test train'], {'tool-right records': 2, 'all scores 0.00': 1, 'share': 0.5})


if __name__ == '__main__':
    unittest.main()
