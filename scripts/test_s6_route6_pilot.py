#!/usr/bin/env python3
"""Tests of scripts/s6-route6-pilot.py on invented dumps: v5's rules kept, and the blind rules 3
and 4 on both sides of 1/7 + 0.05 = 0.1929. python3 -I scripts/test_s6_route6_pilot.py"""
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route6_pilot', os.path.join(HERE, 's6-route6-pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def write(path, rows):
    with open(path, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)


def slates(n, follow=False, none_every=5):
    out = []
    for j in range(n):
        g = 7 if none_every and j % none_every == 0 else j % 7
        t = [0.0] * 8
        t[g] = 1.0
        r = {'uid': 'u%d' % j, 'opts': ['score 0.00 rank %d; t%d: d.' % (i + 1, i) for i in range(7)] + ['none'],
             'target': t}
        if follow:
            r['follow'] = (g + 1) % 7
        out.append(r)
    return out


def dump(recs, pick):
    out = []
    for r in recs:
        p = [0.01] * 8
        p[pick(r)] = 0.93
        out.append({'probs': p, 'target': r['target']})
    return out


class Pilot(unittest.TestCase):
    def build(self, d, blind_right):
        """blind_right[name]: the number of the 100 tool-right blind slates of name whose blind
        pick is the right tool."""
        val = os.path.join(d, 'val')
        os.makedirs(val)
        for name in pilot.p5.MODEL_FILES:
            write(os.path.join(val, name + '.model.jsonl'), slates(100, follow='rw' in name))
        for name in pilot.p5.PAIR_FILES:
            write(os.path.join(val, name + '.jsonl'), slates(100))
        for name in pilot.BLIND_FILES:
            for suf in ('.blind.model.jsonl', '.blind-bare.model.jsonl', '.blind.jsonl'):
                write(os.path.join(val, name + suf), slates(100, none_every=0))
        c = os.path.join(d, 'pilot', 'last')
        os.makedirs(c)
        for name in pilot.p5.MODEL_FILES:
            write(os.path.join(c, name + '.dump.jsonl'), dump(slates(100, follow='rw' in name), lambda r: r['target'].index(1.0)))
        for name in pilot.p5.PAIR_FILES:
            write(os.path.join(c, 'pair_%s.jsonl' % name), dump(slates(100), lambda r: r['target'].index(1.0)))
        for name in pilot.BLIND_FILES:
            n = blind_right[name]

            def pick(r, n=n):
                g = r['target'].index(1.0)
                return g if int(r['uid'][1:]) < n else (g + 1) % 7
            for suf in ('.blind.dump.jsonl', '.blind-bare.dump.jsonl'):
                write(os.path.join(c, name + suf), dump(slates(100, none_every=0), pick))
            write(os.path.join(c, 'pair_%s.blind.jsonl' % name), dump(slates(100, none_every=0), pick))
        pilot.read(c, val)
        return os.path.join(d, 'pilot')

    def test_blind_accuracy_at_0_19_goes_on(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d, {'v5_tools': 19, 'v5_tools_masked': 19})
            with open(os.path.join(p, 'last', 'readings.json')) as f:
                r = json.load(f)
            self.assertAlmostEqual(r['blind']['model B0 v5_tools']['blind accuracy'], 0.19)
            self.assertAlmostEqual(r['blind']['model B0 v5_tools']['chance c'], 1 / 7)
            self.assertIn('pair B0 v5_tools_masked', r['blind'])
            s = pilot.summary(p, 0.5, os.path.join(d, 'summary.json'))
            self.assertFalse(s['abort'], s['why'])
            self.assertAlmostEqual(s['blind bar (rules 3 and 4)'], 0.192857, places=6)

    def test_blind_accuracy_at_0_20_aborts_on_rule_3_or_4(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d, {'v5_tools': 20, 'v5_tools_masked': 19})
            s = pilot.summary(p, 0.5, os.path.join(d, 'summary.json'))
            self.assertEqual([w[0] for w in s['why']], ['3'])
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d, {'v5_tools': 19, 'v5_tools_masked': 20})
            s = pilot.summary(p, 0.5, os.path.join(d, 'summary.json'))
            self.assertEqual([w[0] for w in s['why']], ['4'])
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d, {'v5_tools': 20, 'v5_tools_masked': 20})
            # v5's rule 1 with it
            s = pilot.summary(p, 1.01, os.path.join(d, 'summary.json'))
            self.assertEqual([w[0] for w in s['why']], ['1', '3', '4'])

    def test_fresh_files_are_refused(self):
        with self.assertRaises(SystemExit):
            pilot.load('/x/fresh/t5_tools.blind.jsonl')


if __name__ == '__main__':
    unittest.main()
