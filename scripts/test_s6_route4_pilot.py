#!/usr/bin/env python3
"""Tests of scripts/s6-route4-pilot.py on invented dumps: the readings, and both abort rules.
python3 -I scripts/test_s6_route4_pilot.py"""
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route4_pilot', os.path.join(HERE, 's6-route4-pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def write(path, rows):
    with open(path, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)


def slates(n, follow=False):
    out = []
    for j in range(n):
        g = 7 if j % 5 == 0 else j % 7
        t = [0.0] * 8
        t[g] = 1.0
        r = {'uid': 'u%d' % j, 'opts': ['o'] * 8, 'target': t}
        if follow:
            r['follow'] = (g + 1) % 7
        out.append(r)
    return out


def dump(recs, pick):
    out = []
    for r in recs:
        a = pick(r)
        p = [0.01] * 8
        p[a] = 0.93
        out.append({'probs': p, 'target': r['target']})
    return out


class Pilot(unittest.TestCase):
    def build(self, d, follow_share):
        val = os.path.join(d, 'val')
        os.makedirs(val)
        for name in pilot.MODEL_FILES:
            rw = 'rw' in name
            write(os.path.join(val, name + '.model.jsonl'), slates(100, follow=rw))
        for name in pilot.PAIR_FILES:
            write(os.path.join(val, name + '.jsonl'), slates(100))
        for ck in ('ckpt-5', 'ckpt-10', 'last'):
            c = os.path.join(d, 'pilot', ck)
            os.makedirs(c)
            for name in pilot.MODEL_FILES:
                rw = 'rw' in name
                recs = slates(100, follow=rw)
                if rw:
                    k = [0]

                    def pick(r, k=k):
                        g = r['target'].index(1.0)
                        if g == 7:
                            return 7
                        k[0] += 1
                        return r['follow'] if k[0] <= follow_share * 80 else g
                else:
                    def pick(r):
                        return r['target'].index(1.0)
                write(os.path.join(c, name + '.dump.jsonl'), dump(recs, pick))
            for name in pilot.PAIR_FILES:
                write(os.path.join(c, 'pair_%s.jsonl' % name), dump(slates(100), lambda r: r['target'].index(1.0)))
            pilot.read(c, val)
        return os.path.join(d, 'pilot')

    def test_readings_and_no_abort(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d, 0.1)
            with open(os.path.join(p, 'ckpt-5', 'readings.json')) as f:
                r = json.load(f)
            self.assertEqual(r['model']['v4_tools']['accuracy'], 1.0)
            self.assertAlmostEqual(r['model']['v4_tools_masked_rw']['answer = the rewritten distractor (tool-gold slates)'], 0.1)
            self.assertEqual(r['renamed - natural (model)'], 0.0)
            s = pilot.summary(p, 0.5, os.path.join(d, 'summary.json'))
            self.assertEqual(s['last checkpoint'], 'last')
            self.assertEqual([c['checkpoint'] for c in s['checkpoints']], ['ckpt-5', 'ckpt-10', 'last'])
            self.assertFalse(s['abort'])

    def test_both_abort_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d, 0.5)
            s = pilot.summary(p, 0.5, os.path.join(d, 'summary.json'))
            self.assertTrue(s['abort'])
            self.assertTrue(s['why'][0].startswith('2:'))
            s = pilot.summary(p, 1.01, os.path.join(d, 'summary.json'))
            self.assertEqual([w[0] for w in s['why']], ['1', '2'])

    def test_fresh_files_are_refused(self):
        with self.assertRaises(SystemExit):
            pilot.load('/x/fresh/t4_tools.jsonl')


if __name__ == '__main__':
    unittest.main()
