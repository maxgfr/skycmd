#!/usr/bin/env python3
"""Tests of scripts/s6-route6-fit.py (beta and tau_ask6 on v5_tools by v3's rules) and of the
refusals of scripts/route6-diagnostic.sh, on invented dumps. python3 -I scripts/test_s6_route6_fit.py"""
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route6_fit', os.path.join(HERE, 's6-route6-fit.py'))
fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit)


def rows(n, right, field='target'):
    out = []
    for j in range(n):
        g = j % 8
        p = [0.02] * 8
        p[g if right(j) else (g + 1) % 8] = 0.86
        r = {'probs': p}
        if field == 'target':
            t = [0.0] * 8
            t[g] = 1.0
            r['target'] = t
        else:
            r['gold'] = g
        out.append(r)
    return out


class Fit(unittest.TestCase):
    def test_beta_and_tau_on_invented_dumps(self):
        model = rows(400, lambda j: j % 4 != 0)
        exit0 = rows(400, lambda j: j % 3 != 0)
        partner = rows(400, lambda j: j % 5 != 0, field='gold')
        out = fit.fit(model, exit0, partner)
        self.assertTrue(0.01 <= out['beta'] <= 4.0)
        self.assertIn('tau', out['tau_ask6'])
        self.assertGreaterEqual(out['system']['accuracy'], 0.75)
        self.assertAlmostEqual(out['alone']['model'], 0.75)
        with self.assertRaises(SystemExit):
            fit.fit(model, exit0, partner[:-1] + [dict(partner[-1], gold=(partner[-1]['gold'] + 1) % 8)])

    def test_blind_or_fresh_dumps_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'v5_tools.blind.dump.jsonl')
            open(p, 'w').close()
            with self.assertRaises(SystemExit):
                fit.main(['--model', p, '--exit0', p, '--partner', p, '--out', os.path.join(d, 'f.json')])


class Diagnostic(unittest.TestCase):
    def run_dry(self, *files):
        env = dict(os.environ, DRY_RUN='1')
        return subprocess.run(['bash', os.path.join(HERE, 'route6-diagnostic.sh'), *files], env=env,
                              capture_output=True, text=True)

    def test_its_fixed_list(self):
        for ok in ('v_tools', 'f_tools_masked'):
            r = self.run_dry(ok)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('--render bm25-line', r.stdout)
        r = self.run_dry('f_tools')
        self.assertIn('FRESH-OPEN', r.stdout)
        for bad in ('t5_tools', 'v5_tools', 't_tools', 'f_clinc'):
            r = self.run_dry(bad)
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn('refusing', r.stderr)


if __name__ == '__main__':
    unittest.main()
