"""Tests of scripts/s6-route7-reference.py (s6-route v7, G4): the reference of the pair under D4
on invented dumps, and the shape of G4's JSON from a bench line.

    python3 -I scripts/test_s6_route7_reference.py
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 's6-route7-reference.py')
_spec = importlib.util.spec_from_file_location('ref7', SCRIPT)
ref7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref7)


def slate(uid, gold, rendered=False):
    opts = ['get_a: Gets a.', 'get_b: Gets b.', 'none of these tools fits the request']
    if rendered:
        opts = ['score 1.25 rank 1; get_a: Gets a.', 'score 0.00 rank 2; get_b: Gets b.', opts[2]]
    t = [0.0, 0.0, 0.0]
    t[gold] = 1.0
    return {'uid': uid, 'opts': opts, 'state': 'User: x', 'target': t}


def write(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


BENCH = {'n': 200, 'max_abs_diff': 2.1e-6, 'argmax_agree': 200, 'median_cached_us': 31250.0,
         'median_first_us': 48000.0, 'bytes_model': 2_400_000, 'bytes_embedding': 1_273_248,
         'build': 'simd128', 'fast': False, 'render': 'scores-in-options', 'model': {}, 'embedding': {},
         'static_hits': 200, 'tools_hits': 200}


class Reference(unittest.TestCase):
    def fixture(self, t, rendered=True):
        sl = [slate('u1', 0), slate('u2', 2)]
        rd = [slate('u1', 0, rendered), slate('u2', 2, rendered)]
        model = [{'uid': 'u1', 'target': [1.0, 0.0, 0.0], 'gold': 0, 'probs': [0.6, 0.3, 0.1]},
                 {'uid': 'u2', 'target': [0.0, 0.0, 1.0], 'gold': 2, 'probs': [0.2, 0.2, 0.6]}]
        emb = [{'uid': 'u1', 'gold': 0, 'probs': [0.5, 0.4, 0.1]},
               {'uid': 'u2', 'gold': 2, 'probs': [0.3, 0.3, 0.4]}]
        paths = {}
        for k, rows in (('slates', sl), ('rendered', rd), ('model', model), ('embedding', emb)):
            paths[k] = os.path.join(t, k + '.jsonl')
            write(paths[k], rows)
        paths['fit'] = os.path.join(t, 'fit.json')
        with open(paths['fit'], 'w') as f:
            json.dump({'beta': 0.81, 'tau_ask6': {'tau': 0.7}}, f)
        return paths, model, emb

    def run_ref(self, paths, t):
        out = os.path.join(t, 'fused.jsonl')
        args = ['reference'] + sum([['--' + k, v] for k, v in paths.items()], []) + ['--out', out]
        return subprocess.run([sys.executable, '-I', SCRIPT] + args, capture_output=True, text=True), out

    def test_the_fused_probabilities_of_v3s_rule_at_the_fit_s_beta(self):
        with tempfile.TemporaryDirectory() as t:
            paths, model, emb = self.fixture(t)
            r, out = self.run_ref(paths, t)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [json.loads(l) for l in open(out)]
            self.assertEqual([x['uid'] for x in rows], ['u1', 'u2'])
            for x, m, e in zip(rows, model, emb):
                z = [(a * b) ** 0.81 for a, b in zip(m['probs'], e['probs'])]
                for p, q in zip(x['probs'], z):
                    self.assertTrue(math.isclose(p, q / sum(z), rel_tol=1e-9))
                self.assertEqual(x['gold'], m['gold'])

    def test_a_copy_not_rendered_by_d4_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            paths, _, _ = self.fixture(t, rendered=False)
            r, _ = self.run_ref(paths, t)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('D4', r.stderr)

    def test_other_slates_or_another_order_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            paths, model, _ = self.fixture(t)
            write(paths['model'], model[::-1])
            r, _ = self.run_ref(paths, t)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('not the same slate', r.stderr)


class G4(unittest.TestCase):
    def test_the_shape_of_g4s_json(self):
        with tempfile.TemporaryDirectory() as t:
            b, out = os.path.join(t, 'wasm7.json'), os.path.join(t, 'g4.json')
            with open(b, 'w') as f:
                f.write(json.dumps(BENCH) + '\n')
            r = subprocess.run([sys.executable, '-I', SCRIPT, 'g4', '--bench', b, '--out', out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            g = json.load(open(out))
            for k in ('max_abs_diff', 'argmax_agree', 'n', 'median_ms', 'bytes'):
                self.assertIn(k, g)
            self.assertEqual((g['n'], g['argmax_agree']), (200, 200))
            self.assertAlmostEqual(g['median_ms'], 31.25)
            self.assertEqual(g['bytes'], 3_673_248)
            self.assertIsInstance(g['max_abs_diff'], float)
            # the gate reads these fields as they are
            spec = importlib.util.spec_from_file_location('gate6', os.path.join(HERE, 's6-route6-gate.py'))
            gate6 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gate6)
            holds = (g['max_abs_diff'] <= gate6.G4_DP and g['n'] >= gate6.G4_N and g['argmax_agree'] >= gate6.G4_AGREE
                     and g['median_ms'] <= gate6.G4_MS and g['bytes'] <= gate6.G4_BYTES)
            self.assertTrue(holds)

    def test_a_bench_not_by_d4_or_without_parity_is_refused(self):
        for bad in ({**BENCH, 'render': 'bm25-line'}, {**BENCH, 'fast': True},
                    {k: v for k, v in BENCH.items() if k != 'max_abs_diff'}):
            with self.assertRaises(SystemExit):
                ref7.g4(bad)


if __name__ == '__main__':
    unittest.main()
