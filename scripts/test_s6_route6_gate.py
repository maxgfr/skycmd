#!/usr/bin/env python3
"""Tests of scripts/s6-route6-gate.py on synthetic dumps: G6 on both sides of each of its bars
(the blind accuracy at c + 0.05, its upper bound with servers resampled at c + 0.08, the blind
none gap at 0.05 and its upper bound at 0.15), the first sentence when G6 fails only on the slates
that show server-mates, and G1 to G5 passing and failing. python3 -I scripts/test_s6_route6_gate.py"""
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route6_gate', os.path.join(HERE, 's6-route6-gate.py'))
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
NONE = 'none of these tools fits the request'
K = 7
C = 1 / 7


def spread(j, f):
    """A deterministic share f of the indices j, spread over every server."""
    return (j * 7919 + 13) % 1000 < f * 1000


def slates(n=2000, mates_every=0):
    out = []
    for j in range(n):
        g = K if j % 5 == 0 else (j * 3) % K
        t = [0.0] * (K + 1)
        t[g] = 1.0
        roles = ['right' if i == g else 'drawn' for i in range(K)]
        if mates_every and g < K and j % mates_every == 1:
            roles[(g + 1) % K] = 'exempt'
        out.append({'uid': 'u%d' % j, 'opts': ['t%d: d%s.' % (i, ' w' * i) for i in range(K)] + [NONE],
                    'target': t, 'roles': roles, 'follow': (g + 2) % K if g < K else 1})
    return out


def probs(k, pick, none_p=0.0):
    p = [0.05] * (k + 1)
    p[pick] = 0.6
    p[k] = max(p[k], none_p)
    return p


def blind_dump(sl, right, say_none):
    """right(j, r): whether the blind pick is the right tool; say_none(j, r): whether none is
    answered (none's probability above every tool's)."""
    out = []
    for j, r in enumerate(sl):
        g = r['target'].index(1.0)
        pick = g if g < K and right(j, r) else (g + 1) % K if g < K else 0
        p = probs(K, pick, 0.9 if say_none(j, r) else 0.0)
        out.append({'probs': p, 'model': p, 'embedding': p})
    return out


def servers_of(sl, n):
    return {r['uid']: 's%d' % (j % n) for j, r in enumerate(sl)}


class G6(unittest.TestCase):
    def g6(self, right, say_none, n_servers=200, sl=None, servers=None):
        sl = sl or slates()
        return gate.g6(sl, blind_dump(sl, right, say_none), servers or servers_of(sl, n_servers), C)

    def test_at_chance_it_holds(self):
        r = self.g6(lambda j, r: spread(j, C), lambda j, r: spread(j, 0.5))
        self.assertTrue(r['holds'], r)
        self.assertAlmostEqual(r['(a) the model part']['blind accuracy'], C, delta=0.01)

    def test_the_accuracy_on_both_sides_of_c_plus_0_05(self):
        r = self.g6(lambda j, r: spread(j, 0.185), lambda j, r: spread(j, 0.5))
        a = r['(a) the model part']
        self.assertLessEqual(a['blind accuracy'], C + 0.05)
        self.assertTrue(a['accuracy holds'] and r['holds'])
        r = self.g6(lambda j, r: spread(j, 0.20), lambda j, r: spread(j, 0.5))
        a = r['(a) the model part']
        self.assertGreater(a['blind accuracy'], C + 0.05)
        self.assertFalse(a['conditions']['blind accuracy <= c + 0.05'])
        self.assertFalse(r['holds'])

    def test_the_upper_bound_with_servers_resampled(self):
        # 0.19 on average, all of it on 38 of 200 servers: the point holds, the upper bound not
        r = self.g6(lambda j, r: j % 200 < 38, lambda j, r: spread(j, 0.5))
        a = r['(b) the pair']
        self.assertLessEqual(a['blind accuracy'], C + 0.05)
        self.assertTrue(a['conditions']['blind accuracy <= c + 0.05'])
        self.assertGreater(a['interval of the blind accuracy (servers)'][1], C + 0.08)
        self.assertFalse(a['conditions']['its upper bound (servers) <= c + 0.08'])
        self.assertFalse(r['holds'])

    def test_the_none_gap_on_both_sides_of_0_05(self):
        def say(a, b):
            return lambda j, r: spread(j, a if r['target'][-1] == 1.0 else b)
        r = self.g6(lambda j, r: spread(j, C), say(0.50, 0.46))
        self.assertLessEqual(r['(a) the model part']['blind none gap'], 0.05)
        self.assertTrue(r['holds'], r['(a) the model part']['conditions'])
        r = self.g6(lambda j, r: spread(j, C), say(0.50, 0.43))
        self.assertGreater(r['(a) the model part']['blind none gap'], 0.05)
        self.assertFalse(r['(c) none without the request']['model part'])
        self.assertFalse(r['holds'])

    def test_the_none_gap_upper_bound_at_0_15(self):
        # a gap of about 0.03, carried by the none-right slates of 11 of 20 servers
        sl = slates()
        servers = {r['uid']: 's%d' % ((j // 5) % 20) for j, r in enumerate(sl)}

        def say(j, r):
            if r['target'][-1] == 1.0:
                return (j // 5) % 20 < 11
            return spread(j, 0.52)
        r = self.g6(lambda j, r: spread(j, C), say, sl=sl, servers=servers)
        m = r['(a) the model part']
        self.assertLessEqual(m['blind none gap'], 0.05)
        self.assertGreater(m['interval of the blind none gap (servers)'][1], 0.15)
        self.assertFalse(m['gap holds'])

    def test_a_failure_only_on_the_slates_that_show_server_mates(self):
        sl = slates(mates_every=3)
        mates = lambda r: 'exempt' in r['roles']
        r = self.g6(lambda j, r: mates(r) or spread(j, C), lambda j, r: spread(j, 0.5), sl=sl)
        self.assertFalse(r['holds'])
        self.assertTrue(r['without the slates that show server-mates']['holds'])
        self.assertTrue(r['fails only on the slates that show server-mates'])
        # a failure everywhere is not that case
        r = self.g6(lambda j, r: True, lambda j, r: spread(j, 0.5), sl=sl)
        self.assertFalse(r['fails only on the slates that show server-mates'])


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)


def ordinary(sl, acc_pair, acc_model, acc_emb):
    out = []
    for j, r in enumerate(sl):
        g = r['target'].index(1.0)
        wrong = (g + 1) % (K + 1)
        p = [0.01] * (K + 1)
        p[g if spread(j, acc_pair) else wrong] = 0.9
        m = [0.01] * (K + 1)
        m[g if spread(j, acc_model) else wrong] = 0.9
        e = [0.01] * (K + 1)
        e[g if spread(j + 1, acc_emb) else wrong] = 0.9
        out.append({'uid': r['uid'], 'probs': p, 'model': m, 'embedding': e})
    return out


class Gate(unittest.TestCase):
    def build(self, d, t2_acc=0.9, mates_blind=False):
        sl = slates(mates_every=3)
        sd, root = os.path.join(d, 'fresh'), os.path.join(d, 'eval')
        for f in gate.FILES:
            write(os.path.join(sd, f + '.jsonl'), sl)
        write(os.path.join(sd, 't5_tools.servers.jsonl'), [{'uid': r['uid'], 'server': 's%d' % (j % 200)} for j, r in enumerate(sl)])
        for f in ('t5_tools', 't5_tools_masked', 't5_tools_rw', 't5_tools_masked_rw'):
            write(os.path.join(root, 'pair', f + '.jsonl'), ordinary(sl, 0.9, 0.8, 0.7))
        right = (lambda j, r: 'exempt' in r['roles'] or spread(j, C)) if mates_blind else (lambda j, r: spread(j, C))
        write(os.path.join(root, 'pair', 't5_tools.blind.jsonl'), blind_dump(sl, right, lambda j, r: spread(j, 0.5)))
        bm = [{'probs': x['probs'] if spread(j, 0.6) else x['embedding'], 'gold': r['target'].index(1.0)}
              for j, (r, x) in enumerate(zip(sl, ordinary(sl, 0.5, 0.5, 0.5)))]
        write(os.path.join(root, 'bm25', 't5_tools.jsonl'), bm)
        t2 = slates(945)
        g4 = {'max_abs_diff': 4e-4, 'argmax_agree': 200, 'n': 200, 'median_ms': 31.0, 'bytes': 3_700_000}
        servers = {r['uid']: 's%d' % (j % 200) for j, r in enumerate(sl)}
        return gate.gate(root, sd, (t2, ordinary(t2, t2_acc, 0.5, 0.5)), g4, servers)

    def test_every_condition_holds(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.build(d)
            for g in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6'):
                self.assertTrue(out[g]['holds'], (g, out[g]))
            self.assertTrue(out['verdict']['passed'])
            self.assertTrue(out['verdict']['report'][0].startswith('The gate passes'))

    def test_g5_fails_alone(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.build(d, t2_acc=0.85)
            self.assertFalse(out['G5']['holds'])
            self.assertFalse(out['verdict']['passed'])

    def test_the_first_sentence_when_g6_fails_on_the_server_mates_only(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.build(d, mates_blind=True)
            self.assertFalse(out['G6']['holds'])
            self.assertTrue(out['verdict']['report'][0].startswith('G6 fails only on the slates that show server-mates'))
            self.assertIn('G6 fails', out['verdict']['report'][1])

    def test_a_fresh_file_needs_fresh_open_and_its_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't5_tools.jsonl')
            write(p, [{'a': 1}])
            tf = os.path.join(d, 'test-files.json')
            with open(tf, 'w') as f:
                json.dump({'files': {'t5_tools.jsonl': gate.sha256(p)}}, f)
            with self.assertRaises(SystemExit):
                gate.guard(d, tf, os.path.join(d, 'FRESH-OPEN'))
            open(os.path.join(d, 'FRESH-OPEN'), 'w').close()
            gate.guard(d, tf, os.path.join(d, 'FRESH-OPEN'))
            write(p, [{'a': 2}])
            with self.assertRaises(SystemExit):
                gate.guard(d, tf, os.path.join(d, 'FRESH-OPEN'))


if __name__ == '__main__':
    unittest.main()
