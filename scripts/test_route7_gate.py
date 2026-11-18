"""Tests of scripts/route7-gate.sh's plumbing (s6-route v7), on invented files under a temporary
folder (RUNS, MODELS, PILOT, ROUTE_DATA and T5 point there; nothing of v7's real run is read):

- every stage refuses without the marker of the one before it;
- `open` refuses an aborted pilot, a pilot read at another bar and a changed T5 file, writes
  FRESH-OPEN once, and refuses a second time;
- once T5 is open, `fit` and `embed` refuse; once gate.json exists, `open`, `read`, `g4` and
  `gate` refuse;
- `embed` (with SKY_BIN holding a built sky-route) writes the embedding file of the fit's values
  and leaves the published file byte-identical.

    python3 -I scripts/test_route7_gate.py
"""
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, 'route7-gate.sh')
PUBLISHED = os.path.join(REPO, 'site', 'public', 'models', 's6-route-embed.gguf')
BIN = os.environ.get('SKY_BIN') or os.path.join(REPO, 'target', 's3-train', 'release')


def load(p):
    with open(p) as f:
        return json.load(f)


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class Fixture:
    def __init__(self, tmp):
        self.tmp = tmp
        self.data = os.path.join(tmp, 'data')
        self.t5 = os.path.join(self.data, 'toucan5')
        fresh = os.path.join(self.t5, 'fresh')
        os.makedirs(fresh)
        files = {}
        for name in ('t5_tools', 't5_tools_masked', 't5_tools_rw', 't5_tools_masked_rw', 't5_tools.servers'):
            p = os.path.join(fresh, name + '.jsonl')
            with open(p, 'w') as f:
                f.write(json.dumps({'uid': 'u1', 'split': 'fresh'}) + '\n')
            files[name + '.jsonl'] = sha(p)
        self.fresh = fresh
        self.runs = os.path.join(tmp, 'runs', 's6-route7')
        self.models = os.path.join(tmp, 'models', 's6-route7')
        os.makedirs(self.runs)
        os.makedirs(self.models)
        self.write('test-files.json', {'about': 'invented', 'files': files})
        self.write('placeholder.json', {'placeholder': '?'})
        self.write('fit.json', {'beta': 1.11, 'tau_ask6': {'tau': 0.66, 'accuracy': 0.95, 'coverage': 0.5}})
        for f in ('skycmd-s6-route7-nano-ternary.gguf', 'skycmd-s6-route7-embed.gguf'):
            with open(os.path.join(self.models, f), 'wb') as g:
                g.write(b'invented')
        self.pilot = os.path.join(tmp, 'summary.json')
        self.summary({'abort': False, 'why': [], 'bar (rule 1, from the amendment of 4.5)': 0.578147})

    def write(self, name, obj):
        with open(os.path.join(self.runs, name), 'w') as f:
            json.dump(obj, f)

    def summary(self, s):
        with open(self.pilot, 'w') as f:
            json.dump(s, f)

    def marker(self, stage):
        d = os.path.join(self.runs, 'gate-stages')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, stage + '.done'), 'w') as f:
            f.write('invented\n')

    def run(self, *stages, bin_dir=None):
        env = dict(os.environ, ROUTE_DATA=self.data, T5=self.t5, RUNS=self.runs, MODELS=self.models,
                   PILOT=self.pilot, SKY_BIN=bin_dir or os.path.join(self.tmp, 'no-bin'), MIN_FREE_GB='0')
        return subprocess.run(['bash', SCRIPT, *stages], env=env, capture_output=True, text=True, cwd=REPO)


class Stages(unittest.TestCase):
    def test_each_stage_needs_the_one_before(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(t)
            for stage, before in (('embed', 'fit'), ('open', 'embed'), ('read', 'open'), ('g4', 'read'), ('gate', 'g4')):
                r = fx.run(stage)
                self.assertNotEqual(r.returncode, 0, stage)
                self.assertIn('gate-stages/%s.done' % before, r.stderr)

    def test_open_writes_fresh_open_once(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(t)
            fx.marker('fit')
            fx.marker('embed')
            r = fx.run('open')
            self.assertEqual(r.returncode, 0, r.stderr)
            o = load(os.path.join(fx.runs, 'FRESH-OPEN'))
            self.assertEqual(len(o['T5']), 5)
            self.assertIn('sha256', o['model'])
            r = fx.run('open')
            self.assertEqual(r.returncode, 3)
            self.assertIn('read once', r.stderr)
            # once open, the system is fixed
            for stage in ('fit', 'embed'):
                r = fx.run(stage)
                self.assertEqual(r.returncode, 3, stage)
                self.assertIn('FRESH-OPEN exists', r.stderr)

    def test_open_refuses_an_aborted_pilot_another_bar_or_a_changed_file(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(t)
            fx.marker('fit')
            fx.marker('embed')
            fx.summary({'abort': True, 'why': ['1: below the bar'], 'bar (rule 1, from the amendment of 4.5)': 0.578147})
            r = fx.run('open')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('aborted', r.stderr)
            fx.summary({'abort': False, 'why': [], 'bar (rule 1, from the amendment of 4.5)': 0.0})
            r = fx.run('open')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('0.578147', r.stderr)
            fx.summary({'abort': False, 'why': [], 'bar (rule 1, from the amendment of 4.5)': 0.578147})
            with open(os.path.join(fx.fresh, 't5_tools_rw.jsonl'), 'a') as f:
                f.write('\n')
            r = fx.run('open')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('not the file of', r.stderr)
            self.assertFalse(os.path.exists(os.path.join(fx.runs, 'FRESH-OPEN')))

    def test_nothing_reads_t5_again_once_gated(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(t)
            for s in ('fit', 'embed', 'open', 'read', 'g4'):
                fx.marker(s)
            fx.write('gate.json', {'verdict': {'passed': False}})
            for stage in ('open', 'read', 'g4', 'gate'):
                r = fx.run(stage)
                self.assertEqual(r.returncode, 3, stage)
                self.assertIn('gate.json exists', r.stderr)

    def test_the_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(t)
            before = sorted(os.walk(t))
            env = dict(os.environ, DRY_RUN='1', ROUTE_DATA=fx.data, T5=fx.t5, RUNS=fx.runs, MODELS=fx.models,
                       PILOT=fx.pilot)
            r = subprocess.run(['bash', SCRIPT, 'fit', 'embed', 'open', 'read', 'g4', 'gate'], env=env,
                               capture_output=True, text=True, cwd=REPO)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(os.walk(t)), before)
            self.assertIn('--split t5', r.stdout)
            self.assertIn('--render scores-in-options', r.stdout)

    @unittest.skipUnless(os.path.exists(os.path.join(BIN, 'sky-route')) and os.path.exists(PUBLISHED),
                         'needs a built sky-route (SKY_BIN or target/s3-train/release) and the published embedding file')
    def test_embed_writes_the_fit_s_values_and_leaves_the_published_file(self):
        with tempfile.TemporaryDirectory() as t:
            fx = Fixture(t)
            os.remove(os.path.join(fx.models, 'skycmd-s6-route7-embed.gguf'))
            fx.marker('fit')
            before = sha(PUBLISHED)
            r = fx.run('embed', bin_dir=BIN)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sha(PUBLISHED), before)
            e = load(os.path.join(fx.runs, 'embed.json'))
            self.assertEqual(e['from_sha256'], before)
            out = os.path.join(fx.models, 'skycmd-s6-route7-embed.gguf')
            self.assertEqual(e['sha256'], sha(out))
            inspect = os.path.join(BIN, 'gguf-inspect')
            if os.path.exists(inspect):
                txt = subprocess.run([inspect, out], capture_output=True, text=True).stdout
                self.assertIn('skycmd-embed.route.beta (f64) = 1.11', txt)
                self.assertIn('skycmd-embed.route.tau_ask3 (f64) = 0.66', txt)
                self.assertIn('general.name (string) = "s6-route7-embed"', txt)


if __name__ == '__main__':
    unittest.main()
