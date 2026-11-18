"""Tests of scripts/build-route6-data.sh's plumbing (s6-route v6 and v7).

`reported`: a check of 4.4 is reported and never stops the stage. Its exit is accepted when the
report it wrote names the same exit; a crash (another exit, or no report) stops the stage.

    python3 -I scripts/test_build_route6_data.py
"""

import json
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'build-route6-data.sh')


def function(name):
    with open(SCRIPT) as f:
        text = f.read()
    a = text.index(f'# BEGIN {name}\n')
    b = text.index(f'# END {name}\n')
    return text[a:b]


def run(fake, tmp):
    """Runs `reported out.json bash fake.sh` with the script's own function, then `echo after`."""
    with open(os.path.join(tmp, 'fake.sh'), 'w') as f:
        f.write(fake)
    prog = 'set -euo pipefail\ndry=0\nnote() { echo "# $*"; }\nx() { "$@"; }\nnice() { shift 2; "$@"; }\n'
    prog += function('reported') + 'reported out.json bash fake.sh\necho after\n'
    return subprocess.run(['bash', '-c', prog], cwd=tmp, capture_output=True, text=True)


def check(exit_code, write, said):
    """A fake check: writes its report (or not), then exits."""
    w = "echo '%s' > out.json\n" % json.dumps({'verdict': {'exit': said}}) if write else ''
    return f'{w}exit {exit_code}\n'


class Reported(unittest.TestCase):
    def test_checks_that_fail_are_reported_and_the_stage_goes_on(self):
        with tempfile.TemporaryDirectory() as t:
            r = run(check(1, True, 1), t)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('after', r.stdout)
            self.assertIn('do not stop', r.stdout)

    def test_checks_that_pass(self):
        with tempfile.TemporaryDirectory() as t:
            r = run(check(0, True, 0), t)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('after', r.stdout)

    def test_a_crash_without_a_report_stops(self):
        with tempfile.TemporaryDirectory() as t:
            # a stale report of an earlier run is removed first, so it cannot hide the crash
            open(os.path.join(t, 'out.json'), 'w').write(json.dumps({'verdict': {'exit': 1}}))
            r = run(check(1, False, None), t)
            self.assertEqual(r.returncode, 1)
            self.assertNotIn('after', r.stdout)
            self.assertIn('crashed', r.stderr)

    def test_another_exit_than_the_report_s_stops(self):
        with tempfile.TemporaryDirectory() as t:
            r = run(check(3, True, 1), t)
            self.assertEqual(r.returncode, 3)
            self.assertNotIn('after', r.stdout)


if __name__ == '__main__':
    unittest.main()
