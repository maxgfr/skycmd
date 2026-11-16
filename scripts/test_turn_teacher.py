#!/usr/bin/env python3
"""Tests of scripts/turn-teacher.py's preparation and record layout (no model is run):
uv run --python 3.12 --with numpy python -I scripts/test_turn_teacher.py"""

import importlib.util
import os
import struct
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("turn_teacher", os.path.join(HERE, "turn-teacher.py"))
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)


class Teacher(unittest.TestCase):
    def test_left_padding_keeps_the_end(self):
        x = np.arange(1, 1001, dtype=np.float32)
        y = tt.left_pad(x, 1500)
        self.assertEqual(len(y), 1500)
        self.assertTrue((y[:500] == 0).all() and (y[500:] == x).all())
        z = tt.left_pad(np.arange(3000, dtype=np.float32), 1500)
        self.assertEqual((z[0], z[-1]), (1500, 2999))

    def test_states_pool_the_last_four_seconds(self):
        s = np.zeros((400, tt.DIM), dtype=np.float32)
        s[200:] = np.repeat(np.arange(25, dtype=np.float32), 8)[:, None]
        p = tt.pool_states(s[None])
        self.assertEqual(p.shape, (25, tt.DIM))
        self.assertTrue(np.allclose(p[:, 0], np.arange(25)))
        s[:200] = 99.0  # the first 4 s of the 8 s input are not read
        self.assertTrue(np.allclose(tt.pool_states(s), p))

    def test_the_record_layout_of_the_store(self):
        rng = np.random.default_rng(0)
        pooled = rng.standard_normal(tt.DIM).astype(np.float32)
        states = (rng.standard_normal((25, tt.DIM)) * np.arange(1, 26)[:, None]).astype(np.float32)
        b = tt.encode_record(-2.5, pooled, states)
        self.assertEqual(len(b), 10472)  # sky_infer::turn::store::TEACHER_RECORD
        self.assertEqual(struct.unpack("<f", b[:4])[0], -2.5)
        z, p2, s2 = tt.decode_record(b)
        self.assertTrue(np.allclose(p2, pooled, atol=2e-3))
        scale = np.abs(states).max(axis=1) / 127
        self.assertTrue((np.abs(s2 - states) <= scale[:, None] * 0.51).all())
        zero = tt.encode_record(0.0, np.zeros(tt.DIM), np.zeros((25, tt.DIM)))
        self.assertEqual(tt.decode_record(zero)[2].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
