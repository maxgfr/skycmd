#!/usr/bin/env python3
"""Tests of scripts/all-report.py on files in the real formats: python3 -I scripts/test_all_report.py"""

import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("all_report", os.path.join(HERE, "all-report.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

HEAD = "| source | n | accuracy | NLL | Brier | ECE |\n|---|---:|---:|---:|---:|---:|\n"


def table(exit_no, secret, service):
    return (f"#### Exit {exit_no} (after {exit_no + 2} state blocks), per source\n\n{HEAD}"
            f"| all | 7984 | 0.5000 | 1.0 | 0.1 | 0.1 |\n"
            f"| secret_bench:secret | 1996 | {secret:.4f} | 1.0 | 0.1 | 0.1 |\n"
            f"| secret_bench:service | 1996 | {service:.4f} | 1.0 | 0.1 | 0.1 |\n\n")


def report(gguf_last=(0.98, 0.80)):
    """An s2-eval report with the order of crates/sky-train/src/s2/eval.rs: only the GGUF's last exit
    holds gguf_last, every other table holds accuracies that would pass or fail the other way."""
    md = "# Evaluation report\n\n## Summary\n\n| model | last-exit acc |\n|---|---:|\n| x | 0.1 |\n\n"
    md += "## latent f32 (final QAT master weights)\n\n" + table(0, 0.11, 0.12) + table(1, 0.13, 0.14)
    md += "#### Early exit per source\n\n| source | n | accuracy |\n|---|---:|---:|\n| secret_bench:secret | 1996 | 0.15 |\n\n"
    md += "## ternary (fake-quant)\n\n" + table(0, 0.16, 0.17) + table(1, 0.18, 0.19)
    md += "#### Early exit per source\n\n| source | n | accuracy |\n|---|---:|---:|\n| secret_bench:secret | 1996 | 0.20 |\n\n"
    md += "## GGUF `skycmd-s6-secrets-pico-ternary.gguf`\n\n" + table(0, 0.21, 0.22) + table(1, *gguf_last)
    md += "#### Early exit per source\n\n| source | n | accuracy |\n|---|---:|---:|\n| secret_bench:secret | 1996 | 0.23 |\n\n"
    return md


class Secrets(unittest.TestCase):
    def test_reads_the_ggufs_last_exit(self):
        self.assertEqual(rep.secrets_acc(report(), "secret"), 0.98)
        self.assertEqual(rep.secrets_acc(report(), "service"), 0.80)

    def test_a_report_without_gguf_is_unreadable(self):
        self.assertIsNone(rep.secrets_acc(report().split("\n## GGUF `")[0], "secret"))

    def run_gate(self, md):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/runs/s6-secrets-pico")
            with open(f"{d}/runs/s6-secrets-pico/report.md", "w") as f:
                f.write(md)
            return rep.secrets(f"{d}/")[0]

    def test_gate(self):
        self.assertEqual(self.run_gate(report()), "PASS")
        self.assertEqual(self.run_gate(report((0.97, 0.80))), "FAIL")
        self.assertEqual(self.run_gate("# empty\n"), "unreadable")


class Cast(unittest.TestCase):
    def run_gate(self, summary):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/data/gift-eval-out/s6-cast-pico")
            with open(f"{d}/data/gift-eval-out/s6-cast-pico/summary.json", "w") as f:
                json.dump(summary, f)
            return rep.cast(f"{d}/")[0]

    def summary(self, g=0.8, w=0.5, n=97):
        return {"nGMASE": g, "nWQL": w, "nMSIS": 0.5, "configs": {"nGMASE": n, "nWQL": n, "nMSIS": n}, "unmatched": []}

    def test_gate(self):
        self.assertEqual(self.run_gate(self.summary()), "PASS")
        self.assertEqual(self.run_gate(self.summary(g=0.9)), "FAIL")

    def test_partial_or_empty_summaries_never_pass(self):
        self.assertEqual(self.run_gate(self.summary(n=40)), "unreadable")
        self.assertEqual(self.run_gate(self.summary(g=None, n=0)), "unreadable")


class Missing(unittest.TestCase):
    def test_no_file_is_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            for got in (rep.ear(f"{d}/"), rep.secrets(f"{d}/"), rep.size(f"{d}/", "nano"), rep.cast(f"{d}/")):
                self.assertEqual(got[0], "not run")


if __name__ == "__main__":
    unittest.main()
