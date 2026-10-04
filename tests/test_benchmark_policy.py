"""Synthetic parser fixtures: descriptive NGRAM must never relax suffix gates."""

import json
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "analysis/benchmark_report.py"
spec = importlib.util.spec_from_file_location("benchmark_report_policy", module_path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class BenchmarkPolicyTests(unittest.TestCase):
    def run_fixture(self, changed_mode, allow):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            environment = dict(config={"model_path": "synthetic-parser-fixture"},
                               workload_sha256="synthetic", integration_patch_sha256="synthetic",
                               source_lock={"model": "synthetic"}, torch="synthetic", cuda="synthetic",
                               pip_freeze="synthetic", kernel_environment={},
                               gpu="NVIDIA H100 80GB HBM3", nvidia_smi="GPU UUID : SYNTHETIC_TEST_ONLY")
            for trial in range(5):
                for mode in ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local"):
                    dest = directory / "gpu" / f"{mode}-{trial}"
                    dest.mkdir(parents=True)
                    (dest / "environment.json").write_text(json.dumps(environment))
                    rows = []
                    for block, count in (("independent", 52), ("refinement", 84), ("repeat", 104)):
                        for index in range(count):
                            rows.append(dict(block=block, index=index, question_id=index,
                                             kind="refinement" if block == "refinement" and index >= 52 else "initial",
                                             category="synthetic", input_tokens=2,
                                             truncated=False, elapsed_ns=1000,
                                             chunks=[{"elapsed_ns": 500}], response={"output_ids": [1, 2]}))
                    if mode == changed_mode:
                        rows[0]["response"]["output_ids"] = [1, 3]
                    (dest / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            output = directory / "synthetic-report.json"
            command = [sys.executable, str(root / "analysis/benchmark_report.py"), str(directory), "--output", str(output)]
            if allow:
                command.append("--allow-ngram-numerical-differences")
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return result.returncode, json.loads(output.read_text())

    def test_ngram_default_still_fails(self):
        code, report = self.run_fixture("ngram", False)
        self.assertNotEqual(code, 0)
        self.assertFalse(report["exact_ids_passed"])
        with self.assertRaises(ValueError):
            policy.exact_speedup_modes(report)

    def test_descriptive_ngram_has_no_speedup_field(self):
        code, report = self.run_fixture("ngram", True)
        self.assertEqual(code, 0)
        self.assertFalse(report["exact_ids_passed"])
        self.assertTrue(report["exact_suffix_ids_passed"])
        rows = [r for r in report["aggregate"] if r["mode"] == "ngram"]
        self.assertTrue(all("pooled_speedup" not in r and "descriptive_latency_ratio" in r for r in rows))
        self.assertEqual(len(report["mismatches"]), 5)
        turns = [r for r in report["request_kinds"] if r["kind"] == "refinement"]
        self.assertEqual(len(turns), 5)
        self.assertTrue(all(r["requests"] == 160 for r in turns))
        ngram = next(r for r in turns if r["mode"] == "ngram")
        self.assertNotIn("pooled_speedup", ngram)
        self.assertEqual(ngram["descriptive_latency_ratio"], 1.0)
        self.assertNotIn("ngram", policy.exact_speedup_modes(report))

    def test_suffix_difference_still_fails_with_flag(self):
        code, report = self.run_fixture("suffix", True)
        self.assertNotEqual(code, 0)
        self.assertFalse(report["exact_suffix_ids_passed"])
        with self.assertRaises(ValueError):
            policy.exact_speedup_modes(report)

    def test_clean_report_can_include_ngram(self):
        report = dict(exact_ids_passed=True, exact_suffix_ids_passed=True, mismatches=[])
        self.assertIn("ngram", policy.exact_speedup_modes(report))


if __name__ == "__main__":
    unittest.main()
