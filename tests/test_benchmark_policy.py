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
    def run_fixture(self, changed_mode, allow, draft_override=None, elapsed_by_mode=None):
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
                    recorded = dict(environment, config=dict(environment["config"]))
                    if mode != "ordinary":
                        recorded["config"].update(
                            speculative_algorithm="SUFFIX" if mode.startswith("suffix") else "NGRAM",
                            speculative_num_draft_tokens=33, speculative_ngram_match_type="PROB",
                            speculative_ngram_max_trie_depth=64, speculative_ngram_max_bfs_breadth=1)
                    recorded["draft_environment"] = dict(
                        SUFFIX_FACTOR="1.0", SUFFIX_OFFSET="0.0", SUFFIX_MIN_PROB="0.1",
                        SUFFIX_CACHE_REQUESTS="0" if mode == "suffix-local" else "128",
                        SUFFIX_FIXED="1" if mode == "suffix-fixed" else "0", SUFFIX_ALLOW_WIDTH_PROBE="1")
                    if draft_override and mode == draft_override[0]:
                        recorded["draft_environment"][draft_override[1]] = draft_override[2]
                    (dest / "environment.json").write_text(json.dumps(recorded))
                    rows = []
                    for block, count in (("independent", 52), ("refinement", 84), ("repeat", 104)):
                        for index in range(count):
                            rows.append(dict(block=block, index=index, question_id=index,
                                             kind="refinement" if block == "refinement" and index >= 52 else "initial",
                                             category="synthetic", input_tokens=2,
                                             truncated=False, elapsed_ns=(elapsed_by_mode or {}).get(mode, 1000),
                                             chunks=[{"elapsed_ns": 500}], response={"output_ids": [1, 2]}))
                    if mode == changed_mode:
                        rows[0]["response"]["output_ids"] = [1, 3]
                    (dest / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            output = directory / "synthetic-report.json"
            command = [sys.executable, str(root / "analysis/benchmark_report.py"), str(directory), "--output", str(output)]
            if allow:
                command.append("--allow-ngram-numerical-differences")
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return result.returncode, json.loads(output.read_text()) if output.exists() else None

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

    def test_ablation_ratios_use_same_trial_policy_wall_times(self):
        code, report = self.run_fixture(None, False, elapsed_by_mode={
            "ordinary": 4000, "suffix": 1000, "suffix-fixed": 500, "suffix-local": 2000})
        self.assertEqual(code, 0)
        self.assertEqual(len(report["ablation_comparisons"]), 6)
        for row in report["ablation_comparisons"]:
            expected = 0.5 if row["comparator"] == "suffix-fixed" else 2.0
            self.assertEqual(row["pooled_speedup_ratio"], expected)
            self.assertEqual(row["trial_bootstrap_95"], [expected, expected])
            self.assertEqual(row["paired_trials"], 5)

    def test_local_ablation_cannot_silently_keep_global_cache(self):
        code, report = self.run_fixture(None, True, ("suffix-local", "SUFFIX_CACHE_REQUESTS", "128"))
        self.assertNotEqual(code, 0)
        self.assertIsNone(report)

    def test_unbounded_ablation_cannot_silently_keep_adaptive_bound(self):
        code, report = self.run_fixture(None, True, ("suffix-fixed", "SUFFIX_FIXED", "0"))
        self.assertNotEqual(code, 0)
        self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
