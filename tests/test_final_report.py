"""Synthetic publication fixtures, never model or performance evidence."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ("suffix", "suffix-fixed", "suffix-local")
BLOCKS = {"independent": 52, "refinement": 84, "repeat": 104}


def fixtures():
    aggregate = []
    for mode in (*MODES, "ngram"):
        for block in BLOCKS:
            ratio = {"independent": 0.9, "refinement": 1.0, "repeat": 3.0}[block]
            aggregate.append(dict(mode=mode, block=block, pooled_speedup=ratio,
                                  trial_bootstrap_95=[ratio - 0.02, ratio + 0.02]))
    benchmark = dict(
        exact_suffix_ids_passed=True, measured_requests=6000,
        mismatches=[dict(mode="ngram", block="independent")],
        aggregate=aggregate,
        ablation_comparisons=[dict(comparator=mode, block=block, paired_trials=5,
                                  pooled_speedup_ratio=0.9 if mode == "suffix-fixed" else 1.1,
                                  trial_bootstrap_95=[0.88, 0.92] if mode == "suffix-fixed" else [1.08, 1.12])
                              for mode in ("suffix-fixed", "suffix-local") for block in BLOCKS],
        request_kinds=[dict(mode=mode, block="refinement", kind="refinement", requests=160,
                            pooled_speedup=1.0, trial_bootstrap_95=[0.98, 1.02]) for mode in MODES],
        trials=[dict(mode="ngram", block=block, ordinary_output_tokens=100,
                     output_tokens=99) for block in BLOCKS],
    )
    trace = dict(summaries=[dict(
        variant=mode, block=block, requests=count, mean_verify_rows=2.0,
        target_draft_acceptance=0.5, mean_committed_tokens_per_round=1.5,
        one_row_fraction=0.25,
    ) for mode in MODES for block, count in BLOCKS.items()])
    width = dict(
        rows=[dict(context_tokens=context, verify_rows=rows, median_kernel_ms=1 + rows / 100,
                   families={name: dict(grids=[[rows, 1, 1]]) for name in
                             ("kv_store", "argmax", "attention", "lm_head")})
              for context in (126, 128, 512) for rows in (1, 2, 4, 8, 16, 33)],
        traces=["SYNTHETIC_NO_GPU_TRACE"] * 54,
    )
    route = dict(exact_ids_passed=True, measured_requests=24, paired_trials=6,
                 pooled_shared_over_original_cost_ratio=1.0, trial_bootstrap_95=[0.98, 1.02])
    sources = {name: "SYNTHETIC_NO_RUNTIME" for name in
               ("patches/sglang-suffix.patch", "configs/frozen-workload.jsonl", "configs/gpu-requirements.lock")}
    campaign = dict(
        success=True, completed_trials=list(range(5)), audited_rounds=416,
        source_sha256=sources, run_id="SYNTHETIC_CAMPAIGN", image_id="SYNTHETIC_IMAGE",
        completed_modes=[dict(mode=mode, trial=trial, mismatches=[])
                         for trial in range(5) for mode in ("ordinary", "ngram", *MODES)],
        resume_lineage=dict(run_id="SYNTHETIC_INTERRUPTED_ORIGIN"),
    )
    diagnostics = dict(success=True, source_sha256=sources, run_id="SYNTHETIC_DIAGNOSTICS",
                       completed_phases=["SYNTHETIC_PHASE"] * 16,
                       portable_runner_smoke=dict(success=True))
    return {
        "benchmark-report.json": benchmark,
        "natural-trace-report.json": trace,
        "width-probe-summary.json": width,
        "decode-control-report.json": route,
        "execution-provenance.json": dict(campaign=campaign, diagnostics=diagnostics,
                                           controller=dict(success=True)),
    }


class FinalReportTests(unittest.TestCase):
    def run_fixture(self, mutate=None):
        evidence = fixtures()
        if mutate:
            mutate(evidence)
        with tempfile.TemporaryDirectory(prefix="suffix-synthetic-report-") as temporary:
            directory = Path(temporary)
            for name, value in evidence.items():
                (directory / name).write_text(json.dumps(value))
            output = directory / "SYNTHETIC_NOT_MEASURED.md"
            result = subprocess.run(
                [sys.executable, str(ROOT / "analysis/write_final_report.py"),
                 "--results-directory", str(directory), "--output", str(output)],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode, output.read_text() if output.exists() else None, result.stderr

    def test_negative_results_and_repetition_only_gain_are_explicit(self):
        code, report, error = self.run_fixture()
        self.assertEqual(code, 0, error)
        self.assertIn("| Independent | Slower |", report)
        self.assertIn("| Follow-up only | Inconclusive |", report)
        self.assertIn("| Repeated | Faster |", report)
        self.assertIn("descriptive latency ratios alongside the output differences", report)
        self.assertIn("0.900× [0.880–0.920]", report)
        self.assertIn("1.100× [1.080–1.120]", report)

    def test_incomplete_diagnostics_cannot_produce_results_document(self):
        code, report, _ = self.run_fixture(lambda r: r["execution-provenance.json"]
                                         ["diagnostics"]["completed_phases"].pop())
        self.assertNotEqual(code, 0)
        self.assertIsNone(report)

    def test_suffix_mismatch_cannot_produce_results_document(self):
        def fail_suffix(reports):
            reports["benchmark-report.json"]["exact_suffix_ids_passed"] = False

        code, report, _ = self.run_fixture(fail_suffix)
        self.assertNotEqual(code, 0)
        self.assertIsNone(report)

    def test_failed_portable_smoke_cannot_produce_results_document(self):
        def fail_smoke(reports):
            reports["execution-provenance.json"]["diagnostics"]["portable_runner_smoke"]["success"] = False

        code, report, _ = self.run_fixture(fail_smoke)
        self.assertNotEqual(code, 0)
        self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
