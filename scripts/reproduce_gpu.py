"""Portable frozen-workload reproduction on one Linux H100.

Creates its own gates/reference in a fresh artifact directory. It needs the
pinned local source/native/GPU installation, never a private Modal image or
historical remote output. The original checkout and its results are preserved.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local")


def source_fingerprints(gate):
    recorded = json.loads(gate.read_text())
    assert recorded["success"], "Published source gate did not pass"
    expected = recorded["source_sha256"]
    for name, digest in expected.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Frozen GPU source differs: {name}")
    return expected


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def differences(reference, measured):
    assert len(reference) == len(measured) and reference, "Incomplete output comparison"
    mismatches = []
    for a, b in zip(reference, measured):
        keys = ("block", "index", "question_id") if "block" in a else ("name", "repeat")
        assert all(a[k] == b[k] for k in keys), "Request identities differ"
        x, y = a["response"]["output_ids"], b["response"]["output_ids"]
        if x != y:
            first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            mismatches.append(dict(identity={k: b[k] for k in keys}, first_difference=first,
                                   ordinary_ids=x[first:first + 8], mode_ids=y[first:first + 8],
                                   ordinary_output_tokens=len(x), mode_output_tokens=len(y)))
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; existing paths are refused")
    parser.add_argument("--dry-run", action="store_true", help="Validate source hashes and print stages without GPU work")
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / "configs/gpu-source-manifest.json")
    parser.add_argument("--smoke", action="store_true", help="Check isolated Linux execution on one public input, without timing trials")
    args = parser.parse_args()
    expected = source_fingerprints(args.source_manifest)
    if args.dry_run:
        print(json.dumps(dict(source_sha256=expected, output_dir=str(args.output_dir.resolve()),
                              stages=["plain gates", "416-round KV gate", "fresh ordinary/SUFFIX public gate",
                                      "five rotated trials / 6000 requests", "strict/descriptive benchmark report"],
                              gpu_allocated=False), indent=2))
        return
    if sys.platform != "linux":
        raise RuntimeError("GPU execution requires Linux; --dry-run is available on the Mac")
    destination = args.output_dir.resolve()
    if any(destination.is_relative_to(ROOT / name) for name in ("scripts", "configs", "sglang", "native", "patches")):
        raise ValueError("Output directory cannot be inside a source directory")
    destination.mkdir(parents=True, exist_ok=False)
    work = destination / "project"
    work.mkdir()
    for name in ("scripts", "configs"):
        shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "model-api.json"))
    for name in ("sglang", "native", "patches"):
        (work / name).symlink_to(ROOT / name, target_is_directory=True)
    results = destination / "results"
    results.mkdir()
    (work / "results").symlink_to(results, target_is_directory=True)
    env = dict(os.environ)
    for key in ("SUFFIX_FACTOR", "SUFFIX_OFFSET", "SUFFIX_MIN_PROB", "SUFFIX_CACHE_REQUESTS",
                "SUFFIX_FIXED", "SUFFIX_TRACE", "SUFFIX_TRACE_LOGITS", "SUFFIX_ALLOW_WIDTH_PROBE"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="0", SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE="4096",
               PYTHONPATH=os.pathsep.join(str(work / name) for name in ("native", "sglang/python", "scripts")))
    status = dict(success=False, run_id=destination.name, source_sha256=expected,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  completed_modes=[], completed_trials=[],
                  policy="ordinary/SUFFIX/ablations exact; upstream NGRAM differences descriptive",
                  smoke_only=args.smoke)
    started = time.monotonic()

    def save(name, value):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")

    try:
        with (destination / "campaign.log").open("x", buffering=1) as log:
            def run(arguments):
                command = [sys.executable, *arguments]
                log.write("COMMAND: " + json.dumps(command) + "\n")
                with subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith("PROGRESS:"):
                            print(line.rstrip(), flush=True)
                    if process.wait():
                        raise RuntimeError(f"Stage failed: {arguments}; inspect campaign.log")

            run(["-c", "import torch; assert torch.cuda.device_count()==1; assert torch.cuda.get_device_name(0)=='NVIDIA H100 80GB HBM3'"])
            if args.smoke:
                program = """import json, sys
from pathlib import Path
import sglang as sgl
from gpu_common import engine_config, sampling
row = json.loads(Path('configs/frozen-workload.jsonl').read_text().splitlines()[0])
engine = sgl.Engine(**engine_config(sys.argv[1]))
try:
    responses = [engine.generate(input_ids=row['input_ids'], sampling_params=sampling(33)) for _ in range(2)]
    Path('results/portable-smoke-' + sys.argv[1] + '.json').write_text(json.dumps(responses) + '\\n')
finally:
    engine.shutdown()
"""
                for mode in ("ordinary", "suffix"):
                    run(["-c", program, mode])
                ordinary = json.loads((results / "portable-smoke-ordinary.json").read_text())
                suffix = json.loads((results / "portable-smoke-suffix.json").read_text())
                assert len(ordinary) == len(suffix) == 2
                assert all(a["output_ids"] == b["output_ids"] and a["output_ids"] for a, b in zip(ordinary, suffix))
                status["smoke_requests"] = 4
                status["success"] = True
                print("PROGRESS: portable runner cold/warm smoke passed, four requests with exact IDs", flush=True)
                return
            for mode in MODES[:3]:
                run(["scripts/gpu_correctness.py", "--mode", mode])
            plain = read_rows(results / "correctness-ordinary.jsonl")
            assert len(plain) == 30
            for mode in ("ngram", "suffix"):
                diff = differences(plain, read_rows(results / f"correctness-{mode}.jsonl"))
                save(f"plain-{mode}-equality.json", diff)
                if mode == "suffix" and diff:
                    raise ValueError("Plain SUFFIX gate differs")
            (results / "correctness-suffix.jsonl").rename(results / "correctness-plain-suffix.jsonl")
            run(["scripts/gpu_correctness.py", "--mode", "suffix", "--audit"])
            audit = read_rows(results / "audit-suffix-trace.jsonl")
            assert audit and all(row.get("existing_kv_prefix_unchanged") and row.get("accepted_kv_slots_checked")
                                 and row.get("gpu_layout_and_acceptance_checked") for row in audit)
            assert not differences(plain, read_rows(results / "correctness-suffix.jsonl"))
            status["audited_rounds"] = len(audit)
            for mode in ("ordinary", "suffix"):
                run(["scripts/gpu_public_gate.py", "--mode", mode, "--shared", "1"])
            reference = read_rows(results / "public-gate-ordinary-shared1.jsonl")
            assert len(reference) == 240
            diff = differences(reference, read_rows(results / "public-gate-suffix-shared1.jsonl"))
            save("public-suffix-equality.json", diff)
            assert not diff, "Fresh full-workload SUFFIX gate differs"
            status["gates_passed"] = True
            frozen = (work / "configs/frozen-workload.jsonl").read_bytes()
            assert hashlib.sha256(frozen).hexdigest() == expected["configs/frozen-workload.jsonl"]
            (results / "workload.jsonl").write_bytes(frozen)
            (results / "workload.sha256").write_text(expected["configs/frozen-workload.jsonl"] + "\n")
            for trial in range(5):
                for mode in MODES[trial:] + MODES[:trial]:
                    run(["scripts/gpu_bench.py", "--mode", mode, "--trial", str(trial)])
                    measured = read_rows(results / "gpu" / f"{mode}-{trial}/requests.jsonl")
                    assert len(measured) == 240
                    diff = differences(reference, measured)
                    save(f"benchmark-{mode}-{trial}-equality.json", dict(requests=240, mismatches=diff))
                    if mode != "ngram" and diff:
                        raise ValueError(f"Timed {mode}/{trial} differs")
                    status["completed_modes"].append(dict(mode=mode, trial=trial, mismatches=len(diff)))
                    save("campaign-progress.json", status)
                    print(f"PROGRESS: {len(status['completed_modes'])}/25 completed mode runs; {mode}/{trial} differences={len(diff)}", flush=True)
                baseline = read_rows(results / "gpu" / f"ordinary-{trial}/requests.jsonl")
                for mode in MODES[2:]:
                    assert not differences(baseline, read_rows(results / "gpu" / f"{mode}-{trial}/requests.jsonl"))
                status["completed_trials"].append(trial)
                save("campaign-progress.json", status)
            run([str(ROOT / "analysis/benchmark_report.py"), str(results), "--output",
                 str(destination / "benchmark-report.json"), "--allow-ngram-numerical-differences"])
            status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        save("campaign-status.json", status)


if __name__ == "__main__":
    main()
