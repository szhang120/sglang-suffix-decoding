"""Separate single-GPU diagnostics after all strict serving comparisons pass.

Keeps the frozen model/runtime sources; only adds the ordinary-route control.
Each phase commits artifacts. No timings here enter serving estimates.
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = os.environ.get("SUFFIX_MODAL_V12_IMAGE", "im-96vGJgQDElG1UZq72yqyhI")
app = modal.App("sglang-suffix-final-diagnostics")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id(BASE_IMAGE).add_local_file(
    ROOT / "analysis/decode_control_gpu.py", "/project/analysis/decode_control_gpu.py", copy=True
)
report_files = ("benchmark_report.py", "natural_trace_report.py", "decode_control_report.py", "width_report.py")
cpu_image = modal.Image.debian_slim(python_version="3.12")
for filename in report_files:
    cpu_image = cpu_image.add_local_file(ROOT / "analysis" / filename, f"/analysis/{filename}", copy=True)
cpu_image = cpu_image.add_local_file(
    ROOT / "scripts/analyze_width_probe.py", "/analysis/analyze_width_probe.py", copy=True
)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=10800, startup_timeout=1800,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, reference_run, expected, control_sha, runner_sha, image_id):
    import hashlib
    import subprocess
    import time

    for identifier in (run_id, reference_run):
        if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
            raise ValueError("Invalid run ID")
    directory = Path("/artifacts") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    results = directory / "results"
    results.mkdir()
    started = time.monotonic()
    status = dict(success=False, run_id=run_id, reference_run=reference_run,
                  base_image=BASE_IMAGE, image_id=image_id, source_sha256=expected,
                  runner_sha256=runner_sha, control_sha256=control_sha,
                  instrumented=True, completed_phases=[])

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def same(reference, measured):
        assert len(reference) == len(measured)
        mismatches = []
        for a, b in zip(reference, measured):
            keys = ("block", "index", "question_id")
            assert all(a[k] == b[k] for k in keys)
            if a["response"]["output_ids"] != b["response"]["output_ids"]:
                mismatches.append({k: b[k] for k in keys})
        return mismatches

    def reset_env():
        for key in ("SUFFIX_FIXED", "SUFFIX_CACHE_REQUESTS", "SUFFIX_TRACE",
                    "SUFFIX_TRACE_LOGITS", "SUFFIX_ALLOW_WIDTH_PROBE"):
            os.environ.pop(key, None)

    def point_results(path):
        path.mkdir(parents=True, exist_ok=False)
        link = Path("/project/results")
        if link.is_symlink():
            link.unlink()
        assert not link.exists()
        link.symlink_to(path, target_is_directory=True)

    try:
        for name, digest in expected.items():
            assert hashlib.sha256((Path("/project") / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256(Path("/project/analysis/decode_control_gpu.py").read_bytes()).hexdigest() == control_sha
        reference_dir = Path("/artifacts") / reference_run
        campaign = json.loads((reference_dir / "campaign-status.json").read_text())
        assert campaign["success"] and campaign["source_sha256"] == expected
        assert campaign["completed_trials"] == list(range(5)) and len(campaign["completed_modes"]) == 25
        assert all(not row["mismatches"] for row in campaign["completed_modes"] if row["mode"] != "ngram")
        reference = rows(reference_dir / "results/gpu/ordinary-0/requests.jsonl")
        assert len(reference) == 240
        frozen = Path("/project/configs/frozen-workload.jsonl").read_bytes()
        assert hashlib.sha256(frozen).hexdigest() == expected["configs/frozen-workload.jsonl"]
        with (directory / "diagnostics.log").open("x", buffering=1) as log:

            def run(command):
                log.write("COMMAND: " + json.dumps(command) + "\n")
                with subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith("PROGRESS:"):
                            print(line.rstrip(), flush=True)
                    if process.wait():
                        raise RuntimeError(f"Diagnostic failed: {command}; see diagnostics.log")

            def checkpoint(phase):
                status["completed_phases"].append(phase)
                log.flush()
                (directory / "diagnostics-progress.json").write_text(json.dumps(status, indent=2) + "\n")
                artifacts.commit()
                model_cache.commit()

            for variant in ("suffix", "suffix-fixed", "suffix-local"):
                reset_env()
                destination = results / "natural-trace" / variant / "results"
                point_results(destination)
                os.environ["SUFFIX_TRACE"] = str(destination / "suffix-rounds.jsonl")
                os.environ["SUFFIX_FIXED"] = "1" if variant == "suffix-fixed" else "0"
                os.environ["SUFFIX_CACHE_REQUESTS"] = "0" if variant == "suffix-local" else "128"
                run(["python", "scripts/gpu_public_gate.py", "--mode", "suffix", "--shared", "1"])
                measured = rows(destination / "public-gate-suffix-shared1.jsonl")
                differences = same(reference, measured)
                (destination / "trace-equality.json").write_text(json.dumps(dict(
                    reference_run=reference_run, variant=variant, requests=len(measured),
                    mismatches=differences), indent=2) + "\n")
                assert not differences, f"Instrumented {variant} IDs differ"
                checkpoint(f"natural-trace-{variant}")

            reset_env()
            destination = results / "profile" / "results"
            point_results(destination)
            (destination / "workload.jsonl").write_bytes(frozen)
            for mode in ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local"):
                reset_env()
                run(["python", "scripts/gpu_bench.py", "--mode", mode, "--trial", "0", "--profile"])
                measured = rows(destination / "gpu" / f"{mode}-0-profile/requests.jsonl")
                assert len(measured) == 12
                selected = []
                for block in ("independent", "refinement", "repeat"):
                    block_rows = [r for r in reference if r["block"] == block]
                    selected.extend(block_rows[:2] * 2 if block == "repeat" else block_rows[:4])
                # The repeated profile uses input identities 0,1,0,1 at indices
                # 0,1,2,3; align those inputs, rather than the full workload's 2,3.
                differences = []
                for a, b in zip(selected, measured):
                    assert a["block"] == b["block"] and a["question_id"] == b["question_id"]
                    if a["response"]["output_ids"] != b["response"]["output_ids"]:
                        differences.append({k: b[k] for k in ("block", "index", "question_id")})
                (destination / f"profile-{mode}-equality.json").write_text(json.dumps(
                    dict(requests=12, mismatches=differences, exact_output_required=mode != "ngram"), indent=2) + "\n")
                if mode != "ngram":
                    assert not differences, f"Profiled {mode} IDs differ"
                checkpoint(f"profile-{mode}")

            reset_env()
            destination = results / "route-control" / "results"
            destination.mkdir(parents=True)
            for trial in range(6):
                for shared in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    run(["python", "analysis/decode_control_gpu.py", "--shared", str(shared),
                         "--trial", str(trial), "--output-dir", str(destination)])
                a = rows(destination / f"shared0-{trial}/requests.jsonl")
                b = rows(destination / f"shared1-{trial}/requests.jsonl")
                assert len(a) == len(b) == 2
                assert all(x["question_id"] == y["question_id"] and
                           x["response"]["output_ids"] == y["response"]["output_ids"] for x, y in zip(a, b))
                checkpoint(f"route-control-{trial}")

            reset_env()
            point_results(results / "width-probe" / "results")
            run(["python", "scripts/gpu_width_probe.py"])
            checkpoint("width-probe")
            status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        (directory / "diagnostics-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.function(image=cpu_image, cpu=2, memory=4096, gpu=None, timeout=64800,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def wait_and_execute(run_id, reference_run, expected, control_sha, runner_sha, image_id):
    """Wait without a GPU; analyze complete timings, then serialize diagnostics."""
    import hashlib
    import subprocess
    import time

    for identifier in (run_id, reference_run):
        if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
            raise ValueError("Invalid run ID")
    controller = Path("/artifacts") / f"{run_id}-controller"
    controller.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(success=False, run_id=run_id, reference_run=reference_run,
                  runner_sha256=runner_sha, waiting_owns_no_gpu=True)
    status["analysis_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("/analysis").glob("*.py"))
    }
    try:
        campaign_status = Path("/artifacts") / reference_run / "campaign-status.json"
        while True:
            artifacts.reload()
            if campaign_status.exists():
                campaign = json.loads(campaign_status.read_text())
                if not campaign["success"]:
                    raise ValueError(f"Campaign failed; no follow-up GPU allocated: {campaign.get('error')}")
                assert campaign["source_sha256"] == expected
                break
            if time.monotonic() - started > 43200:
                raise TimeoutError("Campaign has no completion status after 12 hours; no follow-up GPU allocated")
            time.sleep(30)
        # The campaign writes status immediately before its final volume
        # commits. Leave time for those commits and the 2s idle scale-down.
        time.sleep(30)
        artifacts.reload()
        reports = controller / "reports"
        reports.mkdir()

        def report(command):
            with (controller / "analysis.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

        report(["python", "/analysis/benchmark_report.py", f"/artifacts/{reference_run}/results",
                "--output", str(reports / "benchmark-report.json"), "--allow-ngram-numerical-differences"])
        benchmark = json.loads((reports / "benchmark-report.json").read_text())
        assert benchmark["exact_suffix_ids_passed"] and benchmark["measured_requests"] == 6000
        status["serving_report_passed"] = True
        artifacts.commit()
        print("PROGRESS: all6000 serving records validated; starting separate single-GPU diagnostics", flush=True)
        result = execute.remote(run_id, reference_run, expected, control_sha, runner_sha, image_id)
        artifacts.reload()
        assert result["success"], result.get("error")
        directory = Path("/artifacts") / run_id / "results"
        report(["python", "/analysis/natural_trace_report.py", str(directory / "natural-trace"),
                "--output", str(reports / "natural-trace-report.json")])
        report(["python", "/analysis/decode_control_report.py", str(directory / "route-control/results"),
                "--output", str(reports / "decode-control-report.json")])
        width = directory / "width-probe/results/width-probe"
        report(["python", "/analysis/analyze_width_probe.py", str(width)])
        report(["python", "/analysis/width_report.py", str(width)])
        status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        (controller / "controller-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
    return status


@app.local_entrypoint()
def main(run_id: str, reference_run: str = "modal-20261004-final-campaign", wait_for_campaign: bool = False):
    import hashlib

    gate = ROOT / "results/modal/modal-20261004-v12-suffix-isolated/suffix-gate-status.json"
    expected = json.loads(gate.read_text())["source_sha256"]
    control_sha = hashlib.sha256((ROOT / "analysis/decode_control_gpu.py").read_bytes()).hexdigest()
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    function = wait_and_execute if wait_for_campaign else execute
    status = function.remote(run_id, reference_run, expected, control_sha, runner_sha, image.object_id)
    directory = ROOT / "results/modal" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "diagnostics-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
