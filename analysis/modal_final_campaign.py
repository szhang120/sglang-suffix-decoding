"""One detached H100 campaign; exact suffix comparisons, descriptive NGRAM drift.

Uses unchanged v12 model/runtime scripts. Runs gates and five rotated trials
remotely so Mac/client disconnects cannot cancel a partially launched sequence.
No automatic retry, no parallel GPU work, and no changed NGRAM proposer.
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = os.environ.get("SUFFIX_MODAL_V12_IMAGE", "im-96vGJgQDElG1UZq72yqyhI")
app = modal.App("sglang-suffix-validated-campaign")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id(IMAGE_ID)
MODES = ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local")


@app.function(image=image, gpu="H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=43200, startup_timeout=1800,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, expected, runner_sha, image_id, resume_run=""):
    import hashlib
    import subprocess
    import time

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    directory = Path("/artifacts") / run_id
    results = directory / "results"
    results.mkdir(parents=True, exist_ok=False)
    Path("/project/results").symlink_to(results, target_is_directory=True)
    started = time.monotonic()
    status = dict(success=False, run_id=run_id, image_id=image_id,
                  source_sha256=expected, runner_sha256=runner_sha, timeout_seconds=43200,
                  policy="exact ordinary/SUFFIX/ablations; unchanged upstream NGRAM with every divergence recorded descriptively",
                  completed_trials=[], completed_modes=[])

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def equality(reference, measured):
        assert len(reference) == len(measured)
        differences = []
        for a, b in zip(reference, measured):
            keys = ("block", "index", "question_id") if "block" in a else ("name", "repeat")
            assert all(a[k] == b[k] for k in keys)
            x, y = a["response"]["output_ids"], b["response"]["output_ids"]
            if x != y:
                first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                differences.append(dict(identity={k: a[k] for k in keys}, first_difference=first,
                                        ordinary_ids=x[first:first + 8], mode_ids=y[first:first + 8],
                                        ordinary_output_tokens=len(x), mode_output_tokens=len(y)))
        return differences

    try:
        for name, digest in expected.items():
            assert hashlib.sha256((Path("/project") / name).read_bytes()).hexdigest() == digest, name
        if resume_run:
            import shutil

            assert resume_run != run_id and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in resume_run)
            previous_dir = Path("/artifacts") / resume_run
            previous_path = previous_dir / "campaign-status.json"
            previous = json.loads(previous_path.read_text())
            assert not previous["success"] and previous.get("error") == "Interrupted before completion"
            assert previous["source_sha256"] == expected
            completed = previous["completed_trials"]
            assert completed and completed == list(range(len(completed))) and len(completed) < 5
            copied = []
            for trial in completed:
                for mode in MODES:
                    entry = next(row for row in previous["completed_modes"] if row["trial"] == trial and row["mode"] == mode)
                    assert mode == "ngram" or not entry["mismatches"]
                    assert len(rows(previous_dir / f"results/gpu/{mode}-{trial}/requests.jsonl")) == 240
                    shutil.copytree(previous_dir / f"results/gpu/{mode}-{trial}", results / f"gpu/{mode}-{trial}")
                    filename = f"benchmark-{mode}-{trial}-equality.json"
                    shutil.copyfile(previous_dir / "results" / filename, results / filename)
                    copied.append(entry)
            status["completed_modes"] = copied
            status["completed_trials"] = completed.copy()
            status["resume_lineage"] = dict(run_id=resume_run, completed_trials=completed,
                                           original_runner_sha256=previous["runner_sha256"],
                                           original_status_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                                           incomplete_trials_discarded=True,
                                           allocation_policy="Every five-mode paired trial stays on one GPU; allocations can differ between trials")
            history = directory / "resume-source"
            history.mkdir()
            for filename in ("campaign-status.json", "campaign.log"):
                shutil.copyfile(previous_dir / filename, history / filename)
            complete_names = {f"{mode}-{trial}" for trial in completed for mode in MODES}
            for old in sorted((previous_dir / "results/gpu").iterdir()):
                if old.is_dir() and old.name not in complete_names:
                    shutil.copytree(old, history / "discarded-trials" / old.name)
            print(f"PROGRESS: preserved {len(completed)} complete trials from {resume_run}; rerunning incomplete trials", flush=True)
        suffix_gate = Path("/artifacts/modal-20261004-v12-suffix-isolated/results/suffix-gate-summary.json")
        gate = json.loads(suffix_gate.read_text())
        assert gate["complete"] and gate["requests"] == 240 and not gate["mismatches"]
        reference_status = json.loads(Path("/artifacts/modal-20261004-v12-suffix-isolated/suffix-gate-status.json").read_text())
        assert reference_status["success"] and reference_status["source_sha256"] == expected
        branch = json.loads(Path("/artifacts/modal-20261004-branch-attention/branch-attention-report.json").read_text())
        assert all(branch[k]["bitwise_equal"] for k in ("q", "visible_keys", "visible_values"))
        assert branch["controls"]["tree_replay"]["bitwise_equal"]
        assert branch["controls"]["compact_vs_causal"]["bitwise_equal"]
        status["suffix_gate_reference"] = "modal-20261004-v12-suffix-isolated"
        status["branch_replay_reference"] = "modal-20261004-branch-attention"
        frozen = Path("/project/configs/frozen-workload.jsonl").read_bytes()
        lock = json.loads(Path("/project/configs/source-lock.json").read_text())
        assert hashlib.sha256(frozen).hexdigest() == lock["frozen_workload"]["sha256"]
        (results / "workload.jsonl").write_bytes(frozen)
        (results / "workload.sha256").write_text(hashlib.sha256(frozen).hexdigest() + "\n")
        public_reference = rows(Path("/artifacts/modal-20261004-v12-full/results/public-gate-ordinary-shared1.jsonl"))
        assert len(public_reference) == 240
        with (directory / "campaign.log").open("x", buffering=1) as log:

            def run(command):
                log.write("COMMAND: " + json.dumps(command) + "\n")
                with subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith("PROGRESS:"):
                            print(line.rstrip(), flush=True)
                    if process.wait():
                        raise RuntimeError(f"Execution failed: {command}; see campaign.log")

            # Recheck stop/length/reuse contracts in this same configuration.
            for mode in MODES[:3]:
                run(["python", "scripts/gpu_correctness.py", "--mode", mode])
            plain = rows(results / "correctness-ordinary.jsonl")
            assert len(plain) == 30
            for mode in ("ngram", "suffix"):
                differences = equality(plain, rows(results / f"correctness-{mode}.jsonl"))
                (results / f"correctness-{mode}-equality.json").write_text(json.dumps(differences, indent=2) + "\n")
                if mode == "suffix" and differences:
                    raise ValueError("Plain suffix correctness gate differs")
            (results / "correctness-suffix.jsonl").rename(results / "correctness-plain-suffix.jsonl")
            run(["python", "scripts/gpu_correctness.py", "--mode", "suffix", "--audit"])
            audit = rows(results / "audit-suffix-trace.jsonl")
            assert audit and all(r.get("existing_kv_prefix_unchanged")
                                 and r.get("accepted_kv_slots_checked")
                                 and r.get("gpu_layout_and_acceptance_checked") for r in audit)
            assert not equality(plain, rows(results / "correctness-suffix.jsonl")), "Audited suffix outputs differ"
            status["audited_rounds"] = len(audit)
            status["gates_passed"] = True
            log.flush()
            artifacts.commit()
            for trial in range(len(status["completed_trials"]), 5):
                order = MODES[trial:] + MODES[:trial]
                for mode in order:
                    run(["python", "scripts/gpu_bench.py", "--mode", mode, "--trial", str(trial)])
                    measured = rows(results / "gpu" / f"{mode}-{trial}" / "requests.jsonl")
                    assert len(measured) == 240
                    # Enforce the previously gated public ordinary IDs even
                    # when the rotated trial runs a speculator first.
                    differences = equality(public_reference, measured)
                    report = dict(mode=mode, trial=trial, requests=240, mismatches=differences,
                                  exact_output_required=mode != "ngram",
                                  ngram_descriptive_only=mode == "ngram" and bool(differences))
                    (results / f"benchmark-{mode}-{trial}-equality.json").write_text(json.dumps(report, indent=2) + "\n")
                    if differences and mode != "ngram":
                        raise ValueError(f"Timed {mode}/{trial} differs on {len(differences)} requests")
                    status["completed_modes"].append(dict(mode=mode, trial=trial, mismatches=len(differences)))
                    print(f"PROGRESS: {mode} trial{trial} complete; differences={len(differences)}; strict={mode != 'ngram'}", flush=True)
                    log.flush()
                    (directory / "campaign-progress.json").write_text(json.dumps(status, indent=2) + "\n")
                    artifacts.commit()
                    model_cache.commit()
                baseline = rows(results / "gpu" / f"ordinary-{trial}" / "requests.jsonl")
                for mode in MODES[2:]:
                    assert not equality(baseline, rows(results / "gpu" / f"{mode}-{trial}" / "requests.jsonl"))
                status["completed_trials"].append(trial)
                (directory / "campaign-progress.json").write_text(json.dumps(status, indent=2) + "\n")
                artifacts.commit()
            status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        (directory / "campaign-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.local_entrypoint()
def main(run_id: str = "modal-20261004-final-campaign", resume_run: str = "", submit_only: bool = False):
    import hashlib

    reference = ROOT / "configs/gpu-source-manifest.json"
    expected = json.loads(reference.read_text())["source_sha256"]
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if submit_only:
        call = execute.spawn(run_id, expected, runner_sha, image.object_id, resume_run)
        submission = dict(submitted=True, run_id=run_id, resume_run=resume_run,
                          app_id=app.app_id, function_call_id=call.object_id, runner_sha256=runner_sha)
        directory = ROOT / "results/modal" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
        print(json.dumps(submission, indent=2))
        return
    status = execute.remote(run_id, expected, runner_sha, image.object_id, resume_run)
    directory = ROOT / "results/modal" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "campaign-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
