"""Complete SUFFIX's public gate without requiring NGRAM's tree to be linear.

Uses the identical v12 immutable image and its completed ordinary reference.
This is a correctness comparison across allocations, never a latency comparison.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sglang-suffix-isolated-public-gate")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id("im-96vGJgQDElG1UZq72yqyhI")


@app.function(image=image, gpu="H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=7200, startup_timeout=1800,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, expected):
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
    status = dict(success=False, run_id=run_id, image_id="im-96vGJgQDElG1UZq72yqyhI",
                  source_sha256=expected, ordinary_reference_run="modal-20261004-v12-full",
                  cross_allocation_correctness_only=True)
    try:
        for name, digest in expected.items():
            assert hashlib.sha256((Path("/project") / name).read_bytes()).hexdigest() == digest, name
        with (directory / "suffix-gate.log").open("x", buffering=1) as log:
            command = ["python", "/project/scripts/gpu_public_gate.py", "--mode", "suffix", "--shared", "1", "--limit", "0"]
            log.write("COMMAND: " + json.dumps(command) + "\n")
            with subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith("PROGRESS:"):
                        print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError("Suffix gate execution failed; see log")
        reference = [json.loads(line) for line in Path("/artifacts/modal-20261004-v12-full/results/public-gate-ordinary-shared1.jsonl").read_text().splitlines()]
        suffix = [json.loads(line) for line in (results / "public-gate-suffix-shared1.jsonl").read_text().splitlines()]
        assert len(reference) == len(suffix) == 240
        mismatches = []
        for a, b in zip(reference, suffix):
            assert all(a[k] == b[k] for k in ("block", "index", "question_id"))
            x, y = a["response"]["output_ids"], b["response"]["output_ids"]
            if x != y:
                first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                mismatches.append(dict(block=a["block"], index=a["index"], question_id=a["question_id"], first_difference=first))
        report = dict(complete=True, requests=240, mismatches=mismatches,
                      ordinary_reference_run="modal-20261004-v12-full", same_immutable_image=True,
                      cross_allocation_correctness_only=True)
        (results / "suffix-gate-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        status["execution_complete"] = True
        status["mismatches"] = len(mismatches)
        status["success"] = not mismatches
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status and not status.get("execution_complete"):
            status["error"] = "Interrupted before completion"
        (directory / "suffix-gate-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.local_entrypoint()
def main(run_id: str = "modal-20261004-v12-suffix-isolated"):
    expected = json.loads((ROOT / "results/modal/modal-20261004-v12-full/raw/modal-20261004-v12-full/public-gate-status.json").read_text())["source_sha256"]
    status = execute.remote(run_id, expected)
    directory = ROOT / "results/modal" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "suffix-gate-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
