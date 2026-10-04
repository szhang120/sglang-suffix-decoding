"""One-H100 independent controls, starting from exactly pristine SGLang."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = "im-96vGJgQDElG1UZq72yqyhI"
BASE_COMMIT = "e00930c5489053f26d86b179cee0d087f846acbb"
PATCH_SHA = "19fc2ced20cde74fef8bfb0e7baeaada3430fc2a4d8096597e1778da1d2327bf"
app = modal.App("sglang-suffix-independent-reassessment")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id(BASE_IMAGE).run_commands(
    f"test $(git -C /project/sglang rev-parse HEAD) = {BASE_COMMIT}",
    f"python -c \"import hashlib; assert hashlib.sha256(open('/project/patches/sglang-suffix.patch','rb').read()).hexdigest() == '{PATCH_SHA}'\"",
    "git -C /project/sglang apply --reverse --check /project/patches/sglang-suffix.patch",
    "git -C /project/sglang apply --reverse /project/patches/sglang-suffix.patch",
    "git -C /project/sglang diff --exit-code",
).add_local_file(ROOT / "analysis/reassessment_gpu.py", "/project/analysis/reassessment_gpu.py", copy=True)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=7200, startup_timeout=1800,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, expected, diagnostic_sha, image_id):
    import hashlib
    import subprocess
    import time

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    destination = Path("/artifacts") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    status = dict(success=False, run_id=run_id, image_id=image_id,
                  base_image=BASE_IMAGE, sglang_commit=BASE_COMMIT, patch_sha256=PATCH_SHA,
                  diagnostic_script_sha256=diagnostic_sha)
    started = time.monotonic()
    try:
        # Frozen fixture/dependencies/helpers come from the recorded image.
        # SGLang files are deliberately pristine, not the image's old patch.
        for name, digest in expected.items():
            if not name.startswith("sglang/"):
                assert hashlib.sha256((Path("/project") / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256(Path("/project/analysis/reassessment_gpu.py").read_bytes()).hexdigest() == diagnostic_sha
        subprocess.run(["git", "-C", "/project/sglang", "diff", "--exit-code"], check=True)
        assert not Path("/project/sglang/python/sglang/srt/speculative/suffix_worker.py").exists()
        with (destination / "audit.log").open("x", buffering=1) as log:
            for variant in ("pristine", "head-only", "shared-v12"):
                if variant != "pristine":
                    path = ("python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py"
                            if variant == "head-only" else "python/sglang/srt/layers/attention/triton_backend.py")
                    subprocess.run(["git", "-C", "/project/sglang", "apply", f"--include={path}",
                                    "/project/patches/sglang-suffix.patch"], check=True)
                for mode in ("ordinary", "ngram"):
                    command = ["python", "/project/analysis/reassessment_gpu.py", "--variant", variant,
                               "--mode", mode, "--output-dir", str(destination / "results")]
                    log.write("COMMAND: " + json.dumps(command) + "\n")
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, cwd="/project", bufsize=1) as process:
                        for line in process.stdout:
                            log.write(line)
                            if line.startswith("PROGRESS:"):
                                print(line.rstrip(), flush=True)
                        if process.wait():
                            raise RuntimeError(f"{variant}/{mode} execution failed; see audit.log")
        status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        (destination / "audit-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.local_entrypoint()
def main(run_id: str = "modal-20261004-independent-audit"):
    import hashlib

    reference = ROOT / "results/modal/modal-20261004-v12/public-probe-0-status.json"
    expected = json.loads(reference.read_text())["source_sha256"]
    diagnostic_sha = hashlib.sha256((ROOT / "analysis/reassessment_gpu.py").read_bytes()).hexdigest()
    status = execute.remote(run_id, expected, diagnostic_sha, image.object_id)
    destination = ROOT / "results/modal" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
