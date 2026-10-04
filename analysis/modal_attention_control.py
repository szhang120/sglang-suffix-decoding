"""Standalone SGLang attention equivalence controls; one GPU, no model rental overlap."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sglang-attention-equivalence")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id("im-j008pljEZGTNXdCClQMMbR").add_local_file(
    ROOT / "analysis/attention_equivalence_gpu.py", "/project/analysis/attention_equivalence_gpu.py", copy=True
)


@app.function(image=image, gpu="H100!", cpu=2, memory=8192, timeout=1800,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, script_sha, image_id):
    import hashlib
    import subprocess
    import time

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    directory = Path("/artifacts") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(success=False, image_id=image_id, run_id=run_id, script_sha256=script_sha)
    try:
        assert hashlib.sha256(Path("/project/analysis/attention_equivalence_gpu.py").read_bytes()).hexdigest() == script_sha
        subprocess.run(["git", "-C", "/project/sglang", "diff", "--exit-code"], check=True)
        with (directory / "kernel.log").open("x") as log:
            subprocess.run(["python", "/project/analysis/attention_equivalence_gpu.py",
                            "--output", str(directory / "attention-equivalence.json")],
                           cwd="/project", stdout=log, stderr=subprocess.STDOUT, check=True)
        status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        (directory / "kernel-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.local_entrypoint()
def main(run_id: str = "modal-20261004-attention-equivalence"):
    import hashlib

    digest = hashlib.sha256((ROOT / "analysis/attention_equivalence_gpu.py").read_bytes()).hexdigest()
    status = execute.remote(run_id, digest, image.object_id)
    directory = ROOT / "results/modal" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
