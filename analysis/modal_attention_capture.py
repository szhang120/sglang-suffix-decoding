"""Capture and replay the first differing real attention instance on one H100."""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sglang-real-attention-capture")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id(os.environ.get("SUFFIX_MODAL_DIAGNOSTIC_BASE_IMAGE", "im-Sw0TTFndEnpfYhEZKoDfsG"))
for name in ("model_tensor_trace_gpu.py", "attention_capture_hook.py", "attention_capture_report_gpu.py"):
    image = image.add_local_file(ROOT / "analysis" / name, "/project/analysis/" + name, copy=True)
image = image.run_commands(
    "python -c \"from pathlib import Path; p=Path('/project/sglang/python/sglang/kernels/ops/attention/extend_attention.py'); p.write_text(p.read_text()+'\\n'+Path('/project/analysis/attention_capture_hook.py').read_text())\""
)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=1800, startup_timeout=1800,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, script_hashes, image_id):
    import hashlib
    import subprocess
    import time

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    directory = Path("/artifacts") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(success=False, image_id=image_id, run_id=run_id, script_sha256=script_hashes)
    try:
        for name, digest in script_hashes.items():
            assert hashlib.sha256((Path("/project/analysis") / name).read_bytes()).hexdigest() == digest
        assert not Path("/project/sglang/python/sglang/srt/speculative/suffix_worker.py").exists()
        with (directory / "capture.log").open("x", buffering=1) as log:
            for mode in ("ordinary", "ngram"):
                command = ["python", "/project/analysis/model_tensor_trace_gpu.py", "--mode", mode,
                           "--output-dir", str(directory / "results"), "--capture-only"]
                log.write("COMMAND: " + json.dumps(command) + "\n")
                subprocess.run(command, cwd="/project", stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"PROGRESS: attention capture {mode} complete", flush=True)
            command = ["python", "/project/analysis/attention_capture_report_gpu.py", str(directory / "results"),
                       "--output", str(directory / "attention-capture-report.json")]
            subprocess.run(command, cwd="/project", stdout=log, stderr=subprocess.STDOUT, check=True)
        status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        (directory / "capture-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.local_entrypoint()
def main(run_id: str = "modal-20261004-attention-capture"):
    import hashlib

    hashes = {name: hashlib.sha256((ROOT / "analysis" / name).read_bytes()).hexdigest()
              for name in ("model_tensor_trace_gpu.py", "attention_capture_hook.py", "attention_capture_report_gpu.py")}
    status = execute.remote(run_id, hashes, image.object_id)
    directory = ROOT / "results/modal" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capture-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
