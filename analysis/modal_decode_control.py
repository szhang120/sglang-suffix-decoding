"""One-GPU shared-attention baseline-cost control, separate from the campaign."""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = os.environ["SUFFIX_MODAL_PREBUILT_IMAGE"]
app = modal.App("sglang-suffix-decode-control")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
model_cache = modal.Volume.from_name("sglang-suffix-model-cache")
image = modal.Image.from_id(BASE_IMAGE).env({"SUFFIX_MODAL_PREBUILT_IMAGE": BASE_IMAGE}).add_local_file(
    ROOT / "analysis/decode_control_gpu.py", "/project/analysis/decode_control_gpu.py", copy=True
)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=7200, startup_timeout=1800,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, expected, control_sha, image_id):
    import hashlib
    import subprocess
    import time

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    destination = Path("/artifacts") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    status = dict(run_id=run_id, base_image=BASE_IMAGE, control_image=image_id,
                  control_script_sha256=control_sha)
    start = time.monotonic()
    try:
        for name, sha in expected.items():
            assert hashlib.sha256((Path("/project") / name).read_bytes()).hexdigest() == sha, name
        assert hashlib.sha256(Path("/project/analysis/decode_control_gpu.py").read_bytes()).hexdigest() == control_sha
        status["source_sha256"] = expected
        import torch

        assert torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0)
        # Six paired trials balance first/second execution order exactly.
        with (destination / "control.log").open("x") as log:
            for trial in range(6):
                order = (0, 1) if trial % 2 == 0 else (1, 0)
                for shared in order:
                    subprocess.run(["python", "/project/analysis/decode_control_gpu.py",
                                    "--shared", str(shared), "--trial", str(trial),
                                    "--output-dir", str(destination / "results")],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, cwd="/project")
                    print(f"PROGRESS: route control trial{trial} shared{shared} complete", flush=True)
                a = [json.loads(line) for line in (destination / f"results/shared0-{trial}/requests.jsonl").read_text().splitlines()]
                b = [json.loads(line) for line in (destination / f"results/shared1-{trial}/requests.jsonl").read_text().splitlines()]
                assert len(a) == len(b) == 2
                assert all(x["question_id"] == y["question_id"] and
                           x["response"]["output_ids"] == y["response"]["output_ids"] for x, y in zip(a, b)), "Route control output mismatch"
        status["success"] = True
    except Exception as exc:
        status.update(success=False, error=str(exc))
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        (destination / "control-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    return status


@app.local_entrypoint()
def main(run_id: str, reference_status: str):
    import hashlib

    expected = json.loads(Path(reference_status).read_text())["source_sha256"]
    control_sha = hashlib.sha256((ROOT / "analysis/decode_control_gpu.py").read_bytes()).hexdigest()
    status = execute.remote(run_id, expected, control_sha, image.object_id)
    destination = ROOT / "results/modal" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "control-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
