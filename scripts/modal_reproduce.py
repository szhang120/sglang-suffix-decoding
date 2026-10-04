"""Build public pinned sources and run the portable Linux reproduction on Modal.

No project-private image IDs or historical remote outputs are required. Image
construction reuses modal_runner's public build recipe, not its older phases.
"""

import importlib.util
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/project")
CPU_ONLY = os.environ.get("SUFFIX_PORTABLE_CPU_ONLY") == "1"
if modal.is_local():
    spec = importlib.util.spec_from_file_location("portable_image_builder", ROOT / "scripts/modal_runner.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    image = builder.image.add_local_file(
        ROOT / "scripts/reproduce_gpu.py", "/project/scripts/reproduce_gpu.py", copy=True
    ).add_local_file(
        ROOT / "analysis/benchmark_report.py", "/project/analysis/benchmark_report.py", copy=True
    ).add_local_file(
        ROOT / "results/modal/modal-20261004-v12-suffix-isolated/suffix-gate-status.json",
        "/project/portable-source-gate.json", copy=True
    ).env({"SUFFIX_PORTABLE_CPU_ONLY": "1" if CPU_ONLY else "0"})
else:
    # Container hydration imports this module; local build files are not
    # mounted there. The deployed function already has its built image.
    image = modal.Image.debian_slim(python_version="3.12")
app = modal.App("sglang-suffix-portable-reproduction")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts", create_if_missing=True)
model_cache = modal.Volume.from_name("sglang-suffix-model-cache", create_if_missing=True)


@app.function(image=image, gpu=None if CPU_ONLY else "H100!", cpu=8, memory=65536,
              ephemeral_disk=512 * 1024, timeout=43200, startup_timeout=1800,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts, "/model-cache": model_cache})
def execute(run_id, dry_run, smoke):
    import subprocess

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run ID")
    assert dry_run == CPU_ONLY, "Use SUFFIX_PORTABLE_CPU_ONLY=1 with --dry-run; GPU execution must not be CPU-only"
    directory = Path("/artifacts") / run_id
    if directory.exists():
        raise ValueError("Preserving existing artifacts; use a fresh run ID")
    command = ["python", "/project/scripts/reproduce_gpu.py", "--output-dir", str(directory),
               "--source-manifest", "/project/portable-source-gate.json"]
    if dry_run:
        command.append("--dry-run")
    if smoke:
        command.append("--smoke")
    try:
        subprocess.run(command, check=True)
        if dry_run:
            return dict(success=True, gpu_allocated=False, source_hash_dry_run=True)
        return json.loads((directory / "campaign-status.json").read_text())
    finally:
        artifacts.commit()
        model_cache.commit()


@app.local_entrypoint()
def main(run_id: str, dry_run: bool = False, smoke: bool = False):
    assert dry_run == CPU_ONLY, "Dry runs require SUFFIX_PORTABLE_CPU_ONLY=1 before import/deployment"
    status = execute.remote(run_id, dry_run, smoke)
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
