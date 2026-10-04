"""CPU-only artifact analysis; no GPU allocation."""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sglang-kv-origin-analysis")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
image = modal.Image.from_id(os.environ.get("SUFFIX_MODAL_PRISTINE_IMAGE", "im-j008pljEZGTNXdCClQMMbR")).add_local_file(
    ROOT / "analysis/kv_origin_report.py", "/project/analysis/kv_origin_report.py", copy=True
)


@app.function(image=image, cpu=4, memory=32768, timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def execute(digest):
    import hashlib
    import subprocess

    assert hashlib.sha256(Path("/project/analysis/kv_origin_report.py").read_bytes()).hexdigest() == digest
    output = Path("/artifacts/modal-20261004-attention-capture/kv-origin-report.json")
    assert not output.exists()
    result = subprocess.run(["python", "/project/analysis/kv_origin_report.py",
                             "--trace", "/artifacts/modal-20261004-tensor-equivalence/results",
                             "--capture", "/artifacts/modal-20261004-attention-capture/results",
                             "--output", str(output)], check=True, capture_output=True, text=True)
    artifacts.commit()
    return json.loads(result.stdout)


@app.local_entrypoint()
def main():
    import hashlib

    result = execute.remote(hashlib.sha256((ROOT / "analysis/kv_origin_report.py").read_bytes()).hexdigest())
    (ROOT / "results/kv-origin-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
