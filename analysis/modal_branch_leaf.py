"""CPU-only branch artifact analysis."""

import hashlib
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sglang-branch-leaf-analysis")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
image = modal.Image.from_id(os.environ.get("SUFFIX_MODAL_PRISTINE_IMAGE", "im-j008pljEZGTNXdCClQMMbR")).add_local_file(
    ROOT / "analysis/branch_leaf_report.py", "/project/analysis/branch_leaf_report.py", copy=True
)


@app.function(image=image, cpu=4, memory=32768, timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/artifacts": artifacts})
def execute(digest):
    import subprocess

    assert hashlib.sha256(Path("/project/analysis/branch_leaf_report.py").read_bytes()).hexdigest() == digest
    result = subprocess.run(["python", "/project/analysis/branch_leaf_report.py"],
                            check=True, capture_output=True, text=True)
    artifacts.commit()
    return json.loads(result.stdout)


@app.local_entrypoint()
def main():
    result = execute.remote(hashlib.sha256((ROOT / "analysis/branch_leaf_report.py").read_bytes()).hexdigest())
    (ROOT / "results/branch-leaf-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(first_unequal=result["first_unequal"], metadata=result["metadata"]), indent=2))
