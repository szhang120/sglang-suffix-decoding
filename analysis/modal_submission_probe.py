"""Short CPU-only check that async submission outlives the local CLI."""

import json
from pathlib import Path

import modal

app = modal.App("sglang-suffix-submission-probe")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), gpu=None,
              cpu=1, memory=1024, timeout=120, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def execute(run_id):
    import time

    assert run_id and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id)
    directory = Path("/artifacts") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    time.sleep(30)
    status = dict(success=True, gpu_allocated=False, local_cli_already_returned=True)
    (directory / "status.json").write_text(json.dumps(status) + "\n")
    artifacts.commit()
    return status


@app.local_entrypoint()
def main(run_id: str):
    call = execute.spawn(run_id)
    print(json.dumps(dict(submitted=True, app_id=app.app_id, function_call_id=call.object_id,
                          run_id=run_id, gpu_allocated=False)))
