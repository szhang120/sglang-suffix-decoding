"""Continue an existing trial0, then serialize remaining trials and profiling.

Each completed trial is downloaded and checked before allocating the next GPU.
Run from the repository root. No retries or concurrent GPU functions.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

MODES = ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local")


def verify(directory, trial):
    rows = {}
    for mode in MODES:
        path = directory / "gpu" / f"{mode}-{trial}" / "requests.jsonl"
        rows[mode] = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows[mode]) == 240, (mode, trial, "incomplete")
    mismatches = []
    for mode in MODES[1:]:
        for reference, row in zip(rows["ordinary"], rows[mode]):
            assert all(reference[k] == row[k] for k in ("block", "index", "question_id", "kind", "category", "input_tokens"))
            if reference["response"]["output_ids"] != row["response"]["output_ids"]:
                mismatches.append(dict(mode=mode, trial=trial, block=row["block"],
                                       index=row["index"], question_id=row["question_id"]))
    (directory / f"trial-{trial}-equality.json").write_text(
        json.dumps(dict(exact_ids_passed=not mismatches, mismatches=mismatches), indent=2) + "\n"
    )
    if mismatches:
        raise RuntimeError(f"Trial{trial}: {len(mismatches)} output mismatches; preserve data and investigate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--modal", default="/tmp/sglang-modal-cli/bin/modal")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    statuses = root / "results/modal" / args.run_id
    dest = root / "results/gpu/campaign" / args.run_id / "results"
    (dest / "gpu").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SUFFIX_MODAL_PREBUILT_IMAGE=args.image)

    def run(phase, trial=0):
        status = statuses / f"{phase}-{trial}-status.json"
        if status.exists():
            raise RuntimeError(f"Preserving {status}")
        log = root / f"results/setup/modal-{phase}-{args.run_id}-{trial}.log"
        with log.open("x") as f:
            subprocess.run([args.modal, "run", "scripts/modal_runner.py", "--phase", phase,
                            "--run-id", args.run_id, "--trial", str(trial)],
                           env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    initial = statuses / "benchmark-0-status.json"
    print("Waiting for the already-running trial0; no second GPU allocated", flush=True)
    while not initial.exists():
        time.sleep(10)
    if not json.loads(initial.read_text())["success"]:
        raise RuntimeError("Existing trial0 failed; preserve status and investigate")
    for trial in range(5):
        if trial:
            print(f"Starting trial{trial}", flush=True)
            run("benchmark", trial)
        for mode in MODES:
            subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                            f"{args.run_id}/results/gpu/{mode}-{trial}", str(dest / "gpu")], check=True)
        verify(dest, trial)
        print(f"Trial{trial} completed: all240 requests exactly equal in every mode", flush=True)
    print("Starting separate profile run", flush=True)
    run("profile")
    for mode in MODES[:3]:
        subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                        f"{args.run_id}/results/gpu/{mode}-0-profile", str(dest / "gpu")], check=True)
    for path in (root / "results/final").iterdir():
        shutil.copyfile(path, dest / path.name)
    subprocess.run([str(root / ".venv/bin/python"), "analysis/benchmark_report.py", str(dest),
                    "--output", "results/final/benchmark-report.json"], check=True)
    print("Campaign completed; final raw outputs and benchmark report saved locally", flush=True)


if __name__ == "__main__":
    main()
