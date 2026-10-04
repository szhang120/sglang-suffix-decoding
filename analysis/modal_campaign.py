"""Continue an existing trial0, then serialize remaining trials and profiling.

Each completed trial is downloaded and checked before allocating the next GPU.
Run from the repository root. No retries or concurrent GPU functions.
"""

import argparse
import json
import os
import re
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
    parser.add_argument("--after-public-gate", action="store_true",
                        help="Wait for the running full gate, then audit frozen inputs and start trial0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    statuses = root / "results/modal" / args.run_id
    dest = root / "results/gpu/campaign" / args.run_id / "results"
    (dest / "gpu").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SUFFIX_MODAL_PREBUILT_IMAGE=args.image)

    def fetch_status(phase, trial=0):
        local = statuses / f"{phase}-{trial}-status.json"
        if local.exists():
            return json.loads(local.read_text())
        statuses.mkdir(parents=True, exist_ok=True)
        temporary = statuses / f".{phase}-{trial}-remote-status.json"
        remote_key = f"{phase}-{trial}" if phase == "benchmark" else phase
        fetched = subprocess.run(
            [args.modal, "volume", "get", "sglang-suffix-artifacts",
             f"{args.run_id}/{remote_key}-status.json", str(temporary), "--force"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if fetched.returncode:
            if temporary.exists():
                temporary.unlink()
            if "not found" not in fetched.stdout.lower() and "does not exist" not in fetched.stdout.lower():
                raise RuntimeError(f"Cannot recover remote completion status: {fetched.stdout[-2000:]}")
            return None
        status = json.loads(temporary.read_text())
        if status["run_id"] != args.run_id or status["phase"] != phase or status["trial"] != trial:
            raise RuntimeError("Remote completion status identity differs")
        temporary.replace(local)
        return status

    def run(phase, trial=0):
        status = statuses / f"{phase}-{trial}-status.json"
        if status.exists():
            raise RuntimeError(f"Preserving {status}")
        log = root / f"results/setup/modal-{phase}-{args.run_id}-{trial}.log"
        with log.open("x") as f:
            # Detachment keeps the single remote function alive if the CLI
            # connection drops. Completion is authoritative in the volume.
            process = subprocess.Popen([args.modal, "run", "--detach", "scripts/modal_runner.py", "--phase", phase,
                                        "--run-id", args.run_id, "--trial", str(trial)],
                                       env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + int(env.get("SUFFIX_MODAL_TIMEOUT", "14400")) + 2400
                stopped_checks = 0
                while True:
                    completion = fetch_status(phase, trial)
                    if completion is not None:
                        if not completion.get("success", False):
                            raise RuntimeError(f"{phase}/{trial} failed: {completion.get('error', 'interrupted')}")
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Remote completion status missing after timeout; inspect the detached app before any further GPU allocation")
                    if process.poll() is not None:
                        app_ids = re.findall(r"ap-[A-Za-z0-9]+", log.read_text())
                        if not app_ids:
                            raise RuntimeError(f"Modal launch exited {process.returncode} before an app ID was recorded")
                        listed = subprocess.run([args.modal, "app", "list", "--json"],
                                                check=True, capture_output=True, text=True)
                        app_state = next((a for a in json.loads(listed.stdout) if a["app_id"] == app_ids[0]), None)
                        stopped_checks = stopped_checks + 1 if app_state and app_state["state"] == "stopped" else 0
                        if stopped_checks >= 3:
                            raise RuntimeError("Detached app stopped without completion status; preserve artifacts and investigate")
                    time.sleep(30)
            finally:
                if process.poll() is None:
                    # End only the detached CLI connection. A running remote
                    # function remains alive and must be inspected separately.
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def wait_for(phase):
        deadline = time.monotonic() + 6 * 3600
        print(f"Waiting for the already-running {phase}; no second GPU allocated", flush=True)
        completion = None
        while completion is None:
            completion = fetch_status(phase)
            if completion is not None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"No {phase} completion status after six hours; inspect the existing detached app")
            time.sleep(30)
        if not completion.get("success", False):
            raise RuntimeError(f"Existing {phase} failed; preserve status and investigate")

    if args.after_public_gate:
        wait_for("public-gate")
        print("Full public gate passed; starting plain and direct-KV audit on frozen inputs", flush=True)
        run("audit-frozen")
        print("Audit passed; starting trial0", flush=True)
        run("benchmark")
    else:
        wait_for("benchmark")
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
    # Fetch this campaign's evidence; do not silently relabel old-image gates.
    evidence = ["workload.jsonl", "workload.sha256", "public-gate-summary.json",
                "audit-summary.json", "audit-suffix-trace.jsonl"]
    evidence += [f"correctness-{mode}.jsonl" for mode in MODES[:3]]
    evidence += [f"correctness-plain-{mode}.jsonl" for mode in MODES[:3]]
    evidence += [f"public-gate-{mode}-shared1.jsonl" for mode in MODES[:3]]
    evidence += [f"public-gate-{mode}-shared1-environment.json" for mode in MODES[:3]]
    for name in evidence:
        subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                        f"{args.run_id}/results/{name}", str(dest / name)], check=True)
    subprocess.run([str(root / ".venv/bin/python"), "analysis/benchmark_report.py", str(dest),
                    "--output", "results/final/benchmark-report.json"], check=True)
    print("Campaign completed; final raw outputs and benchmark report saved locally", flush=True)


if __name__ == "__main__":
    main()
