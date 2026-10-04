"""Serialize separate diagnostics after a fully successful serving campaign.

Waiting owns no GPU. Any failed earlier gate stops follow-up allocation.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--modal", default="/tmp/sglang-modal-cli/bin/modal")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = dict(os.environ, SUFFIX_MODAL_PREBUILT_IMAGE=args.image)
    statuses = root / "results/modal" / args.run_id
    report_path = root / "results/final/benchmark-report.json"
    dest = root / "results/gpu/campaign" / args.run_id
    deadline = time.monotonic() + 36 * 3600
    print("Waiting for complete exact-ID serving results; no GPU allocated", flush=True)
    while not report_path.exists():
        for name in ("public-gate-0", "audit-frozen-0", "profile-0",
                     *(f"benchmark-{i}" for i in range(5))):
            status = statuses / f"{name}-status.json"
            if status.exists() and not json.loads(status.read_text())["success"]:
                raise RuntimeError(f"Earlier {name} failed; no follow-up GPU allocated")
        if time.monotonic() >= deadline:
            raise RuntimeError("No complete report after36 hours; inspect the existing campaign")
        time.sleep(10)
    report = json.loads(report_path.read_text())
    if not report["exact_ids_passed"] or report["measured_requests"] != 6000:
        raise RuntimeError("Serving report fails the required equality/count gate")
    # Allow the preceding function's two-second idle scale-down to finish.
    time.sleep(3)

    def run(run_id, phase, extra=()):
        log = root / f"results/setup/modal-{phase}-{run_id}.log"
        with log.open("x") as f:
            subprocess.run([args.modal, "run", "scripts/modal_runner.py", "--run-id", run_id,
                            "--phase", phase, *extra], env=env, stdout=f,
                           stderr=subprocess.STDOUT, check=True)

    variants = ("suffix", "suffix-fixed", "suffix-local")
    for variant in variants:
        run_id = f"{args.run_id}-trace-{variant}"
        print(f"Starting separate full-workload {variant} trace", flush=True)
        run(run_id, "natural-trace", ("--variant", variant, "--reference-run", args.run_id))
        directory = dest / "natural-trace" / variant
        directory.mkdir(parents=True, exist_ok=False)
        subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                        f"{run_id}/results", str(directory)], check=True)
    subprocess.run([str(root / ".venv/bin/python"), "analysis/natural_trace_report.py",
                    str(dest / "natural-trace"), "--output", "results/final/natural-trace-report.json"], check=True)

    run_id = f"{args.run_id}-route-control"
    print("Starting the narrow, matched-output ordinary route-cost control", flush=True)
    with (root / f"results/setup/modal-route-control-{run_id}.log").open("x") as log:
        subprocess.run([args.modal, "run", "analysis/modal_decode_control.py", "--run-id", run_id,
                        "--reference-status", str(statuses / "public-gate-0-status.json")],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    directory = dest / "route-control"
    directory.mkdir(parents=True, exist_ok=False)
    subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                    f"{run_id}/results", str(directory)], check=True)
    subprocess.run([str(root / ".venv/bin/python"), "analysis/decode_control_report.py",
                    str(directory / "results"), "--output", "results/final/decode-control-report.json"], check=True)

    run_id = f"{args.run_id}-width"
    print("Starting the final-candidate controlled-width profile", flush=True)
    run(run_id, "width-probe")
    directory = dest / "width-probe"
    directory.mkdir(parents=True, exist_ok=False)
    subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                    f"{run_id}/results", str(directory)], check=True)
    probe = directory / "results/width-probe"
    subprocess.run([str(root / ".venv/bin/python"), "scripts/analyze_width_probe.py", str(probe)], check=True)
    subprocess.run([str(root / ".venv/bin/python"), "analysis/width_report.py", str(probe)], check=True)
    print("Follow-up traces, route-cost control and final widths completed", flush=True)


if __name__ == "__main__":
    main()
