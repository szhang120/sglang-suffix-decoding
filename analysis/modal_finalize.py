"""CPU-only completion: measurement tables, figures and distributable raw data.

Waits for the diagnostic controller; never allocates a GPU or changes inputs.
GitHub publication uses the local authenticated CLI after downloading assets.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sglang-suffix-final-analysis")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    str(ROOT / "configs/analysis-requirements.txt")
)
for name in ("benchmark_report.py", "natural_trace_report.py", "write_final_report.py",
             "plot_results.py", "package_artifacts.py"):
    image = image.add_local_file(ROOT / "analysis" / name, f"/analysis/{name}", copy=True)
image = image.add_local_file(ROOT / "configs/analysis-requirements.txt",
                             "/analysis/analysis-requirements.txt", copy=True)


@app.function(image=image, gpu=None, cpu=2, memory=8192, timeout=64800,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def execute(run_id, campaign_run, diagnostics_run, runner_sha):
    import hashlib
    import platform
    import shutil
    import subprocess
    import time

    for name in (run_id, campaign_run, diagnostics_run):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
            raise ValueError("Invalid run ID")
    directory = Path("/artifacts") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(success=False, run_id=run_id, campaign_run=campaign_run,
                  diagnostics_run=diagnostics_run, gpu_allocated=False,
                  runner_sha256=runner_sha, python=platform.python_version())
    status["analysis_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("/analysis").glob("*")) if path.is_file()
    }
    (directory / "finalize-progress.json").write_text(json.dumps(
        dict(status, phase="waiting for diagnostic completion; no GPU allocated"), indent=2) + "\n")
    artifacts.commit()
    try:
        campaign_dir = Path("/artifacts") / campaign_run
        diagnostic_dir = Path("/artifacts") / diagnostics_run
        controller_dir = Path("/artifacts") / f"{diagnostics_run}-controller"
        completion = controller_dir / "controller-status.json"
        while True:
            artifacts.reload()
            if completion.exists():
                controller = json.loads(completion.read_text())
                if not controller["success"]:
                    raise ValueError(f"Diagnostic controller failed; no final performance claim: {controller.get('error')}")
                break
            campaign_status = campaign_dir / "campaign-status.json"
            if campaign_status.exists():
                campaign = json.loads(campaign_status.read_text())
                if not campaign["success"]:
                    raise ValueError(f"Campaign failed; no final performance claim: {campaign.get('error')}")
            if time.monotonic() - started > 57600:
                raise TimeoutError("No diagnostic completion after 16 hours")
            time.sleep(30)

        reports = directory / "results/final"
        reports.mkdir(parents=True)
        for source, name in ((campaign_dir / "campaign-status.json", "campaign"),
                             (diagnostic_dir / "diagnostics-status.json", "diagnostics"),
                             (completion, "controller")):
            assert source.is_file()
            status[name] = json.loads(source.read_text())
        (reports / "execution-provenance.json").write_text(json.dumps(
            {k: status[k] for k in ("campaign", "diagnostics", "controller")}, indent=2) + "\n")
        shutil.copyfile(controller_dir / "reports/decode-control-report.json", reports / "decode-control-report.json")
        shutil.copyfile(diagnostic_dir / "results/width-probe/results/width-probe/summary.json", reports / "width-probe-summary.json")
        (directory / "analysis-environment.txt").write_text(subprocess.check_output(
            ["python", "-m", "pip", "freeze"], text=True))
        os_env = dict(__import__("os").environ,
                      MPLCONFIGDIR=str(directory / "plot-cache"),
                      XDG_CACHE_HOME=str(directory / "plot-cache"))
        with (directory / "analysis.log").open("x") as log:

            def run(command):
                log.write("COMMAND: " + json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=os_env)

            run(["python", "/analysis/benchmark_report.py", str(campaign_dir / "results"),
                 "--output", str(reports / "benchmark-report.json"), "--allow-ngram-numerical-differences"])
            run(["python", "/analysis/natural_trace_report.py", str(diagnostic_dir / "results/natural-trace"),
                 "--output", str(reports / "natural-trace-report.json")])
            run(["python", "/analysis/write_final_report.py", "--results-directory", str(reports),
                 "--output", str(directory / "measurement-tables.md")])
            run(["python", "/analysis/plot_results.py", "--root", str(directory),
                 "--width-summary", str(reports / "width-probe-summary.json"), "--width-stem", "verify-width-v12"])

            archive_dir = directory / "archives"
            archive_dir.mkdir()
            specs = [("serving", campaign_dir),
                     ("natural-traces", diagnostic_dir / "results/natural-trace"),
                     ("route-control", diagnostic_dir / "results/route-control"),
                     ("width-profiles", diagnostic_dir / "results/width-probe"),
                     ("portable-runner-smoke", diagnostic_dir / "results/portable-runner-smoke")]
            for mode in ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local"):
                specs.append((f"profile-{mode}", diagnostic_dir / f"results/profile/results/gpu/{mode}-0-profile"))
            for label, source in specs:
                output = archive_dir / f"{label}.tar.gz"
                run(["python", "/analysis/package_artifacts.py", "--input", f"{label}={source}",
                     "--output", str(output), "--manifest", str(reports / f"{label}-manifest.json")])
                assert output.stat().st_size < 2_000_000_000, "Preserve oversized raw archive; split before GitHub upload"
                print(f"PROGRESS: packaged {label}; {output.stat().st_size} bytes", flush=True)
                artifacts.commit()
        # Publish the exact analysis sources with the generated evidence.
        shutil.copytree("/analysis", directory / "analysis-source")
        status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        (directory / "finalize-status.json").write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
    return status


@app.local_entrypoint()
def main(run_id: str = "modal-20261004-final-analysis",
         campaign_run: str = "modal-20261004-final-campaign",
         diagnostics_run: str = "modal-20261004-final-diagnostics", submit_only: bool = False):
    import hashlib

    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if submit_only:
        call = execute.spawn(run_id, campaign_run, diagnostics_run, runner_sha)
        submission = dict(submitted=True, run_id=run_id, campaign_run=campaign_run,
                          diagnostics_run=diagnostics_run, app_id=app.app_id, function_call_id=call.object_id)
        destination = ROOT / "results/modal" / run_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
        print(json.dumps(submission, indent=2))
        return
    status = execute.remote(run_id, campaign_run, diagnostics_run, runner_sha)
    destination = ROOT / "results/modal" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "finalize-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        raise SystemExit(1)
