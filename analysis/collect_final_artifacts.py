"""Collect completed measured evidence; optionally publish the public release.

No GPU allocation. Publication requires unchanged HEAD and documentation from
launch, preserving concurrent user work. All archive members are hash-checked.
"""

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="modal-20261004-final-analysis")
    parser.add_argument("--modal", default="/tmp/sglang-modal-cli/bin/modal")
    parser.add_argument("--gh", default="/opt/homebrew/bin/gh")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert args.run_id and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id)
    working_files = ("PROJECT_PLAN.md", "README.md", "docs/TECHNICAL_REPORT.md",
                     "results/local-verification.json")
    initial = {name: digest(root / name) for name in working_files}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    destination = root / "results/gpu/final-download" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status_path = destination / "polled-status.json"
    print("Waiting for completed measurements/diagnostics/analysis; no GPU allocated", flush=True)
    while True:
        result = subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                                 f"{args.run_id}/finalize-status.json", str(status_path), "--force"],
                                cwd=root, capture_output=True, text=True)
        if result.returncode == 0:
            status = json.loads(status_path.read_text())
            if not status["success"]:
                raise RuntimeError(f"Final analysis failed; artifacts preserved: {status.get('error')}")
            break
        error = result.stderr + result.stdout
        if not any(message in error.lower() for message in ("not found", "does not exist", "no such file or directory")):
            raise RuntimeError(f"Cannot poll completion; inspect running Modal apps: {error}")
        if time.monotonic() - start > 18 * 3600:
            raise TimeoutError("No completed final analysis after 18 hours; inspect existing apps")
        time.sleep(30)
    raw = destination / "raw"
    raw.mkdir()
    subprocess.run([args.modal, "volume", "get", "sglang-suffix-artifacts",
                    args.run_id, str(raw)], cwd=root, check=True)
    found = list(raw.rglob("finalize-status.json"))
    assert len(found) == 1
    completed = found[0].parent
    reports = completed / "results/final"
    archives = sorted((completed / "archives").glob("*.tar.gz"))
    assert len(archives) == 9
    for archive in archives:
        manifest = json.loads((reports / f"{archive.name.removesuffix('.tar.gz')}-manifest.json").read_text())
        assert manifest["sha256"] == digest(archive) and manifest["bytes"] == archive.stat().st_size
        with tarfile.open(archive, "r:gz") as contents:
            seen = set()
            for member in contents:
                assert member.isfile() and member.name in manifest["files"] and member.name not in seen
                assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
                expected = manifest["files"][member.name]
                assert member.size == expected["bytes"]
                with contents.extractfile(member) as f:
                    assert hashlib.file_digest(f, "sha256").hexdigest() == expected["sha256"], member.name
                seen.add(member.name)
            assert seen == set(manifest["files"])
    print("Verified every archive and included raw-file hash", flush=True)
    # Model/runtime hashes remain frozen; analysis sources must be the reviewed
    # local versions that generated the extended second-turn reports.
    for name, sha in status["analysis_source_sha256"].items():
        path = root / ("configs" if name.endswith(".txt") else "analysis") / name
        assert digest(path) == sha, f"Analysis source changed: {name}"
    benchmark = json.loads((reports / "benchmark-report.json").read_text())
    assert benchmark["exact_suffix_ids_passed"] and benchmark["measured_requests"] == 6000
    assert len(list((completed / "docs/figures").glob("*.png"))) == 2
    import shutil

    # Historical initial-gate fixtures already live here; preserve them and
    # refuse collisions with any existing measured report.
    final_dir = root / "results/final"
    final_dir.mkdir(exist_ok=True)
    for report in reports.iterdir():
        target = final_dir / report.name
        assert report.is_file() and not target.exists(), f"Preserving {target}"
        shutil.copyfile(report, target)
    shutil.copyfile(found[0], root / "results/final/finalize-status.json")
    assert not (root / "docs/RESULTS.md").exists(), "Preserving an existing results document"
    shutil.copyfile(completed / "docs/RESULTS.md", root / "docs/RESULTS.md")
    for figure in (completed / "docs/figures").iterdir():
        target = root / "docs/figures" / figure.name
        assert not target.exists(), f"Preserving {target}"
        shutil.copyfile(figure, target)
    print("Collected measured report, figures and provenance", flush=True)
    if not args.publish:
        return
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == head, "HEAD changed; review collected evidence before publication"
    assert {name: digest(root / name) for name in working_files} == initial, "Documentation changed; preserving concurrent work"
    assert not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True), "Preserving staged work"
    ci = json.loads(subprocess.check_output([
        args.gh, "run", "list", "--repo", "szhang120/sglang-suffix-decoding", "--commit", head,
        "--json", "status,conclusion,url", "--limit", "10"], cwd=root, text=True))
    assert ci and all(r["status"] == "completed" and r["conclusion"] == "success" for r in ci), "Source checkpoint CI has not passed"
    ngram = len(benchmark["mismatches"])
    p = root / "README.md"
    text = p.read_text()
    text = text.replace("A five-trial H100 campaign is running; no serving speedup is claimed yet.",
                        "The five-trial H100 campaign is complete: all 4,800 ordinary/SUFFIX/ablation outputs match exactly. See [measured results](docs/RESULTS.md) for serving ratios, intervals, actual second turns, ablations and negative results.")
    text = text.replace("any differing-output latency comparison will be labeled descriptive.",
                        f"its {ngram}/1,200 differing timed responses are reported descriptively.")
    text = text.replace("Final serving results remain pending.", "Final measured results and raw artifacts are published.")
    p.write_text(text)
    p = root / "docs/TECHNICAL_REPORT.md"
    text = p.read_text()
    text = text.replace("Five serving trials are running.", "Five serving trials and separate diagnostics are complete; see [measured results](RESULTS.md).")
    text = text.replace("The final profile uses contexts", "The v11 post-head-fix profile uses contexts")
    text = text.replace("**Performance behavior is still unresolved.**", "**The completed v12 experiment is reported in [measured results](RESULTS.md).** The v11 diagnostics here are retained as history.")
    text = text.replace("Serving results must distinguish", "Serving results distinguish")
    p.write_text(text)
    p = root / "PROJECT_PLAN.md"
    text = p.read_text().replace("A single detached H100 campaign now runs five rotated trials.",
                                "Five serving trials and all separate diagnostic stages have completed. The gated measured report, figures and raw artifacts are published.")
    text = text.replace("7. **Running:**", "7. **Completed:**")
    text = text.replace("**Still required:** five serving trials, final-candidate width and baseline-cost controls, full-workload acceptance analysis and final report including negative results and the distinction from the paper's Llama/vLLM experimental setup.",
                        "**Completed:** five serving trials, final-candidate width and baseline-cost controls, full-workload acceptance analysis and final measured report including negative results and the distinction from the paper's original experimental setup. See `docs/RESULTS.md` and `results/final/`.")
    p.write_text(text)
    p = root / "results/local-verification.json"
    verification = json.loads(p.read_text())
    verification["latest_ci"] = dict(commit=head, linux_and_macos="passed", url=ci[0]["url"])
    verification["independent_reassessment"]["serving_benchmarks"] = dict(
        status="complete", measured_requests=6000, exact_ordinary_suffix_ablation_requests=4800,
        ngram_differing_responses=ngram, campaign_run=status["campaign_run"],
        diagnostics_run=status["diagnostics_run"], final_analysis_run=status["run_id"],
        performance_report="docs/RESULTS.md", raw_artifact_manifests="results/final")
    p.write_text(json.dumps(verification, indent=2) + "\n")
    paths = [*working_files, "docs/RESULTS.md", "results/final"]
    paths += [str(path.relative_to(root)) for path in (root / "docs/figures").iterdir()
              if path.name.startswith(("verify-width-v12.", "benchmark-speedup."))]
    subprocess.run(["git", "add", "--", *paths], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "Publish complete five-trial measured reproduction and limitations"], cwd=root, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=root, check=True)
    release_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    notes = destination / "release-notes.md"
    notes.write_text("Five controlled H100 trials completed: all 4,800 ordinary/SUFFIX/ablation outputs match exactly. "
                     f"Upstream NGRAM differs on {ngram}/1,200 measured responses and is reported descriptively. "
                     "Separate artifacts contain 720 full-workload suffix traces, five profiles, 24 route-control requests and 54 width probes. "
                     "Every archive has a per-file SHA256 manifest. No model weights or credentials are included. "
                     "See docs/RESULTS.md and docs/TECHNICAL_REPORT.md at this release commit for all measured ratios, intervals, negative results, source pins and limitations.\n")
    manifests = sorted(reports.glob("*-manifest.json"))
    subprocess.run([args.gh, "release", "create", "v0.3.0-measured-reproduction", *map(str, archives), *map(str, manifests),
                    "--repo", "szhang120/sglang-suffix-decoding", "--target", release_head,
                    "--title", "Measured SGLang SuffixDecoding reproduction", "--notes-file", str(notes)], cwd=root, check=True)
    print("Published measured write-up and all raw benchmark/profile assets", flush=True)


if __name__ == "__main__":
    main()
