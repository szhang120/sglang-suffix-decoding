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
    working_files = ("README.md", "NOTICE", "configs/gpu-source-manifest.json")
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
    assert len(archives) == 10
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
    provenance = json.loads((reports / "execution-provenance.json").read_text())
    smoke = provenance["diagnostics"]["portable_runner_smoke"]
    assert smoke["success"] and smoke["smoke_only"] and smoke["smoke_requests"] == 4
    assert smoke["runner_sha256"] == digest(root / "scripts/reproduce_gpu.py"), "Portable runner source changed"
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
    for figure in (completed / "docs/figures").iterdir():
        target = root / "docs/figures" / figure.name
        assert not target.exists(), f"Preserving {target}"
        shutil.copyfile(figure, target)
    print("Collected measured data, figures and provenance", flush=True)
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
    # Keep the README human-reviewed. Collection never creates project plans,
    # status pages or narrative reports in the public checkout.
    paths = ["results/final"]
    paths += [str(path.relative_to(root)) for path in (root / "docs/figures").iterdir()
              if path.name.startswith(("verify-width-v12.", "benchmark-speedup."))]
    subprocess.run(["git", "add", "--", *paths], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "Publish verified benchmark data and figures"], cwd=root, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=root, check=True)
    release_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    notes = destination / "release-notes.md"
    notes.write_text("Five controlled H100 trials completed: all 4,800 ordinary/SUFFIX/ablation outputs match exactly. "
                     f"Upstream NGRAM differs on {ngram}/1,200 measured responses and is reported descriptively. "
                     "Separate artifacts contain 720 full-workload suffix traces, five profiles, 24 route-control requests, 54 width probes and four portable-runner smoke requests. "
                     "Every archive has a per-file SHA256 manifest. No model weights or credentials are included. "
                     "See README.md at this release commit for measurements, configuration and limitations.\n")
    manifests = sorted(reports.glob("*-manifest.json"))
    subprocess.run([args.gh, "release", "create", "v0.3.0-measured-reproduction", *map(str, archives), *map(str, manifests),
                    "--repo", "szhang120/sglang-suffix-decoding", "--target", release_head,
                    "--title", "Measured SGLang SuffixDecoding reproduction", "--notes-file", str(notes)], cwd=root, check=True)
    print("Published verified benchmark/profile assets", flush=True)


if __name__ == "__main__":
    main()
