"""Modal execution: one explicit H100, staged gates, persistent raw artifacts.

Run with the isolated local Modal CLI environment:
  modal run scripts/modal_runner.py --phase preflight
  modal run scripts/modal_runner.py --phase correctness --run-id RUN_ID
  modal run scripts/modal_runner.py --phase workload --run-id RUN_ID
  modal run scripts/modal_runner.py --phase benchmark --run-id RUN_ID --trial 0
  modal run scripts/modal_runner.py --phase profile --run-id RUN_ID
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CPU_ONLY = os.environ.get("SUFFIX_MODAL_CPU_ONLY") == "1"
FUNCTION_TIMEOUT = int(os.environ.get("SUFFIX_MODAL_TIMEOUT", "14400"))
app = modal.App("sglang-suffix-reproduction")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts", create_if_missing=True)
model_cache = modal.Volume.from_name(
    "sglang-suffix-model-cache", create_if_missing=True
)

if rust_base := os.environ.get("SUFFIX_MODAL_RUST_BASE_IMAGE"):
    # This reverses only the known patch in an immutable project-owned image.
    # Mac checkouts and user work are never reset. Cargo artifacts stay cached.
    image = modal.Image.from_id(rust_base).run_commands(
        "test $(git -C /project/sglang rev-parse HEAD) = e00930c5489053f26d86b179cee0d087f846acbb",
        "git -C /project/sglang apply --reverse --check /project/patches/sglang-suffix.patch",
        "git -C /project/sglang apply --reverse /project/patches/sglang-suffix.patch",
    )
else:
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04", add_python="3.12"
        )
        .entrypoint([])
        .apt_install(
            "git", "curl", "build-essential", "pkg-config", "libssl-dev", "libnuma-dev"
        )
        .uv_pip_install(
            requirements=[str(ROOT / "configs/gpu-requirements.lock")],
            uv_version="0.11.18",
            extra_options="--require-hashes --only-binary=:all: --index-strategy unsafe-best-match",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh",
            "sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.92.0",
        )
        .env(
            {
                "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                "HF_HOME": "/model-cache/huggingface",
                "CUDA_VISIBLE_DEVICES": "0",
                "TORCH_CUDA_ARCH_LIST": "9.0",
                "CARGO_BUILD_JOBS": "4",
            }
        )
        .workdir("/project")
    )
    image = image.run_commands(
        "git clone --branch v0.5.21 --depth 1 https://github.com/sgl-project/sglang.git sglang",
        "git -C sglang switch -c codex/suffix-decoding",
        "test $(git -C sglang rev-parse HEAD) = e00930c5489053f26d86b179cee0d087f846acbb",
        "python -m pip install -v --no-deps --no-build-isolation -e ./sglang/python",
        "python -m pip check",
    )

image = image.env(
    {
        "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE": "4096",
        "SGLANG_CACHE_DIR": "/model-cache/kernels/sglang",
        "TRITON_CACHE_DIR": "/model-cache/kernels/triton",
        "TORCH_EXTENSIONS_DIR": "/model-cache/kernels/torch",
        "CUDA_CACHE_PATH": "/model-cache/kernels/cuda",
        "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE": "4096",
        "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE": "4096",
    }
)
for directory in ("configs", "native", "patches", "scripts", "tests"):
    image = image.add_local_dir(
        ROOT / directory,
        f"/project/{directory}",
        copy=True,
        ignore=[
            "build/**",
            "**/__pycache__/**",
            "**/*.so",
            "**/*.pyc",
            "model-api.json",
            "modal_runner.py",
        ],
    )
image = image.run_commands(
    "bash scripts/bootstrap.sh",
    "python -m cmake -S native -B native/build -DPython_EXECUTABLE=$(command -v python) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/project/native/suffix_native",
    "python -m cmake --build native/build -j 4",
    "PYTHONPATH=/project/native python -m unittest discover -s tests -v",
    "python -m pip check",
).env({"PYTHONPATH": "/project/native:/project/sglang/python"})


# Optional immutable development image; runtime verifies source fingerprints.
if prebuilt := os.environ.get("SUFFIX_MODAL_PREBUILT_IMAGE"):
    image = modal.Image.from_id(prebuilt)


@app.function(
    image=image,
    gpu=None if CPU_ONLY else "H100!",
    cpu=8,
    memory=65536,
    ephemeral_disk=512 * 1024,
    volumes={"/artifacts": artifacts, "/model-cache": model_cache},
    timeout=FUNCTION_TIMEOUT,
    startup_timeout=1800,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def execute(
    phase: str,
    run_id: str,
    trial: int = 0,
    image_id: str = "",
    expected_sources: dict = None,
    gpu_requested: bool = True,
    function_timeout: int = 14400,
    public_gate_limit: int = 0,
    trace_variant: str = "suffix",
    reference_run: str = "",
):
    import hashlib
    import subprocess
    import time

    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in run_id
    ):
        raise ValueError(
            "run_id must contain only letters, digits, hyphens or underscores"
        )
    run_dir = Path("/artifacts") / run_id
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    link = Path("/project/results")
    if not link.exists():
        link.symlink_to(results, target_is_directory=True)
    phase_key = f"{phase}-{trial}" if phase == "benchmark" else phase
    log = run_dir / f"{phase_key}.log"
    status_file = run_dir / f"{phase_key}-status.json"
    if status_file.exists():
        raise ValueError(
            f"Preserving completed phase {phase_key}; use a new run ID to repeat"
        )
    started = time.time()
    status = {"success": False, "phase": phase, "trial": trial, "run_id": run_id, "start_unix": started}
    try:
        with log.open("a", buffering=1) as f:

            def run(command):
                f.write("COMMAND: " + json.dumps(command) + "\n")
                f.flush()
                print("Running", command, flush=True)
                with subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd="/project", text=True, bufsize=1,
                ) as process:
                    for line in process.stdout:
                        f.write(line)
                        if line.startswith("PROGRESS: "):
                            print(line.rstrip(), flush=True)
                    code = process.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)

            if not expected_sources:
                raise ValueError("Source fingerprints are required")
            actual_sources = {
                name: hashlib.sha256((Path("/project") / name).read_bytes()).hexdigest()
                for name in expected_sources
            }
            status["source_sha256"] = actual_sources
            if actual_sources != expected_sources:
                raise ValueError(
                    "Prebuilt image source fingerprint mismatch; rebuild the image"
                )
            status["gpu_requested"] = gpu_requested
            status["function_timeout_seconds"] = function_timeout
            if phase == "cpu-validate":
                run(
                    [
                        "python",
                        "-c",
                        "import torch; from sglang.srt.server_args import ServerArgs; from sglang.srt.entrypoints.engine import Engine; from suffix_native.cache import SuffixDecodingCache; print(torch.__version__, torch.version.cuda, 'SGLang Engine import OK')",
                    ]
                )
                run(["python", "-m", "unittest", "discover", "-s", "tests", "-v"])
                run(["python", "-m", "pip", "check"])
                status["success"] = True
                status["log_tail"] = log.read_text()[-12000:]
                return_status = True
            else:
                return_status = False
            if return_status:
                return status
            if not gpu_requested:
                raise ValueError("CPU-only deployment supports only cpu-validate")
            run(["nvidia-smi", "-q"])
            run(
                [
                    "python",
                    "-c",
                    (
                        "import torch; from suffix_native.cache import SuffixDecodingCache; "
                        "assert torch.cuda.device_count()==1; "
                        "assert 'H100' in torch.cuda.get_device_name(0); "
                        "print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0)); "
                        "x=torch.randn(32,4096,device='cuda',dtype=torch.bfloat16); "
                        "print((x@x.T).shape); torch.cuda.synchronize(); "
                        "from sglang.srt.server_args import ServerArgs; print('SGLang import OK')"
                    ),
                ]
            )
            if phase == "preflight":
                run(["python", "-m", "pip", "freeze"])
            elif phase in ("correctness", "prepare"):
                for mode in ("ordinary", "ngram", "suffix"):
                    run(["python", "scripts/gpu_correctness.py", "--mode", mode])
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
                if phase == "prepare":
                    run(["python", "scripts/materialize_workload.py"])
            elif phase == "audit":
                # Numerical diagnostics are intentionally allowed after a failed
                # correctness gate; they never produce benchmark claims.
                for mode in ("ordinary", "ngram", "suffix"):
                    run(["python", "scripts/gpu_correctness.py", "--mode", mode, "--audit"])
                run(["python", "scripts/analyze_audit.py", str(results)])
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
            elif phase == "audit-frozen":
                # Validate the new target route without regenerating second
                # turns: timings must use the inputs checked by public-gate.
                checks = json.loads((results / "public-gate-summary.json").read_text())["checks"]
                current = [c for c in checks if c["shared_decode"] == 1]
                if ({c["mode"] for c in current} != {"ngram", "suffix"}
                        or any(not c["complete"] or c["mismatches"] for c in current)):
                    raise ValueError("Full public equality gate must pass before audit-frozen")
                frozen = Path("/project/configs/frozen-workload.jsonl").read_bytes()
                frozen_lock = json.loads(Path("/project/configs/source-lock.json").read_text())["frozen_workload"]
                if hashlib.sha256(frozen).hexdigest() != frozen_lock["sha256"]:
                    raise ValueError("Frozen public workload hash differs")
                if (results / "workload.jsonl").exists():
                    raise ValueError("Preserving existing workload")
                for mode in ("ordinary", "ngram", "suffix"):
                    run(["python", "scripts/gpu_correctness.py", "--mode", mode])
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
                for mode in ("ordinary", "ngram", "suffix"):
                    (results / f"correctness-{mode}.jsonl").rename(
                        results / f"correctness-plain-{mode}.jsonl"
                    )
                for mode in ("ordinary", "ngram", "suffix"):
                    run(["python", "scripts/gpu_correctness.py", "--mode", mode, "--audit"])
                run(["python", "scripts/analyze_audit.py", str(results)])
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
                (results / "workload.jsonl").write_bytes(frozen)
                (results / "workload.sha256").write_text(frozen_lock["sha256"] + "\n")
                status["frozen_workload_sha256"] = frozen_lock["sha256"]
            elif phase == "workload":
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
                run(["python", "scripts/materialize_workload.py"])
            elif phase == "benchmark":
                source_lock = json.loads(Path("/project/configs/source-lock.json").read_text())
                if source_lock["gpu_candidate"].get("unified_decode"):
                    checks = json.loads((results / "public-gate-summary.json").read_text())["checks"]
                    current = [c for c in checks if c["shared_decode"] == 1]
                    if ({c["mode"] for c in current} != {"ngram", "suffix"}
                            or any(not c["complete"] or c["mismatches"] for c in current)):
                        raise ValueError("Full public equality gate must pass before timings")
                    if hashlib.sha256((results / "workload.jsonl").read_bytes()).hexdigest() != source_lock["frozen_workload"]["sha256"]:
                        raise ValueError("Timings must use the public-gated frozen inputs")
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
                modes = ["ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local"]
                if not 0 <= trial < 5:
                    raise ValueError("trial must be 0..4")
                modes = modes[trial:] + modes[:trial]
                for mode in modes:
                    run(
                        [
                            "python",
                            "scripts/gpu_bench.py",
                            "--mode",
                            mode,
                            "--trial",
                            str(trial),
                        ]
                    )
                    if source_lock["gpu_candidate"].get("unified_decode"):
                        reference = [json.loads(line) for line in
                                     (results / "public-gate-ordinary-shared1.jsonl").read_text().splitlines()]
                        measured = [json.loads(line) for line in
                                    (results / "gpu" / f"{mode}-{trial}" / "requests.jsonl").read_text().splitlines()]
                        if len(reference) != 240 or len(measured) != 240:
                            raise ValueError("Incomplete timed mode or public reference")
                        mismatches = []
                        for a, b in zip(reference, measured):
                            if any(a[k] != b[k] for k in ("block", "index", "question_id")):
                                raise ValueError("Timed/public request identities differ")
                            if a["response"]["output_ids"] != b["response"]["output_ids"]:
                                mismatches.append(dict(block=b["block"], index=b["index"], question_id=b["question_id"]))
                        (results / f"benchmark-{mode}-{trial}-equality.json").write_text(
                            json.dumps(dict(reference="public-gate-ordinary-shared1.jsonl",
                                            requests=240, mismatches=mismatches), indent=2) + "\n"
                        )
                        if mismatches:
                            raise ValueError(f"Timed {mode} differs on {len(mismatches)} requests; stopping allocation")
                        print(f"PROGRESS: timed {mode} trial{trial} matches all240 public reference outputs", flush=True)
            elif phase == "width-probe":
                run(["python", "scripts/gpu_width_probe.py"])
            elif phase == "natural-trace":
                if trace_variant not in ("suffix", "suffix-fixed", "suffix-local"):
                    raise ValueError("Unknown suffix trace variant")
                if not reference_run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in reference_run):
                    raise ValueError("A validated reference run ID is required")
                reference_dir = Path("/artifacts") / reference_run / "results"
                checks = json.loads((reference_dir / "public-gate-summary.json").read_text())["checks"]
                current = [c for c in checks if c["shared_decode"] == 1]
                if ({c["mode"] for c in current} != {"ngram", "suffix"}
                        or any(not c["complete"] or c["mismatches"] for c in current)):
                    raise ValueError("Natural traces require a passing public gate")
                os.environ["SUFFIX_TRACE"] = str(results / "suffix-rounds.jsonl")
                os.environ["SUFFIX_FIXED"] = "1" if trace_variant == "suffix-fixed" else "0"
                os.environ["SUFFIX_CACHE_REQUESTS"] = "0" if trace_variant == "suffix-local" else "128"
                status["trace_variant"] = trace_variant
                status["reference_run"] = reference_run
                run(["python", "scripts/gpu_public_gate.py", "--mode", "suffix", "--shared", "1"])
                reference = [json.loads(line) for line in
                             (reference_dir / "public-gate-ordinary-shared1.jsonl").read_text().splitlines()]
                traced = [json.loads(line) for line in
                          (results / "public-gate-suffix-shared1.jsonl").read_text().splitlines()]
                if len(reference) != 240 or len(traced) != 240:
                    raise ValueError("Incomplete natural trace workload")
                mismatches = []
                for a, b in zip(reference, traced):
                    if any(a[k] != b[k] for k in ("block", "index", "question_id")):
                        raise ValueError("Traced/public request identities differ")
                    if a["response"]["output_ids"] != b["response"]["output_ids"]:
                        mismatches.append(dict(block=b["block"], index=b["index"], question_id=b["question_id"]))
                (results / "trace-equality.json").write_text(json.dumps(
                    dict(reference_run=reference_run, variant=trace_variant,
                         requests=240, mismatches=mismatches), indent=2) + "\n")
                if mismatches:
                    raise ValueError("Natural trace output differs from public reference")
            elif phase in ("public-probe", "public-gate"):
                variants = [(1, "ordinary"), (1, "ngram"), (1, "suffix")]
                if phase == "public-probe":
                    variants = [(0, "ordinary"), (0, "ngram")] + variants
                for shared, mode in variants:
                    run(["python", "scripts/gpu_public_gate.py", "--mode", mode,
                         "--shared", str(shared), "--limit", str(public_gate_limit)])
                command = ["python", "scripts/analyze_public_gate.py", str(results)]
                if phase == "public-gate" and public_gate_limit == 0:
                    command.append("--require-complete")
                run(command)
            elif phase == "profile":
                for mode in ("ordinary", "ngram", "suffix"):
                    run(
                        [
                            "python",
                            "scripts/gpu_bench.py",
                            "--mode",
                            mode,
                            "--trial",
                            "0",
                            "--profile",
                        ]
                    )
            elif phase == "analyze":
                run(["python", "scripts/analyze_gpu.py"])
            else:
                raise ValueError(f"Unknown phase: {phase}")
        status["success"] = True
    except Exception as exc:
        status.update(success=False, error=str(exc))
    finally:
        if not status["success"] and "error" not in status:
            status["error"] = "Interrupted before completion"
        status["elapsed_seconds"] = time.time() - started
        # Indicative list-price compute estimate, not an account invoice.
        status["compute_estimate_usd"] = status["elapsed_seconds"] * (
            (0.001097 if gpu_requested else 0) + 8 * 0.0000131 + 64 * 0.00000222
        )
        status["pricing_checked"] = "2026-10-04"
        status["image_id"] = image_id
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        artifacts.commit()
        model_cache.commit()
    status["log_tail"] = log.read_text()[-12000:]
    return status


@app.local_entrypoint()
def main(phase: str = "preflight", run_id: str = "", trial: int = 0, limit: int = 0,
         variant: str = "suffix", reference_run: str = ""):
    import datetime
    import hashlib

    run_id = run_id or datetime.datetime.now(datetime.timezone.utc).strftime(
        "modal-%Y%m%d-%H%M%S"
    )
    if (phase == "cpu-validate") != CPU_ONLY:
        raise ValueError("Set SUFFIX_MODAL_CPU_ONLY=1 only for cpu-validate")
    source_names = [
        "configs/source-lock.json",
        "configs/gpu-requirements.lock",
        "patches/sglang-suffix.patch",
        "native/suffix_native/cache.py",
    ]
    if (ROOT / "configs/frozen-workload.jsonl").exists():
        source_names.append("configs/frozen-workload.jsonl")
    source_names += [
        "sglang/" + line.split(" b/", 1)[1]
        for line in (ROOT / "patches/sglang-suffix.patch").read_text().splitlines()
        if line.startswith("diff --git ")
    ]
    source_names += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "scripts").glob("*.py"))
        if p.name != "modal_runner.py"
    ]
    expected = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_names
    }
    output = execute.remote(
        phase, run_id, trial, image.object_id, expected, not CPU_ONLY, FUNCTION_TIMEOUT,
        limit, variant, reference_run
    )
    print(json.dumps(output, indent=2))
    destination = ROOT / "results/modal" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{phase}-{trial}-status.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    if not output["success"]:
        raise SystemExit(1)
