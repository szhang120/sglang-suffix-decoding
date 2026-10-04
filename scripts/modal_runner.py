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
    timeout=7200,
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
    status = {"phase": phase, "trial": trial, "run_id": run_id, "start_unix": started}
    try:
        with log.open("a", buffering=1) as f:

            def run(command):
                f.write("COMMAND: " + json.dumps(command) + "\n")
                f.flush()
                print("Running", command, flush=True)
                subprocess.run(
                    command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd="/project",
                    check=True,
                )

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
            elif phase == "correctness":
                for mode in ("ordinary", "ngram", "suffix"):
                    run(["python", "scripts/gpu_correctness.py", "--mode", mode])
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
            elif phase == "audit":
                # Numerical diagnostics are intentionally allowed after a failed
                # correctness gate; they never produce benchmark claims.
                for mode in ("ordinary", "ngram", "suffix"):
                    run(["python", "scripts/gpu_correctness.py", "--mode", mode, "--audit"])
                run(["python", "scripts/analyze_audit.py", str(results)])
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
            elif phase == "workload":
                run(["python", "scripts/analyze_gpu.py", "--correctness-only"])
                run(["python", "scripts/materialize_workload.py"])
            elif phase == "benchmark":
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
            elif phase == "width-probe":
                run(["python", "scripts/gpu_width_probe.py"])
                run(["python", "scripts/analyze_width_probe.py", str(results / "width-probe")])
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
def main(phase: str = "preflight", run_id: str = "", trial: int = 0):
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
        "sglang/python/sglang/srt/speculative/suffix_worker.py",
        "sglang/python/sglang/srt/speculative/ngram_worker.py",
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
        phase, run_id, trial, image.object_id, expected, not CPU_ONLY
    )
    print(json.dumps(output, indent=2))
    destination = ROOT / "results/modal" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{phase}-{trial}-status.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    if not output["success"]:
        raise SystemExit(1)
