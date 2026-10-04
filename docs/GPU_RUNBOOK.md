# Provisioning and execution

Modal authentication/payment setup is complete; net spend limit is$0. The final Triton/FP32-logit configuration passes30 exact-ID cases across both speculators and direct GPU KV checks. A stride-preserving head matmul removes a measured1.09GB weight-copy operation. No serving benchmark result is established yet. All earlier numerical and diagnostic failures are retained.

## Modal execution

Use Modal CLI 1.6.1 in the isolated Mac environment `/tmp/sglang-modal-cli`. Token setup stays outside this repository. The launcher requests one explicit `H100!`, 8 CPU cores, 64GiB RAM and 512GiB ephemeral disk (Modal’s minimum for an explicit request). Persistent named volumes hold model cache and raw artifacts. The current default timeout is four hours, with no automatic retries and scale-down after two seconds. `SUFFIX_MODAL_TIMEOUT` overrides that operational guard. The frozen v11 image source lock and original trial0 used a two-hour guard; later phase statuses record the actual timeout. Increasing headroom changes no inference settings. Decision score: **93/100**, based on the measured 22-minute ordinary baseline within a five-mode trial.

```sh
/tmp/sglang-modal-cli/bin/modal run scripts/modal_runner.py --phase preflight --run-id RUN_ID
/tmp/sglang-modal-cli/bin/modal run scripts/modal_runner.py --phase correctness --run-id RUN_ID
/tmp/sglang-modal-cli/bin/modal run scripts/modal_runner.py --phase workload --run-id RUN_ID
/tmp/sglang-modal-cli/bin/modal run scripts/modal_runner.py --phase benchmark --run-id RUN_ID --trial 0
```

Use phase `audit` for target-logit/acceptance/KV diagnostics after a failure, and `width-probe` for controlled row-count profiling. After downloading width artifacts, run `python scripts/analyze_width_probe.py PATH/width-probe` locally. The streaming reader selects CPU annotations and correlates their launches with GPU kernels; matching GPU annotation labels are excluded. Phase `prepare` combines plain correctness and workload freezing in one GPU allocation. Repeat benchmark trials1–4, then run phases `profile` and `analyze`. Each trial runs every mode sequentially on the same GPU. Commands refuse to overwrite completed phases. Download artifacts from volume `sglang-suffix-artifacts`, under RUN_ID. Create the destination directory before a recursive download; the CLI otherwise treats it as a single filename:

```sh
mkdir -p results/modal/RUN_ID/raw
/tmp/sglang-modal-cli/bin/modal volume get sglang-suffix-artifacts RUN_ID/results results/modal/RUN_ID/raw
```

The downloaded tree includes a `results/` subdirectory. Immutable project image IDs can accelerate development with `SUFFIX_MODAL_RUST_BASE_IMAGE`; portable builds do not depend on those IDs. Runtime compares exact source fingerprints before execution. Image builds happen before GPU execution; installation failures are retained in `results/setup/`. The first build caught a CUDA-tile downloader-stub hash in the resolver output. The direct NVIDIA CPython 3.12 wheel is now pinned through `configs/gpu-overrides.in`; its downloaded bytes match the NVIDIA index hash. Hash checking remains mandatory.

List-price compute estimate for the requested resources is approximately $4.84/hour, excluding image builds, storage and other billable usage; this is not an account invoice. See [Modal pricing](https://modal.com/pricing). No always-on deployment is created.

Public correctness/profile artifacts include the actual source snapshots from immutable images, not just image identifiers that require access to the original Modal workspace. `analysis/export_modal_sources.py` exports those files using CPU only and checks recorded source hashes. `analysis/package_artifacts.py` packages explicit evidence directories with normalized headers and a per-file SHA256 manifest. The first source-export helper attempt lacked a remote environment variable; its failed log is retained, and the repaired export passes every source-hash check.

## Steps on the Mac / rental dashboard

1. Choose a rental service with Linux SSH access. Select **one NVIDIA H100 80GB**, Ubuntu 24.04 x86_64, and preferably **200GB disk** (model, CUDA/PyTorch, native builds and traces). A 100GB disk may require managing caches. H100 fit score: **86/100**, because it matches the paper's single-GPU comparison hardware and has generous BF16/KV headroom. Exact availability and rental price depend on the provider; none is assumed here.
2. Choose a CUDA 13 development environment with an NVIDIA R580-or-newer host driver. CUDA 13.0.3's packaged Linux driver is 580.126.20; the host must satisfy the CUDA compatibility requirement. See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html). The “CUDA Version” in `nvidia-smi` describes driver support, not an installed toolkit.
3. Add an SSH public key in the provider dashboard. If you already have a suitable key, upload its `.pub` file. Otherwise create a separate key in a Mac terminal:

   ```sh
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_suffixdecoding
   cat ~/.ssh/id_ed25519_suffixdecoding.pub
   ```

   Enter a passphrase if desired. Paste only the public-key line into the provider dashboard. The private file stays on the Mac.
4. Start the host. Copy the provider's SSH command, including its username and port. Example format:

   ```sh
   ssh -i ~/.ssh/id_ed25519_suffixdecoding -p 12345 root@HOST
   ```

5. Give this chat that command and the key **path**, plus any prescribed remote workspace directory. Do not paste key contents, passwords or API secrets. A password-only host requires setting up key authentication first. I can then handle transfer, environment validation and execution.
6. Keep the rental running through correctness, benchmarks and result download. After the artifacts have been saved locally, stop/delete the instance in the provider dashboard to end billing. Confirm whether attached storage is billed separately.

## Remote environment gate

Before downloading model weights, capture `nvidia-smi -q`, `uname -a`, Python version, `nvcc --version`, compiler version and Rust version. Require exactly one visible GPU, Python 3.12, CUDA 13 development tools, C++20 support, Rust **1.92** (the pinned SGLang source toolchain), and sufficient disk/RAM. Build dependencies need outbound GitHub/PyPI/Hugging Face access. The selected model is public and ungated; no training occurs.

`configs/gpu-requirements.lock` pins **212 packages with hashes**, resolved for Ubuntu 24.04 glibc 2.39 / x86_64 / Python 3.12 with uv 0.11.18. Resolution disallows source builds and uses PyPI plus NVIDIA’s public wheel index, with an explicit NVIDIA CUDA-tile wheel to avoid the PyPI downloader stub. Installation requires binary wheels and checks hashes. Resolver success is not a binary runtime guarantee. Core upstream versions: PyTorch 2.13.0, FlashInfer 0.6.18, SGLang kernel 0.4.7, Transformers 5.12.1. Preserve upstream pins first, run `pip check`, then test import/model startup. Report any failure and revise the lock explicitly; do not silently substitute versions.

Transfer the root project and bootstrap source checkouts remotely. Exclude `.venv`, `native/build`, compiled Mac `.so` files and `.git` directories from a file transfer. `scripts/bootstrap.sh` recreates exact source commits and applies the integration patch while preserving pre-existing checkouts. An alternative is cloning this repository once it has been published. Author reference checkout remains separate; vLLM is not installed.

From the project directory in a fresh Python 3.12 virtual environment:

```sh
bash scripts/gpu_setup.sh
bash scripts/run_gpu.sh
```

Setup assumes CUDA/compiler/Rust are already available. Package installs use the hash lock; native suffix tests precede an editable SGLang build. SGLang's Rust workspace carries its own Cargo lock. Save any compilation logs if this gate fails.

## Execution and evidence

`run_gpu.sh` runs real SGLang engines in separate processes. First, length caps 1/2/3/17/33/65/128, EOS, stop strings, cold/warm cache, repeated requests and cache flush must match ordinary output token IDs exactly. A failure stops execution before timings. These cases exercise later-token predictions after rejection and request reuse, providing evidence about KV correctness; they do not replace a direct KV audit if a mismatch appears.

Materialize one frozen workload: tokenizer IDs and second-turn inputs include assistant responses generated once by ordinary decoding. Every timed mode receives the same IDs. Truncations are recorded. Workload hash, source lock, resolved settings, installed package freeze and complete NVIDIA information are saved alongside results.

Five trials rotate ordinary, linear NGRAM PROB (breadth1), suffix, unbounded-match-length suffix and local-only suffix modes. All use graphs/overlap/radix cache disabled; each workload block starts with a cache reset. Independent, refinement and repeated-identical-prompt blocks are reported separately. Repetition is a diagnostic upper bound and must never be presented as an agent benchmark.

Separate profile runs add per-round suffix traces and SGLang's real scheduler CPU/GPU profiler with tensor shapes. Inspect projection/MLP GEMM row counts and attention query lengths against `verify_rows`. Compare GPU event time across naturally observed lengths with matched context lengths. Inspect CPU draft, host transfers, allocation and launch overhead too. Shorter tensors establish a different execution shape; only traces/timings establish how much GPU work/time it saves. If natural widths are too narrow, add a documented controlled-width profiling experiment before drawing that conclusion.

Streaming request timings include host/tokenization output handling. Record TTFT, total wall time, tokens/sec and per-trial aggregate speedup. Prefer paired trial summaries and uncertainty ranges; do not mix profiler runs into timing results. Root analysis script refuses output-ID divergence and refuses to produce a summary without real measurements.

Download `results/gpu`, correctness files, `workload.jsonl`, its hash and environment logs. Inspect logs for unintended GPU contention, thermal/clocks changes and errors. Update the technical report and plan with results, including failures. Preserve raw artifacts; the `.gitignore` excludes bulky GPU traces so they can be published as release artifacts rather than accidentally committed.
