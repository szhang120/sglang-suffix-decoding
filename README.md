# SuffixDecoding in SGLang

A reproducible **linear, greedy, batch-one** implementation using the author CPU suffix tree and SGLang target verification. **SUFFIX matches ordinary decoding on all 240 public-workload requests** in the retained v12 configuration. Initial GPU checks include 30 exact-ID cases and 416 verification/KV assertions. Controlled-width profiling shows some saved GPU work and substantial tile padding. A five-trial H100 campaign is running; no serving speedup is claimed yet. Earlier failures are preserved. Independent pristine controls and exact branch replay identify a numerical attention-layout effect in one NGRAM case. The upstream NGRAM baseline stays unchanged; any differing-output latency comparison will be labeled descriptive.

- [Project plan](PROJECT_PLAN.md): code-grounded design, invariants, milestones and 0–100 decision scores.
- [Technical report](docs/TECHNICAL_REPORT.md): inference/KV reasoning, local evidence and limits.
- [GPU runbook](docs/GPU_RUNBOOK.md): Modal and SSH execution workflows.
- [Portable GPU runner](scripts/reproduce_gpu.py): fresh gates and frozen-input trials, preserving existing results; [Modal entry point](scripts/modal_reproduce.py) builds public pinned sources without private image IDs.
- [Source lock](configs/source-lock.json) and [hashed Linux dependency lock](configs/gpu-requirements.lock).
- [Public audit release](https://github.com/szhang120/sglang-suffix-decoding/releases/tag/v0.1.0-audit): raw width traces, invalid broader run and exact source snapshots, with SHA256 manifests.
- [Passing correctness records and frozen workload](results/final/) and [controlled-width profile summaries](results/width-probe-final/).

## Local CPU verification

Python 3.12 and a C++20 compiler are required. On the development Mac, `.venv` already contains the pinned CPU dependencies and the native extension has been built.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/bootstrap.sh
bash scripts/build_native.sh
PYTHONPATH=native python scripts/cpu_profile.py
```

The bootstrap preserves modified checkouts. SGLang lives in `sglang/` on `codex/suffix-decoding`, based on v0.5.21 commit `e00930c5489053f26d86b179cee0d087f846acbb`. Author code lives separately in `reference/ArcticInference/`. Checkouts and native binaries are excluded from this root repository; the committed patch reconstructs the integration.

Current verification: ten CPU/host contract tests and six reporting-policy checks pass. The GPU runners compare ordinary decoding, NGRAM PROB and suffix decoding, record raw outputs/configurations and enforce the disclosed exact-output policy. The analysis also checks recorded proposer/ablation settings. See the runbook before renting a GPU.

## Generate on the configured Linux GPU

After the Linux dependency/native/SGLang setup in the runbook:

```sh
export PYTHONPATH="$PWD/native:$PWD/sglang/python:$PWD/scripts"
export SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=4096
python - <<'PY'
import sglang as sgl
from gpu_common import engine_config, sampling

engine = sgl.Engine(**engine_config("suffix"))
try:
    print(engine.generate(prompt="Explain causal attention briefly.",
                          sampling_params=sampling(128)))
finally:
    engine.shutdown()
PY
```

The intended scope is one greedy request, one CUDA GPU, dense Qwen, eager execution and the common deterministic Triton/FP32-head configuration. Unsupported batching, graphs, overlap, sampling/history penalties and grammar/logprob requests are rejected. This is a research adaptation with bounded caches, not a general production speculator.

The [independent attribution audit](https://github.com/szhang120/sglang-suffix-decoding/releases/tag/v0.2.0-attribution-audit) publishes selected operator tensors, exact branch Q/K/V and masks, source snapshots and per-file hashes. Earlier failures and the superseded trace analysis remain available.

## Sources and licensing

[Paper v3](https://arxiv.org/abs/2411.04975v3), [ArcticInference](https://github.com/snowflakedb/ArcticInference), [SGLang](https://github.com/sgl-project/sglang), and [Spec-Bench](https://github.com/hemingkx/Spec-Bench). Apache-2.0 source notices are retained; see [third-party provenance](THIRD_PARTY.md). This is an adapted SGLang reproduction project, with no dependency on upstream PR acceptance. Final serving results remain pending.
