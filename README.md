# SuffixDecoding in SGLang

A linear, greedy, batch-one implementation using ArcticInference's CPU suffix trees and SGLang target verification. Benchmarked with Qwen2.5-7B-Instruct on one H100 80GB.

## Setup

CPU checks require Python 3.12, Git and a C++20 compiler. They run on macOS and Linux.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/bootstrap.sh
bash scripts/build_native.sh
```

Bootstrap preserves existing changes and creates pinned SGLang/ArcticInference checkouts.

For GPU reproduction on Modal:

```sh
python -m pip install modal==1.6.1
modal setup
SUFFIX_PORTABLE_CPU_ONLY=1 SUFFIX_MODAL_CPU_ONLY=1 \
  modal run scripts/modal_reproduce.py --run-id cpu-check --dry-run
modal run --detach scripts/modal_reproduce.py \
  --run-id reproduction-$(date +%Y%m%d-%H%M%S) --submit-only
```

The image builds public pinned sources and uses one H100. Use fresh run IDs. Check with `modal app list`; download using the submitted ID:

```sh
mkdir -p results/runs/download
modal volume get sglang-suffix-artifacts RUN_ID results/runs/download
```

Existing Linux H100: Ubuntu 24.04, Python 3.12, CUDA 13.0.3, compatible driver, Rust 1.92 and C++20 tools:

```sh
bash scripts/gpu_setup.sh
python scripts/reproduce_gpu.py --output-dir results/runs/new-run
```

## Implementation

- Local cache: prompt and committed output. Global cache: 128 committed responses, FIFO. Completion frees local state; abort removes the response; flush clears both.
- Author CPU code matches recent tokens and scores continuations by occurrence counts. Draft cap: `min(32, remaining_output - 1, match_length)`; probability threshold: 0.1.
- SGLang verifies the pending token plus drafts, accepts the matching prefix and emits a correction/bonus. Only accepted KV slots are retained; the correction/bonus remains pending. Caches receive committed output only.

Unsupported: batching, sampling, graphs, overlap, grammar and logprobs. No draft model or training.

<!-- BEGIN MEASUREMENTS -->

## Results

Qwen2.5-7B-Instruct; one H100 80GB; greedy batch 1; BF16 with an FP32 output head.

Deterministic Triton attention; eager execution; seed 42; output limit 256. Graphs, overlap and radix caching disabled.

Five rotated trials × five modes × 240 requests = 6,000 measurements. Inputs: 52 initial prompts and 32 follow-ups. Blocks: 52 independent, 84 first/follow-up, 104 repeated requests; caches start empty per block.

Speedup = ordinary/mode summed request latency, including host/streaming overhead. Above 1 is faster. Brackets: 95% paired bootstrap intervals, 10,000 trial resamples, seed 42; fixed-workload timing variation.

| Mode | Independent | First + follow-up | Follow-up only | Repeated |
|---|---:|---:|---:|---:|
| Adaptive dual cache | 0.999× [0.983–1.025] | 1.113× [1.086–1.141] | 1.280× [1.250–1.310] | 1.902× [1.859–1.947] |
| Without match-length bound | 1.055× [1.032–1.079] | 1.189× [1.160–1.219] | 1.393× [1.358–1.430] | 1.993× [1.915–2.066] |
| Local cache only | 0.977× [0.950–1.005] | 1.063× [1.047–1.078] | 1.225× [1.207–1.244] | 0.970× [0.952–0.993] |

Repeated includes both identical-prompt passes, a favorable reuse test. The bound-removal variant still uses probability, available-continuation and output-budget limits.

| Adaptive SUFFIX vs ordinary | Result |
|---|---|
| Independent | Inconclusive |
| Follow-up only | Faster |
| Repeated | Faster |

Ablation ratios: above 1 favors adaptive dual cache.

| Block | Adaptive / without match-length bound | Dual cache / local only |
|---|---:|---:|
| Independent | 0.947× [0.915–0.985] | 1.023× [0.996–1.046] |
| First + follow-up | 0.936× [0.932–0.940] | 1.048× [1.024–1.071] |
| Repeated | 0.954× [0.935–0.980] | 1.960× [1.935–1.985] |

Adaptive bound vs removal (independent / first + follow-up / repeated): slower / slower / slower. Candidate selection also changes; the cause is not isolated.

| Correctness check | Passing comparisons |
|---|---:|
| Ordinary, SUFFIX and suffix ablations in timed runs | 4,800 / 4,800 |
| Separate suffix traces | 720 / 720 |
| Direct verification/KV assertions | 416 / 416 |
| Controlled-width outputs | 54 / 54 |
| Ordinary-route outputs | 24 / 24 |
| Portable cold/warm smoke outputs | 4 / 4 |

### NGRAM PROB

NGRAM differs from ordinary output on 50/1,200 timed requests. The ratios below are descriptive latency ratios, not exact-output speedups.

| Block | Ordinary / NGRAM latency | Differing outputs |
|---|---:|---:|
| Independent | 0.965× | 15 |
| First + follow-up | 1.122× | 15 |
| Repeated | 1.938× | 20 |

NGRAM uses different caches and can branch at fanout 1. SUFFIX output checks remain strict.

### GPU verification width

| Context tokens | One-row kernel sum | 33-row kernel sum | Reduction |
|---:|---:|---:|---:|
| 126 | 7.865 ms | 8.962 ms | 12.2% |
| 128 | 7.745 ms | 9.006 ms | 14.0% |
| 512 | 8.865 ms | 10.342 ms | 14.3% |

Three profiling trials. KV-store/argmax grids shrink; attention/head grids remain unchanged. Profiled kernel sums are neither request latency nor FLOPs.

Two-prompt, six-pair route control: shared/original latency 0.983× [0.952–1.019]; above 1 means the shared route is slower. General baseline cost remains unestablished.

<!-- END MEASUREMENTS -->

## Limitations

- This uses Qwen/SGLang and a small public workload, rather than the paper's Llama/vLLM and live agentic applications. Tree speculation and hybrid fallback are untested.
- The ordinary baseline shares the deterministic attention route and head-stride fix with speculative modes; it is not optimized upstream ordinary decoding.
- Exact outputs on the tested suite do not prove arbitrary-input correctness. Five trials do not establish workload generalization.
- The cache fits every measured block; eviction pressure and large-cache scalability were not benchmarked. CPU proposal/cache costs were not separately isolated.
- Public-image CPU checks and four GPU smoke requests passed. The complete standalone public five-trial runner was not separately rerun.

## Data and sources

[Raw measurements and SHA256 manifests](https://github.com/szhang120/sglang-suffix-decoding/releases/tag/v0.3.0-measured-reproduction). Saved JSON: [serving](results/final/benchmark-report.json), [proposal traces](results/final/natural-trace-report.json), [GPU widths](results/final/width-probe-summary.json), [route control](results/final/decode-control-report.json) and [runtime provenance](results/final/execution-provenance.json).

[Source versions](configs/source-lock.json), [runtime fingerprints](configs/gpu-source-manifest.json) and [hashed Linux dependencies](configs/gpu-requirements.lock). The source lock is the byte-identical measured-runtime snapshot.

[Paper](https://arxiv.org/abs/2411.04975v3) · [ArcticInference](https://github.com/snowflakedb/ArcticInference) · [SGLang](https://github.com/sgl-project/sglang) · [Spec-Bench](https://github.com/hemingkx/Spec-Bench) · [Attribution](NOTICE) · [License](LICENSE)
