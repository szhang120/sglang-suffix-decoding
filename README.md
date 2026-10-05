# SuffixDecoding in SGLang

This implementation uses ArcticInference's CPU suffix trees to propose tokens. SGLang verifies the proposals with the target model. Decoding is linear and greedy, with a batch size of 1. The benchmarks use Qwen2.5-7B-Instruct on one H100 80GB.

## Setup

The CPU checks run on macOS and Linux. Install Python 3.12, Git and a C++20 compiler. Then run:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/bootstrap.sh
bash scripts/build_native.sh
```

The bootstrap script creates separate SGLang and ArcticInference checkouts at fixed commits. It preserves existing changes.

To repeat the GPU benchmarks on Modal, run:

```sh
python -m pip install modal==1.6.1
modal setup
SUFFIX_PORTABLE_CPU_ONLY=1 SUFFIX_MODAL_CPU_ONLY=1 \
  modal run scripts/modal_reproduce.py --run-id cpu-check --dry-run
modal run --detach scripts/modal_reproduce.py \
  --run-id reproduction-$(date +%Y%m%d-%H%M%S) --submit-only
```

The image uses public source code at fixed versions. The GPU function uses one H100. Use a new run ID for each run. Check progress with `modal app list`. Replace `RUN_ID` below with the submitted run ID, then download the artifacts:

```sh
mkdir -p results/runs/download
modal volume get sglang-suffix-artifacts RUN_ID results/runs/download
```

A Linux H100 host needs Ubuntu 24.04, Python 3.12 and CUDA 13.0.3. It also needs a compatible NVIDIA driver, Rust 1.92 and C++20 tools. Run:

```sh
bash scripts/gpu_setup.sh
python scripts/reproduce_gpu.py --output-dir results/runs/new-run
```

## Implementation

- The local cache stores the prompt and committed output. The global cache stores up to 128 committed responses in first-in, first-out order. Completion removes the local cache. An abort removes the response; a flush clears both caches.
- The author CPU code matches recent tokens and scores continuations by occurrence counts. The draft limit is `min(32, remaining_output - 1, match_length)`. The probability threshold is 0.1.
- SGLang verifies the last emitted token and the draft tokens. It emits the accepted prefix and a correction or bonus token. It keeps key and value (KV) cache slots for accepted inputs. The correction or bonus token remains pending until the next pass. Only committed output enters the suffix caches.

This implementation does not support batching, sampling, CUDA graphs, overlap, grammar constraints or logprob requests. It uses no draft model or training.

<!-- BEGIN MEASUREMENTS -->

## Results

The model is Qwen2.5-7B-Instruct on one H100 80GB. Decoding is greedy, with a batch size of 1. Weights and activations use BF16. The output head produces FP32 values.

All modes use deterministic Triton attention, eager execution and seed 42. The output limit is 256 tokens. CUDA graphs, overlap and radix caching are disabled.

Five trials × five modes × 240 requests = 6,000 measurements. The mode order rotates between trials. Inputs include 52 initial prompts and 32 follow-ups. Each block has 52 independent, 84 first and follow-up, or 104 repeated requests. Each block starts with empty algorithm caches.

Speedup is total ordinary request latency divided by total mode latency. It includes host and streaming overhead. Values above 1 mean faster execution. Brackets show 95% paired bootstrap intervals from 10,000 trial resamples with seed 42. The intervals describe timing variation on this workload.

| Mode | Independent | First + follow-up | Follow-up only | Repeated |
|---|---:|---:|---:|---:|
| Adaptive dual cache | 0.999× [0.983–1.025] | 1.113× [1.086–1.141] | 1.280× [1.250–1.310] | 1.902× [1.859–1.947] |
| Without match-length bound | 1.055× [1.032–1.079] | 1.189× [1.160–1.219] | 1.393× [1.358–1.430] | 1.993× [1.915–2.066] |
| Local cache only | 0.977× [0.950–1.005] | 1.063× [1.047–1.078] | 1.225× [1.207–1.244] | 0.970× [0.952–0.993] |

The repeated block includes the first and second passes of each identical prompt. This test gives favorable conditions for cache reuse. Without the match-length bound, probability, available continuations and the output limit still constrain proposals.

| Adaptive SUFFIX vs ordinary | Result |
|---|---|
| Independent | Inconclusive |
| Follow-up only | Faster |
| Repeated | Faster |

Each ablation removes one component. Ratios above 1 favor the adaptive method with both caches.

| Block | Adaptive / without match-length bound | Dual cache / local only |
|---|---:|---:|
| Independent | 0.947× [0.915–0.985] | 1.023× [0.996–1.046] |
| First + follow-up | 0.936× [0.932–0.940] | 1.048× [1.024–1.071] |
| Repeated | 0.954× [0.935–0.980] | 1.960× [1.935–1.985] |

Decoding with the adaptive bound is slower than decoding without it in all three blocks. Removing the bound can change the chosen candidate and its length. These tests do not identify which change caused the latency difference.

| Correctness check | Passing comparisons |
|---|---:|
| Ordinary, SUFFIX and suffix ablations in timed runs | 4,800 / 4,800 |
| Separate suffix traces | 720 / 720 |
| Direct verification and KV assertions | 416 / 416 |
| Controlled-width outputs | 54 / 54 |
| Attention route comparison outputs | 24 / 24 |
| Portable smoke outputs with empty and populated caches | 4 / 4 |

### NGRAM PROB

NGRAM output differs from ordinary output on 50 of 1,200 timed requests. The table gives descriptive latency ratios, not exact-output speedups.

| Block | Ordinary / NGRAM latency | Differing outputs |
|---|---:|---:|
| Independent | 0.965× | 15 |
| First + follow-up | 1.122× | 15 |
| Repeated | 1.938× | 20 |

NGRAM and SUFFIX use different cache policies. NGRAM can merge branches even when each suffix anchor has a fanout of 1. SUFFIX outputs must match ordinary outputs.

### GPU verification width

| Context tokens | Kernel time, 1 row | Kernel time, 33 rows | Reduction |
|---:|---:|---:|---:|
| 126 | 7.865 ms | 8.962 ms | 12.2% |
| 128 | 7.745 ms | 9.006 ms | 14.0% |
| 512 | 8.865 ms | 10.342 ms | 14.3% |

The table shows median sums of GPU kernel times from three profiling trials. The launch grids for KV storage and argmax become smaller. The attention and output-head grids stay the same. Kernel time includes profiler overhead. It does not measure request latency or operation counts.

The attention route comparison uses two prompts and six paired trials. Shared-route latency / original-route latency is 0.983× [0.952–1.019]. Values above 1 mean the shared route is slower. This test does not establish the baseline cost for other inputs.

<!-- END MEASUREMENTS -->

## Limitations

- The tests use Qwen in SGLang and a small public workload. The paper uses Llama in vLLM and agentic applications. These tests do not cover tree speculation or hybrid fallback.
- All modes use the same deterministic attention route and output-head memory-layout fix. The ordinary baseline differs from optimized upstream SGLang decoding.
- Matching outputs on this test set do not prove correctness for all inputs. Five trials do not establish performance on other workloads.
- Each workload block fits in the cache. The benchmarks do not test cache eviction or large caches. CPU proposal and cache costs were not measured separately.
- The public image passed CPU checks. The portable runner passed four GPU smoke requests. A separate full five-trial run of the public runner was not completed.

## Data and sources

Download the [raw measurements and SHA256 manifests](https://github.com/szhang120/sglang-suffix-decoding/releases/tag/v0.3.0-measured-reproduction). Each manifest lists file hashes for data verification.

The JSON files contain [timing summaries](results/final/benchmark-report.json), [proposal summaries](results/final/natural-trace-report.json), [GPU width measurements](results/final/width-probe-summary.json), [attention route comparisons](results/final/decode-control-report.json) and [execution records](results/final/execution-provenance.json).

Use the recorded [source versions](configs/source-lock.json), [runtime file hashes](configs/gpu-source-manifest.json) and [Linux dependency hashes](configs/gpu-requirements.lock). The source lock is unchanged from the benchmark execution.

[Paper](https://arxiv.org/abs/2411.04975v3) · [ArcticInference](https://github.com/snowflakedb/ArcticInference) · [SGLang](https://github.com/sgl-project/sglang) · [Spec-Bench](https://github.com/hemingkx/Spec-Bench) · [Attribution](NOTICE) · [License](LICENSE)
