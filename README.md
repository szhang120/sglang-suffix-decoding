# SuffixDecoding in SGLang

SuffixDecoding reuses cached token sequences to propose several next tokens at once. This implementation uses ArcticInference's CPU suffix trees for proposals. SGLang runs the target model to verify them. It implements the paper's linear, greedy variant with a batch size of 1. The benchmarks use Qwen2.5-7B-Instruct on one H100 80GB.

## Setup

First, build the CPU suffix cache and run its checks on macOS or Linux. Install Python 3.12, Git and a C++20 compiler. Run these commands from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/bootstrap.sh
bash scripts/build_native.sh
```

The bootstrap script creates separate SGLang and ArcticInference checkouts at fixed commits, then applies the SGLang patch. It preserves existing changes. The build script compiles the suffix cache and runs the CPU tests.

Next, repeat the GPU benchmarks on Modal. The following commands install the Modal client, authenticate your account and check the public image on CPU. The last command submits the GPU run:

```sh
python -m pip install modal==1.6.1
modal setup
SUFFIX_PORTABLE_CPU_ONLY=1 SUFFIX_MODAL_CPU_ONLY=1 \
  modal run scripts/modal_reproduce.py --run-id cpu-check --dry-run
modal run --detach scripts/modal_reproduce.py \
  --run-id reproduction-$(date +%Y%m%d-%H%M%S) --submit-only
```

The image uses public source code at fixed versions. Each GPU run uses one H100. Use a new run ID for each run. Check the app status with `modal app list`. After the run completes, download its artifacts. Replace `RUN_ID` below with the submitted run ID:

```sh
mkdir -p results/runs/download
modal volume get sglang-suffix-artifacts RUN_ID results/runs/download
```

You can also run the benchmarks directly on a Linux H100 host. This host needs Ubuntu 24.04, Python 3.12 and CUDA 13.0.3. It also needs a compatible NVIDIA driver, Rust 1.92 and C++20 tools. Install the GPU dependencies and start the run:

```sh
bash scripts/gpu_setup.sh
python scripts/reproduce_gpu.py --output-dir results/runs/new-run
```

## Implementation

Two suffix caches provide the token sequences for reuse. The local cache stores the current prompt and committed output. The global cache stores committed output from up to 128 requests. It removes the oldest response first when it reaches this limit.

The ArcticInference CPU code matches the end of the current token sequence against both caches. It estimates continuation probabilities from occurrence counts and selects the continuation with the highest summed probability. The draft limit is `min(32, remaining_output - 1, match_length)`. Here, `remaining_output` is the number of output tokens still allowed, and `match_length` is the number of matched tokens. The probability threshold is 0.1.

SGLang verifies the last emitted token and the proposed draft tokens in one target-model pass. It accepts consecutive draft tokens that match the target model's greedy choices. At the first mismatch, it emits the target model's correction token. If all draft tokens match, it emits a bonus token. It keeps key and value (KV) cache slots for accepted inputs and releases rejected slots. The correction or bonus token enters the KV cache on the next pass.

Only committed output enters the suffix caches. Completion removes the local cache and retains the global response. An abort removes the local cache and the global response for that request. A cache flush clears both suffix caches.

This implementation does not support batching, sampling, CUDA graphs, overlap, grammar constraints or logprob requests. It uses no draft model or training.

<!-- BEGIN MEASUREMENTS -->

## Results

The benchmarks compare ordinary decoding, three SUFFIX configurations and SGLang NGRAM PROB. The target model is Qwen2.5-7B-Instruct on one H100 80GB. All modes use greedy decoding with a batch size of 1. Weights and activations use BF16. The output head produces FP32 values.

All modes use deterministic Triton attention, eager execution and seed 42. Each request can generate up to 256 tokens. CUDA graphs, overlap and radix caching are disabled.

Each trial runs five modes on the same 240 requests. Five trials produce 6,000 measurements, with the mode order rotated between trials. The workload uses 52 initial prompts and 32 follow-ups. Each mode runs three blocks, with empty suffix or NGRAM caches at the start of each block:

- Independent: 52 initial prompts, each run once.
- First + follow-up: 52 initial prompts and 32 follow-ups that change an earlier prompt.
- Repeated: 52 initial prompts, each run twice, for 104 requests.

### Decoding latency

Ordinary decoding is the baseline. Adaptive dual cache is SUFFIX with both caches and the match-length bound. The other SUFFIX configurations remove either the bound or the global cache.

Speedup is total ordinary request latency divided by total request latency for the compared mode. Request latency includes host and streaming overhead. Values above 1 mean faster execution. Brackets show 95% paired bootstrap intervals from 10,000 trial resamples with seed 42. These intervals describe timing variation on this fixed workload.

| Mode | Independent | First + follow-up | Follow-up only | Repeated |
|---|---:|---:|---:|---:|
| Adaptive dual cache | 0.999× [0.983–1.025] | 1.113× [1.086–1.141] | 1.280× [1.250–1.310] | 1.902× [1.859–1.947] |
| Without match-length bound | 1.055× [1.032–1.079] | 1.189× [1.160–1.219] | 1.393× [1.358–1.430] | 1.993× [1.915–2.066] |
| Local cache only | 0.977× [0.950–1.005] | 1.063× [1.047–1.078] | 1.225× [1.207–1.244] | 0.970× [0.952–0.993] |

The follow-up column includes only the 32 follow-up requests from each trial. The repeated column includes both passes of each identical prompt. Repeated prompts give favorable conditions for cache reuse. The adaptive SUFFIX results against ordinary decoding are:

| Adaptive SUFFIX vs ordinary | Result |
|---|---|
| Independent | Inconclusive |
| Follow-up only | Faster |
| Repeated | Faster |

The next comparison tests the value of the bound and the global cache separately. Each ablation removes one component. Removing the bound keeps the 32-token limit, probability threshold and output limit. Available continuations can also limit proposal length.

Each ratio divides the ablated mode's latency by the adaptive dual-cache latency. Values above 1 favor the adaptive configuration.

| Block | Bound vs no bound, speedup | Dual vs local cache, speedup |
|---|---:|---:|
| Independent | 0.947× [0.915–0.985] | 1.023× [0.996–1.046] |
| First + follow-up | 0.936× [0.932–0.940] | 1.048× [1.024–1.071] |
| Repeated | 0.954× [0.935–0.980] | 1.960× [1.935–1.985] |

Decoding with the adaptive bound is slower than decoding without it in all three blocks. Removing the bound can change both the selected continuation and its length. These tests do not separate their effects on latency.

### Output checks

All timed SUFFIX configurations produce the same output token IDs as ordinary decoding. Separate checks cover suffix traces, verification, KV updates and the portable runner. The table lists passing comparisons and assertions:

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

NGRAM and SUFFIX use different cache policies. NGRAM can merge branches even when each suffix anchor has only one continuation. The output differences prevent an exact-output speed comparison with NGRAM.

### GPU verification width

This test holds the context fixed and changes the number of input rows in a verification pass. One row contains the last emitted token. A 33-row pass adds 32 draft tokens. The table shows median sums of GPU kernel times from three profiling trials:

| Context tokens | Kernel time, 1 row | Kernel time, 33 rows | Reduction |
|---:|---:|---:|---:|
| 126 | 7.865 ms | 8.962 ms | 12.2% |
| 128 | 7.745 ms | 9.006 ms | 14.0% |
| 512 | 8.865 ms | 10.342 ms | 14.3% |

Shorter verification passes reduce measured kernel time. The launch grids for KV storage and argmax become smaller. The attention and output-head grids stay the same. Kernel time includes profiler overhead. It does not measure request latency or operation counts.

### Attention route

All benchmark modes share the same attention route. A separate check compares this route with the original decode route on two prompts and six paired trials. Shared-route latency divided by original-route latency is 0.983× [0.952–1.019]. Values above 1 mean the shared route is slower. The interval includes 1, so the result is inconclusive. This check does not establish the baseline cost for other inputs.

<!-- END MEASUREMENTS -->

## Limitations

- These results apply to Qwen in SGLang on a small public workload. The paper tests Llama in vLLM on agentic applications. This implementation covers linear speculation; it does not cover tree speculation or hybrid fallback.
- All modes share the deterministic attention route and the fix to the output head's memory layout. The ordinary baseline therefore differs from optimized upstream SGLang decoding.
- Matching token IDs on this test set does not prove correctness for all inputs. Five trials measure timing variation on this workload, not performance on other workloads.
- Each workload block fits in the cache. The benchmarks do not test cache eviction or large caches. The measurements do not separate CPU proposal and cache costs from other request costs.
- The public image passed CPU checks, and the portable runner passed four GPU smoke requests. The portable runner has not completed a separate full five-trial reproduction.

## Data and sources

To inspect the measurements, download the [raw data and SHA256 manifests](https://github.com/szhang120/sglang-suffix-decoding/releases/tag/v0.3.0-measured-reproduction). Use the file hashes in each manifest to verify the downloaded data.

The saved JSON summaries support the tables above:

- [Request timing](results/final/benchmark-report.json): latency ratios and output comparisons.
- [Suffix proposals](results/final/natural-trace-report.json): proposal lengths and acceptance rates.
- [GPU verification width](results/final/width-probe-summary.json): kernel times and launch grids.
- [Attention routes](results/final/decode-control-report.json): the shared and original route comparison.
- [Execution records](results/final/execution-provenance.json): run configurations and completed checks.

To repeat the setup, use the recorded [source versions](configs/source-lock.json), [runtime file hashes](configs/gpu-source-manifest.json) and [Linux dependency hashes](configs/gpu-requirements.lock). These files identify the runtime used for the measurements.

[Paper](https://arxiv.org/abs/2411.04975v3) · [ArcticInference](https://github.com/snowflakedb/ArcticInference) · [SGLang](https://github.com/sgl-project/sglang) · [Spec-Bench](https://github.com/hemingkx/Spec-Bench) · [Attribution](NOTICE) · [License](LICENSE)
