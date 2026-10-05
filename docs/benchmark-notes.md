# Benchmark notes

These notes define the scope of the implementation and the comparisons in the [README](../README.md).

## Execution

- The implementation supports linear, greedy decoding with a batch size of 1. Batching, sampling, CUDA graphs, overlap, grammar constraints and logprob requests are unsupported.
- The suffix caches supply proposals. The implementation uses no draft model, training or fine-tuning.
- The benchmark configuration disables CUDA graphs, overlap and radix caching.
- Tree speculation and hybrid fallback are outside this implementation's scope.

## Comparisons

- These tests use Qwen in SGLang on a small public workload. The paper tests Llama in vLLM on agentic applications. This is an adapted reproduction of the linear variant.
- All modes share a deterministic attention route and a fix to the output head's memory layout. The ordinary baseline differs from optimized upstream SGLang decoding.
- NGRAM differs from ordinary output on 50 of 1,200 timed requests. Its latency ratios are descriptive, rather than exact-output speedups.
- Removing the adaptive bound can change both the selected continuation and its length. The experiments do not separate their effects on latency.

## Measurement scope

- Matching token IDs on the test set does not prove correctness for all inputs. Five trials measure timing variation on this workload, not performance on other workloads.
- Each workload block fits in the cache. The benchmarks do not test cache eviction or large caches.
- Request latency includes CPU proposal and cache costs. Those costs were not measured separately.
- GPU kernel times include profiler overhead. They do not measure request latency or operation counts.
- The attention route comparison uses two prompts and six paired trials. It does not establish the baseline cost for other inputs.

## Reproduction checks

The public image passed CPU checks. The portable runner passed four GPU smoke requests with empty and populated caches. A separate full five-trial reproduction with the portable runner has not been completed.
