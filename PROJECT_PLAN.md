# SuffixDecoding in SGLang

Final measurements complete. Five serving trials and all separate diagnostic stages passed their gates. See [measured results](docs/RESULTS.md), [final status](docs/STATUS.md) and [preserved history](docs/PROJECT_HISTORY.md).

## Scope and integration

Implement the paper's linear greedy batch-one algorithm in SGLang, retaining the author's CPU suffix implementation, dual caches and adaptive proposal lengths. No draft model, training, tree/batch expansion or upstream PR dependency. Broader scope requires measured value.

The dedicated SGLang branch is `codex/suffix-decoding`; the ArcticInference reference checkout stays separate. [Source lock](configs/source-lock.json) records exact engine, author, workload and model commits. Retain SGLang v0.5.21 (`e00930c…`) and the tested Linux dependency lock: PyTorch 2.13.0/CUDA13, FlashInfer 0.6.18, SGLang kernel 0.4.7, Python 3.12 and Rust 1.92. The [runbook](docs/GPU_RUNBOOK.md) records compatible provisioning and reproduction steps.

`SuffixWorker` subclasses the existing NGRAM worker: replace proposal generation, construct an explicit causal chain and reuse target verification/acceptance/KV settlement. Lifecycle hooks handle completion, prefill-only completion, abort and flush. The unchanged author C++20/nanobind cache has no vLLM dependency. All target-model execution uses SGLang.

## Correctness invariants

- Local cache contains the full prompt plus committed output. Global cache contains response tokens only, bounded to 128 responses with FIFO eviction. Completion frees local state; abort removes its response; flush clears both caches. Insert only scheduler-committed `output_ids_through_stop`, never rejected drafts or a discarded terminal tail.
- Propose at most `min(32, remaining_output-1, floor(alpha*p+offset))` drafts for match length `p`, with alpha 1, offset 0 and cumulative continuation-probability threshold 0.1. Retain author scoring/tie behavior and its exclusive match-context bound.
- Verify the pending emitted token followed by `k` drafts using exactly `k+1` rows. Runtime width, positions, chain mask, retrieval links and result stride must agree. Scheduler reservation is distinct from executed rows. Zero drafts still execute one row.
- Accept a contiguous matching prefix and one target correction/bonus. For `A` emitted tokens, settle `A` processed input KV slots; the new correction stays pending. Preserve the existing KV prefix and free rejected slots before the next forward.
- Reject unsupported batching, sampling/history penalties, grammar/logprobs, overlap and CUDA graphs. Every timed ordinary/SUFFIX/ablation output must match the gated ordinary reference and its paired trial baseline. Any SUFFIX mismatch blocks performance claims.

## Frozen experiment

One H100 80GB, Qwen2.5-7B-Instruct, BF16 with FP32 head, TP 1/PP 1, greedy batch 1, cap 256/EOS, eager deterministic Triton, overlap/graphs/radix off. The head-stride correction and shared ordinary-attention route apply to every mode and must be disclosed; the baseline is not optimized upstream ordinary decoding.

The public Spec-Bench subset freezes 52 initial inputs and 32 second turns. Each mode/trial runs 52 independent, 84 initial/refinement and 104 repeated-prompt requests. Five rotated trials compare ordinary, unchanged upstream NGRAM PROB, adaptive dual-cache SUFFIX, unbounded-match-cap SUFFIX and local-only SUFFIX: 6000 timed requests. Report the 32 actual second turns separately. Repetition is a diagnostic upper bound. Blocks begin with empty caches and receive no future output seeds. Their maximum 104 requests does not exercise eviction pressure at bound 128.

Upstream NGRAM fanout 1 is per suffix anchor; merged paths can branch. Independent pristine controls reproduce selected divergences without the adapter. One exact accepted-branch replay isolates a BF16 attention-layout difference, not all divergences. Retain every NGRAM difference, label differing-output timings descriptive and exclude them from exact-output speedup figures. The default reporter remains strict; the explicit NGRAM exception never relaxes SUFFIX checks.

Each five-mode trial stays on one physical GPU. Recovery copies only complete paired trials and reruns incomplete trials in full; GPU allocations may differ between trials, with UUIDs recorded. Preserve the original failed status/log and partial records under `resume-source/`. Model/runtime/workload hashes stay unchanged. Submit long jobs asynchronously; inspect completion before launching the next allocation.

## Milestones and evidence

1. **Complete:** pinned workspace, public repository/CI, native reuse, cache lifecycle, proposer, verifier, acceptance and KV path. Ten host contracts and seven benchmark-policy checks pass locally; Linux/Mac CI passes at source checkpoint `f98928aa13cd905835cf18687bd6c1f05bfb8274`; four publication-report checks also pass on Mac/Linux and in the public Linux image. Synthetic fixtures check refusal of incomplete/failed evidence and explicit negative findings; they supply no model or performance measurements. Host stubs do not establish GPU correctness.
2. **Complete:** plain 30-case GPU checks, 416 direct verification/KV assertions, separate 240/240 exact SUFFIX gate, independent upstream attribution, and public raw audit releases. One captured NGRAM layout effect is explained; arbitrary-input equivalence and all NGRAM errors remain unproven.
3. **Complete:** five rotated serving trials, 6000 measured requests and all 4800 strict ordinary/SUFFIX/ablation comparisons. Upstream NGRAM differs on 50/1200 responses; descriptive results retain every mismatch.
4. **Complete:** 720 exact-output natural traces, five 12-request profiles, 24 paired ordinary-route controls, 54 controlled-width probes and four portable-runner smoke requests. Instrumented times remain separate from serving estimates. See the measured GPU-work and baseline-cost findings.
5. **Complete:** paired trial analysis, actual second turns, proposal/acceptance denominators, scientific figures, per-file/archive hashes, technical write-up and public measured release. Negative results and differences from the paper remain explicit. CPU public-build and GPU smoke validation do not establish a separate complete standalone five-trial rerun.

**Completed:** serving trials, final-candidate diagnostics, full-workload acceptance analysis, measured report, figures and raw artifacts. See `docs/RESULTS.md` and `results/final/`.

## Decision scores

Scores are engineering judgments, not probabilities.

| Decision | Score /100 | Evidence |
|---|---:|---|
| Retain reviewed implementation | 96 | Explicit chain, lifecycle/host checks, direct KV assertions and full SUFFIX gate support continuing. |
| Restart the complete workspace | 35 | Would discard useful pins and verified evidence without an identified adapter contract failure. |
| Retain v0.5.21 | 82 | Tested build/interfaces and locked execution; changing versions would confound comparisons. |
| Preserve complete paired trials and submit asynchronously | 97 | Saved trial revalidates; normal submitter exit was verified by a CPU probe. Explicit app cancellation remains possible. |
| Keep upstream NGRAM with descriptive mismatches | 93 | Independent reproduction supports retaining the requested baseline while restricting exact-output claims. |
| Finish public portable runner before publication | 97 | Fresh gates remove private output dependencies; full standalone orchestration still needs an explicit validation limitation. |

The unsolicited Codex scheduled follow-up was removed after the user's clarification. All authorized execution stages are complete; no scheduled task or idle cloud waiter is active.
