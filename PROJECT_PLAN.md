# SuffixDecoding in SGLang

Checkpoint: 2026-10-04 20:31 EDT. A single detached H100 campaign now runs five rotated trials. The complete first trial is preserved; trials 1–4 are running after an explicit cancellation whose source is unknown. SUFFIX passes the separate 240-request exact-ID gate. Final performance conclusions remain pending. See [current status](docs/STATUS.md) and [preserved design/diagnosis history](docs/PROJECT_HISTORY.md).

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

1. **Complete:** pinned workspace, public repository/CI, native reuse, cache lifecycle, proposer, verifier, acceptance and KV path. Ten host contracts and six reporting-policy checks pass on Mac/Linux; host stubs do not establish GPU correctness.
2. **Complete:** plain 30-case GPU checks, 416 direct verification/KV assertions, separate 240/240 exact SUFFIX gate, independent upstream attribution, and public raw audit releases. One captured NGRAM layout effect is explained; arbitrary-input equivalence and all NGRAM errors remain unproven.
3. **Running:** five paired serving trials. Trial 0 has 1200 records: all 960 ordinary/SUFFIX/ablation outputs match; NGRAM differs 10/240. Resumed trial 1 NGRAM completes with 10/240 differences and adaptive SUFFIX with 0/240; seven modes and 1680 timed requests are validated. Require five complete trials and 4800 strict comparisons before aggregation.
4. **Next:** one separately tracked diagnostic allocation after campaign completion:720 strict suffix traces, five 12-request profiles, 24 paired ordinary-route controls, 54 controlled-width probes and four portable-runner smoke requests. Existing v11 profiles show smaller KV-store/reduction launches while attention/head tiles remain padded; repeat on the final configuration. Instrumented times never enter serving estimates.
5. **Next:** CPU analysis, paired trial intervals, actual proposal/acceptance denominators, scientific figures, raw-file/archive hashes, technical write-up and measured GitHub release. Include negative results and differences from the paper's original Llama/vLLM, proprietary AgenticSQL and live-agent experiments. Portable source-hash dry runs pass on Mac/Linux; a smoke will not establish a separate complete standalone rerun.

**Still required:** five serving trials, final-candidate width and baseline-cost controls, full-workload acceptance analysis and final report including negative results and the distinction from the paper's Llama/vLLM experimental setup.

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
