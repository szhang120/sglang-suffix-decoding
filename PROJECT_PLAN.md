# SuffixDecoding in SGLang

Updated 2026-10-04 during the independent reassessment. **Broad correctness remains unresolved. Serving benchmarks are stopped.** The v12 public gate ended on client disconnect after 240 ordinary and 136 NGRAM requests; six NGRAM mismatches remain on q112, q114 and q151. No suffix results were completed in that run. Raw records are preserved. The custom-mask prototype is archived and removed from active development after fixed-tensor controls found no evidence for it.

## Objective and frozen scope

Public, reproducible linear greedy batch-one SuffixDecoding in SGLang, on one H100 80GB with Qwen2.5-7B-Instruct. Retain dual CPU caches and adaptive lengths; no training, tree/batch/graph expansion or upstream PR dependency. `configs/source-lock.json` pins SGLang v0.5.21 (`e00930c...`), ArcticInference (`aca5d9a...`), Spec-Bench, model revision and 212 hashed Linux packages. Author checkout stays separate and clean; SGLang development branch is `codex/suffix-decoding`.

Decision scores are engineering judgments, not probabilities:

| Decision | Score | Reason |
|---|---:|---|
| Independently compare pristine SGLang before more patches | 98 | Both NGRAM and suffix failed; shared patches/configuration prevent attribution to upstream or the adapter. |
| Reuse author CPU cache provisionally | 92 | Native files are unchanged, integrity and oracle checks pass; source review confirms dual-cache scoring and adaptive bounds. |
| Reuse NGRAM verifier and KV settlement provisionally | 85 | Limits duplicated correctness logic; 416 audited rounds pass, but the tested contexts do not establish broad equality. |
| Retain v0.5.21 while diagnosing | 82 | Compatible build, speculative interfaces and locked text execution work. Changing engine versions now would confound the comparison. |
| Keep common eager deterministic settings | 88 | Inspectable actual widths and fewer scheduling variables, at a cost to production performance. |
| Continue speculative attention fixes before pristine comparison | 35 | Kernel names alone do not establish identical numerical execution; this already failed on three public prompts. |
| Preserve head strides if independently validated | 90 | Removes a measured 1.09GB copy; separate it from the suffix algorithm and test output drift directly. |
| Use detached, single-GPU runs | 96 | Prior client disconnect canceled the gate. Detached execution preserves remote completion; no concurrent GPU jobs or automatic retries. |
| Rebuild the entire workspace | 35 | Existing pins, fixtures, raw failures and native cache are useful. Rewrite only components whose contracts fail review. |

## Integration design and invariants

`SuffixWorker` is a 259-line NGRAM subclass. `spec_info.py` adds SUFFIX dispatch and NGRAM scheduler semantics; initialization omits the NGRAM corpus only for SUFFIX. Completion/abort hooks manage suffix lifecycle. The author C++20/nanobind module executes CPU lookup without a vLLM dependency.

Local cache: full prompt plus committed output. Global cache: response tokens only, FIFO bounded to 128 requests. Completion frees local state, including prefill-only completion; abort removes its response; flush clears both caches. Synchronize only `output_ids_through_stop`, never a rejected tail. Preserve native tie behavior and exclusive context-match bound. Proposal cap is `min(32, remaining_output-1, floor(alpha*p+offset))`, with alpha1, offset0 and cumulative probability threshold0.1.

Verify `[pending last emitted token] + drafts` with actual width k+1 and a causal chain mask. Scoped runtime width, metadata, retrieval links and result stride must agree. Scheduler capacity reservation is separate from executed rows. Empty proposal still executes one target row.

Accept the contiguous matching prefix and one target correction/bonus. For A emitted tokens, settle A processed input KV slots; the new correction remains pending. Preserve the existing KV prefix and release rejected slots. Cache insertion occurs after scheduler commitment. Reject batching, sampling/history penalties, grammar/logprobs, overlap and graphs in this initial scope.

## Immediate reassessment

1. **Completed:** independent pristine/head-only/shared-v12 comparison, with no suffix adapter: pristine and head-only each match NGRAM on 5/8 selected cold/warm requests; shared-v12 matches on 7/8. The head fix changes neither mode's eight outputs. Shared attention changes only two ordinary outputs. This establishes that the adapter is unnecessary for these particular failures, not that all remaining failures are upstream. Full source diffs, IDs and environments are preserved in `results/modal/modal-20261004-independent-audit/`; aggregate in `results/reassessment-summary.json`.
2. **Reviewed:** cache, pending-token, sampler and KV contracts rederived from paper/author/upstream code independently of earlier plan claims. No structural adapter defect identified yet. Host stubs omit real-engine initialization and GPU attention metadata, so retain direct GPU assertions and broader gates.
3. **Completed:** all 3,840 fixed-Q/K/V query-row fixtures match bitwise between causal single-query, custom single-query and custom multi-query verification attention. Widths1/2/4/8/16/33, packed/contiguous queries, linear/permuted KV addressing, five contexts and three seeds. Synthetic equality does not establish real-model equivalence. Removed the unsupported custom-mask prototype; preserved it in `analysis/hypotheses/`.
4. **Completed:** q112 cold ordinary/NGRAM upstream tensor dumps reproduce output 125 divergence. Among correctly conditioned query rows, the first reliable differing leaf is layer 3 attention at logical position 120 (width33); preceding QKV projection agrees. Shared cached RoPE modules cause misleading repeated layer labels in the upstream dumper, so exclude those labels. Preserve original diagnostic source and raw tensor hashes in `results/tensor-equivalence-summary.json`.
5. **Running:** capture actual post-RoPE Q, logical KV and attention output at that instance, then replay with query width/KV length/future values varied independently. This tests whether synthetic controls missed a real-input numerical effect or cached conditioning differs. Changing ordinary execution requires a baseline-cost control and explicit reporting.
6. Reconstruct the distributable patch and rerun relevant host/GPU checks only after choosing the validated design. Pass all 240 public requests for ordinary/NGRAM/SUFFIX before timing. Check every completed timed mode immediately; fail closed on divergence.

## Experiments and completion milestones

The frozen 84-row workload contains 52 initial requests across13 categories and32 refinements. SHA256 `8b94953aa6e1c8ec18e4f9c405f82cf87d4c330ad172be118e0547e4c657132c`. Refinement text was generated once under v11 ordinary decode. Historical outputs are provenance, never cache seeds or the candidate reference. Each block starts empty and caches persist only causally within it.

Five rotated-order trials compare ordinary, linear NGRAM PROB (breadth1,33slots), dual-cache suffix, unbounded-match-length suffix, and local-only suffix. Blocks:52 independent,84 initial/refinement,104 repeated-prompt requests. Repetition is a diagnostic upper bound. Record streamed wall latency, first-chunk latency, token IDs, resolved configuration, hashes, package freeze, GPU UUID/driver. Require6000 equal measured outputs before aggregation; paired trial intervals characterize this fixed subset only. NGRAM and suffix cache policies differ and must be disclosed.

Separate profiling must show actual GPU launches/work, not only smaller masks. Existing v11 profiles match54 controlled-width outputs at contexts126/128/512: KV writes/reductions shrink, while 128-row attention/head tiles remain padded. They are valid diagnostic evidence for that configuration, not final serving results. Collect full-workload suffix traces for actual proposal/acceptance denominators; capacity-based SGLang metadata is misleading. Repeat final-candidate width profiles and quantify any changed ordinary route's cost.

**Completed assets:** public repository, Linux/Mac CI, pinned native reuse, ten host checks, initial30-case GPU gates,416 KV assertions, controlled profiles, public raw audit release with exact source snapshots. **Still required:** independent attribution, broad equality, five serving trials, workload/acceptance analysis and final report including negative results and the distinction from the paper's Llama/vLLM experimental setup.
