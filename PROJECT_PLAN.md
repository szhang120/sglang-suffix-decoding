# SuffixDecoding in SGLang

Updated 2026-10-04. The v11 configuration passes 30 GPU token-equality cases and the direct KV audit, but fails the broader public-workload gate. Controlled-width profiling is complete. The v12 shared-attention control passes eight public prompts for both speculators; the full 240-request gate is running before timing trials resume.

## Objective and scope

Public-ready reproduction artifacts: linear, greedy, one request, one target GPU; dual CPU suffix caches, adaptive proposal lengths, exact target verification and KV settlement. SGLang executes Qwen2.5-7B-Instruct. No training or upstream acceptance dependency. Paper reference is arXiv:2411.04975 **v3**; see `configs/source-lock.json` for exact source/model revisions.

Decision scores are engineering confidence/fit, 0–100, not measured probabilities:

| Decision | Score | Evidence / tradeoff |
|---|---:|---|
| Start at SGLang v0.5.21 | 82 | NGRAM V2 verifier, completion hooks, runtime config overrides already exist. 212-package resolution succeeds. First real installation caught a CUDA-tile stub/wheel hash mismatch; direct NVIDIA wheel now pinned. Locked Linux installation, Rust build, model execution and text-path checks pass. |
| Batch-invariant BF16 execution | 70 | Default kernels diverged at tides token127 in both speculators, repeated 3 times. Common deterministic ops and 4096 splits fix all 30 NGRAM cases, but warm SUFFIX diverges at token113. Audit target margins/acceptance before attributing this to numerics or KV. Internal tiles may still pad. |
| FP32 logits with Triton attention | 62 | Improves the initial30-case gate, but broader v11 equality fails123/240 NGRAM and13/28 suffix requests. This setting alone is insufficient. |
| Preserve LM-head weight strides | 94 | Profile identifies a 1.09GB transpose copy per forward (~4.8ms kernel). Existing stride-aware Triton math removes it; the final trace confirms its absence and all 30 cases and KV checks pass. The change applies equally to all baselines. |
| Shared decode/verify attention experiment | 95 | Ordinary decode and verification currently use different Triton kernels. Both speculators fail broad public prompts. An opt-in eager batch-one route now uses the existing unified verification kernel for ordinary decode. The eight-prompt ablation reproduces six NGRAM mismatches with the original decode path and zero for either speculator with shared attention. New ordinary outputs equal old NGRAM outputs on all eight. Full-workload validation remains necessary; the baseline cost changes too. |
| Linear NGRAM PROB baseline | 92 | Breadth1 matches the chosen suffix topology and passes all30 cases. Branching breadth10 failures remain recorded. |
| Reuse author CPU suffix implementation | 96 | Standalone C++20/nanobind module builds without vLLM or PyTorch; native integrity tests pass. |
| Reuse NGRAM verification and KV movement | 90 | Avoid a second acceptance/KV implementation. Variable-width dispatch and416 GPU layout/acceptance/KV-preservation checks pass. |
| Disable graphs and overlap first | 95 | True verify width is inspectable; avoids padded capture and delayed host output. Costs ordinary-decode performance too; applies to every baseline. |
| Qwen2.5-7B-Instruct on one H100 80GB | 86 | Public ungated weights; ample headroom, dense BF16 model. Differs from paper's Llama-3.1-8B. |
| Public Spec-Bench plus second-turn refinement | 82 | 52 frozen prompts across 13 categories. Tests limitations and refinement; not OpenHands/SWE-Bench or proprietary AgenticSQL. |
| Cache Rust build before integration files | 94 | Python-only corrections reuse immutable compiled artifacts. Fresh builds remain portable; cached-image source hashes are checked at runtime. |
| Modal staged GPU functions | 90 | Existing credits/authentication; explicit H100!, persistent model/artifact volumes, correctness gate before timings. Image/source fingerprints and host driver are recorded. |
| Defer trees, batching, graphs, hybrid drafts | 98 | Establish correctness and actual saved work before broadening. |

## Code-grounded design

`spec_info.py`: new `SUFFIX` algorithm uses NGRAM scheduler semantics (`is_ngram`, no draft KV). `suffix_worker.py` subclasses `NGRAMWorker`; author CPU module is vendored under `native/` with Apache notices. No author vLLM plugin is installed.

Start a local suffix tree with the full prompt. Append only scheduler-committed output to local and global trees. Global tree stores outputs only, bounded FIFO by 128 requests; local tree is removed on completion, including prefill-only completion. Abort drops the response. Flush resets both caches. Author score chooses between the two caches; proposals are linear, with maximum draft length `min(32, floor(alpha*p+offset))`, probability threshold 0.1, alpha initially 1, depth 64. Preserve author's native search/tie behavior; fix wrapper's nonexistent `max_depth` default to `max_tree_depth`.

Each verification input is `[last committed token] + proposals`: one root row plus k proposal rows. Root is already emitted but its KV is pending. Causal lower-triangular chain mask, actual positions/retrieval links and actual `NgramVerifyInput.draft_token_num=k+1`. Capacity-sized storage is sliced, not padded. Scoped runtime width matches backend metadata. Empty proposal is a one-row target verification. Scheduler reserves maximum KV slots; reservation is distinct from rows computed.

Inherited `eagle_sample` accepts a contiguous greedy prefix and one correction/bonus token. Inherited `move_accept_tokens_to_target_kvcache` settles corresponding target input KV; correction/bonus becomes the pending root for the next iteration. Result stride carries actual width. CPU cache ingestion occurs after scheduler commit; never add speculative tails or output beyond stop.

## Correctness invariants and gates

- Every emitted ID equals ordinary greedy decoding for the same tokenized prompt, model, dtype and stop settings; report any numerical divergence rather than relaxing the gate.
- Verification has k+1 rows, positions increase by one, and each row sees only prefix plus preceding chain nodes.
- Accept only the prefix before the first mismatch, then target correction (or full-match bonus). Always progress, obey output limit and EOS/stop.
- Local/global caches never contain rejected candidates; global never contains prompts; IDs can be reused after lifecycle cleanup.
- KV contains processed accepted prefix; pending token is processed exactly once next round. Rejected rows cannot corrupt later requests. GPU tests must establish this, not infer it from CPU tests.
- Batch >1, nondeterministic ops, nongreedy, nondefault history penalties/minimum-output constraints, grammar/logprob, overlap, graph execution and non-Triton configurations are outside initial scope.

## Experiments

Freeze model revision, tokenizer input IDs, refinement inputs derived once from ordinary outputs, workload hash, package freeze, resolved engine config, GPU/driver and raw per-request data. All modes: BF16, TP1, batch1, page1, Triton, graphs/overlap/radix prefix cache disabled, batch-invariant execution FP32-output logits and4096-token Triton decode splits, 256 output cap, seed42, identical order. Warm kernel execution then reset algorithm caches before each workload block. Cache persists causally within block.

Compare ordinary, linear NGRAM PROB (33 verify slots, breadth1, trie64), suffix (33-slot capacity), unconstrained-match-length suffix ablation, and local-only suffix ablation. Rotate five mode orders across five trials. Distinguish independent requests, actual second-turn refinement prompts, and repeated-identical-prompt diagnostic upper bound. Record output IDs, wall latency, first streamed chunk latency, token counts and metadata. Reject cross-mode ID mismatches before aggregation. These are serving timings including host/streaming overhead, not isolated kernel time.

Profile separately: suffix per-round widths/match lengths/accepted counts plus SGLang CPU/GPU profiler with input shapes. Inspect target projections/GEMMs, attention query lengths and GPU time vs width. A smaller host mask or KV reserve alone is insufficient. Trace mode adds synchronization and is excluded from timing results. GPU evidence remains required.

## Milestones / current evidence

1. **Done locally:** clean-workspace inspection; separate author checkout; SGLang development branch; exact revision lock and licensing.
2. **Done locally:** CPU native build; dual-cache and adaptive-length tests; randomized oracle acceptance test; ten passing host tests and raw CPU draft profile.
3. **Implemented and GPU-checked:** SUFFIX dispatch, variable target width, inherited greedy verifier/KV movement, completion/abort hooks. Patch clean-apply/syntax checks; common FP32-output batch-invariant matmul preserves weight strides.
4. **GPU gate passes:** final Triton/batch-invariant/FP32-logit configuration with linear NGRAM PROB matches30/30 cases for both speculators. Across416 suffix verify rounds, actual positions/links/acceptance/sequence advancement, unchanged existing KV prefix and accepted KV slots pass independent assertions. A diagnostic layer-bound error was repaired and its failed run retained. H10080GB HBM3/driver580.95.05; Linux native/Rust builds, ten host tests, pip check, Engine import and CUDA math pass. A126-token controlled-width profile passes18 exact-output requests. KV-store grid grows2→66 blocks and argmax19→528 blocks from1→33 rows; attention stays one 128-row query tile. FP32 configuration kernel sum12.93→14.00ms; context boundary and fixed weight-copy costs confound broad interpretation. Aligned 128/512-context controls with the copy fix are now complete.
5. **Controlled widths complete:** final configuration matches all 54 requests at contexts 126/128/512 and widths 1/2/4/8/16/33. At aligned contexts 128 and 512, median summed kernel time is 7.98/8.86ms for one row versus 8.99/10.26ms for 33 rows. KV writes and argmax execute fewer blocks; attention and head launch grids remain fixed. Internal tile padding limits the benefit. The large weight-copy kernel is absent. These profiler measurements are separate from serving latency.
6. **Public repository established:** `https://github.com/szhang120/sglang-suffix-decoding`; Linux and Mac CI pass native build, ten host checks and patch reconstruction. Exact before-copy/final source snapshots are packaged with raw width traces and manifests.
7. **Public gate failed:** trial0 NGRAM differs on 123/240 requests (27 independent, 43 refinement, 53 repetition). The interrupted suffix run differs on 13/28 completed independent requests. Ordinary initial outputs match the frozen workload. Both modes fail beyond the initial diagnostic prompts, so no exact-output speedup comparison is valid. The GPU app and future-trial controller were stopped; raw records are preserved. Investigate ordinary decode versus target verification attention paths before restarting timings (95/100).
8. **Shared-attention diagnostic:** v12 ordinary decode uses the existing unified verification attention kernel, including the just-written pending token in the original decode KV index list. No proposer, acceptance or KV-movement change. On eight prompts, NGRAM has six mismatches with the switch off; NGRAM and SUFFIX have zero with it on. New ordinary equals old NGRAM for all eight. The full 240-request gate is running; do not infer broad correctness from this subset. The shared ordinary route can cost more than optimized upstream decode.
9. **Remaining:** pass a representative public-workload gate; five rotated-order serving trials, workload-sensitive analysis, natural-width profiles and final write-up. The frozen 84-row workload is in `results/final/workload.jsonl`, SHA256 `8b94953aa6e1c8ec18e4f9c405f82cf87d4c330ad172be118e0547e4c657132c`. The ordinary trial0 blocks total approximately 22 minutes of measured request latency, motivating four-hour guards for future five-mode allocations (93/100). Original trial0 retains its existing two-hour guard. Model controls are unchanged.

If verification or shape evidence fails, revise integration before timing. If speedup is absent, retain negative results and attribute cost to CPU lookup, synchronization, short matches, eager launch overhead or attention/MLP scaling only when traces support it. No automatic expansion of scope.
