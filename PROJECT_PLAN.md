# SuffixDecoding in SGLang

Updated 2026-10-04. Modal H100 model execution passed; token divergences remain under investigation. No GPU performance claim is established.

## Objective and scope

Public-ready reproduction artifacts: linear, greedy, one request, one target GPU; dual CPU suffix caches, adaptive proposal lengths, exact target verification and KV settlement. SGLang executes Qwen2.5-7B-Instruct. No training or upstream acceptance dependency. Paper reference is arXiv:2411.04975 **v3**; see `configs/source-lock.json` for exact source/model revisions.

Decision scores are engineering confidence/fit, 0–100, not measured probabilities:

| Decision | Score | Evidence / tradeoff |
|---|---:|---|
| Start at SGLang v0.5.21 | 82 | NGRAM V2 verifier, completion hooks, runtime config overrides already exist. 212-package resolution succeeds. First real installation caught a CUDA-tile stub/wheel hash mismatch; direct NVIDIA wheel now pinned. Binary validation remains open. |
| Batch-invariant BF16 execution | 70 | Default kernels diverged at tides token127 in both speculators, repeated 3 times. Common deterministic ops and 4096 splits fix all 30 NGRAM cases, but warm SUFFIX diverges at token113. Audit target margins/acceptance before attributing this to numerics or KV. Internal tiles may still pad. |
| FP32-output LM head | 45 | Removes final BF16 ties but does not restore equality: all three suffix tides repetitions differ at113. Keep failed evidence; do not claim a remedy. |
| Common deterministic Triton attention | 85 | All30 suffix cases match; branching NGRAM has one warm length128 mismatch. FP32 logits and4096 decode splits remain common. |
| Linear NGRAM PROB baseline | 88 | Breadth1 matches the chosen suffix topology and isolates proposal policy. Branching breadth10 results remain recorded as failed exact-ID evidence, rather than discarded. |
| Reuse author CPU suffix implementation | 96 | Standalone C++20/nanobind module builds without vLLM or PyTorch; native integrity tests pass. |
| Reuse NGRAM verification and KV movement | 90 | Avoid a second acceptance/KV implementation. Variable-width dispatch still needs real-GPU validation. |
| Disable graphs and overlap first | 95 | True verify width is inspectable; avoids padded capture and delayed host output. Costs ordinary-decode performance too; applies to every baseline. |
| Qwen2.5-7B-Instruct on one H100 80GB | 86 | Public ungated weights; ample headroom, dense BF16 model. Differs from paper's Llama-3.1-8B. |
| Public Spec-Bench plus second-turn refinement | 82 | 52 frozen prompts across 13 categories. Tests limitations and refinement; not OpenHands/SWE-Bench or proprietary AgenticSQL. |
| Cache Rust build before integration files | 94 | Python-only corrections reuse immutable compiled artifacts. Fresh builds remain portable; cached-image source hashes are checked at runtime. |
| Modal staged GPU functions | 90 | Existing credits/authentication; explicit H100!, persistent model/artifact volumes, correctness gate before timings. Image provenance and host driver still require validation. |
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
3. **Implemented, GPU-unverified:** SUFFIX dispatch, variable target width, inherited greedy verifier/KV movement, completion/abort hooks. Patch clean-apply check and syntax checks.
4. **GPU gate incomplete:** all modes execute Qwen; 30 length/EOS/stop/repetition/flush cases per mode are saved. NGRAM matches all under batch-invariant execution; one warm SUFFIX tides case diverges. The audit found an ordinary exact BF16 tie and suffix gap0.125 at position113. FP32 logits alone fail all three tides repetitions; actual GPU layout/acceptance assertions pass on every verify round. Triton matches all30 suffix cases, but branching NGRAM differs on one warm length128 case. Linear NGRAM PROB breadth1 plus direct existing-KV-prefix/accepted-slot checks are now under test. Timings and profiler captures remain pending. Modal H100! preflight passed on H100 80GB HBM3 / driver580.95.05. Linux native and Rust builds, ten host tests, pip check, Engine import and BF16 CUDA math passed.
5. **Pending evidence:** GPU compatibility lock, KV correctness gate, raw benchmarks, workload-sensitive speedup analysis and final write-up. Current technical report is explicitly provisional.

If verification or shape evidence fails, revise integration before timing. If speedup is absent, retain negative results and attribute cost to CPU lookup, synchronization, short matches, eager launch overhead or attention/MLP scaling only when traces support it. No automatic expansion of scope.
