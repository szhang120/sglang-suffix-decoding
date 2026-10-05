# Status at 2026-10-04 23:14 EDT

The 20:29 EDT check confirms: the public Modal image build completed without private image overrides, passed all 16 host/report checks and all 25 frozen source fingerprints, and returned `success: true` on CPU. It allocated no GPU and its app has stopped. The serving campaign remains unchanged; adaptive SUFFIX trial1 completed with zero differences.

The agent-created scheduled follow-up was deleted at the user's clarification on 2026-10-04 20:50 EDT. Its creation lacked an explicit scheduling request. There is no scheduled continuation. Work continues in the current chat; the asynchronously submitted Modal campaign runs remotely. Deleting the follow-up does not stop that GPU job. No idle cloud waiter was restarted.

The 20:42 EDT CPU/public-build check passes all 20 tests, including four synthetic publication-report fixtures. They reject incomplete diagnostics, SUFFIX differences and failed portable smoke, and require explicit negative/repetition-only conclusions. They are not measured model or performance evidence. The previously latest source checkpoint passed Linux/Mac CI.

This is a saved checkpoint, not a live dashboard. The current job was submitted asynchronously and its local caller has exited. Explicit Modal app cancellation still stops it; inspect remote status after any interruption.

## Complete

- Public repository, pinned dependencies/source revisions, separate author checkout, dedicated SGLang development branch, CI, implementation and technical draft.
- Linear greedy batch-one SUFFIX with native author CPU lookup, dual caches, adaptive proposal lengths, verification, acceptance and KV settlement.
- Twenty-one local host/report checks and passing Linux/Mac CI at source checkpoint `f98928aa13cd905835cf18687bd6c1f05bfb8274`, the complete 240-request exact-token-ID SUFFIX gate, and 416 direct verification/KV assertions.
- Independent upstream/adapter attribution and public raw audit release `v0.2.0-attribution-audit`. One captured NGRAM branch-layout numerical difference was isolated; this does not explain every NGRAM divergence.
- Earlier controlled-width GPU profiles: shorter execution reduces some kernel work, while attention/head tiles remain padded. Final-candidate repetition remains required.

## Active

Only `ap-TawwrpzDJ2lkv2krq66I9s` (`sglang-suffix-validated-campaign`) remains active: one H100 running `modal-20261004-final-campaign-resumed`. It preserves the complete first trial and reruns trials 1–4 under unchanged GPU source/workload hashes. GPU allocation continues until completion, failure, explicit stop or its 12-hour timeout. The local submit command has exited; monitoring is read-only.

Latest checked progress: fifteen of 25 mode/trial combinations, 3600/6000 timed requests validated. Three complete paired trials are available. All 2880 completed ordinary/SUFFIX/ablation outputs match. NGRAM differs on 30/720 completed requests. The first trial is complete:

| Mode | Completed requests | Token-ID differences |
|---|---:|---:|
| Ordinary | 240 | 0 |
| NGRAM PROB | 240 | 10 |
| SUFFIX | 240 | 0 |
| SUFFIX without the adaptive match-length bound | 240 | 0 |
| Local-cache-only SUFFIX | 240 | 0 |

The resumed allocation has passed fresh plain gates and the direct KV audit. Second-trial NGRAM completed with 10/240 differences; adaptive SUFFIX completed with 0/240 differences. The unbounded-match-cap and local-only ablations each completed with 0/240 differences. Trials 0–2 are complete; the fourth trial has started with unbounded-match-cap SUFFIX. The progress file's `success: false` is an initial completion flag; the final `campaign-status.json` determines success or failure. Earlier incomplete trial records do not count toward the 6000 measured requests.

## Stopped deliberately

- `ap-3gwZdWJPQpUSik3MqoSlDf`: idle CPU diagnostics waiter. Its future H100 launch is now disabled.
- `ap-b0TnY33l1yFsOZ1YU0dHdS`: idle CPU analysis waiter.
- Local `collect_final_artifacts.py --publish` watcher (PID 93149).

These were intentionally sequenced controllers, not three concurrent GPU jobs. Stopping them removes idle CPU allocation and simplifies tracking. Source, model cache and committed artifact volumes are preserved. No benchmark implementation or inputs were changed.

Decision score: **95/100** for retaining the progressing benchmark and stopping idle follow-up controllers. Existing evidence supports retaining the implementation (**96/100**) over a complete restart (**35/100**).

## Remaining

1. Finish the remaining trials, inspect campaign completion and all 6000 output comparisons; preserve NGRAM differences as descriptive results.
2. Launch diagnostics as one separately tracked stage: full-workload acceptance traces, final-candidate GPU width profiles, and ordinary-route cost controls. Use a fresh run ID; the stopped controller directories already exist.
3. Generate measured reports, inspect figures, verify raw archives and finish the technical write-up.
4. Publish the measured release only after reviewing the completed evidence. No final paper-performance reproduction conclusion exists yet.

Future execution should announce each stage, report completed/total counts and any failure, and avoid silent polling. The live GPU campaign must be inspected before allocating any further GPU.

## Resumed execution

The user authorized continuation after the status reset. Continue the active campaign, then launch diagnostics, final analysis and publication sequentially. No persistent cloud waiter or automatic publisher was restarted. The runbook now states the required completion checks. The analysis verifies recorded proposer settings and rejects silent SUFFIX ablation misconfiguration; all twenty-one host/report checks pass. GPU code and workload remain frozen.

The public portable runner passes source-hash dry runs on Mac and Linux. Its brief CPU app stopped after completion; a fixed container-import error and an earlier image-ID typo are recorded. A four-request Linux/GPU smoke will run in the final diagnostic allocation (16 phases total); it does not establish a separate complete five-trial rerun. Both published source checkpoints have passed Linux/Mac CI. A transient automatic approval-service usage error interrupted a CI read; the normal reviewed retry succeeded after the user resumed.

## Cancellation and recovery

The original campaign received an explicit input cancellation signal at 19:38:47 EDT and stopped at 19:39:33. Its status says `Interrupted before completion`; there is no recorded correctness/CUDA failure. The logs do not identify the cancellation source. Its complete 1200-request trial and partial second trial are saved locally and in the artifact volume.

The resumed campaign copies only complete paired trials, preserves the failed status/log and incomplete records under `resume-source/`, and reruns incomplete trials in full. No partial timings are mixed across allocations. Local revalidation confirms all 960 strict first-trial outputs match, NGRAM has 10/240 differences, and every mode's recorded configuration is valid.

All long stage launchers now support `--submit-only`, using async `spawn()` and returning a function-call receipt. A CPU-only probe's local CLI exited at 19:45:26; its remote completion file was written at 19:45:48. The probe app stopped. This validates continuation after normal submitter exit, not immunity to explicit app cancellation. Decision score: **97/100** for preserving complete paired work and replacing the long-lived waiting caller.

## Ablation interpretation

The reporting code now compares adaptive SUFFIX directly with both ablations using the same five paired trial wall times and bootstrap intervals. It reports negative/inconclusive policy effects explicitly. Changing the adaptive cap can also change the winning continuation in the author's cumulative-score search; the ablation is not merely truncating one fixed proposal. All 21 local checks pass. No runtime, model, workload or GPU source fingerprint changed. First-trial ratios are exploratory and are not published as final performance conclusions.

## Preliminary three-trial performance checkpoint

Three complete paired trials yield adaptive SUFFIX wall-latency ratios of 1.008× for independent inputs, 1.130× for the combined initial/follow-up block, 1.299× for actual follow-ups alone (a subset of that block), and 1.929× for repeated inputs. Independent per-trial ratios span 0.986–1.051×; all three follow-up and repetition ratios exceed 1. Removing the match-length bound yields 1.064×, 1.204×, 1.411× and 2.003× respectively. Local-only repetition remains 0.983×. These are exploratory pooled ratios, not a final five-trial analysis or a reproduction claim. Final uncertainty intervals and diagnostics remain required. The frozen runtime and inputs are unchanged.
