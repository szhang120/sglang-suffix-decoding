# Status at 2026-10-04 20:13 EDT

Later check at 20:29 EDT: the public Modal image build completed without private image overrides, passed all 16 host/report checks and all 25 frozen source fingerprints, and returned `success: true` on CPU. It allocated no GPU. The serving campaign remains unchanged; adaptive SUFFIX trial1 is processing repetition.

The same-chat follow-up `finish-suffixdecoding-experiments` checks every 15 minutes and continues completed stages sequentially. It stays quiet on unchanged checks and removes itself after project completion. Keep the Mac on and the desktop app running for these local follow-ups; the submitted Modal GPU work itself runs remotely. No idle cloud waiter was restarted.

This is a saved checkpoint, not a live dashboard. The current job was submitted asynchronously and its local caller has exited. Explicit Modal app cancellation still stops it; inspect remote status after any interruption.

## Complete

- Public repository, pinned dependencies/source revisions, separate author checkout, dedicated SGLang development branch, CI, implementation and technical draft.
- Linear greedy batch-one SUFFIX with native author CPU lookup, dual caches, adaptive proposal lengths, verification, acceptance and KV settlement.
- Sixteen host/report checks, the complete 240-request exact-token-ID SUFFIX gate, and 416 direct verification/KV assertions.
- Independent upstream/adapter attribution and public raw audit release `v0.2.0-attribution-audit`. One captured NGRAM branch-layout numerical difference was isolated; this does not explain every NGRAM divergence.
- Earlier controlled-width GPU profiles: shorter execution reduces some kernel work, while attention/head tiles remain padded. Final-candidate repetition remains required.

## Active

Only `ap-TawwrpzDJ2lkv2krq66I9s` (`sglang-suffix-validated-campaign`) remains active: one H100 running `modal-20261004-final-campaign-resumed`. It preserves the complete first trial and reruns trials 1–4 under unchanged GPU source/workload hashes. GPU allocation continues until completion, failure, explicit stop or its 12-hour timeout. The local submit command has exited; monitoring is read-only.

Latest checked progress: six of 25 mode/trial combinations, 1440/6000 timed requests validated. The first trial is complete:

| Mode | Completed requests | Token-ID differences |
|---|---:|---:|
| Ordinary | 240 | 0 |
| NGRAM PROB | 240 | 10 |
| SUFFIX | 240 | 0 |
| SUFFIX without the adaptive match-length bound | 240 | 0 |
| Local-cache-only SUFFIX | 240 | 0 |

The resumed allocation has passed fresh plain gates and the direct KV audit. Second-trial NGRAM completed with 10/240 differences; adaptive SUFFIX is next. The progress file's `success: false` is an initial completion flag; the final `campaign-status.json` determines success or failure. Earlier incomplete trial records do not count toward the 6000 measured requests.

## Stopped deliberately

- `ap-3gwZdWJPQpUSik3MqoSlDf`: idle CPU diagnostics waiter. Its future H100 launch is now disabled.
- `ap-b0TnY33l1yFsOZ1YU0dHdS`: idle CPU analysis waiter.
- Local `collect_final_artifacts.py --publish` watcher (PID 93149).

These were intentionally sequenced controllers, not three concurrent GPU jobs. Stopping them removes idle CPU allocation and simplifies tracking. Source, model cache and committed artifact volumes are preserved. No benchmark implementation or inputs were changed.

Decision score: **95/100** for retaining the progressing benchmark and stopping idle follow-up controllers. Existing evidence supports retaining the implementation (**96/100**) over a complete restart (**35/100**).

## Remaining

1. Finish the remaining four trials, inspect campaign completion and all 6000 output comparisons; preserve NGRAM differences as descriptive results.
2. Launch diagnostics as one separately tracked stage: full-workload acceptance traces, final-candidate GPU width profiles, and ordinary-route cost controls. Use a fresh run ID; the stopped controller directories already exist.
3. Generate measured reports, inspect figures, verify raw archives and finish the technical write-up.
4. Publish the measured release only after reviewing the completed evidence. No final paper-performance reproduction conclusion exists yet.

Future execution should announce each stage, report completed/total counts and any failure, and avoid silent polling. The live GPU campaign must be inspected before allocating any further GPU.

## Resumed execution

The user authorized continuation after the status reset. Continue the active campaign, then launch diagnostics, final analysis and publication sequentially. No persistent cloud waiter or automatic publisher was restarted. The runbook now states the required completion checks. The analysis verifies recorded proposer settings and rejects silent SUFFIX ablation misconfiguration; all sixteen host/report checks pass. GPU code and workload remain frozen.

The public portable runner passes source-hash dry runs on Mac and Linux. Its brief CPU app stopped after completion; a fixed container-import error and an earlier image-ID typo are recorded. A four-request Linux/GPU smoke will run in the final diagnostic allocation (16 phases total); it does not establish a separate complete five-trial rerun. Both published source checkpoints have passed Linux/Mac CI. A transient automatic approval-service usage error interrupted a CI read; the normal reviewed retry succeeded after the user resumed.

## Cancellation and recovery

The original campaign received an explicit input cancellation signal at 19:38:47 EDT and stopped at 19:39:33. Its status says `Interrupted before completion`; there is no recorded correctness/CUDA failure. The logs do not identify the cancellation source. Its complete 1200-request trial and partial second trial are saved locally and in the artifact volume.

The resumed campaign copies only complete paired trials, preserves the failed status/log and incomplete records under `resume-source/`, and reruns incomplete trials in full. No partial timings are mixed across allocations. Local revalidation confirms all 960 strict first-trial outputs match, NGRAM has 10/240 differences, and every mode's recorded configuration is valid.

All long stage launchers now support `--submit-only`, using async `spawn()` and returning a function-call receipt. A CPU-only probe's local CLI exited at 19:45:26; its remote completion file was written at 19:45:48. The probe app stopped. This validates continuation after normal submitter exit, not immunity to explicit app cancellation. Decision score: **97/100** for preserving complete paired work and replacing the long-lived waiting caller.
