# Status at 2026-10-05 05:45 EDT

The 20:29 EDT check confirms: the public Modal image build completed without private image overrides, passed all 16 host/report checks and all 25 frozen source fingerprints, and returned `success: true` on CPU. It allocated no GPU and its app has stopped. The frozen serving campaign completed successfully on October 5; all five paired trials are validated.

The agent-created scheduled follow-up was deleted at the user's clarification on 2026-10-04 20:50 EDT. Its creation lacked an explicit scheduling request. There is no scheduled continuation. Work continues in the current chat; the asynchronously submitted Modal diagnostics run remotely; the serving campaign has stopped. Deleting the follow-up does not stop that GPU job. No idle cloud waiter was restarted.

The 20:42 EDT CPU/public-build check passes all 20 tests, including four synthetic publication-report fixtures. They reject incomplete diagnostics, SUFFIX differences and failed portable smoke, and require explicit negative/repetition-only conclusions. They are not measured model or performance evidence. The previously latest source checkpoint passed Linux/Mac CI.

This is a saved checkpoint, not a live dashboard. The current job was submitted asynchronously and its local caller has exited. Explicit Modal app cancellation still stops it; inspect remote status after any interruption.

## Complete

- Public repository, pinned dependencies/source revisions, separate author checkout, dedicated SGLang development branch, CI, implementation and technical draft.
- Linear greedy batch-one SUFFIX with native author CPU lookup, dual caches, adaptive proposal lengths, verification, acceptance and KV settlement.
- Twenty-one local host/report checks and passing Linux/Mac CI at source checkpoint `f98928aa13cd905835cf18687bd6c1f05bfb8274`, the complete 240-request exact-token-ID SUFFIX gate, and 416 direct verification/KV assertions.
- Independent upstream/adapter attribution and public raw audit release `v0.2.0-attribution-audit`. One captured NGRAM branch-layout numerical difference was isolated; this does not explain every NGRAM divergence.
- Earlier controlled-width GPU profiles: shorter execution reduces some kernel work, while attention/head tiles remain padded. Final-candidate repetition remains required.

## Serving complete; diagnostics active

`modal-20261004-final-campaign-resumed` completed with `success: true`, five trials, 25 modes and 6000 timed records. All 4800 ordinary/SUFFIX/ablation outputs match. NGRAM differs on 50/1200 requests; its times remain descriptive. All 25 frozen fingerprints match. The campaign app `ap-TawwrpzDJ2lkv2krq66I9s` stopped at 02:32:53 EDT, and an independent 03:30 check found no containers.

Diagnostics and controller both completed with `success: true`: all 16 phases, 720 exact-output traces, five profiles, 24 route-control requests, 54 width probes and the four-request portable smoke. The diagnostic app `ap-LH7AwrtpEfMxOZyroo0MGW` stopped at 05:21:01 EDT. No GPU containers remain.

CPU-only `modal-20261005-final-analysis` (`ap-Ff0ldElxatswz5b389bTad`) completed successfully in 80.5 seconds. Its report and two figures have been reviewed; raw archives and SHA256 manifests are generated. Source CI and archive/member hashes must pass before publication. Decision score: **97/100** for publishing the complete evidence with explicit negative findings and limitations.

## Stopped deliberately

- `ap-3gwZdWJPQpUSik3MqoSlDf`: idle CPU diagnostics waiter. Its future H100 launch is now disabled.
- `ap-b0TnY33l1yFsOZ1YU0dHdS`: idle CPU analysis waiter.
- Local `collect_final_artifacts.py --publish` watcher (PID 93149).

These were intentionally sequenced controllers, not three concurrent GPU jobs. Stopping them removes idle CPU allocation and simplifies tracking. Source, model cache and committed artifact volumes are preserved. No benchmark implementation or inputs were changed.

Decision score: **95/100** for retaining the progressing benchmark and stopping idle follow-up controllers. Existing evidence supports retaining the implementation (**96/100**) over a complete restart (**35/100**).

## Remaining

1. Verify raw archive/member hashes and passing CI for the final publication source checkpoint.
2. Collect the reviewed report/figures and publish the measured release.
3. Confirm all project apps stopped and provide the public release and measured conclusions.

No GPU is running. No scheduled follow-up or automatic publisher is active.

## Resumed execution

The user authorized continuation after the status reset. Serving, diagnostics and CPU analysis are complete; final publication checks remain. No persistent cloud waiter or automatic publisher was restarted. The runbook now states the required completion checks. The analysis verifies recorded proposer settings and rejects silent SUFFIX ablation misconfiguration; all twenty-one host/report checks pass. GPU code and workload remain frozen.

The public portable runner passes source-hash dry runs on Mac and Linux. Its brief CPU app stopped after completion; a fixed container-import error and an earlier image-ID typo are recorded. A four-request Linux/GPU smoke will run in the final diagnostic allocation (16 phases total); it does not establish a separate complete five-trial rerun. Both published source checkpoints have passed Linux/Mac CI. A transient automatic approval-service usage error interrupted a CI read; the normal reviewed retry succeeded after the user resumed.

## Cancellation and recovery

The original campaign received an explicit input cancellation signal at 19:38:47 EDT and stopped at 19:39:33. Its status says `Interrupted before completion`; there is no recorded correctness/CUDA failure. The logs do not identify the cancellation source. Its complete 1200-request trial and partial second trial are saved locally and in the artifact volume.

The resumed campaign copies only complete paired trials, preserves the failed status/log and incomplete records under `resume-source/`, and reruns incomplete trials in full. No partial timings are mixed across allocations. Local revalidation confirms all 960 strict first-trial outputs match, NGRAM has 10/240 differences, and every mode's recorded configuration is valid.

All long stage launchers now support `--submit-only`, using async `spawn()` and returning a function-call receipt. A CPU-only probe's local CLI exited at 19:45:26; its remote completion file was written at 19:45:48. The probe app stopped. This validates continuation after normal submitter exit, not immunity to explicit app cancellation. Decision score: **97/100** for preserving complete paired work and replacing the long-lived waiting caller.

## Ablation interpretation

The reporting code now compares adaptive SUFFIX directly with both ablations using the same five paired trial wall times and bootstrap intervals. It reports negative/inconclusive policy effects explicitly. Changing the adaptive cap can also change the winning continuation in the author's cumulative-score search; the ablation is not merely truncating one fixed proposal. All 21 local checks pass. No runtime, model, workload or GPU source fingerprint changed. First-trial ratios are exploratory and are not published as final performance conclusions.

## Five-trial serving results; final diagnostic interpretation pending

The validated serving report records adaptive SUFFIX ratios (ordinary/mode summed request wall latency) of 0.999× independent, 1.113× combined initial/follow-up, 1.280× actual follow-ups alone and 1.902× repeated inputs. Corresponding 95% paired trial bootstrap intervals are [0.983,1.025], [1.086,1.141], [1.250,1.310] and [1.859,1.947]. Intervals describe repeat variation on this fixed subset, not generalization to other workloads. The bound-removal ablation yields 1.055×, 1.189×, 1.393× and 1.993×. Direct paired comparisons favor removing the bound in all three blocks, with intervals excluding parity. Changing the bound can also change the selected continuation; this is not a pure length-only intervention. Local-only repetition is 0.970×. Independent adaptive decoding has no demonstrated improvement; follow-up and repetition gains persist. Profiling and final report review remain required before the public measured release.

Report: `results/modal/modal-20261005-final-diagnostics-controller/reports/benchmark-report.json`. The original failed campaign and complete-trial recovery lineage remain preserved.
