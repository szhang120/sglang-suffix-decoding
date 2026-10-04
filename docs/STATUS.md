# Status at 2026-10-04 19:12 EDT

This is a saved checkpoint, not a live dashboard. Pausing Codex does not stop detached Modal apps.

## Complete

- Public repository, pinned dependencies/source revisions, separate author checkout, dedicated SGLang development branch, CI, implementation and technical draft.
- Linear greedy batch-one SUFFIX with native author CPU lookup, dual caches, adaptive proposal lengths, verification, acceptance and KV settlement.
- Sixteen host/report checks, the complete 240-request exact-token-ID SUFFIX gate, and 416 direct verification/KV assertions.
- Independent upstream/adapter attribution and public raw audit release `v0.2.0-attribution-audit`. One captured NGRAM branch-layout numerical difference was isolated; this does not explain every NGRAM divergence.
- Earlier controlled-width GPU profiles: shorter execution reduces some kernel work, while attention/head tiles remain padded. Final-candidate repetition remains required.

## Active

Only `ap-xoBixARmigL9i5dgUso8Je` (`sglang-suffix-validated-campaign`) remains active: one H100 running the frozen five-mode, five-trial campaign. GPU allocation continues until completion, failure, explicit stop or its 12-hour timeout.

Latest committed progress: four of 25 mode/trial combinations, 960/6000 timed requests. All are trial 0:

| Mode | Completed requests | Token-ID differences |
|---|---:|---:|
| Ordinary | 240 | 0 |
| NGRAM PROB | 240 | 10 |
| SUFFIX | 240 | 0 |
| SUFFIX without the adaptive match-length bound | 240 | 0 |

Local-cache-only SUFFIX was starting at the last live log read. No complete five-mode trial was yet recorded. The progress file's `success: false` is an initial completion flag; the final `campaign-status.json` determines success or failure.

## Stopped deliberately

- `ap-3gwZdWJPQpUSik3MqoSlDf`: idle CPU diagnostics waiter. Its future H100 launch is now disabled.
- `ap-b0TnY33l1yFsOZ1YU0dHdS`: idle CPU analysis waiter.
- Local `collect_final_artifacts.py --publish` watcher (PID 93149).

These were intentionally sequenced controllers, not three concurrent GPU jobs. Stopping them removes idle CPU allocation and simplifies tracking. Source, model cache and committed artifact volumes are preserved. No benchmark implementation or inputs were changed.

Decision score: **95/100** for retaining the progressing benchmark and stopping idle follow-up controllers. Existing evidence supports retaining the implementation (**96/100**) over a complete restart (**35/100**).

## Remaining

1. Inspect campaign completion and all 6000 output comparisons; preserve NGRAM differences as descriptive results.
2. Launch diagnostics as one separately tracked stage: full-workload acceptance traces, final-candidate GPU width profiles, and ordinary-route cost controls. Use a fresh run ID; the stopped controller directories already exist.
3. Generate measured reports, inspect figures, verify raw archives and finish the technical write-up.
4. Publish the measured release only after reviewing the completed evidence. No final paper-performance reproduction conclusion exists yet.

Future execution should announce each stage, report completed/total counts and any failure, and avoid silent polling. The live GPU campaign must be inspected before allocating any further GPU.

## Resumed execution

The user authorized continuation after the status reset. Continue the active campaign, then launch diagnostics, final analysis and publication sequentially. No persistent cloud waiter or automatic publisher was restarted. The runbook now states the required completion checks. The analysis verifies recorded proposer settings and rejects silent SUFFIX ablation misconfiguration; all sixteen host/report checks pass. GPU code and workload remain frozen.
