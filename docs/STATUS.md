# Completed measured reproduction

The five serving trials, separate diagnostics and CPU analysis are complete. See [measured results](RESULTS.md), [technical design](TECHNICAL_REPORT.md) and [preserved investigation history](PROJECT_HISTORY.md).

- Serving: 6000 requests; all 4800 ordinary/SUFFIX/ablation outputs match exactly.
- Upstream NGRAM: 50/1200 differing responses, retained as descriptive comparisons.
- Separate checks: 720 exact suffix traces, 54 width probes, 24 ordinary-route controls and four portable-runner smoke requests.
- Raw timing/profile archives have per-file SHA256 manifests in the measured release.

Campaign: `modal-20261004-final-campaign-resumed`. Diagnostics: `modal-20261005-final-diagnostics`. Analysis: `modal-20261005-final-analysis`. The interrupted campaign's complete trial was preserved; incomplete trials were rerun without mixing partial timings. GPU UUIDs and cancellation lineage remain in the provenance.

There is no known open SUFFIX correctness defect on the finite tested suite. This does not prove arbitrary-input equivalence. One upstream NGRAM numerical layout case was isolated; the remaining divergences are not individually attributed. The complete standalone portable orchestration was not separately rerun.

Decision scores: retain the reviewed implementation 96/100; restart the complete workspace 35/100; preserve complete paired trials with explicit lineage 97/100.
