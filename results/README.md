# Evidence and provenance

`final/` contains the plain, uninstrumented 30-case correctness gate for each mode and the frozen 84-row public workload from `modal-20261004-v11`. Workload rows include tokenizer IDs, ordinary outputs and whether inputs were truncated. No future answer is inserted into speculative caches.

`width-probe-final/` contains 54 controlled-width requests and correlated GPU-kernel summaries for the final stride-preserving FP32-head configuration. `width-probe-fp32/` contains the earlier 18-request study with the weight copy still present. The three contexts in the final study distinguish query-width effects from crossing a key-tile boundary. Profiler timing is separate from serving timing.

To regenerate final kernel summaries from downloaded traces:

```sh
python scripts/analyze_width_probe.py PATH/width-probe
python analysis/width_report.py PATH/width-probe
```

Both commands preserve existing outputs; use a fresh extraction directory when repeating analysis. The summary records individual trace SHA256 hashes. Large traces are excluded from Git and published in the [audit prerelease](https://github.com/szhang120/sglang-suffix-decoding/releases/tag/v0.1.0-audit), together with exact source snapshots and per-file/archive SHA256 manifests.

Generate scientific figures with Python 3.12 and the plotting-only pins in `configs/analysis-requirements.txt`:

```sh
python -m pip install -r configs/analysis-requirements.txt
MPLCONFIGDIR=/tmp/suffix-matplotlib python analysis/plot_results.py
```

`analysis/benchmark_report.py PATH/results --output REPORT.json` validates five complete, unprofiled trials, records exact-ID mismatches and reports paired latency ratios with trial bootstrap intervals. If equality fails, its saved report is diagnostic and its exit status fails; it cannot establish an exact-output performance comparison. Bootstrap intervals describe repeated execution on this fixed subset.

`modal/` and `setup/` retain earlier attempts, failures and diagnostic logs. A phase status can fail after successful GPU execution if its subsequent analysis failed. The initial width study had that outcome; its corrected local analysis is retained. The ambiguous single file `modal/modal-20261004-v2/raw` resulted from an incorrectly specified recursive download destination and is not a complete artifact collection. Explicit per-mode correctness files in that directory are authoritative.

`local-verification.json` records the current verification status. CPU microbenchmarks use synthetic integer sequences and establish no model speedup. The first serving trial failed the broader equality gate: see `invalid-benchmark-v11/equality-failure.json`. Completed ordinary/NGRAM records and 28 partial suffix records are preserved; no comparative speedup is valid from that attempt.

`modal/modal-20261004-v12/` contains the eight-prompt attention-path control. It reproduces six NGRAM mismatches with the original decode route and zero with the shared route for either speculator. These are diagnostic subset records; they do not constitute the full public-workload gate.
