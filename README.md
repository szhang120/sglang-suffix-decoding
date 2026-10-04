# SuffixDecoding in SGLang

A reproducible **linear, greedy, batch-one** implementation candidate using the author CPU suffix tree and SGLang target verification. **GPU correctness and performance are not yet validated.** No GPU rental has been provisioned, and no speedup is claimed.

- [Project plan](PROJECT_PLAN.md): code-grounded design, invariants, milestones and 0–100 decision scores.
- [Technical report](docs/TECHNICAL_REPORT.md): inference/KV reasoning, local evidence and limits.
- [GPU runbook](docs/GPU_RUNBOOK.md): rental/SSH steps and exact execution workflow.
- [Source lock](configs/source-lock.json) and [hashed Linux dependency lock](configs/gpu-requirements.lock).

## Local CPU verification

Python 3.12 and a C++20 compiler are required. On the development Mac, `.venv` already contains the pinned CPU dependencies and the native extension has been built.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/bootstrap.sh
bash scripts/build_native.sh
PYTHONPATH=native python scripts/cpu_profile.py
```

The bootstrap preserves modified checkouts. SGLang lives in `sglang/` on `codex/suffix-decoding`, based on v0.5.21 commit `e00930c5489053f26d86b179cee0d087f846acbb`. Author code lives separately in `reference/ArcticInference/`. Checkouts and native binaries are excluded from this root repository; the committed patch reconstructs the integration.

Current verification: eight CPU/host contract tests pass. The GPU runners compare ordinary decoding, NGRAM PROB and suffix decoding, record raw outputs/configurations and refuse mismatched token IDs. See the runbook before renting a GPU.

## Sources and licensing

[Paper v3](https://arxiv.org/abs/2411.04975v3), [ArcticInference](https://github.com/snowflakedb/ArcticInference), [SGLang](https://github.com/sgl-project/sglang), and [Spec-Bench](https://github.com/hemingkx/Spec-Bench). Apache-2.0 source notices are retained; see [third-party provenance](THIRD_PARTY.md). This is an adapted SGLang reproduction project, with no dependency on upstream PR acceptance. Publication and final results remain pending GPU validation.
