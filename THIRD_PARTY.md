# Third-party provenance

- `native/{bindings.cc,suffix_tree.cc,suffix_tree.h,int32_map.h,CMakeLists.txt}` and `native/suffix_native/cache.py` derive from Snowflake ArcticInference commit `aca5d9a8a62474035c15d114d40a01abc8c94b51`, Apache-2.0. Original copyright notices and `native/LICENSE` retained. Wrapper changes: import namespace and `max_depth` default fix. `__init__.py` is empty.
- `patches/sglang-suffix.patch` modifies SGLang v0.5.21 commit `e00930c5489053f26d86b179cee0d087f846acbb`, Apache-2.0. Full upstream checkout is separate and ignored.
- `configs/specbench-subset.jsonl` selects original question records from Spec-Bench commit `fd2c1cd7d2201ef71db4c5f4e455008f017967bf`, `data/spec_bench/question.jsonl`. Its Apache-2.0 repository license is retained in `configs/SPECBENCH_LICENSE`; original dataset categories/questions may have their own underlying provenance. The source file hash and deterministic selection are recorded in the source lock.
- Model: Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, downloaded separately from Hugging Face at GPU execution time; no weights are redistributed here.
- Paper: arXiv:2411.04975v3, referenced rather than redistributed. This repository's original integration/scripts/documentation are Apache-2.0.
