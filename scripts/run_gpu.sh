#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/native:$PWD/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
for mode in ordinary ngram suffix; do
  python3 scripts/gpu_correctness.py --mode "$mode"
done
python3 scripts/analyze_gpu.py --correctness-only
python3 scripts/materialize_workload.py
# Balanced rotation reduces temporal bias; each process owns a fresh engine.
orders=("ordinary ngram suffix suffix-fixed suffix-local" \
        "ngram suffix suffix-fixed suffix-local ordinary" \
        "suffix suffix-fixed suffix-local ordinary ngram" \
        "suffix-fixed suffix-local ordinary ngram suffix" \
        "suffix-local ordinary ngram suffix suffix-fixed")
for trial in 0 1 2 3 4; do
  for mode in ${orders[$trial]}; do
    python3 scripts/gpu_bench.py --mode "$mode" --trial "$trial"
  done
done
python3 scripts/analyze_gpu.py
for mode in ordinary ngram suffix; do
  python3 scripts/gpu_bench.py --mode "$mode" --trial 0 --profile
done
