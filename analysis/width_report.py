"""Summarize correlated kernels; timings are diagnostic, not serving latency."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def family(name):
    if "store_kvcache_kernel" in name:
        return "kv_store"
    if "ArgMaxOps" in name:
        return "argmax"
    if name == "_fwd_kernel_unified":
        return "attention"
    if name == "matmul_kernel_persistent":
        return "lm_head"
    if "gemm" in name.lower():
        return "dense_gemm"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    source = args.directory / "kernel-summary.json"
    rows = json.loads(source.read_text())
    summary = []
    for context in sorted({r["context_tokens"] for r in rows}):
        for width in sorted({r["verify_rows"] for r in rows}):
            group = [r for r in rows if (r["context_tokens"], r["verify_rows"]) == (context, width)]
            assert len(group) == 3 and {r["trial"] for r in group} == {0, 1, 2}
            families = {}
            for label in ("kv_store", "argmax", "attention", "lm_head", "dense_gemm", "other"):
                samples = [sum(k["gpu_us"] for name, k in r["kernels"].items()
                               if family(name) == label) for r in group]
                grids = sorted({tuple(grid) for r in group
                                for name, k in r["kernels"].items() if family(name) == label
                                for grid in k["grids"]})
                families[label] = dict(median_gpu_us=statistics.median(samples), grids=grids)
            samples = [r["gpu_kernel_sum_us"] / 1000 for r in group]
            summary.append(dict(context_tokens=context, verify_rows=width,
                                median_kernel_ms=statistics.median(samples),
                                range_kernel_ms=[min(samples), max(samples)],
                                samples_kernel_ms=samples, families=families))
    traces = []
    for path in sorted((args.directory / "profile").rglob("*.gz")):
        traces.append(dict(path=str(path.relative_to(args.directory)), bytes=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(kernel_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  metric="Sum of GPU kernel durations correlated to the first verification CPU span",
                  caveat="Profiler overhead; sums are not wall latency, hardware FLOP counts, or production speedups.",
                  rows=summary, traces=traces)
    destination = args.directory / "summary.json"
    if destination.exists():
        raise SystemExit(f"Preserving {destination}")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
