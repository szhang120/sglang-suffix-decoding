"""Attribute GPU kernels to matched first-verify CPU launch spans.

Compare actual launch counts/time; tensor M alone is not evidence of saved work.
Chrome traces are profiler diagnostics, not serving latency measurements.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


def read_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def correlation(event):
    args = event.get("args", {})
    return args.get("correlation", args.get("Correlation ID"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    seen = set()
    paths = sorted((args.directory / "profile").rglob("*.json"))
    paths += sorted((args.directory / "profile").rglob("*.json.gz"))
    pattern = re.compile(r"suffix_verify\[rows=(\d+) out=1 rid=probe-(\d+)-(\d+)\]")
    for path in paths:
        events = read_trace(path)
        launches = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
        kernels = defaultdict(list)
        for event in events:
            if event.get("cat") == "kernel" and correlation(event) is not None:
                kernels[correlation(event)].append(event)
        for span in events:
            match = pattern.fullmatch(span.get("name", ""))
            if not match or span.get("ph") != "X":
                continue
            rows, trial, requested = map(int, match.groups())
            assert rows == requested
            key = (trial, rows)
            if key in seen:
                raise AssertionError(f"Duplicate first-verify span: {key}")
            seen.add(key)
            selected = []
            for launch in launches:
                if launch.get("pid") != span.get("pid") or launch.get("tid") != span.get("tid"):
                    continue
                if span["ts"] <= launch.get("ts", -1) < span["ts"] + span["dur"]:
                    selected.extend(kernels.get(correlation(launch), []))
            if not selected:
                raise AssertionError("No correlated GPU kernels; inspect the profiler format")
            by_name = defaultdict(lambda: dict(count=0, gpu_us=0.0, grids=[]))
            for event in selected:
                item = by_name[event["name"]]
                item["count"] += 1
                item["gpu_us"] += event["dur"]
                grid = event.get("args", {}).get("grid")
                if grid is not None and grid not in item["grids"]:
                    item["grids"].append(grid)
            records.append(dict(trial=trial, verify_rows=rows, cpu_span_us=span["dur"],
                                gpu_kernel_count=len(selected),
                                gpu_kernel_sum_us=sum(e["dur"] for e in selected),
                                kernels=dict(by_name), trace=str(path)))
    assert seen == {(trial, width) for trial in range(3) for width in (1, 2, 4, 8, 16, 33)}, (
        "Incomplete width profile", seen
    )
    destination = args.directory / "kernel-summary.json"
    if destination.exists():
        raise SystemExit(f"Preserving {destination}")
    destination.write_text(json.dumps(sorted(records, key=lambda r: (r["verify_rows"], r["trial"])), indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "kernels"} for row in records], indent=2))


if __name__ == "__main__":
    main()
