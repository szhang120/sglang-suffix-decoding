"""Attribute GPU kernels to matched first-verify CPU launch spans.

Compare actual launch counts/time; tensor M alone is not evidence of saved work.
Chrome traces are profiler diagnostics, not serving latency measurements.
"""

import argparse
import gzip
import json
import re
from pathlib import Path


def read_trace(path):
    # Stream multi-million-event traces instead of materializing Python objects
    # for the entire JSON. Only selected spans/correlations remain in memory.
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    chunk = 1024 * 1024
    with opener(path, "rt") as f:
        buffer = f.read(chunk)
        match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
        if match is None:
            raise ValueError("Missing traceEvents in trace header")
        buffer = buffer[match.end():]
        position = 0
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position == len(buffer):
                buffer, position = f.read(chunk), 0
                if not buffer:
                    raise ValueError("Truncated profiler trace")
                continue
            if buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buffer, position = buffer[position:] + more, 0
                continue
            yield event
            position = end
            if position > chunk:
                buffer, position = buffer[position:], 0


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
    reference = json.loads((args.directory / "reference.json").read_text())
    contexts = [case["input_tokens"] for case in reference["contexts"]] if "contexts" in reference else [len(reference["input_ids"])]
    pattern = re.compile(r"suffix_verify\[rows=(\d+) out=1 rid=probe-(?:(\d+)-)?(\d+)-(\d+)\]")
    for path in paths:
        spans = []
        for span in read_trace(path):
            match = pattern.fullmatch(span.get("name", ""))
            # Torch emits a GPU user annotation with the same label. Only CPU
            # annotations contain the launch-thread scope used for correlation.
            if not match or span.get("ph") != "X" or span.get("cat") != "user_annotation":
                continue
            rows, context, trial, requested = match.groups()
            rows, trial, requested = map(int, (rows, trial, requested))
            context = int(context) if context is not None else contexts[0]
            assert rows == requested
            key = (context, trial, rows)
            if key in seen:
                raise AssertionError(f"Duplicate CPU first-verify span: {key}")
            seen.add(key)
            spans.append(dict(span=span, context=context, trial=trial, rows=rows, kernels={}))
        assignments = {}
        for launch in read_trace(path):
            if launch.get("cat") not in ("cuda_runtime", "cuda_driver"):
                continue
            identifier = correlation(launch)
            if identifier is None:
                continue
            for index, item in enumerate(spans):
                span = item["span"]
                if launch.get("pid") != span.get("pid") or launch.get("tid") != span.get("tid"):
                    continue
                if span["ts"] <= launch.get("ts", -1) < span["ts"] + span["dur"]:
                    assert identifier not in assignments or assignments[identifier] == index
                    assignments[identifier] = index
        for event in read_trace(path):
            if event.get("cat") != "kernel":
                continue
            index = assignments.get(correlation(event))
            if index is None:
                continue
            by_name = spans[index]["kernels"]
            item = by_name.setdefault(event["name"], dict(count=0, gpu_us=0.0, grids=[]))
            item["count"] += 1
            item["gpu_us"] += event["dur"]
            grid = event.get("args", {}).get("grid")
            if grid is not None and grid not in item["grids"]:
                item["grids"].append(grid)
        for item in spans:
            by_name = item["kernels"]
            if not by_name:
                raise AssertionError("No correlated GPU kernels; inspect the profiler format")
            records.append(dict(context_tokens=item["context"], trial=item["trial"], verify_rows=item["rows"],
                                cpu_span_us=item["span"]["dur"],
                                gpu_kernel_count=sum(k["count"] for k in by_name.values()),
                                gpu_kernel_sum_us=sum(k["gpu_us"] for k in by_name.values()),
                                kernels=by_name, trace=str(path)))
    assert seen == {(context, trial, width) for context in contexts for trial in range(3) for width in (1, 2, 4, 8, 16, 33)}, (
        "Incomplete width profile", seen
    )
    destination = args.directory / "kernel-summary.json"
    if destination.exists():
        raise SystemExit(f"Preserving {destination}")
    destination.write_text(json.dumps(sorted(records, key=lambda r: (r["context_tokens"], r["verify_rows"], r["trial"])), indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "kernels"} for row in records], indent=2))


if __name__ == "__main__":
    main()
