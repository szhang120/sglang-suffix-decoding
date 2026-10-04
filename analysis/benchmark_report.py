"""Paired latency ratios and exact-ID checks from complete, unprofiled trials.

Confidence intervals resample trial blocks, not individual tokens/requests.
They describe run-to-run variation on this fixed workload, not generalization.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

MODES = ("ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local")
BLOCKS = {"independent": 52, "refinement": 84, "repeat": 104}


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(sorted_values, fraction):
    index = (len(sorted_values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def intervals(pairs):
    rng = random.Random(42)
    ratios = sorted(sum(pairs[i][0] for i in indices) / sum(pairs[i][1] for i in indices)
                    for indices in ([rng.randrange(len(pairs)) for _ in pairs] for _ in range(10000)))
    return [percentile(ratios, 0.025), percentile(ratios, 0.975)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Downloaded results directory containing gpu/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mismatches = []
    summaries = []
    hashes = {}
    environments = {}
    controls = None
    identity = None
    grouped_pairs = defaultdict(list)
    categories = defaultdict(lambda: [0, 0, 0])
    for trial in range(5):
        records = {}
        for mode in MODES:
            path = args.directory / "gpu" / f"{mode}-{trial}" / "requests.jsonl"
            records[mode] = load(path)
            hashes[str(path.relative_to(args.directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            env_path = path.parent / "environment.json"
            environment = json.loads(env_path.read_text())
            hashes[str(env_path.relative_to(args.directory))] = hashlib.sha256(env_path.read_bytes()).hexdigest()
            common = {k: v for k, v in environment["config"].items() if not k.startswith("speculative_")}
            provenance = (environment["workload_sha256"], environment["integration_patch_sha256"],
                          environment["source_lock"]["model"], environment["torch"], environment["cuda"],
                          environment["pip_freeze"], environment["kernel_environment"])
            if controls is None:
                controls, identity = common, provenance
            assert common == controls and provenance == identity, (mode, trial, "environment differs")
            assert environment["gpu"] == "NVIDIA H100 80GB HBM3"
            match = re.search(r"GPU UUID\s*:\s*(\S+)", environment["nvidia_smi"])
            assert match, "Missing GPU UUID"
            environments[(mode, trial)] = match.group(1)
            assert len(records[mode]) == sum(BLOCKS.values()), (mode, trial, "incomplete")
            assert {block: sum(r["block"] == block for r in records[mode]) for block in BLOCKS} == BLOCKS
        base = records["ordinary"]
        assert len({environments[(mode, trial)] for mode in MODES}) == 1, "GPU changed within a paired trial"
        for mode, rows in records.items():
            for reference, row in zip(base, rows):
                assert all(reference[k] == row[k] for k in ("block", "index", "question_id", "kind", "category", "input_tokens", "truncated"))
                x, y = reference["response"]["output_ids"], row["response"]["output_ids"]
                if x != y:
                    first = next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), min(len(x), len(y)))
                    mismatches.append(dict(mode=mode, trial=trial, block=row["block"], index=row["index"],
                                           question_id=row["question_id"], first_difference=first,
                                           ordinary_ids=x[first:first + 8], mode_ids=y[first:first + 8]))
                key = (mode, row["block"], row["category"])
                categories[key][0] += reference["elapsed_ns"]
                categories[key][1] += row["elapsed_ns"]
                categories[key][2] += 1
            for block in BLOCKS:
                b = [r for r in base if r["block"] == block]
                subset = [r for r in rows if r["block"] == block]
                base_ns = sum(r["elapsed_ns"] for r in b)
                mode_ns = sum(r["elapsed_ns"] for r in subset)
                tokens = sum(len(r["response"]["output_ids"]) for r in subset)
                grouped_pairs[(mode, block)].append((base_ns, mode_ns))
                summaries.append(dict(mode=mode, trial=trial, block=block,
                                      wall_seconds=mode_ns / 1e9, output_tokens=tokens,
                                      tokens_per_second=tokens * 1e9 / mode_ns,
                                      aggregate_speedup=base_ns / mode_ns,
                                      median_ttft_ms=statistics.median(r["chunks"][0]["elapsed_ns"] / 1e6 for r in subset),
                                      median_request_ms=statistics.median(r["elapsed_ns"] / 1e6 for r in subset)))
    aggregate = []
    for mode in MODES:
        for block in BLOCKS:
            pairs = grouped_pairs[(mode, block)]
            speedups = [a / b for a, b in pairs]
            aggregate.append(dict(mode=mode, block=block,
                                  pooled_speedup=sum(a for a, _ in pairs) / sum(b for _, b in pairs),
                                  median_trial_speedup=statistics.median(speedups),
                                  observed_trial_range=[min(speedups), max(speedups)],
                                  trial_bootstrap_95=intervals(pairs),
                                  median_trial_ttft_ms=statistics.median(r["median_ttft_ms"] for r in summaries
                                                                       if (r["mode"], r["block"]) == (mode, block))))
    report = dict(exact_ids_passed=not mismatches, mismatches=mismatches,
                  measured_requests=5 * len(MODES) * sum(BLOCKS.values()),
                  metric="Paired ratio of summed request wall latency, ordinary/mode; includes host and streaming overhead",
                  uncertainty="10,000 paired trial-block bootstrap samples, seed42; fixed workload, five trials, percentile interval",
                  common_config=controls,
                  gpu_uuid_by_trial={str(trial): environments[("ordinary", trial)] for trial in range(5)},
                  aggregate=aggregate, trials=summaries,
                  categories=[dict(mode=m, block=b, category=c, requests=n, pooled_speedup=a / d)
                              for (m, b, c), (a, d, n) in sorted(categories.items())],
                  raw_sha256=hashes)
    if args.output.exists():
        raise SystemExit(f"Preserving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(exact_ids_passed=not mismatches, mismatches=len(mismatches), aggregate=aggregate), indent=2))
    if mismatches:
        raise SystemExit("Measured outputs differ; results cannot establish an exact-output performance comparison")


if __name__ == "__main__":
    main()
