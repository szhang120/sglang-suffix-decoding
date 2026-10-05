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


def validate_proposer(mode, environment):
    """Require the recorded configuration to implement the named experiment."""
    config = environment["config"]
    if mode == "ordinary":
        assert "speculative_algorithm" not in config, "Ordinary baseline enables speculation"
        return
    expected = dict(speculative_algorithm="SUFFIX" if mode.startswith("suffix") else "NGRAM",
                    speculative_num_draft_tokens=33, speculative_ngram_match_type="PROB",
                    speculative_ngram_max_trie_depth=64, speculative_ngram_max_bfs_breadth=1)
    assert all(config.get(k) == v for k, v in expected.items()), (mode, "Speculative configuration differs")
    if mode.startswith("suffix"):
        draft = environment["draft_environment"]
        expected_values = dict(SUFFIX_FACTOR=1.0, SUFFIX_OFFSET=0.0, SUFFIX_MIN_PROB=0.1,
                               SUFFIX_CACHE_REQUESTS=0 if mode == "suffix-local" else 128,
                               SUFFIX_FIXED=1 if mode == "suffix-fixed" else 0,
                               SUFFIX_ALLOW_WIDTH_PROBE=1)
        assert all(float(draft.get(k, "nan")) == v for k, v in expected_values.items()), (mode, "Suffix ablation configuration differs")


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


def exact_speedup_modes(report):
    """Reject failed reports; exclude explicitly permitted differing NGRAM."""
    if not report.get("exact_suffix_ids_passed", report.get("exact_ids_passed", False)):
        raise ValueError("Suffix output equality failed")
    mismatches = report.get("mismatches", [])
    if mismatches:
        if (not report.get("ngram_numerical_differences_permitted", False)
                or any(row["mode"] != "ngram" for row in mismatches)):
            raise ValueError("Report failed its explicit numerical policy")
        return ("suffix", "suffix-fixed", "suffix-local")
    if not report.get("exact_ids_passed", False):
        raise ValueError("Report lacks a passing exact-output gate")
    return ("ngram", "suffix", "suffix-fixed", "suffix-local")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Downloaded results directory containing gpu/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-ngram-numerical-differences", action="store_true",
                        help="Keep strict suffix gates; report differing NGRAM outputs as descriptive latency ratios only")
    args = parser.parse_args()
    mismatches = []
    summaries = []
    hashes = {}
    environments = {}
    controls = None
    identity = None
    grouped_pairs = defaultdict(list)
    kind_pairs = defaultdict(list)
    kind_counts = defaultdict(int)
    categories = defaultdict(lambda: [0, 0, 0])
    for trial in range(5):
        records = {}
        for mode in MODES:
            path = args.directory / "gpu" / f"{mode}-{trial}" / "requests.jsonl"
            records[mode] = load(path)
            hashes[str(path.relative_to(args.directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            env_path = path.parent / "environment.json"
            environment = json.loads(env_path.read_text())
            validate_proposer(mode, environment)
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
                                           ordinary_ids=x[first:first + 8], mode_ids=y[first:first + 8],
                                           ordinary_output_tokens=len(x), mode_output_tokens=len(y)))
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
                for kind in sorted({r["kind"] for r in subset}):
                    original = [r for r in b if r["kind"] == kind]
                    measured = [r for r in subset if r["kind"] == kind]
                    kind_pairs[(mode, block, kind)].append((sum(r["elapsed_ns"] for r in original),
                                                           sum(r["elapsed_ns"] for r in measured)))
                    kind_counts[(mode, block, kind)] += len(measured)
                summaries.append(dict(mode=mode, trial=trial, block=block,
                                      wall_seconds=mode_ns / 1e9, output_tokens=tokens,
                                      ordinary_output_tokens=sum(len(r["response"]["output_ids"]) for r in b),
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
    ablations = []
    for comparator in ("suffix-fixed", "suffix-local"):
        for block in BLOCKS:
            adaptive = grouped_pairs[("suffix", block)]
            compared = grouped_pairs[(comparator, block)]
            assert len(adaptive) == len(compared) == 5
            assert all(a[0] == b[0] for a, b in zip(adaptive, compared))
            # Same-trial policy comparison; ordinary cancels in the ratio of
            # the two ordinary/mode speedups. Resample the paired wall times.
            pairs = [(b[1], a[1]) for a, b in zip(adaptive, compared)]
            ratios = [a / b for a, b in pairs]
            ablations.append(dict(
                comparator=comparator, block=block, paired_trials=5,
                requests_per_mode=5 * BLOCKS[block],
                metric="Comparator/adaptive wall latency; above1 means adaptive is faster",
                pooled_speedup_ratio=sum(a for a, _ in pairs) / sum(b for _, b in pairs),
                observed_trial_range=[min(ratios), max(ratios)],
                trial_bootstrap_95=intervals(pairs),
            ))
    exact_modes_passed = not any(m["mode"] != "ngram" for m in mismatches)
    ngram_differs = any(m["mode"] == "ngram" for m in mismatches)
    kind_rows = []
    for (mode, block, kind), pairs in sorted(kind_pairs.items()):
        ratios = [a / b for a, b in pairs]
        kind_rows.append(dict(mode=mode, block=block, kind=kind,
                              requests=kind_counts[(mode, block, kind)],
                              pooled_speedup=sum(a for a, _ in pairs) / sum(b for _, b in pairs),
                              observed_trial_range=[min(ratios), max(ratios)],
                              trial_bootstrap_95=intervals(pairs)))
    if args.allow_ngram_numerical_differences and ngram_differs:
        for row in summaries:
            if row["mode"] == "ngram":
                row["descriptive_latency_ratio"] = row.pop("aggregate_speedup")
                row["different_outputs"] = True
        for row in aggregate:
            if row["mode"] == "ngram":
                row["descriptive_latency_ratio"] = row.pop("pooled_speedup")
                row["median_trial_latency_ratio"] = row.pop("median_trial_speedup")
                row["different_outputs"] = True
        for row in kind_rows:
            if row["mode"] == "ngram":
                row["descriptive_latency_ratio"] = row.pop("pooled_speedup")
                row["different_outputs"] = True
    category_rows = [dict(mode=m, block=b, category=c, requests=n, pooled_speedup=a / d)
                     for (m, b, c), (a, d, n) in sorted(categories.items())]
    if args.allow_ngram_numerical_differences and ngram_differs:
        for row in category_rows:
            if row["mode"] == "ngram":
                row["descriptive_latency_ratio"] = row.pop("pooled_speedup")
                row["different_outputs"] = True
    report = dict(exact_ids_passed=not mismatches, exact_suffix_ids_passed=exact_modes_passed, mismatches=mismatches,
                  ngram_numerical_differences_permitted=args.allow_ngram_numerical_differences,
                  ngram_limitation="Different responses and possibly lengths; descriptive NGRAM latency ratios cannot establish an exact-output speedup. One independently reproduced masked-tree rounding case does not attribute all divergences.",
                  measured_requests=5 * len(MODES) * sum(BLOCKS.values()),
                  metric="Paired ratio of summed request wall latency, ordinary/mode; includes host and streaming overhead",
                  proposer_configurations_checked=True,
                  uncertainty="10,000 paired trial-block bootstrap samples, seed42; fixed workload, five trials, percentile interval",
                  common_config=controls,
                  gpu_uuid_by_trial={str(trial): environments[("ordinary", trial)] for trial in range(5)},
                  aggregate=aggregate, trials=summaries,
                  categories=category_rows, request_kinds=kind_rows, ablation_comparisons=ablations,
                  raw_sha256=hashes)
    if args.output.exists():
        raise SystemExit(f"Preserving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(exact_ids_passed=not mismatches, mismatches=len(mismatches), aggregate=aggregate), indent=2))
    if mismatches and not (args.allow_ngram_numerical_differences and exact_modes_passed):
        raise SystemExit("Measured outputs differ; results cannot establish an exact-output performance comparison")


if __name__ == "__main__":
    main()
