"""Quantify the shared ordinary attention route's cost on matched outputs."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from benchmark_report import intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Contains shared0-N and shared1-N")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials, pairs, hashes = [], [], {}
    reference_outputs = None
    common = None
    uuids = set()
    for trial in range(6):
        records = {}
        for shared in (0, 1):
            directory = args.directory / f"shared{shared}-{trial}"
            request_path = directory / "requests.jsonl"
            env_path = directory / "environment.json"
            records[shared] = [json.loads(line) for line in request_path.read_text().splitlines()]
            env = json.loads(env_path.read_text())
            assert env["shared_decode"] == shared
            assert env["gpu"] == "NVIDIA H100 80GB HBM3"
            identity = (env["config"], env["source_lock"]["model"], env["torch"], env["cuda"],
                        env["workload_sha256"], env["control_script_sha256"])
            if common is None:
                common = identity
            assert identity == common, "Control settings differ"
            match = re.search(r"GPU UUID\s*:\s*(\S+)", env["nvidia_smi"])
            assert match
            uuids.add(match.group(1))
            assert [r["question_id"] for r in records[shared]] == [81, 82]
            outputs = [r["response"]["output_ids"] for r in records[shared]]
            if reference_outputs is None:
                reference_outputs = outputs
            assert outputs == reference_outputs, "Route control outputs differ"
            for path in (request_path, env_path):
                hashes[str(path.relative_to(args.directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        ordinary_ns = sum(r["elapsed_ns"] for r in records[0])
        shared_ns = sum(r["elapsed_ns"] for r in records[1])
        pairs.append((shared_ns, ordinary_ns))
        trials.append(dict(trial=trial, order=[0, 1] if trial % 2 == 0 else [1, 0],
                           original_wall_seconds=ordinary_ns / 1e9, shared_wall_seconds=shared_ns / 1e9,
                           shared_over_original_cost_ratio=shared_ns / ordinary_ns,
                           original_median_ttft_ms=statistics.median(r["chunks"][0]["elapsed_ns"] for r in records[0]) / 1e6,
                           shared_median_ttft_ms=statistics.median(r["chunks"][0]["elapsed_ns"] for r in records[1]) / 1e6))
    assert len(uuids) == 1, "GPU changed within the route control"
    ratios = [a / b for a, b in pairs]
    report = dict(
        exact_ids_passed=True, measured_requests=24, paired_trials=6,
        metric="Shared/original summed ordinary request wall latency; above1 means a slower shared route",
        selection="Two writing prompts q81/q82 selected because both attention routes matched in the prior diagnostic",
        limitation="Small selected control; not a representative public workload or optimized production SGLang comparison",
        output_tokens_per_prompt=[len(ids) for ids in reference_outputs],
        pooled_shared_over_original_cost_ratio=sum(a for a, _ in pairs) / sum(b for _, b in pairs),
        median_trial_cost_ratio=statistics.median(ratios), observed_trial_range=[min(ratios), max(ratios)],
        trial_bootstrap_95=intervals(pairs),
        uncertainty="Paired trial bootstrap on these two fixed prompts, seed42; six trials",
        common_config=common[0], model=common[1], gpu_uuid=next(iter(uuids)),
        trials=trials, raw_sha256=hashes,
    )
    if args.output.exists():
        raise SystemExit(f"Preserving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trials", "raw_sha256", "common_config")}, indent=2))


if __name__ == "__main__":
    main()
