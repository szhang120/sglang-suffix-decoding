"""Fail closed on output mismatches; aggregate measured runs, never invent them."""

import argparse
import json
import statistics

from gpu_common import ROOT


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    reference = ROOT / "results/correctness-ordinary.jsonl"
    if not reference.exists():
        raise SystemExit(
            "GPU results unavailable: run correctness and benchmarks on the rented GPU"
        )
    base = read(reference)
    if not base:
        raise SystemExit("Empty correctness reference")
    expected_names = [f"length-{n}" for n in (1, 2, 3, 17, 33, 65, 128)] + [
        "eos",
        "stop-string",
        "mismatch",
    ]
    expected_cases = [(name, repeat) for repeat in range(3) for name in expected_names]
    assert [(x["name"], x["repeat"]) for x in base] == expected_cases, (
        "Incomplete correctness reference"
    )
    for mode in ["ngram", "suffix"]:
        other = read(ROOT / f"results/correctness-{mode}.jsonl")
        assert len(base) == len(other)
        for x, y in zip(base, other):
            assert (x["name"], x["repeat"]) == (y["name"], y["repeat"])
            assert x["response"]["output_ids"] == y["response"]["output_ids"], (
                mode,
                x["name"],
                x["repeat"],
            )
    if args.correctness_only:
        print("Exact token-ID equality passed across ordinary, NGRAM and SUFFIX")
        return
    summary = []
    for path in sorted((ROOT / "results/gpu").glob("ordinary-*/requests.jsonl")):
        if "-profile" in str(path):
            continue
        trial = path.parent.name.split("-")[-1]
        baseline = read(path)
        expected_counts = {"independent": 52, "refinement": 84, "repeat": 104}
        assert {block: sum(x["block"] == block for x in baseline)
                for block in expected_counts} == expected_counts, (
            "Incomplete benchmark reference", trial
        )
        for mode in ["ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local"]:
            other = read(ROOT / f"results/gpu/{mode}-{trial}/requests.jsonl")
            assert len(other) == len(baseline)
            for x, y in zip(baseline, other):
                assert (x["block"], x["index"], x["question_id"]) == (
                    y["block"],
                    y["index"],
                    y["question_id"],
                )
                assert x["response"]["output_ids"] == y["response"]["output_ids"], (
                    mode,
                    trial,
                    x["block"],
                    x["index"],
                )
            for block in ["independent", "refinement", "repeat"]:
                b = [x for x in baseline if x["block"] == block]
                rows = [x for x in other if x["block"] == block]
                wall = sum(x["elapsed_ns"] for x in rows)
                tokens = sum(len(x["response"]["output_ids"]) for x in rows)
                summary.append(
                    dict(
                        mode=mode,
                        trial=int(trial),
                        block=block,
                        wall_seconds=wall / 1e9,
                        output_tokens=tokens,
                        tokens_per_second=tokens / (wall / 1e9),
                        aggregate_speedup=sum(x["elapsed_ns"] for x in b) / wall,
                        median_ttft_ms=statistics.median(
                            x["chunks"][0]["elapsed_ns"] / 1e6 for x in rows
                        ),
                    )
                )
    if not summary:
        raise SystemExit(
            "Correctness records exist; GPU benchmark results still unavailable"
        )
    assert len(summary) == 75, "Five complete trials required for the planned comparison"
    (ROOT / "results/gpu-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
