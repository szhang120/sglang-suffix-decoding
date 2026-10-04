"""Actual adaptive row/acceptance counts from separate full-workload traces.

Configured SGLang draft-capacity metadata is not an adaptive denominator.
Profiler/trace latencies are deliberately excluded from this analysis.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = ("suffix", "suffix-fixed", "suffix-local")
BLOCKS = {"independent": 52, "refinement": 84, "repeat": 104}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(requests, rounds):
    widths = [r["verify_rows"] for r in rounds]
    accepted = [r["accepted_with_bonus"][0] for r in rounds]
    proposed = sum(w - 1 for w in widths)
    accepted_drafts = sum(a - 1 for a in accepted)
    draft_rounds = sum(w > 1 for w in widths)
    committed = sum(r["committed_tokens"] for r in rounds)
    return dict(
        requests=len(requests), verify_rounds=len(rounds),
        output_tokens=sum(len(r["response"]["output_ids"]) for r in requests),
        mean_verify_rows=statistics.mean(widths) if widths else None,
        median_verify_rows=statistics.median(widths) if widths else None,
        width_histogram=dict(sorted(Counter(widths).items())),
        proposed_drafts=proposed, target_accepted_drafts=accepted_drafts,
        target_draft_acceptance=accepted_drafts / proposed if proposed else None,
        mean_target_emissions_per_round=statistics.mean(accepted) if accepted else None,
        mean_committed_tokens_per_round=committed / len(rounds) if rounds else None,
        one_row_fraction=sum(w == 1 for w in widths) / len(rounds) if rounds else None,
        full_match_fraction_of_draft_rounds=sum(w > 1 and a == w for w, a in zip(widths, accepted)) / draft_rounds if draft_rounds else None,
        terminal_tokens_discarded=sum(accepted) - committed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Contains VARIANT/results/ trace directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        metric="Actual verification rows and accepted drafts, excluding prefill",
        target_acceptance_denominator="Sum of verify_rows - 1, excluding the pending root",
        stop_handling="Target acceptance may include a terminal tail discarded at EOS; committed emissions are clipped to final output length",
        timing="Separate instrumented runs; no serving-latency claims",
        summaries=[], categories=[], request_kinds=[], raw_sha256={},
    )
    provenance = None
    for variant in VARIANTS:
        directory = args.directory / variant / "results"
        equality = json.loads((directory / "trace-equality.json").read_text())
        assert equality["variant"] == variant and equality["requests"] == 240 and not equality["mismatches"]
        paths = [directory / name for name in (
            "public-gate-suffix-shared1.jsonl", "suffix-rounds.jsonl",
            "public-gate-suffix-shared1-environment.json", "trace-equality.json")]
        for path in paths:
            report["raw_sha256"][str(path.relative_to(args.directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        env = json.loads(paths[2].read_text())
        assert env["effective_unified_decode"] == 1
        identity = (env["workload_sha256"], env["config"], equality["reference_run"])
        if provenance is None:
            provenance = identity
        assert identity == provenance, "Trace inputs/config/reference differ"
        requests = read(paths[0])
        assert len(requests) == 240
        assert {b: sum(r["block"] == b for r in requests) for b in BLOCKS} == BLOCKS
        by_rid = {r["response"]["meta_info"]["id"]: r for r in requests}
        assert len(by_rid) == 240
        rounds = read(paths[1])
        grouped = defaultdict(list)
        for row in rounds:
            request = by_rid[row["rid"]]
            width = row["verify_rows"]
            accepted = row["accepted_with_bonus"][0]
            assert 1 <= accepted <= width <= 33
            assert len(row["input_ids"]) == width
            remaining = len(request["response"]["output_ids"]) - row["output_len_before"]
            assert remaining > 0, "Round starts beyond committed completion"
            row["committed_tokens"] = min(accepted, remaining)
            row["block"] = request["block"]
            row["category"] = request["category"]
            row["kind"] = request["kind"]
            grouped[row["rid"]].append(row)
        for rid, request in by_rid.items():
            previous_end = 1  # Prefill emits the first token.
            for row in grouped[rid]:
                assert row["output_len_before"] == previous_end, "Gap or duplicate verification round"
                previous_end += row["committed_tokens"]
            assert previous_end == len(request["response"]["output_ids"]), "Incomplete trace"
        for block in BLOCKS:
            selected_requests = [r for r in requests if r["block"] == block]
            selected_rounds = [r for r in rounds if r["block"] == block]
            report["summaries"].append(dict(variant=variant, block=block,
                                             **summarize(selected_requests, selected_rounds)))
            for kind in sorted({r["kind"] for r in selected_requests}):
                report["request_kinds"].append(dict(
                    variant=variant, block=block, kind=kind,
                    **summarize([r for r in selected_requests if r["kind"] == kind],
                                [r for r in selected_rounds if r["kind"] == kind])))
            for category in sorted({r["category"] for r in selected_requests}):
                report["categories"].append(dict(
                    variant=variant, block=block, category=category,
                    **summarize([r for r in selected_requests if r["category"] == category],
                                [r for r in selected_rounds if r["category"] == category])))
    report["workload_sha256"], report["common_config"], report["reference_run"] = provenance
    if args.output.exists():
        raise SystemExit(f"Preserving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
