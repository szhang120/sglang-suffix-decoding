"""Export compact measurement tables after serving and diagnostic gates pass."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(name):
        return json.loads((args.results_directory / name).read_text())

    benchmark = read("benchmark-report.json")
    trace = read("natural-trace-report.json")
    width = read("width-probe-summary.json")
    route = read("decode-control-report.json")
    provenance = read("execution-provenance.json")
    campaign = provenance["campaign"]
    diagnostics = provenance["diagnostics"]
    assert campaign["success"] and diagnostics["success"]
    assert provenance["controller"]["success"]
    assert campaign["source_sha256"] == diagnostics["source_sha256"]
    assert campaign["completed_trials"] == list(range(5))
    assert len(campaign["completed_modes"]) == 25
    assert all(not r["mismatches"] for r in campaign["completed_modes"] if r["mode"] != "ngram")
    assert len(diagnostics["completed_phases"]) == 16
    assert diagnostics["portable_runner_smoke"]["success"]
    assert benchmark["exact_suffix_ids_passed"] and benchmark["measured_requests"] == 6000
    assert all(r["mode"] == "ngram" for r in benchmark["mismatches"])
    assert len(trace["summaries"]) == 9 and sum(r["requests"] for r in trace["summaries"]) == 720
    assert len(width["rows"]) == 18 and len(width["traces"]) == 54
    assert route["exact_ids_passed"] and route["measured_requests"] == 24
    assert route["paired_trials"] == 6
    assert not args.output.exists(), f"Preserving {args.output}"

    modes = (("suffix", "Adaptive dual cache"), ("suffix-fixed", "Without match-length bound"),
             ("suffix-local", "Local cache only"))
    blocks = (("independent", "Independent"), ("refinement", "First + follow-up"),
              ("repeat", "Repeated"))
    aggregates = {(r["mode"], r["block"]): r for r in benchmark["aggregate"]}
    turns = {(r["mode"], r["block"], r["kind"]): r for r in benchmark["request_kinds"]}

    def ratio(row, key="pooled_speedup"):
        low, high = row["trial_bootstrap_95"]
        return f"{row[key]:.3f}× [{low:.3f}–{high:.3f}]"

    def finding(row):
        low, high = row["trial_bootstrap_95"]
        return "Faster" if low > 1 else "Slower" if high < 1 else "Inconclusive"

    text = [
        "## Results", "",
        "Qwen2.5-7B-Instruct; one H100 80GB; greedy batch 1; BF16 with an FP32 output head.", "",
        "Deterministic Triton attention; eager execution; seed 42; output limit 256. Graphs, overlap and radix caching disabled.", "",
        "Five rotated trials × five modes × 240 requests = 6,000 measurements. Inputs: 52 initial prompts and 32 follow-ups. Blocks: 52 independent, 84 first/follow-up, 104 repeated requests; caches start empty per block.", "",
        "Speedup = ordinary/mode summed request latency, including host/streaming overhead. Above 1 is faster. Brackets: 95% paired bootstrap intervals, 10,000 trial resamples, seed 42; fixed-workload timing variation.", "",
        "| Mode | Independent | First + follow-up | Follow-up only | Repeated |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in modes:
        follow = turns[(mode, "refinement", "refinement")]
        assert follow["requests"] == 160
        cells = [ratio(aggregates[(mode, "independent")]), ratio(aggregates[(mode, "refinement")]),
                 ratio(follow), ratio(aggregates[(mode, "repeat")])]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    text += ["", "Repeated includes both identical-prompt passes, a favorable reuse test. The bound-removal variant still uses probability, available-continuation and output-budget limits.", "",
             "| Adaptive SUFFIX vs ordinary | Result |", "|---|---|"]
    for label, row in (("Independent", aggregates[("suffix", "independent")]),
                       ("Follow-up only", turns[("suffix", "refinement", "refinement")]),
                       ("Repeated", aggregates[("suffix", "repeat")])):
        text.append(f"| {label} | {finding(row)} |")

    ablations = {(r["comparator"], r["block"]): r for r in benchmark["ablation_comparisons"]}
    assert len(ablations) == 6 and all(r["paired_trials"] == 5 for r in ablations.values())
    text += ["", "Ablation ratios: above 1 favors adaptive dual cache.", "",
             "| Block | Adaptive / without match-length bound | Dual cache / local only |", "|---|---:|---:|"]
    for block, label in blocks:
        cells = [ratio(ablations[(comparator, block)], "pooled_speedup_ratio")
                 for comparator in ("suffix-fixed", "suffix-local")]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    bound_findings = [finding(ablations[("suffix-fixed", block)]).lower() for block, _ in blocks]
    text += ["", "Adaptive bound vs removal (independent / first + follow-up / repeated): " + " / ".join(bound_findings) + ". Candidate selection also changes; the cause is not isolated.", "",
             "| Correctness check | Passing comparisons |", "|---|---:|",
             "| Ordinary, SUFFIX and suffix ablations in timed runs | 4,800 / 4,800 |",
             "| Separate suffix traces | 720 / 720 |",
             f"| Direct verification/KV assertions | {campaign['audited_rounds']} / {campaign['audited_rounds']} |",
             "| Controlled-width outputs | 54 / 54 |",
             "| Ordinary-route outputs | 24 / 24 |",
             "| Portable cold/warm smoke outputs | 4 / 4 |", "",
             "### NGRAM PROB", "",
             f"NGRAM differs from ordinary output on {len(benchmark['mismatches'])}/1,200 timed requests. The ratios below are descriptive latency ratios, not exact-output speedups.", "",
             "| Block | Ordinary / NGRAM latency | Differing outputs |", "|---|---:|---:|"]
    mismatches = Counter(r["block"] for r in benchmark["mismatches"])
    for block, label in blocks:
        row = aggregates[("ngram", block)]
        value = row.get("descriptive_latency_ratio", row.get("pooled_speedup"))
        text.append(f"| {label} | {value:.3f}× | {mismatches[block]} |")
    text += ["", "NGRAM uses different caches and can branch at fanout 1. SUFFIX output checks remain strict.", "",
             "### GPU verification width", "",
             "| Context tokens | One-row kernel sum | 33-row kernel sum | Reduction |", "|---:|---:|---:|---:|"]
    widths = {(r["context_tokens"], r["verify_rows"]): r for r in width["rows"]}
    for context in (126, 128, 512):
        one = widths[(context, 1)]["median_kernel_ms"]
        full = widths[(context, 33)]["median_kernel_ms"]
        text.append(f"| {context} | {one:.3f} ms | {full:.3f} ms | {100 * (1 - one / full):.1f}% |")
    text += ["", "Three profiling trials. KV-store/argmax grids shrink; attention/head grids remain unchanged. Profiled kernel sums are neither request latency nor FLOPs.", "",
             f"Two-prompt, six-pair route control: shared/original latency {ratio(route, 'pooled_shared_over_original_cost_ratio')}; above 1 means the shared route is slower. General baseline cost remains unestablished.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text))
    print(f"Wrote verified measurement tables to {args.output}")


if __name__ == "__main__":
    main()
