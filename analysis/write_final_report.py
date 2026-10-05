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
        "The benchmarks measure request latency under different cache-reuse conditions. Separate comparisons test the adaptive bound, the global cache and GPU verification width.", "",
        "The target model is Qwen2.5-7B-Instruct on one H100 80GB. All modes use greedy decoding with a batch size of 1. Weights and activations use BF16. The output head produces FP32 values.", "",
        "All modes share the deterministic Triton attention route and the fix to the output head's memory layout. Execution is eager, with seed 42. Each request can generate up to 256 tokens.", "",
        "See the [benchmark notes](docs/benchmark-notes.md) for execution requirements and comparison limits.", "",
        "Each trial runs ordinary decoding, three SUFFIX configurations and SGLang NGRAM PROB on the same 240 requests. Five trials produce 6,000 measurements, with the mode order rotated between trials.", "",
        "The workload uses 52 initial prompts and 32 follow-up turns from [Spec-Bench](https://github.com/hemingkx/Spec-Bench). Each follow-up includes the initial prompt, a fixed ordinary-decoding answer and the next user turn. Each mode runs three blocks, with empty suffix or NGRAM caches at the start of each block:", "",
        "- Independent: 52 initial prompts, each run once.",
        "- First + follow-up: 52 initial prompts and 32 follow-up conversation turns.",
        "- Repeated: 52 initial prompts, each run twice, for 104 requests.", "",
        "### Decoding latency", "",
        "Ordinary decoding generates one token per decode pass and serves as the baseline. Adaptive dual cache is SUFFIX with both caches and the match-length bound. The other SUFFIX configurations remove either the bound or the global cache.", "",
        "Speedup is total ordinary request latency divided by total request latency for the compared mode. Request latency includes host and streaming overhead. Values above 1 mean faster execution. Brackets show 95% paired bootstrap intervals from 10,000 trial resamples with seed 42. These intervals describe timing variation on this fixed workload.", "",
        "| Mode | Independent | First + follow-up | Follow-up only | Repeated |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in modes:
        follow = turns[(mode, "refinement", "refinement")]
        assert follow["requests"] == 160
        cells = [ratio(aggregates[(mode, "independent")]), ratio(aggregates[(mode, "refinement")]),
                 ratio(follow), ratio(aggregates[(mode, "repeat")])]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    text += ["", "The follow-up column includes only the 32 follow-up requests from each trial. The repeated column includes both passes of each identical prompt. Repeated prompts give favorable conditions for cache reuse. The adaptive SUFFIX results against ordinary decoding are:", "",
             "| Adaptive SUFFIX vs ordinary | Result |", "|---|---|"]
    for label, row in (("Independent", aggregates[("suffix", "independent")]),
                       ("Follow-up only", turns[("suffix", "refinement", "refinement")]),
                       ("Repeated", aggregates[("suffix", "repeat")])):
        text.append(f"| {label} | {finding(row)} |")

    ablations = {(r["comparator"], r["block"]): r for r in benchmark["ablation_comparisons"]}
    assert len(ablations) == 6 and all(r["paired_trials"] == 5 for r in ablations.values())
    text += ["", "The next comparison tests the value of the bound and the global cache separately. Each ablation removes one component. Removing the bound keeps the 32-token limit, probability threshold and output limit. Available continuations can also limit proposal length.", "",
             "Each ratio divides the ablated mode's latency by the adaptive dual-cache latency. Values above 1 favor the adaptive configuration.", "",
             "| Block | Bound vs no bound, speedup | Dual vs local cache, speedup |", "|---|---:|---:|"]
    for block, label in blocks:
        cells = [ratio(ablations[(comparator, block)], "pooled_speedup_ratio")
                 for comparator in ("suffix-fixed", "suffix-local")]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    bound_findings = [finding(ablations[("suffix-fixed", block)]).lower() for block, _ in blocks]
    if len(set(bound_findings)) == 1:
        bound_statement = f"Decoding with the adaptive bound is {bound_findings[0]} than decoding without it in all three blocks." if bound_findings[0] != "inconclusive" else "The adaptive bound comparison is inconclusive in all three blocks."
    else:
        bound_statement = "Adaptive bound results: " + "; ".join(f"{label.lower()}: {result}" for (_, label), result in zip(blocks, bound_findings)) + "."
    global_statements = []
    for block, label in blocks:
        outcome = finding(ablations[("suffix-local", block)])
        if outcome == "Inconclusive":
            global_statements.append(f"The global-cache comparison is inconclusive in the {label.lower()} block.")
        else:
            global_statements.append(f"The dual cache is {outcome.lower()} than the local cache alone in the {label.lower()} block.")
    text += ["", bound_statement + " Removing the bound can change both the selected continuation and its length.", "",
             " ".join(global_statements), "",
             "### Output checks", "",
             "The output checks compare SUFFIX's token IDs with ordinary decoding and check its cache updates. All timed SUFFIX configurations produce the same output token IDs as ordinary decoding. Separate checks cover suffix traces, verification, KV updates and the portable runner. The table lists passing comparisons and assertions:", "",
             "| Correctness check | Passing comparisons |", "|---|---:|",
             "| Ordinary, SUFFIX and suffix ablations in timed runs | 4,800 / 4,800 |",
             "| Separate suffix traces | 720 / 720 |",
             f"| Direct verification and KV assertions | {campaign['audited_rounds']} / {campaign['audited_rounds']} |",
             "| Controlled-width outputs | 54 / 54 |",
             "| Attention route comparison outputs | 24 / 24 |",
             "| Portable smoke outputs with empty and populated caches | 4 / 4 |", "",
             "### NGRAM PROB", "",
             f"SGLang NGRAM PROB also proposes draft tokens from cached sequences. Its output differs from ordinary output on {len(benchmark['mismatches'])} of 1,200 timed requests. The table gives descriptive latency ratios alongside the output differences.", "",
             "| Block | Ordinary / NGRAM latency | Differing outputs |", "|---|---:|---:|"]
    mismatches = Counter(r["block"] for r in benchmark["mismatches"])
    for block, label in blocks:
        row = aggregates[("ngram", block)]
        value = row.get("descriptive_latency_ratio", row.get("pooled_speedup"))
        text.append(f"| {label} | {value:.3f}× | {mismatches[block]} |")
    text += ["", "NGRAM and SUFFIX use different cache policies. NGRAM can merge branches even when each suffix anchor has only one continuation.", "",
             "### GPU verification width", "",
             "This test checks whether shorter verification passes reduce actual GPU kernel time. It holds the context fixed and changes the number of input rows. One row contains the last emitted token. A 33-row pass adds 32 draft tokens. The table shows median sums of GPU kernel times from three profiling trials:", "",
             "| Context tokens | Kernel time, 1 row | Kernel time, 33 rows | Reduction |", "|---:|---:|---:|---:|"]
    widths = {(r["context_tokens"], r["verify_rows"]): r for r in width["rows"]}
    for context in (126, 128, 512):
        one = widths[(context, 1)]["median_kernel_ms"]
        full = widths[(context, 33)]["median_kernel_ms"]
        text.append(f"| {context} | {one:.3f} ms | {full:.3f} ms | {100 * (1 - one / full):.1f}% |")
    route_low, route_high = route["trial_bootstrap_95"]
    route_finding = (
        "The shared route is slower in this comparison." if route_low > 1 else
        "The shared route is faster in this comparison." if route_high < 1 else
        "The interval includes 1, so the result is inconclusive."
    )
    text += ["", "Shorter verification passes reduce measured kernel time. The launch grids for KV storage and argmax become smaller. The attention and output-head grids stay the same. Kernel time includes profiler overhead.", "",
             "### Attention route", "",
             f"All benchmark modes share the same attention route. A separate check compares this route with the original decode route on two prompts and six paired trials. Shared-route latency divided by original-route latency is {ratio(route, 'pooled_shared_over_original_cost_ratio')}. Values above 1 mean the shared route is slower. {route_finding}", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text))
    print(f"Wrote verified measurement tables to {args.output}")


if __name__ == "__main__":
    main()
