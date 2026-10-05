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
        "The model is Qwen2.5-7B-Instruct on one H100 80GB. Decoding is greedy, with a batch size of 1. Weights and activations use BF16. The output head produces FP32 values.", "",
        "All modes use deterministic Triton attention, eager execution and seed 42. The output limit is 256 tokens. CUDA graphs, overlap and radix caching are disabled.", "",
        "Five trials × five modes × 240 requests = 6,000 measurements. The mode order rotates between trials. Inputs include 52 initial prompts and 32 follow-ups. Each block has 52 independent, 84 first and follow-up, or 104 repeated requests. Each block starts with empty algorithm caches.", "",
        "Speedup is total ordinary request latency divided by total mode latency. It includes host and streaming overhead. Values above 1 mean faster execution. Brackets show 95% paired bootstrap intervals from 10,000 trial resamples with seed 42. The intervals describe timing variation on this workload.", "",
        "| Mode | Independent | First + follow-up | Follow-up only | Repeated |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in modes:
        follow = turns[(mode, "refinement", "refinement")]
        assert follow["requests"] == 160
        cells = [ratio(aggregates[(mode, "independent")]), ratio(aggregates[(mode, "refinement")]),
                 ratio(follow), ratio(aggregates[(mode, "repeat")])]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    text += ["", "The repeated block includes the first and second passes of each identical prompt. This test gives favorable conditions for cache reuse. Without the match-length bound, probability, available continuations and the output limit still constrain proposals.", "",
             "| Adaptive SUFFIX vs ordinary | Result |", "|---|---|"]
    for label, row in (("Independent", aggregates[("suffix", "independent")]),
                       ("Follow-up only", turns[("suffix", "refinement", "refinement")]),
                       ("Repeated", aggregates[("suffix", "repeat")])):
        text.append(f"| {label} | {finding(row)} |")

    ablations = {(r["comparator"], r["block"]): r for r in benchmark["ablation_comparisons"]}
    assert len(ablations) == 6 and all(r["paired_trials"] == 5 for r in ablations.values())
    text += ["", "Each ablation removes one component. Ratios above 1 favor the adaptive method with both caches.", "",
             "| Block | Adaptive / without match-length bound | Dual cache / local only |", "|---|---:|---:|"]
    for block, label in blocks:
        cells = [ratio(ablations[(comparator, block)], "pooled_speedup_ratio")
                 for comparator in ("suffix-fixed", "suffix-local")]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    bound_findings = [finding(ablations[("suffix-fixed", block)]).lower() for block, _ in blocks]
    if len(set(bound_findings)) == 1:
        bound_statement = f"Decoding with the adaptive bound is {bound_findings[0]} than decoding without it in all three blocks." if bound_findings[0] != "inconclusive" else "The adaptive bound comparison is inconclusive in all three blocks."
    else:
        bound_statement = "Adaptive bound results: " + "; ".join(f"{label.lower()}: {result}" for (_, label), result in zip(blocks, bound_findings)) + "."
    text += ["", bound_statement + " Removing the bound can change the chosen candidate and its length. These tests do not identify which change caused the latency difference.", "",
             "| Correctness check | Passing comparisons |", "|---|---:|",
             "| Ordinary, SUFFIX and suffix ablations in timed runs | 4,800 / 4,800 |",
             "| Separate suffix traces | 720 / 720 |",
             f"| Direct verification and KV assertions | {campaign['audited_rounds']} / {campaign['audited_rounds']} |",
             "| Controlled-width outputs | 54 / 54 |",
             "| Attention route comparison outputs | 24 / 24 |",
             "| Portable smoke outputs with empty and populated caches | 4 / 4 |", "",
             "### NGRAM PROB", "",
             f"NGRAM output differs from ordinary output on {len(benchmark['mismatches'])} of 1,200 timed requests. The table gives descriptive latency ratios, not exact-output speedups.", "",
             "| Block | Ordinary / NGRAM latency | Differing outputs |", "|---|---:|---:|"]
    mismatches = Counter(r["block"] for r in benchmark["mismatches"])
    for block, label in blocks:
        row = aggregates[("ngram", block)]
        value = row.get("descriptive_latency_ratio", row.get("pooled_speedup"))
        text.append(f"| {label} | {value:.3f}× | {mismatches[block]} |")
    text += ["", "NGRAM and SUFFIX use different cache policies. NGRAM can merge branches even when each suffix anchor has a fanout of 1. SUFFIX outputs must match ordinary outputs.", "",
             "### GPU verification width", "",
             "| Context tokens | Kernel time, 1 row | Kernel time, 33 rows | Reduction |", "|---:|---:|---:|---:|"]
    widths = {(r["context_tokens"], r["verify_rows"]): r for r in width["rows"]}
    for context in (126, 128, 512):
        one = widths[(context, 1)]["median_kernel_ms"]
        full = widths[(context, 33)]["median_kernel_ms"]
        text.append(f"| {context} | {one:.3f} ms | {full:.3f} ms | {100 * (1 - one / full):.1f}% |")
    text += ["", "The table shows median sums of GPU kernel times from three profiling trials. The launch grids for KV storage and argmax become smaller. The attention and output-head grids stay the same. Kernel time includes profiler overhead. It does not measure request latency or operation counts.", "",
             f"The attention route comparison uses two prompts and six paired trials. Shared-route latency / original-route latency is {ratio(route, 'pooled_shared_over_original_cost_ratio')}. Values above 1 mean the shared route is slower. This test does not establish the baseline cost for other inputs.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text))
    print(f"Wrote verified measurement tables to {args.output}")


if __name__ == "__main__":
    main()
