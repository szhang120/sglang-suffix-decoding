"""Write measured results only after serving and diagnostic gates complete."""

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

    modes = (("suffix", "Adaptive dual cache"), ("suffix-fixed", "Unbounded match cap"),
             ("suffix-local", "Local cache only"))
    blocks = (("independent", "Independent"), ("refinement", "Initial + refinement"),
              ("repeat", "Identical-prompt repetition"))
    aggregates = {(r["mode"], r["block"]): r for r in benchmark["aggregate"]}

    def interval(row):
        a, b = row["trial_bootstrap_95"]
        return f"{a:.3f}–{b:.3f}"

    def pct(value):
        return "n/a" if value is None else f"{100 * value:.1f}%"

    text = [
        "# Measured results: SuffixDecoding in SGLang",
        "",
        "All five rotated serving trials completed. Ordinary decoding, adaptive SUFFIX and both suffix ablations match output IDs on all 4,800 measured requests. The unchanged upstream NGRAM baseline is checked separately below. All target-model execution used SGLang; instrumented runs are excluded from serving estimates.",
        "",
        f"The campaign also passed its plain stop/length/cache gates and {campaign['audited_rounds']} direct verification/KV assertions. Full-workload traces add 720 strictly matching suffix requests. Controlled-width profiles match 54 outputs; the ordinary-route control matches all 24 outputs. The portable Linux runner additionally passes four cold/warm smoke requests in its isolated workspace; its complete standalone five-trial orchestration was not separately rerun.",
        "",
        "## Controlled serving timings",
        "",
        "Qwen2.5-7B-Instruct BF16 weights/activations, FP32-output head, one H100 80GB, TP1/PP1, greedy batch 1. All modes use the shared deterministic Triton attention route, with graphs, overlap and radix caching disabled. The frozen public subset has 52 initial inputs and 32 second turns; each block starts with empty algorithm caches. Warmup is excluded. Ratios are summed ordinary request wall latency divided by mode latency, including host and streaming overhead. Above 1 means faster. Intervals resample the five paired trial blocks; they describe repeat variation on this fixed subset.",
        "",
        "| Mode | Independent | Initial + refinement | Identical-prompt repetition |",
        "|---|---:|---:|---:|",
    ]
    for mode, label in modes:
        cells = [f"{aggregates[(mode, block)]['pooled_speedup']:.3f}× [{interval(aggregates[(mode, block)])}]"
                 for block, _ in blocks]
        text.append(f"| {label} | " + " | ".join(cells) + " |")
    text += ["", "![Exact-output serving ratios and paired trial intervals](figures/benchmark-speedup.png)",
             "", "Repetition is a diagnostic upper bound, not an agent benchmark. The unbounded-match-cap ablation does not force 33 executed rows: probability, available continuation and output budget still shorten proposals.",
             "", "### Actual second turns", "",
             "The combined refinement block includes first turns. These separate figures use only its 32 second-turn requests per trial (160 measurements per mode).",
             "", "| Mode | Second-turn latency ratio | 95% paired trial interval |",
             "|---|---:|---:|"]
    turns = {(r["mode"], r["block"], r["kind"]): r for r in benchmark["request_kinds"]}
    for mode, label in modes:
        row = turns[(mode, "refinement", "refinement")]
        assert row["requests"] == 160
        text.append(f"| {label} | {row['pooled_speedup']:.3f}× | {interval(row)} |")

    mismatches = Counter(r["block"] for r in benchmark["mismatches"])
    text += ["", "## Upstream NGRAM comparison", "",
             f"NGRAM differs on **{len(benchmark['mismatches'])}/1,200** timed responses. Its per-anchor fanout is 1, but merged suffix anchors can still branch. Cache policy differs from SUFFIX: trailing 64-token windows in a corpus rather than a full-prompt local tree and response-only FIFO. Every divergence is retained. A real accepted-branch replay isolates one BF16 attention-layout effect with bitwise-identical Q and visible K/V; this does not attribute all mismatches.",
             "", "Differing NGRAM results are descriptive latency ratios, not exact-output speedups. Token counts expose response-length differences; response quality was not evaluated.",
             "", "| Block | Descriptive ordinary/NGRAM latency ratio | Differing responses | Ordinary tokens | NGRAM tokens |",
             "|---|---:|---:|---:|---:|"]
    for block, label in blocks:
        row = aggregates[("ngram", block)]
        trials = [r for r in benchmark["trials"] if (r["mode"], r["block"]) == ("ngram", block)]
        ratio = row.get("descriptive_latency_ratio", row.get("pooled_speedup"))
        text.append(f"| {label} | {ratio:.3f}× | {mismatches[block]} | {sum(r['ordinary_output_tokens'] for r in trials)} | {sum(r['output_tokens'] for r in trials)} |")
    text += ["", "## Actual proposals and acceptance", "",
             "These are separate full-workload traces. The denominator is actual drafts (verification rows minus the pending root), not SGLang's configured 32-draft capacity. Committed emissions clip terminal tails at stop/EOS. Prefill's first token is excluded from verification rounds.",
             "", "| Variant | Block | Mean verification rows | Draft acceptance | Committed tokens/round | One-row rounds |",
             "|---|---|---:|---:|---:|---:|"]
    labels = dict(modes)
    block_labels = dict(blocks)
    for row in trace["summaries"]:
        text.append(f"| {labels[row['variant']]} | {block_labels[row['block']]} | {row['mean_verify_rows']:.2f} | {pct(row['target_draft_acceptance'])} | {row['mean_committed_tokens_per_round']:.2f} | {pct(row['one_row_fraction'])} |")
    text += ["", "Controlled ablations change one proposer policy at a time. The following point ratios compare adaptive dual-cache speedup with each ablation's speedup; they are additional descriptions of the same trials, not independent experiments.",
             "", "| Block | Adaptive / unbounded-match-cap speedup | Adaptive / local-only speedup |",
             "|---|---:|---:|"]
    for block, label in blocks:
        adaptive = aggregates[("suffix", block)]["pooled_speedup"]
        text.append(f"| {label} | {adaptive / aggregates[('suffix-fixed', block)]['pooled_speedup']:.3f}× | {adaptive / aggregates[('suffix-local', block)]['pooled_speedup']:.3f}× |")

    text += ["", "## GPU work and the baseline route", "",
             "| Context tokens | One-row kernel sum | 33-row kernel sum | Reduction at one row |",
             "|---:|---:|---:|---:|"]
    widths = {(r["context_tokens"], r["verify_rows"]): r for r in width["rows"]}
    for context in (126, 128, 512):
        one = widths[(context, 1)]["median_kernel_ms"]
        full = widths[(context, 33)]["median_kernel_ms"]
        text.append(f"| {context} | {one:.3f}ms | {full:.3f}ms | {100 * (1 - one / full):.1f}% |")
    text += ["", "![Controlled verification widths and GPU kernel durations](figures/verify-width-v12.png)",
             "", "Correlated launch grids at context 128:", "",
             "| Kernel family | One-row grids | 33-row grids |", "|---|---|---|"]
    for family in ("kv_store", "argmax", "attention", "lm_head"):
        small = widths[(128, 1)]["families"][family]["grids"]
        large = widths[(128, 33)]["families"][family]["grids"]
        text.append(f"| {family} | `{small}` | `{large}` |")
    text += ["", "These medians summarize three randomized controlled-width profiles. Launch grids distinguish work changes from masked padding; the table makes unchanged attention/head tiles visible. Kernel-duration sums include profiler effects and are neither request wall time nor measured FLOPs. Context 126 crosses a 64-key tile boundary. Forced zero candidates remain subject to target verification and are never injected into outputs.",
             "", f"The separate six-pair ordinary control gives a shared/original route cost ratio of **{route['pooled_shared_over_original_cost_ratio']:.3f}×** [{interval(route)}]. Above 1 means the shared route is slower. Only two writing prompts with matching observed outputs were selected; this is a narrow cost control, not a representative optimized-upstream comparison. The common head-stride fix avoids the previously measured 1.09GB copy on every forward in all modes.",
             "", "## Performance behavior and limitations", ""]
    checks = [("independent inputs", aggregates[("suffix", "independent")]),
              ("actual second turns", turns[("suffix", "refinement", "refinement")]),
              ("identical-prompt repetition", aggregates[("suffix", "repeat")])]
    positive = []
    for label, row in checks:
        low, high = row["trial_bootstrap_95"]
        finding = "faster" if low > 1 else "slower" if high < 1 else "inconclusive relative to 1"
        if finding == "faster":
            positive.append(label)
        text.append(f"- Adaptive SUFFIX on {label}: **{finding}**, pooled ratio {row['pooled_speedup']:.3f}× [{interval(row)}].")
    text += [""]
    if positive == ["identical-prompt repetition"]:
        text.append("Only the diagnostic repetition block provides clear evidence of a speed gain. The broader performance behavior on independent inputs and second turns was not established by this experiment.")
    elif positive:
        text.append("This adapted experiment provides evidence of a gain on " + ", ".join(positive) + ". The remaining rows, including negative or inconclusive results, constrain that finding.")
    else:
        text.append("This experiment does not establish the paper's claimed speed-gain behavior on the measured blocks. Negative and inconclusive results remain part of the reproduction.")
    text += ["", "This is an adaptation, not the paper's original numerical result: Qwen2.5-7B in SGLang replaces its Llama/vLLM setup; the output cache is bounded to 128 requests; the public subset is small; execution is eager; no proprietary AgenticSQL or live OpenHands trajectory is reproduced. Each block has at most 104 requests, so the serving experiment does not measure eviction pressure at that bound. Host checks cover FIFO eviction. Five repeated timings do not establish workload generalization. Exact IDs on this finite suite do not prove arbitrary-input hidden-state equivalence. Cache reuse, kernel padding, CPU overhead and the modified ordinary route all affect the result. Request wall latency includes CPU proposal/cache work, which is not separately isolated by the controlled-width kernel sums.",
             "", "## Reproducibility", "",
             f"Campaign: `{campaign['run_id']}`. Diagnostics: `{diagnostics['run_id']}`. Immutable runtime image: `{campaign['image_id']}`; image IDs identify provenance in the original Modal workspace rather than public portable images.",
             "", f"Integration patch SHA256: `{campaign['source_sha256']['patches/sglang-suffix.patch']}`. Workload SHA256: `{campaign['source_sha256']['configs/frozen-workload.jsonl']}`. Dependency lock SHA256: `{campaign['source_sha256']['configs/gpu-requirements.lock']}`.",
             "", "The JSON reports retain raw-file hashes, IDs, resolved configuration, GPU UUIDs, package freezes, trace hashes and per-trial measurements. Exact runtime source snapshots and selected numerical fixtures are public in the attribution audit release. See [GPU runbook](GPU_RUNBOOK.md), [technical design](TECHNICAL_REPORT.md), and the committed `results/final/` reports. Timing and profile artifacts are separate release assets; no model weights or credentials are distributed.", ""]
    if campaign.get("resume_lineage"):
        origin = campaign["resume_lineage"]
        text += [f"The original campaign `{origin['run_id']}` was canceled after its complete first trial. The resumed campaign preserved only complete paired trials and reran incomplete trials on a fresh allocation. Every five-mode trial stayed on one physical GPU; allocations can differ between trials, as recorded by GPU UUID. The original failed status, log and discarded partial records are retained under `resume-source/` in the serving archive. Model, runtime and frozen input hashes are unchanged.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text))
    print(f"Wrote gated measured results to {args.output}")


if __name__ == "__main__":
    main()
