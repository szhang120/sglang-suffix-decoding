"""Generate standalone scientific figures from saved evidence, without GPU work."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "sglang-suffix-decoding"
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    dest = args.root / "docs/figures"
    dest.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.root / "results/width-probe-final/summary.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    for context, color, style in ((126, "#707070", "--"), (128, "#0072B2", "-"), (512, "#D55E00", "-")):
        rows = [r for r in data["rows"] if r["context_tokens"] == context]
        widths = [r["verify_rows"] for r in rows]
        medians = [r["median_kernel_ms"] for r in rows]
        errors = [[r["median_kernel_ms"] - r["range_kernel_ms"][0] for r in rows],
                  [r["range_kernel_ms"][1] - r["median_kernel_ms"] for r in rows]]
        label = f"Context {context}" + (" (key-tile boundary)" if context == 126 else "")
        ax.errorbar(widths, medians, yerr=errors, color=color, linestyle=style,
                    marker="o", markersize=4, capsize=3, label=label)
    ax.set(xlabel="Actual verification rows", ylabel="Correlated GPU kernel sum (ms)",
           xlim=(0, 34), ylim=(0, 12), xticks=[1, 2, 4, 8, 16, 33])
    ax.set_title("Shorter proposals save some work; tile padding limits the gain", fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.015, 0.015, "H100 / Qwen2.5-7B / final FP32 head; median and observed range of 3 profiling trials", fontsize=7)
    for extension in ("png", "svg"):
        metadata = {"Creator": "analysis/plot_results.py"}
        if extension == "svg":
            metadata["Date"] = None
        fig.savefig(dest / f"verify-width.{extension}", dpi=180, metadata=metadata)
    plt.close(fig)
    report = args.root / "results/final/benchmark-report.json"
    if report.exists():
        results = json.loads(report.read_text())
        if not results.get("exact_suffix_ids_passed", results["exact_ids_passed"]):
            raise SystemExit("Output equality failed; refusing an unlabeled performance figure")
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
        blocks = ("independent", "refinement", "repeat")
        modes = (("ngram", "NGRAM PROB", "#707070"), ("suffix", "Suffix adaptive", "#0072B2"),
                 ("suffix-fixed", "Suffix unbounded match cap", "#D55E00"),
                 ("suffix-local", "Suffix local only", "#009E73"))
        ngram_excluded = any(r.get("different_outputs") for r in results["aggregate"] if r["mode"] == "ngram")
        if ngram_excluded:
            modes = tuple(m for m in modes if m[0] != "ngram")
        positions = list(range(len(blocks)))
        for j, (mode, label, color) in enumerate(modes):
            rows = [next(r for r in results["aggregate"] if (r["mode"], r["block"]) == (mode, block)) for block in blocks]
            centers = [r["pooled_speedup"] for r in rows]
            errs = [[r["pooled_speedup"] - r["trial_bootstrap_95"][0] for r in rows],
                    [r["trial_bootstrap_95"][1] - r["pooled_speedup"] for r in rows]]
            ax.bar([x + (j - (len(modes) - 1) / 2) * 0.19 for x in positions], centers, width=0.18,
                   yerr=errs, capsize=2, color=color, label=label)
        ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(positions, ["Independent", "Initial + refinement", "Identical-prompt repetition"])
        ax.set_ylabel("Paired aggregate speedup vs ordinary decoding")
        ax.set_ylim(bottom=0)
        ax.set_title("Fixed public subset, five rotated-order trials", fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        footer = "95% paired trial-bootstrap interval; repeat is a diagnostic upper bound, not an agent workload"
        if ngram_excluded:
            footer += "\nNGRAM excluded from exact-output speedup figure; differing outputs reported separately"
        fig.text(0.015, 0.015, footer, fontsize=7)
        for extension in ("png", "svg"):
            metadata = {"Creator": "analysis/plot_results.py"}
            if extension == "svg":
                metadata["Date"] = None
            fig.savefig(dest / f"benchmark-speedup.{extension}", dpi=180, metadata=metadata)
        plt.close(fig)


if __name__ == "__main__":
    main()
