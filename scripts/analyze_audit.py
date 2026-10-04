"""Summarize saved correctness divergences and target-logit margins."""

import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    base = read(args.directory / "correctness-ordinary.jsonl")
    trace_path = args.directory / "audit-suffix-trace.jsonl"
    traces = read(trace_path) if trace_path.exists() else []
    summary = []
    for mode in ("ngram", "suffix"):
        records = read(args.directory / f"correctness-{mode}.jsonl")
        assert len(base) == len(records) == 30
        for x, y in zip(base, records):
            assert (x["name"], x["repeat"]) == (y["name"], y["repeat"])
            a, b = x["response"]["output_ids"], y["response"]["output_ids"]
            if a == b:
                continue
            first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
            item = dict(mode=mode, name=x["name"], repeat=x["repeat"], first_difference=first,
                        ordinary_length=len(a), other_length=len(b),
                        ordinary_token=a[first] if first < len(a) else None,
                        other_token=b[first] if first < len(b) else None)
            top = x["response"]["meta_info"].get("output_top_logprobs")
            if top and first < len(top):
                item["ordinary_top2_logprobs"] = top[first]
                item["ordinary_logprob_margin"] = top[first][0][0] - top[first][1][0]
            if mode == "suffix":
                rid = f"case-{x['repeat']}-{x['name']}"
                candidates = [r for r in traces if r["rid"] == rid and
                              r["output_len_before"] <= first < r["output_len_before"] + r["accepted_with_bonus"][0]]
                assert len(candidates) == 1, candidates
                row = candidates[0]
                offset = first - row["output_len_before"]
                item["suffix_round"] = row
                item["suffix_prediction_row"] = offset
                values = row["top2_logits"][offset]
                item["suffix_logit_margin"] = values[0] - values[1]
                item["suffix_top2_ids"] = row["top2_ids"][offset]
            summary.append(item)
    dest = args.directory / "audit-summary.json"
    if dest.exists():
        raise SystemExit(f"Preserving {dest}")
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
