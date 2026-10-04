"""Locate differing cached values in earlier model forwards, using CPU only."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    captures = {mode: torch.load(args.capture / mode / "attention.pt", weights_only=True)
                for mode in ("ordinary", "ngram")}
    request = json.loads((args.trace / "ordinary/request.json").read_text())
    expected = request["input_ids"] + request["response"]["output_ids"]
    ordinary, ngram = captures["ordinary"], captures["ngram"]
    n = ordinary["keys"].shape[0]
    differing_positions = sorted(set(torch.where((ordinary["keys"] != ngram["keys"][:n]).flatten(1).any(1))[0].tolist())
                                 | set(torch.where((ordinary["values"] != ngram["values"][:n]).flatten(1).any(1))[0].tolist()))

    def difference(a, b):
        delta = (a.float() - b.float()).abs()
        return dict(bitwise_equal=torch.equal(a, b), unequal_elements=int((a != b).sum()),
                    max_abs=float(delta.max()), mean_abs=float(delta.mean()))

    rows = {position: dict(position=position, expected_input_id=expected[position],
                           cached_key_difference=difference(ordinary["keys"][position], ngram["keys"][position]),
                           cached_value_difference=difference(ordinary["values"][position], ngram["values"][position]),
                           forwards=[]) for position in differing_positions}
    reference_qkv = {}
    for mode in ("ordinary", "ngram"):
        files = sorted((args.trace / mode / "tensors").glob("*/Pass*.pt"))
        assert files
        for path in files:
            tensors = torch.load(path, weights_only=True)
            positions = tensors["model.forward_batch_info.positions"].tolist()
            ids = tensors["model.forward_batch_info.input_ids"].tolist()
            correct_prefix = True
            for row, (position, token) in enumerate(zip(positions, ids)):
                correct_prefix &= position < len(expected) and token == expected[position]
                if position not in rows:
                    continue
                qkv = tensors["model.layers.3.self_attn.qkv_proj"][row]
                if mode == "ordinary":
                    reference_qkv[position] = qkv
                entry = dict(mode=mode, file=path.name, row=row, width=len(positions),
                             input_id=token, correct_prefix=correct_prefix,
                             projected_value_vs_cached=difference(qkv[-512:].view(4, 128), captures[mode]["values"][position]))
                if mode == "ngram":
                    entry["qkv_vs_ordinary"] = difference(qkv, reference_qkv[position])
                    entry["projected_value_vs_ordinary_cache"] = difference(qkv[-512:].view(4, 128), ordinary["values"][position])
                rows[position]["forwards"].append(entry)
    report = dict(diagnostic=True, differing_positions=differing_positions, rows=list(rows.values()),
                  limitation="CPU comparison of previously captured GPU tensors; does not execute a substitute target model")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
