"""Align leaf activations by logical input position and correct causal prefix.

The upstream dumper saves top-model/logits outputs in the following pass;
exclude those lagged keys and compare the current leaf outputs only.
"""

import argparse
import hashlib
import json
from pathlib import Path


def compare(directory):
    import torch

    requests = {mode: json.loads((directory / mode / "request.json").read_text())
                for mode in ("ordinary", "ngram")}
    ordinary = requests["ordinary"]
    prompt = ordinary["input_ids"]
    outputs = ordinary["response"]["output_ids"]
    ngram_outputs = requests["ngram"]["response"]["output_ids"]
    first_output_difference = next((i for i, (a, b) in enumerate(zip(outputs, ngram_outputs)) if a != b), None)
    expected = prompt + outputs
    prefixes = {}
    hashes = {}
    ordinary_files = sorted((directory / "ordinary/tensors").glob("*/Pass*.pt"))
    ngram_files = sorted((directory / "ngram/tensors").glob("*/Pass*.pt"))
    assert ordinary_files and ngram_files
    for path in ordinary_files:
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        positions = tensors["model.forward_batch_info.positions"].tolist()
        ids = tensors["model.forward_batch_info.input_ids"].tolist()
        assert all(expected[p] == token for p, token in zip(positions, ids))
        for i, position in enumerate(positions):
            prefixes[position] = (path, i)
        hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    mismatches = []
    eligible_rows = comparisons = 0
    cached_path = cached_tensors = None
    for path in ngram_files:
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        positions = tensors["model.forward_batch_info.positions"].tolist()
        ids = tensors["model.forward_batch_info.input_ids"].tolist()
        correct_path = True
        hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row, (position, token) in enumerate(zip(positions, ids)):
            correct_path &= position < len(expected) and expected[position] == token
            if not correct_path or position not in prefixes:
                continue
            eligible_rows += 1
            reference_path, reference_row = prefixes[position]
            if reference_path != cached_path:
                cached_tensors = torch.load(reference_path, map_location="cpu", weights_only=True)
                cached_path = reference_path
            for name, value in tensors.items():
                if name == "model" or "forward_batch_info" in name or not name.startswith("model."):
                    continue
                reference = cached_tensors[name]
                parts = value if isinstance(value, list) else [value]
                references = reference if isinstance(reference, list) else [reference]
                assert len(parts) == len(references), name
                for part, (a, b) in enumerate(zip(parts, references)):
                    assert a.shape[0] == len(positions), (name, a.shape, len(positions))
                    assert b.shape[0] == len(cached_tensors["model.forward_batch_info.positions"])
                    a, b = a[row], b[reference_row]
                    comparisons += 1
                    if not torch.equal(a, b):
                        delta = (a.float() - b.float()).abs()
                        mismatches.append(dict(position=position, output_input_index=position - len(prompt),
                                               ngram_pass=path.name, query_row=row, width=len(positions),
                                               operator=name, part=part, unequal_elements=int((a != b).sum()),
                                               max_abs=float(delta.max()), mean_abs=float(delta.mean())))
    return dict(diagnostic=True, first_output_difference=first_output_difference,
                ordinary_output_ids=outputs, ngram_output_ids=ngram_outputs,
                eligible_rows=eligible_rows, tensor_comparisons=comparisons,
                unequal_comparisons=len(mismatches), first_mismatches=mismatches[:40],
                mismatches=mismatches, tensor_sha256=hashes,
                limitation="Leaf output comparison; correct proposal prefixes only; hooks synchronize and suppress server warmup; not a timing result")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.directory)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("mismatches", "tensor_sha256", "ordinary_output_ids", "ngram_output_ids")}, indent=2))


if __name__ == "__main__":
    main()
