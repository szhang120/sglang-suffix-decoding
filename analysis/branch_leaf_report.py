"""Inspect the specific branch that produced the differing cached value."""

import hashlib
import json
from pathlib import Path


def main():
    import torch

    base = Path("/artifacts/modal-20261004-tensor-equivalence/results")
    request = json.loads((base / "ordinary/request.json").read_text())
    expected = request["input_ids"] + request["response"]["output_ids"]
    files = {"ordinary": next((base / "ordinary/tensors").glob("*/Pass00040.pt")),
             "ngram": next((base / "ngram/tensors").glob("*/Pass00025.pt"))}
    tensors = {mode: torch.load(path, weights_only=True, mmap=True) for mode, path in files.items()}
    metadata = {}
    for mode, saved in tensors.items():
        ids = saved["model.forward_batch_info.input_ids"].tolist()
        positions = saved["model.forward_batch_info.positions"].tolist()
        metadata[mode] = [dict(row=i, position=p, input_id=t, expected_id=expected[p] if p < len(expected) else None)
                          for i, (p, t) in enumerate(zip(positions, ids))]
    assert metadata["ordinary"][0]["position"] == metadata["ngram"][4]["position"] == 118
    assert metadata["ordinary"][0]["input_id"] == metadata["ngram"][4]["input_id"] == expected[118]
    rows = []
    for name, value in tensors["ngram"].items():
        if not name.startswith("model.") or "forward_batch_info" in name or name.endswith("rotary_emb"):
            continue
        reference = tensors["ordinary"][name]
        parts = value if isinstance(value, list) else [value]
        references = reference if isinstance(reference, list) else [reference]
        for part, (a, b) in enumerate(zip(parts, references)):
            a, b = a[4], b[0]
            delta = (a.float() - b.float()).abs()
            rows.append(dict(operator=name, part=part, equal=torch.equal(a, b),
                             unequal_elements=int((a != b).sum()), max_abs=float(delta.max())))
    report = dict(diagnostic=True, metadata=metadata, comparisons=rows,
                  first_unequal=next((r for r in rows if not r["equal"]), None),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  correction="NGRAM PROB top_k1 restricts each anchor's children; multiple anchors can merge branching paths. A contiguous-prefix trace filter incorrectly excluded row4 after the sibling at row3.")
    output = Path("/artifacts/modal-20261004-attention-capture/branch-leaf-report.json")
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
