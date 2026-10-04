"""Separate tree-mask/key-order numerics from query-row shape with real inputs."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd_unified

    captures = {mode: torch.load(args.directory / mode / "attention.pt", weights_only=True)
                for mode in ("ordinary", "ngram")}
    ordinary, ngram = captures["ordinary"], captures["ngram"]
    assert ordinary["prefix"] == 118 and ngram["prefix"] == 116
    assert ordinary["width"] == 1 and ngram["width"] == 33
    mask = ngram["custom_mask"].view(33, -1)
    visible = mask[4].nonzero().flatten()
    assert len(visible) == 119
    assert visible.tolist() == list(range(116)) + [116, 118, 120]

    def diff(a, b):
        delta = (a.float() - b.float()).abs()
        return dict(bitwise_equal=torch.equal(a, b), unequal_elements=int((a != b).sum()),
                    max_abs=float(delta.max()), mean_abs=float(delta.mean()))

    def run(qcpu, qstride, keys, values, prefix, custom):
        width, length = qcpu.shape[0], keys.shape[0]
        q = torch.empty_strided(qcpu.shape, qstride, dtype=qcpu.dtype, device="cuda")
        q.copy_(qcpu)
        output = torch.empty((width, 28, 128), dtype=q.dtype, device="cuda")
        extend_attention_fwd_unified(
            q, output, keys.cuda(), values.cuda(), 1.0, 1.0,
            torch.tensor([0, width], dtype=torch.int32, device="cuda"),
            torch.tensor([0, length], dtype=torch.int64, device="cuda"),
            torch.arange(length, dtype=torch.int64, device="cuda"),
            torch.tensor([prefix], dtype=torch.int32, device="cuda"), width,
            custom_mask=custom.cuda().flatten() if custom is not None else None,
            mask_indptr=torch.zeros(2, dtype=torch.int64, device="cuda") if custom is not None else None,
            sm_scale=ordinary["sm_scale"], is_causal=True, page_size=1,
        )
        return output.cpu()

    causal = run(ordinary["q"], ordinary["q_stride"], ordinary["keys"], ordinary["values"], 118, None)
    tree = run(ngram["q"], ngram["q_stride"], ngram["keys"], ngram["values"], 116, mask)
    single_tree = run(ngram["q"][4:5], ordinary["q_stride"], ngram["keys"], ngram["values"], 118, mask[4:5])
    compact = run(ngram["q"][4:5], ordinary["q_stride"], ngram["keys"][visible], ngram["values"][visible], 118, None)
    full_compact = run(ngram["q"], ngram["q_stride"], ngram["keys"][visible], ngram["values"][visible], 118,
                       torch.ones(33, 119, dtype=torch.bool))
    report = dict(
        diagnostic=True, layer_ordinal=2, query_position=118, ngram_row=4,
        visible_physical_columns=visible.tolist(),
        q=diff(ordinary["q"][0], ngram["q"][4]),
        visible_keys=diff(ordinary["keys"], ngram["keys"][visible]),
        visible_values=diff(ordinary["values"], ngram["values"][visible]),
        captured_output=diff(ordinary["output"][0], ngram["output"][4]),
        controls=dict(ordinary_replay=diff(ordinary["output"], causal),
                      tree_replay=diff(ngram["output"], tree),
                      tree_vs_causal=diff(tree[4], causal[0]),
                      single_tree_vs_causal=diff(single_tree[0], causal[0]),
                      compact_vs_causal=diff(compact[0], causal[0]),
                      full_compact_vs_causal=diff(full_compact[4], causal[0])),
        capture_sha256={mode: hashlib.sha256((args.directory / mode / "attention.pt").read_bytes()).hexdigest() for mode in captures},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="One accepted branch with exact visible inputs and actual mask; no serving or broad correctness claim",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
