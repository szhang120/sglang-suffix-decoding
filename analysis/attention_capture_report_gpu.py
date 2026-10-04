"""Replay captured real-model Q/K/V with one variable changed at a time."""

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

    def difference(a, b):
        assert a.shape == b.shape, (a.shape, b.shape)
        delta = (a.float() - b.float()).abs()
        return dict(bitwise_equal=torch.equal(a, b), unequal_elements=int((a != b).sum()),
                    max_abs=float(delta.max()), mean_abs=float(delta.mean()))

    def run(capture, width=None, kv_length=None, custom=True, zero_future=False):
        qcpu = capture["q"]
        width = width or capture["width"]
        q = torch.empty_strided((width, 28, 128), capture["q_stride"], dtype=qcpu.dtype, device="cuda")
        q.copy_(qcpu[:width])
        keys, values = capture["keys"].cuda(), capture["values"].cuda()
        prefix = capture["prefix"]
        length = kv_length or keys.shape[0]
        if zero_future:
            keys[prefix + 1:].zero_()
            values[prefix + 1:].zero_()
        output = torch.empty_like(q)
        mask = None
        if custom:
            mask = torch.arange(length, device="cuda")[None, :] <= prefix + torch.arange(width, device="cuda")[:, None]
            mask = mask.flatten()
        extend_attention_fwd_unified(
            q, output, keys, values, 1.0, 1.0,
            torch.tensor([0, width], dtype=torch.int32, device="cuda"),
            torch.tensor([0, length], dtype=torch.int64, device="cuda"),
            torch.arange(length, dtype=torch.int64, device="cuda"),
            torch.tensor([prefix], dtype=torch.int32, device="cuda"), width,
            custom_mask=mask, mask_indptr=torch.zeros(2, dtype=torch.int64, device="cuda") if custom else None,
            sm_scale=capture["sm_scale"], is_causal=True, page_size=1,
        )
        return output.cpu()

    ordinary, ngram = captures["ordinary"], captures["ngram"]
    assert ordinary["prefix"] == ngram["prefix"] == 120
    assert ordinary["width"] == 1 and ngram["width"] == 33
    root_length = ordinary["keys"].shape[0]
    comparisons = {
        "model_q": difference(ordinary["q"][0], ngram["q"][0]),
        "model_visible_keys": difference(ordinary["keys"], ngram["keys"][:root_length]),
        "model_visible_values": difference(ordinary["values"], ngram["values"][:root_length]),
        "model_attention_output": difference(ordinary["output"][0], ngram["output"][0]),
    }
    causal = run(ordinary, custom=False)
    full = run(ngram)
    controls = dict(
        original_replay=difference(ordinary["output"], causal),
        ngram_replay=difference(ngram["output"], full),
        full_vs_causal=difference(full[0], causal[0]),
        truncated_custom_vs_causal=difference(run(ngram, width=1, kv_length=root_length)[0], causal[0]),
        truncated_causal_vs_causal=difference(run(ngram, width=1, kv_length=root_length, custom=False)[0], causal[0]),
        full_queries_short_kv_vs_causal=difference(run(ngram, kv_length=root_length)[0], causal[0]),
        full_zero_future_vs_causal=difference(run(ngram, zero_future=True)[0], causal[0]),
    )
    report = dict(diagnostic=True, layer_ordinal=3, prefix=120,
                  ordinary_q_stride=ordinary["q_stride"], ngram_q_stride=ngram["q_stride"],
                  captured_input_comparisons=comparisons, replay_controls=controls,
                  finite={mode: {name: bool(torch.isfinite(capture[name]).all())
                                 for name in ("q", "keys", "values", "output")} for mode, capture in captures.items()},
                  capture_sha256={mode: hashlib.sha256((args.directory / mode / "attention.pt").read_bytes()).hexdigest() for mode in captures},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation="One real-model attention instance; diagnostic hooks and changed output cap; not a broad correctness or performance result")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
