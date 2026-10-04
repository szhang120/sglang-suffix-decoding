"""Hold Q/K/V fixed while varying query width, layout and mask specialization.

Uses the pinned SGLang attention kernel, not a substitute model engine.
Synthetic fixtures isolate kernel numerics; model-level equality is separate.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from sglang.kernels.ops.attention.extend_attention import (
        _get_block_sizes_for_extend_attention,
        extend_attention_fwd_unified,
    )

    assert torch.cuda.device_count() == 1
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    records = []

    def run(q, keys, values, indices, prefix, mask):
        width = q.shape[0]
        n = prefix + width
        output = torch.empty((width, 28, 128), dtype=dtype, device=device)
        extend_attention_fwd_unified(
            q, output, keys, values, 1.0, 1.0,
            torch.tensor([0, width], dtype=torch.int32, device=device),
            torch.tensor([0, n], dtype=torch.int64, device=device), indices[:n],
            torch.tensor([prefix], dtype=torch.int32, device=device), width,
            custom_mask=mask,
            mask_indptr=torch.zeros(2, dtype=torch.int64, device=device) if mask is not None else None,
            sm_scale=128 ** -0.5, is_causal=True, page_size=1,
        )
        return output

    def difference(a, b):
        delta = (a.float() - b.float()).abs()
        return dict(bitwise_equal=torch.equal(a, b), unequal_elements=int((a != b).sum()),
                    max_abs=float(delta.max()), mean_abs=float(delta.mean()))

    # Three observed failure positions plus aligned tile-boundary controls.
    for prefix in (79, 104, 128, 240, 512):
        for seed in range(3):
            torch.manual_seed(42 + seed)
            total = prefix + 33
            logical_keys = torch.randn(total, 4, 128, dtype=dtype, device=device)
            logical_values = torch.randn_like(logical_keys)
            for layout in ("contiguous", "packed-qkv"):
                # Packed Q has the model's QKV projection stride, 3584+512+512.
                packed = torch.randn(33, 3584 if layout == "contiguous" else 4608,
                                     dtype=dtype, device=device)
                queries = packed[:, :3584].view(33, 28, 128)
                for addressing in ("linear", "permuted"):
                    indices = (torch.arange(total, device=device, dtype=torch.int64)
                               if addressing == "linear" else torch.randperm(total, device=device))
                    keys = torch.empty_like(logical_keys)
                    values = torch.empty_like(logical_values)
                    keys[indices], values[indices] = logical_keys, logical_values
                    # Mirror the backend's single-query reshape/view, which can
                    # normalize an otherwise irrelevant size-one row stride.
                    single = queries[:1].reshape(-1, 3584).view(-1, 28, 128)
                    # Double-precision reference is diagnostic, not a demand
                    # that the production BF16 softmax weights match FP64.
                    q64 = single[0].cpu().double()
                    k64 = logical_keys[:prefix + 1].cpu().double().repeat_interleave(7, dim=1)
                    v64 = logical_values[:prefix + 1].cpu().double().repeat_interleave(7, dim=1)
                    scores = torch.einsum("hd,thd->ht", q64, k64) * (128 ** -0.5)
                    reference = torch.einsum("ht,thd->hd", scores.softmax(-1), v64)
                    for width in (1, 2, 4, 8, 16, 33):
                        mask = torch.cat((torch.ones(width, prefix, dtype=torch.bool, device=device),
                                          torch.ones(width, width, dtype=torch.bool, device=device).tril()), dim=1).flatten()
                        verified = run(queries[:width], keys, values, indices, prefix, mask)
                        for row in range(width):
                            single = queries[row:row + 1].reshape(-1, 3584).view(-1, 28, 128)
                            causal = run(single, keys, values, indices, prefix + row, None)[0]
                            custom = run(single, keys, values, indices, prefix + row,
                                         torch.ones(prefix + row + 1, dtype=torch.bool, device=device))[0]
                            records.append(dict(
                                prefix=prefix, query_row=row, seed=seed, layout=layout,
                                addressing=addressing, width=width,
                                single_query_stride=list(single.stride()), verify_query_stride=list(queries[:width].stride()),
                                causal_vs_verify=difference(causal, verified[row]),
                                custom_vs_verify=difference(custom, verified[row]),
                                causal_vs_custom=difference(causal, custom),
                                verify_max_error_fp64=float((verified[0].cpu().double() - reference).abs().max()) if row == 0 else None,
                            ))
    report = dict(
        synthetic=True, fixtures=len(records), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        blocks=list(_get_block_sizes_for_extend_attention(128, 128)),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        causal_vs_verify_unequal=sum(not r["causal_vs_verify"]["bitwise_equal"] for r in records),
        custom_vs_verify_unequal=sum(not r["custom_vs_verify"]["bitwise_equal"] for r in records),
        causal_vs_custom_unequal=sum(not r["causal_vs_custom"]["bitwise_equal"] for r in records),
        records=records,
    )
    if args.output.exists():
        raise SystemExit(f"Preserving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "nvidia_smi")}, indent=2))


if __name__ == "__main__":
    main()
