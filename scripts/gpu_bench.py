"""One mode/trial per process, one GPU. Raw records precede any aggregation."""

import argparse
import hashlib
import json
import os
import subprocess
import time

from gpu_common import LOCK, ROOT, engine_config, sampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["ordinary", "ngram", "suffix", "suffix-fixed", "suffix-local"],
        required=True,
    )
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    dest = (
        ROOT
        / "results/gpu"
        / f"{args.mode}-{args.trial}{'-profile' if args.profile else ''}"
    )
    dest.mkdir(parents=True, exist_ok=False)
    if args.mode == "suffix-fixed":
        os.environ["SUFFIX_FIXED"] = "1"
    if args.mode == "suffix-local":
        os.environ["SUFFIX_CACHE_REQUESTS"] = "0"
    if args.profile:
        os.environ["SUFFIX_TRACE"] = str(dest / "suffix-rounds.jsonl")
    if args.mode.startswith("suffix"):
        os.environ["SUFFIX_ALLOW_WIDTH_PROBE"] = "1"
    import torch

    import sglang as sgl

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    cfg = engine_config(args.mode)
    environment = dict(
        config=cfg,
        draft_environment={
            name: os.environ.get(name, default)
            for name, default in {
                "SUFFIX_FACTOR": "1.0", "SUFFIX_OFFSET": "0.0",
                "SUFFIX_MIN_PROB": "0.1", "SUFFIX_CACHE_REQUESTS": "128",
                "SUFFIX_FIXED": "0", "SUFFIX_ALLOW_WIDTH_PROBE": "0",
            }.items()
        },
        kernel_environment={
            name: os.environ.get(name)
            for name in (
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE",
                "SGLANG_SUFFIX_UNIFIED_DECODE",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT",
                "SGLANG_CACHE_DIR",
            )
        },
        source_lock=LOCK,
        integration_patch_sha256=hashlib.sha256(
            (ROOT / "patches/sglang-suffix.patch").read_bytes()
        ).hexdigest(),
        workload_sha256=hashlib.sha256(
            (ROOT / "results/workload.jsonl").read_bytes()
        ).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        pip_freeze=subprocess.check_output(
            [__import__("sys").executable, "-m", "pip", "freeze"], text=True
        ),
    )
    (dest / "environment.json").write_text(json.dumps(environment, indent=2))
    rows = [
        json.loads(x)
        for x in (ROOT / "results/workload.jsonl").read_text().splitlines()
    ]
    engine = sgl.Engine(**cfg)
    try:
        (dest / "server-info.json").write_text(
            json.dumps(engine.get_server_info(), indent=2, default=str)
        )
        # Warm every frozen prompt's prefill shape, then all suffix widths.
        # Only one token is emitted; all algorithm caches are cleared below.
        for row in rows:
            engine.generate(input_ids=row["input_ids"], sampling_params=sampling(1))
        if args.mode.startswith("suffix"):
            for width in range(1, 34):
                engine.generate(
                    input_ids=rows[0]["input_ids"],
                    sampling_params=dict(
                        temperature=0, ignore_eos=True, max_new_tokens=width + 1,
                        custom_params=dict(suffix_probe_width=width),
                    ),
                )
        else:
            engine.generate(input_ids=rows[0]["input_ids"], sampling_params=sampling(128))
        engine.flush_cache()
        with (dest / "requests.jsonl").open("w") as f:
            # Independent and refinement blocks are public first/second turns.
            # Repetition is a separate diagnostic upper bound, not a paper workload.
            blocks = [
                ("independent", [r for r in rows if r["kind"] == "initial"]),
                ("refinement", rows),
                ("repeat", [r for r in rows if r["kind"] == "initial"] * 2),
            ]
            for block, requests in blocks:
                engine.flush_cache()
                if args.profile:
                    # Preserve repeated responses in the diagnostic profile.
                    requests = requests[:2] * 2 if block == "repeat" else requests[:4]
                for index, row in enumerate(requests):
                    if args.profile:
                        engine.start_profile(
                            output_dir=str(dest / "profile" / f"{block}-{index}"),
                            activities=["CPU", "GPU"], num_steps=8,
                            record_shapes=True, with_stack=False, detailed_annotations=True,
                        )
                    start = time.perf_counter_ns()
                    chunks = []
                    final = None
                    for response in engine.generate(
                        input_ids=row["input_ids"],
                        sampling_params=sampling(),
                        stream=True,
                    ):
                        now = time.perf_counter_ns()
                        chunks.append(
                            dict(
                                elapsed_ns=now - start,
                                output_tokens=len(response.get("output_ids", [])),
                            )
                        )
                        final = response
                    end = time.perf_counter_ns()
                    if args.profile:
                        from gpu_width_probe import stop_if_active

                        stop_if_active(engine)
                    if final is None or "output_ids" not in final:
                        raise RuntimeError(
                            "Missing token IDs: cannot establish exact equality"
                        )
                    record = dict(
                        block=block,
                        index=index,
                        question_id=row["question_id"],
                        category=row["category"],
                        kind=row["kind"],
                        input_tokens=len(row["input_ids"]),
                        truncated=row["truncated"],
                        elapsed_ns=end - start,
                        chunks=chunks,
                        response=final,
                    )
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    if (index + 1) % 10 == 0 or index + 1 == len(requests):
                        print(
                            f"PROGRESS: {args.mode} trial{args.trial} {block} "
                            f"{index + 1}/{len(requests)}", flush=True,
                        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
