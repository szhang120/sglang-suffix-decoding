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
        if args.profile:
            engine.start_profile(
                output_dir=str(dest / "profile"),
                activities=["CPU", "GPU"],
                record_shapes=True,
                detailed_annotations=True,
            )
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
        if args.profile:
            engine.stop_profile()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
