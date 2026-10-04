"""Ordinary decode route cost on two prompts with equal observed outputs.

The prompts were selected from the eight-prompt attention-path diagnostic,
not as a representative serving workload. No speculative algorithm is used.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=int, choices=(0, 1), required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gpu_common import LOCK, ROOT, engine_config, sampling

    import sglang as sgl
    import torch

    data = (ROOT / "configs/frozen-workload.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOCK["frozen_workload"]["sha256"]
    rows = [r for r in map(json.loads, data.decode().splitlines()) if r["kind"] == "initial"][:2]
    assert [r["question_id"] for r in rows] == [81, 82]
    LOCK["gpu_candidate"]["unified_decode"] = bool(args.shared)
    config = engine_config("ordinary")
    destination = args.output_dir / f"shared{args.shared}-{args.trial}"
    destination.mkdir(parents=True, exist_ok=False)
    environment = dict(
        config=config, shared_decode=args.shared, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        workload_sha256=hashlib.sha256(data).hexdigest(),
        selection="First two independent prompts, q81/q82; outputs match across routes in the prior eight-prompt diagnostic",
        source_lock=LOCK, control_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    engine = sgl.Engine(**config)
    try:
        environment["server_info"] = engine.get_server_info()
        (destination / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n")
        # Full-shape warmup for both inputs and decode sequence lengths.
        for row in rows:
            engine.generate(input_ids=row["input_ids"], sampling_params=sampling())
        engine.flush_cache()
        with (destination / "requests.jsonl").open("x") as f:
            for row in rows:
                start = time.perf_counter_ns()
                chunks = []
                final = None
                for response in engine.generate(input_ids=row["input_ids"], sampling_params=sampling(), stream=True):
                    chunks.append(dict(elapsed_ns=time.perf_counter_ns() - start,
                                       output_tokens=len(response.get("output_ids", []))))
                    final = response
                elapsed = time.perf_counter_ns() - start
                assert final is not None and chunks and final["output_ids"]
                f.write(json.dumps(dict(question_id=row["question_id"], shared_decode=args.shared,
                                        trial=args.trial, elapsed_ns=elapsed, chunks=chunks,
                                        response=final)) + "\n")
    finally:
        engine.shutdown()
    print(f"PROGRESS: ordinary route control shared{args.shared} trial{args.trial} complete", flush=True)


if __name__ == "__main__":
    main()
