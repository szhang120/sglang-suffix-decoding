"""Record pinned SGLang leaf outputs around a reproducible q112 divergence.

Uses the engine's existing tensor-dump facility; diagnostic synchronization
and server warmup changes make this unsuitable for timing measurements.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ordinary", "ngram"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gpu_common import LOCK, ROOT, engine_config, sampling

    LOCK["gpu_candidate"]["unified_decode"] = True
    config = engine_config(args.mode)
    os.environ["SGLANG_SUFFIX_DECODE_CUSTOM_MASK"] = "0"
    data = (ROOT / "configs/frozen-workload.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOCK["frozen_workload"]["sha256"]
    row = next(r for r in map(json.loads, data.decode().splitlines())
               if r["kind"] == "initial" and r["question_id"] == 112)
    destination = args.output_dir / args.mode
    destination.mkdir(parents=True, exist_ok=False)
    config["debug_tensor_dump_output_folder"] = str(destination / "tensors")
    if args.capture_only:
        config["debug_tensor_dump_layers"] = []
        os.environ["SUFFIX_ATTENTION_CAPTURE_FILE"] = str(destination / "attention.pt")
    import torch
    import sglang as sgl

    assert torch.cuda.device_count() == 1
    environment = dict(
        diagnostic=True, mode=args.mode, config=config, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        pip_freeze=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        sglang_diff=subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "diff"], text=True),
        workload_sha256=hashlib.sha256(data).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    engine = sgl.Engine(**config)
    try:
        environment["server_info"] = engine.get_server_info()
        (destination / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n")
        engine.flush_cache()
        output = engine.generate(input_ids=row["input_ids"], sampling_params=sampling(50 if args.capture_only else 126),
                                 rid=f"tensor-trace-{args.mode}")
        (destination / "request.json").write_text(json.dumps(dict(input_ids=row["input_ids"], response=output), indent=2) + "\n")
        print(f"PROGRESS: tensor trace {args.mode} completed", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
