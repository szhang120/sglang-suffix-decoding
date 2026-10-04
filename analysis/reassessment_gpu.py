"""Small ordinary/NGRAM controls on pristine and incrementally patched SGLang.

This diagnostic intentionally records divergences without turning them into
benchmark claims. The suffix adapter is absent in every tested variant.
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
    parser.add_argument("--variant", choices=("pristine", "head-only", "shared-v12"), required=True)
    parser.add_argument("--mode", choices=("ordinary", "ngram"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gpu_common import LOCK, ROOT, engine_config, sampling

    LOCK["gpu_candidate"]["unified_decode"] = args.variant == "shared-v12"
    config = engine_config(args.mode)
    os.environ["SGLANG_SUFFIX_DECODE_CUSTOM_MASK"] = "0"
    data = (ROOT / "configs/frozen-workload.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOCK["frozen_workload"]["sha256"]
    all_rows = [json.loads(line) for line in data.decode().splitlines()]
    selected = [r for r in all_rows if r["kind"] == "initial" and r["question_id"] in (112, 114, 151)]
    selected += [r for r in all_rows if r["kind"] == "refinement" and r["question_id"] == 114]
    assert len(selected) == 4
    dest = args.output_dir / args.variant / args.mode
    dest.mkdir(parents=True, exist_ok=False)
    import torch
    import sglang as sgl

    assert torch.cuda.device_count() == 1
    environment = dict(
        variant=args.variant, mode=args.mode, config=config,
        workload_sha256=hashlib.sha256(data).hexdigest(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        pip_freeze=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        sglang_commit=subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "rev-parse", "HEAD"], text=True).strip(),
        sglang_diff=subprocess.check_output(["git", "-C", str(ROOT / "sglang"), "diff"], text=True),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection="Three failing first-turn prompts plus q114 refinement; cold pass then response-cache-warm pass",
        output_cap=256, diagnostic_instrumentation=False,
    )
    engine = sgl.Engine(**config)
    try:
        environment["server_info"] = engine.get_server_info()
        (dest / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n")
        engine.flush_cache()
        with (dest / "requests.jsonl").open("x") as f:
            for repeat in (0, 1):
                for index, row in enumerate(selected):
                    out = engine.generate(input_ids=row["input_ids"], sampling_params=sampling(),
                                          rid=f"audit-{args.variant}-{args.mode}-{repeat}-{index}")
                    f.write(json.dumps(dict(repeat=repeat, index=index, question_id=row["question_id"],
                                            kind=row["kind"], input_ids=row["input_ids"], response=out)) + "\n")
                    f.flush()
                    print(f"PROGRESS: {args.variant} {args.mode} q{row['question_id']} {row['kind']} repeat{repeat}", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
