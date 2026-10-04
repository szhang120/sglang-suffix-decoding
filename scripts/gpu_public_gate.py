"""Public workload correctness, with an explicit shared-attention ablation."""

import argparse
import hashlib
import json
import os

from gpu_common import LOCK, ROOT, engine_config, sampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ordinary", "ngram", "suffix"), required=True)
    parser.add_argument("--shared", type=int, choices=(0, 1), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("limit must be nonnegative; zero means the full240-request gate")
    name = f"public-gate-{args.mode}-shared{args.shared}"
    dest = ROOT / "results" / f"{name}.jsonl"
    if dest.exists():
        raise SystemExit(f"Preserving {dest}")
    source = ROOT / "configs/frozen-workload.jsonl"
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOCK["frozen_workload"]["sha256"]
    rows = [json.loads(line) for line in data.decode().splitlines()]
    initial = [row for row in rows if row["kind"] == "initial"]
    assert len(rows) == 84 and len(initial) == 52
    LOCK["gpu_candidate"]["unified_decode"] = bool(args.shared)
    config = engine_config(args.mode)
    import sglang as sgl

    engine = sgl.Engine(**config)
    try:
        (ROOT / "results" / f"{name}-environment.json").write_text(json.dumps(dict(
            config=config, effective_unified_decode=args.shared,
            workload_sha256=hashlib.sha256(data).hexdigest(), limit=args.limit,
            server_info=engine.get_server_info(),
            shared_decode_environment=os.environ["SGLANG_SUFFIX_UNIFIED_DECODE"],
        ), indent=2, default=str) + "\n")
        with dest.open("w") as f:
            count = 0
            for block, requests in (("independent", initial), ("refinement", rows), ("repeat", initial * 2)):
                engine.flush_cache()
                for index, row in enumerate(requests):
                    if args.limit and count >= args.limit:
                        return
                    out = engine.generate(input_ids=row["input_ids"], sampling_params=sampling(),
                                          rid=f"{name}-{block}-{index}")
                    f.write(json.dumps(dict(block=block, index=index, question_id=row["question_id"],
                                            category=row["category"], kind=row["kind"], response=out)) + "\n")
                    f.flush()
                    count += 1
                    if count % 8 == 0 or index + 1 == len(requests):
                        print(f"PROGRESS: {name} {count} public requests", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
