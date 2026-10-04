"""Freeze Qwen-tokenized public prompts and baseline-derived refinement inputs."""

import hashlib
import json

from gpu_common import LOCK, ROOT, engine_config, sampling


def main():
    from transformers import AutoTokenizer

    import sglang as sgl

    tokenizer = AutoTokenizer.from_pretrained(
        LOCK["model"]["id"], revision=LOCK["model"]["revision"]
    )
    rows = [
        json.loads(x)
        for x in (ROOT / "configs/specbench-subset.jsonl").read_text().splitlines()
    ]
    output = ROOT / "results/workload.jsonl"
    if output.exists():
        raise SystemExit(
            "Preserving existing workload.jsonl; move it before regenerating"
        )
    engine = sgl.Engine(**engine_config("ordinary"))
    try:
        with output.open("w") as f:
            for r in rows:
                messages = [dict(role="user", content=r["turns"][0])]
                ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                truncated = len(ids) > 15360
                ids = ids[-15360:]
                answer = engine.generate(input_ids=ids, sampling_params=sampling())
                base = dict(
                    question_id=r["question_id"],
                    category=r["category"],
                    input_ids=ids,
                    truncated=truncated,
                    baseline_output_ids=answer["output_ids"],
                )
                f.write(json.dumps(dict(base, kind="initial")) + "\n")
                if len(r["turns"]) > 1:
                    messages += [
                        dict(role="assistant", content=answer["text"]),
                        dict(role="user", content=r["turns"][1]),
                    ]
                    refined_ids = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                    f.write(
                        json.dumps(
                            dict(
                                question_id=r["question_id"],
                                category=r["category"],
                                kind="refinement",
                                input_ids=refined_ids[-15360:],
                                truncated=len(refined_ids) > 15360,
                            )
                        )
                        + "\n"
                    )
        (ROOT / "results/workload.sha256").write_text(
            hashlib.sha256(output.read_bytes()).hexdigest() + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
