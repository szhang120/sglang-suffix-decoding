"""Real-engine stop/length/reuse correctness cases. Compare IDs across modes."""

import argparse
import json

from gpu_common import LOCK, ROOT, engine_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ordinary", "ngram", "suffix"], required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer

    import sglang as sgl

    tok = AutoTokenizer.from_pretrained(
        LOCK["model"]["id"], revision=LOCK["model"]["revision"]
    )
    engine = sgl.Engine(**engine_config(a.mode))
    path = ROOT / "results" / f"correctness-{a.mode}.jsonl"
    if path.exists():
        raise SystemExit(f"Preserving {path}")
    cases = []
    for n in (1, 2, 3, 17, 33, 65, 128):
        cases.append(
            dict(
                name=f"length-{n}",
                content="Repeat exactly: alpha beta gamma delta. " * 12,
                params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
            )
        )
    cases += [
        dict(
            name="eos",
            content="Answer only with the digit 1: what is 1+0?",
            params=dict(temperature=0, max_new_tokens=128),
        ),
        dict(
            name="stop-string",
            content="Count in English: one, two, three, four, five.",
            params=dict(temperature=0, max_new_tokens=128, stop=["three"]),
        ),
        dict(
            name="mismatch",
            content="Write a short explanation of ocean tides.",
            params=dict(temperature=0, max_new_tokens=128),
        ),
    ]
    try:
        with path.open("w") as f:
            for repeat in range(3):
                if repeat == 2:
                    engine.flush_cache()
                for case in cases:
                    ids = tok.apply_chat_template(
                        [dict(role="user", content=case["content"])],
                        add_generation_prompt=True,
                    )
                    out = engine.generate(
                        input_ids=ids,
                        sampling_params=case["params"],
                        rid=f"case-{repeat}-{case['name']}",
                    )
                    if len(out["output_ids"]) > case["params"]["max_new_tokens"]:
                        raise AssertionError("Generation exceeded output cap")
                    f.write(
                        json.dumps(dict(repeat=repeat, name=case["name"], response=out))
                        + "\n"
                    )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
