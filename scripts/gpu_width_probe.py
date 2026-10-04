"""Controlled verify widths at matched contexts; profiles exclude warmup.

All forced candidates are zeroes and remain subject to target verification.
Context126 crosses a64-key tile boundary;128/512 control for that effect.
"""

import json
import os
import random

from gpu_common import LOCK, ROOT, engine_config


def stop_if_active(engine):
    try:
        engine.stop_profile()
    except RuntimeError as exc:
        if "Profiling is not in progress" not in str(exc):
            raise


def main():
    from transformers import AutoTokenizer

    import sglang as sgl

    dest = ROOT / "results/width-probe"
    dest.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(
        LOCK["model"]["id"], revision=LOCK["model"]["revision"]
    )
    ids = tok.apply_chat_template(
        [dict(role="user", content="Repeat exactly: alpha beta gamma delta. " * 12)],
        add_generation_prompt=True, tokenize=True, return_dict=False,
    )
    assert len(ids) == 126, "Pinned tokenizer probe changed"
    space = tok.encode(" ", add_special_tokens=False)
    assert len(space) == 1
    cases = [{"input_tokens": n, "input_ids": ids + space * (n - len(ids))}
             for n in (126, 128, 512)]
    params = dict(temperature=0, max_new_tokens=34, ignore_eos=True)
    ordinary = sgl.Engine(**engine_config("ordinary"))
    try:
        for case in cases:
            case["response"] = ordinary.generate(
                input_ids=case["input_ids"], sampling_params=params
            )
            assert len(case["response"]["output_ids"]) == 34
            assert case["response"]["output_ids"][1] != 0
        (dest / "reference.json").write_text(json.dumps({"contexts": cases}, indent=2) + "\n")
    finally:
        ordinary.shutdown()
    os.environ["SUFFIX_ALLOW_WIDTH_PROBE"] = "1"
    os.environ["SUFFIX_TRACE"] = str(dest / "rounds.jsonl")
    engine = sgl.Engine(**engine_config("suffix"))
    try:
        for width in range(1, 34):
            engine.generate(
                input_ids=ids,
                sampling_params=dict(temperature=0, ignore_eos=True, max_new_tokens=width + 1,
                                     custom_params=dict(suffix_probe_width=width)),
                rid=f"warm-width-{width}",
            )
        for case in cases[1:]:
            engine.generate(input_ids=case["input_ids"], sampling_params=dict(
                **params, custom_params=dict(suffix_probe_width=33)))
        engine.flush_cache()
        with (dest / "requests.jsonl").open("w") as f:
            for case in cases:
                context = case["input_tokens"]
                for trial in range(3):
                    widths = [1, 2, 4, 8, 16, 33]
                    random.Random(42 + trial).shuffle(widths)
                    for width in widths:
                        engine.flush_cache()
                        label = f"probe-{context}-{trial}-{width}"
                        # Bounded traces: prefill and the first verify rounds.
                        engine.start_profile(
                            output_dir=str(dest / "profile" / label),
                            activities=["CPU", "GPU"], num_steps=2,
                            record_shapes=True, with_stack=False, detailed_annotations=True,
                        )
                        out = engine.generate(
                            input_ids=case["input_ids"],
                            sampling_params=dict(**params, custom_params=dict(suffix_probe_width=width)),
                            rid=label,
                        )
                        stop_if_active(engine)
                        f.write(json.dumps(dict(context_tokens=context, trial=trial, width=width,
                                                response=out)) + "\n")
                        f.flush()
                        if out["output_ids"] != case["response"]["output_ids"]:
                            raise AssertionError(f"Forced-width output differs: {context=}, {trial=}, {width=}")
                print(f"PROGRESS: width profiling completed at context{context}", flush=True)
        (dest / "server-info.json").write_text(
            json.dumps(engine.get_server_info(), indent=2, default=str) + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
