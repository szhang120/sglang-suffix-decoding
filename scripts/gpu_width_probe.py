"""Controlled verify-width diagnostics, outside benchmark timing results.

Candidate zeroes deliberately cause rejection. Every output must still equal
ordinary decoding. GPU traces align first verify rounds at output position 1,
so all widths see the same committed context. Later rounds are auxiliary data.
"""

import json
import os
import random

from gpu_common import LOCK, ROOT, engine_config


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
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    params = dict(temperature=0, max_new_tokens=34, ignore_eos=True)
    ordinary = sgl.Engine(**engine_config("ordinary"))
    try:
        reference = ordinary.generate(input_ids=ids, sampling_params=params)
        (dest / "reference.json").write_text(
            json.dumps(dict(input_ids=ids, response=reference), indent=2) + "\n"
        )
    finally:
        ordinary.shutdown()
    os.environ["SUFFIX_ALLOW_WIDTH_PROBE"] = "1"
    os.environ["SUFFIX_TRACE"] = str(dest / "rounds.jsonl")
    engine = sgl.Engine(**engine_config("suffix"))
    try:
        # Compile every possible actual width, excluding compilation from traces.
        for width in range(1, 34):
            engine.generate(
                input_ids=ids,
                sampling_params=dict(
                    temperature=0, ignore_eos=True, max_new_tokens=width + 1,
                    custom_params=dict(suffix_probe_width=width),
                ),
                rid=f"warm-width-{width}",
            )
        engine.flush_cache()
        engine.start_profile(
            output_dir=str(dest / "profile"), activities=["CPU", "GPU"],
            record_shapes=True, detailed_annotations=True,
        )
        with (dest / "requests.jsonl").open("w") as f:
            for trial in range(3):
                widths = [1, 2, 4, 8, 16, 33]
                random.Random(42 + trial).shuffle(widths)
                for width in widths:
                    engine.flush_cache()
                    out = engine.generate(
                        input_ids=ids,
                        sampling_params=dict(
                            **params, custom_params=dict(suffix_probe_width=width)
                        ),
                        rid=f"probe-{trial}-{width}",
                    )
                    record = dict(trial=trial, width=width, response=out)
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    if out["output_ids"] != reference["output_ids"]:
                        raise AssertionError(f"Forced-width output differs: {trial=}, {width=}")
        engine.stop_profile()
        (dest / "server-info.json").write_text(
            json.dumps(engine.get_server_info(), indent=2, default=str) + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
