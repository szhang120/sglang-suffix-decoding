import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "configs/source-lock.json").read_text())


def engine_config(mode):
    # Experimental shared-attention switch is frozen in the source lock and
    # applies equally to ordinary/NGRAM/SUFFIX target execution.
    os.environ["SGLANG_SUFFIX_UNIFIED_DECODE"] = (
        "1" if LOCK["gpu_candidate"].get("unified_decode", False) else "0"
    )
    m = LOCK["model"]
    config = dict(
        model_path=m["id"],
        revision=m["revision"],
        dtype="bfloat16",
        attention_backend="triton",
        enable_deterministic_inference=True,
        enable_fp32_lm_head=True,
        tp_size=1,
        pp_size=1,
        page_size=1,
        max_running_requests=1,
        disable_overlap_schedule=True,
        disable_decode_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        disable_radix_cache=True,
        context_length=16384,
        mem_fraction_static=0.75,
        random_seed=42,
    )
    if mode != "ordinary":
        config.update(
            speculative_algorithm="SUFFIX" if mode.startswith("suffix") else "NGRAM",
            speculative_num_draft_tokens=33,
            speculative_ngram_match_type="PROB",
            speculative_ngram_max_trie_depth=64,
            speculative_ngram_max_bfs_breadth=1,
        )
    return config


def sampling(max_new_tokens=256):
    return dict(temperature=0, max_new_tokens=max_new_tokens, ignore_eos=False)
