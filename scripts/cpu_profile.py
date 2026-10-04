"""CPU-only draft microbenchmark. Never interpret this as GPU speedup."""

import json
import platform
import statistics
import time
from pathlib import Path

from suffix_native.cache import SuffixDecodingCache

c = SuffixDecodingCache(64, 128)
raw = []
sequence = list(range(256)) * 4
for rid in range(128):
    t = time.perf_counter_ns()
    c.start_request(rid, sequence)
    c.add_active_response(rid, sequence)
    update_ns = time.perf_counter_ns() - t
    for length in (2, 4, 8, 16, 32):
        context = [999] + list(range(length))
        for repeat in range(20):
            t = time.perf_counter_ns()
            d = c.speculate(rid, context, 32)
            raw.append(
                dict(
                    rid=rid,
                    match_context=length,
                    repeat=repeat,
                    elapsed_ns=time.perf_counter_ns() - t,
                    proposed=len(d.token_ids),
                    match_len=d.match_len,
                )
            )
    c.stop_request(rid)
    raw.append(dict(rid=rid, update_ns=update_ns))
Path("results/cpu-profile.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
print(
    json.dumps(
        dict(
            kind="CPU draft microbenchmark, synthetic periodic tokens",
            platform=platform.platform(),
            python=platform.python_version(),
            median_draft_us=statistics.median(
                r["elapsed_ns"] / 1000 for r in raw if "elapsed_ns" in r
            ),
            samples=sum("elapsed_ns" in r for r in raw),
        ),
        indent=2,
    )
)
