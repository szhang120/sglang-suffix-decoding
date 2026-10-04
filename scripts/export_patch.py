"""Refresh the distributable patch from the dedicated SGLang checkout."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = subprocess.check_output(
    ["git", "-C", str(ROOT / "sglang"), "rev-parse", "HEAD"], text=True
).strip()
assert base == "e00930c5489053f26d86b179cee0d087f846acbb", (
    "Preserve/review changed base before exporting"
)
tracked = subprocess.check_output(
    ["git", "-C", str(ROOT / "sglang"), "diff", "--binary"], text=True
)
worker = ROOT / "sglang/python/sglang/srt/speculative/suffix_worker.py"
added = subprocess.run(
    [
        "git",
        "diff",
        "--no-index",
        "--",
        "/dev/null",
        "sglang/python/sglang/srt/speculative/suffix_worker.py",
    ],
    cwd=ROOT,
    text=True,
    capture_output=True,
)
assert added.returncode == 1
new = added.stdout.replace("b/sglang/python/", "b/python/").replace(
    "a/sglang/python/", "a/python/"
)
(ROOT / "patches/sglang-suffix.patch").write_text(tracked + new)
