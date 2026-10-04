"""Separate diagnostic subset evidence from the full public equality gate."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    checks = []
    for shared in (0, 1):
        reference = args.directory / f"public-gate-ordinary-shared{shared}.jsonl"
        if not reference.exists():
            continue
        base = [json.loads(line) for line in reference.read_text().splitlines()]
        assert base, "Empty reference"
        for mode in ("ngram", "suffix"):
            path = args.directory / f"public-gate-{mode}-shared{shared}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == len(base), (mode, shared, "incomplete")
            errors = []
            for a, b in zip(base, rows):
                assert all(a[k] == b[k] for k in ("block", "index", "question_id"))
                x, y = a["response"]["output_ids"], b["response"]["output_ids"]
                if x != y:
                    first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                    errors.append(dict(block=a["block"], index=a["index"], question_id=a["question_id"],
                                       first_difference=first, ordinary_ids=x[first:first+8], mode_ids=y[first:first+8]))
            checks.append(dict(shared_decode=shared, mode=mode, requests=len(rows),
                               complete=len(rows) == 240, mismatches=errors))
    report = dict(checks=checks)
    output = args.directory / "public-gate-summary.json"
    if output.exists():
        raise SystemExit(f"Preserving {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    current = [c for c in checks if c["shared_decode"] == 1]
    assert {c["mode"] for c in current} == {"ngram", "suffix"}, "Both candidates required"
    assert all(not c["mismatches"] for c in current), "Shared-attention candidate differs"
    if args.require_complete:
        assert all(c["complete"] for c in current), "Full240-request gate required"


if __name__ == "__main__":
    main()
