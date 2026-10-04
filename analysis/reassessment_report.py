"""Compare the independently executed pristine/head/shared controls."""

import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = ("pristine", "head-only", "shared-v12")


def compare(a, b):
    assert len(a) == len(b) == 8
    errors = []
    for x, y in zip(a, b):
        assert all(x[k] == y[k] for k in ("repeat", "index", "question_id", "kind", "input_ids"))
        u, v = x["response"]["output_ids"], y["response"]["output_ids"]
        if u != v:
            first = next((i for i, (l, r) in enumerate(zip(u, v)) if l != r), min(len(u), len(v)))
            errors.append(dict(repeat=x["repeat"], question_id=x["question_id"], kind=x["kind"],
                               first_difference=first, a_ids=u[first:first + 8], b_ids=v[first:first + 8]))
    return dict(requests=8, exact_cases=8 - len(errors), mismatches=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    hashes = {}
    controls = None
    for variant in VARIANTS:
        for mode in ("ordinary", "ngram"):
            directory = args.directory / variant / mode
            path = directory / "requests.jsonl"
            records[(variant, mode)] = [json.loads(line) for line in path.read_text().splitlines()]
            env_path = directory / "environment.json"
            env = json.loads(env_path.read_text())
            cfg = {k: v for k, v in env["config"].items() if not k.startswith("speculative_")}
            identity = (cfg, env["torch"], env["cuda"], env["gpu"], env["pip_freeze"],
                        env["sglang_commit"], env["workload_sha256"], env["script_sha256"])
            if controls is None:
                controls = identity
            assert identity == controls, "Common environment differs"
            if variant == "pristine":
                assert env["sglang_diff"] == "", "Pristine control has tracked source changes"
            assert env["sglang_commit"] == "e00930c5489053f26d86b179cee0d087f846acbb"
            assert len(records[(variant, mode)]) == 8
            for p in (path, env_path):
                hashes[str(p.relative_to(args.directory))] = hashlib.sha256(p.read_bytes()).hexdigest()
    comparisons = {}
    for variant in VARIANTS:
        comparisons[f"{variant}: ordinary_vs_ngram"] = compare(records[(variant, "ordinary")], records[(variant, "ngram")])
    for mode in ("ordinary", "ngram"):
        comparisons[f"{mode}: pristine_vs_head-only"] = compare(records[("pristine", mode)], records[("head-only", mode)])
        comparisons[f"{mode}: head-only_vs_shared-v12"] = compare(records[("head-only", mode)], records[("shared-v12", mode)])
    report = dict(
        diagnostic=True, measured_requests=48, suffix_adapter_present=False,
        common_config=controls[0], comparisons=comparisons, raw_sha256=hashes,
        limitation="Four selected failing contexts, cold and warm passes; not a broad correctness gate or serving benchmark",
    )
    if args.output.exists():
        raise SystemExit(f"Preserving {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: dict(requests=v["requests"], exact_cases=v["exact_cases"],
                              mismatches=[(m["question_id"], m["kind"], m["repeat"], m["first_difference"]) for m in v["mismatches"]])
                      for k, v in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
