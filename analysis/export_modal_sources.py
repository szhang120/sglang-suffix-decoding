"""Export exact experiment sources from an immutable project image; CPU only."""

import json
import os
from pathlib import Path

import modal

IMAGE = os.environ["SUFFIX_SOURCE_IMAGE"]
app = modal.App("sglang-suffix-source-provenance")
artifacts = modal.Volume.from_name("sglang-suffix-artifacts")


@app.function(image=modal.Image.from_id(IMAGE).env({"SUFFIX_SOURCE_IMAGE": IMAGE}), cpu=2, memory=2048, gpu=None,
              timeout=300, retries=0, scaledown_window=2,
              volumes={"/artifacts": artifacts})
def export(expected):
    import hashlib
    import shutil

    root = Path("/project")
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    dest = Path("/artifacts/source-snapshots") / IMAGE
    dest.mkdir(parents=True, exist_ok=False)
    names = set(expected)
    for directory in ("configs", "native", "patches", "scripts", "tests"):
        for path in (root / directory).rglob("*"):
            if (path.is_file() and path.name != "model-api.json"
                and "build" not in path.parts and "__pycache__" not in path.parts
                and path.suffix in (".py", ".sh", ".cc", ".h", ".txt", ".json", ".jsonl", ".lock", ".in", ".patch")
                and path.name != "modal_runner.py"):
                names.add(str(path.relative_to(root)))
    for line in (root / "patches/sglang-suffix.patch").read_text().splitlines():
        if line.startswith("diff --git "):
            names.add("sglang/" + line.split(" b/", 1)[1])
    for name in ("native/LICENSE", "configs/SPECBENCH_LICENSE"):
        names.add(name)
    manifest = {}
    for name in sorted(names):
        source = root / name
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (dest / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    artifacts.commit()
    return dict(image=IMAGE, source_files=len(manifest), path=str(dest))


@app.local_entrypoint()
def main(status: str):
    expected = json.loads(Path(status).read_text())["source_sha256"]
    print(json.dumps(export.remote(expected), indent=2))
