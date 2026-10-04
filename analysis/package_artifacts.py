"""Package explicit evidence directories with normalized headers and file hashes."""

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Archive label=directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Preserving existing archive/manifest; use a fresh output name")
    files = {}
    for item in args.input:
        label, directory = item.split("=", 1)
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", label), "Invalid archive label"
        root = Path(directory)
        assert root.is_dir(), directory
        for path in sorted(root.rglob("*")):
            assert not path.is_symlink(), path
            if path.is_file():
                name = f"{label}/{path.relative_to(root)}"
                assert name not in files
                files[name] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with args.output.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for name, path in sorted(files.items()):
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as f:
                        archive.addfile(info, f)
                    records[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(dict(archive=args.output.name, bytes=args.output.stat().st_size,
                                            sha256=digest(args.output), files=records), indent=2) + "\n")
    print(json.dumps(dict(archive=str(args.output), files=len(records), bytes=args.output.stat().st_size)))


if __name__ == "__main__":
    main()
