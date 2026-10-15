"""Verify archives; --restore creates missing JSON only, never replaces existing files."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main(restore=False):
    records = json.loads((HERE / "EVIDENCE_ARCHIVES.json").read_text(encoding="utf-8"))["records"]
    created = 0
    for row in records:
        source = (ROOT / row["published_repo_path"]).resolve()
        target = (ROOT / row["original_repo_path"]).resolve()
        assert source.is_relative_to(ROOT) and target.is_relative_to(ROOT) and target.suffix == ".json"
        assert digest(source) == row["archive_sha256"]
        sha, size = hashlib.sha256(), 0
        with gzip.open(source, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
                size += len(chunk)
        assert sha.hexdigest() == row["original_sha256"] and size == row["original_bytes"]
        if target.exists():
            assert target.is_file() and digest(target) == row["original_sha256"]
        elif restore:
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(source, "rb") as handle, target.open("xb") as output:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    output.write(chunk)
            assert digest(target) == row["original_sha256"]
            created += 1
    print(json.dumps({"status": "PASS", "archives": len(records), "created_missing_JSON": created,
                      "existing_files_overwritten": False, "scientific_execution": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    main(parser.parse_args().restore)
