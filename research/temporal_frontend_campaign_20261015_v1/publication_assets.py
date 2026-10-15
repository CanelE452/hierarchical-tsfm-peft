"""Verify public assets or restore the four byte-identical cost CSV files.

Standard library only; never imports experiment code or accesses local caches.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "PUBLICATION_FILES.json"
FILE_LIST = HERE / "PUBLICATION_FILES.txt"
EXTENSIONS = {".py", ".json", ".jsonl", ".md", ".csv", ".png", ".svg", ".txt", ".log", ".gz"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def resolve(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"Outside publication directory: {name}")
    return path


def check(path, size, sha):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size or digest(path) != sha:
        raise ValueError(f"Size/SHA mismatch or missing regular file: {path}")


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = manifest["files"]
    names = [item["path"] for item in records]
    if names != sorted(set(names)) or FILE_LIST.read_text(encoding="utf-8").splitlines() != names:
        raise ValueError("Manifest and exact file list differ or contain duplicates")
    omitted = set(manifest["self_excluded"]) | {item["original_path"] for item in manifest["compressed_originals"]}
    expected_self = {p.relative_to(ROOT).as_posix() for p in (MANIFEST, FILE_LIST)}
    if set(manifest["self_excluded"]) != expected_self:
        raise ValueError("Only the manifest and list may exclude their own circular hashes")
    for item in records:
        path = resolve(item["path"])
        if path.suffix.lower() not in EXTENSIONS or "__pycache__" in path.parts or path.is_symlink():
            raise ValueError(f"Disallowed published file: {item['path']}")
        check(path, item["bytes"], item["sha256"])
    actual = {p.relative_to(ROOT).as_posix() for p in HERE.rglob("*")
              if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
              and p.suffix.lower() in EXTENSIONS and p.relative_to(ROOT).as_posix() not in omitted}
    if actual != set(names):
        raise ValueError(f"Inventory mismatch: missing={sorted(set(names)-actual)}, extra={sorted(actual-set(names))}")
    indexed = {item["path"]: item for item in records}
    if len(manifest["compressed_originals"]) != 4:
        raise ValueError("Expected exactly four compressed original cost CSVs")
    for item in manifest["compressed_originals"]:
        resolve(item["original_path"])
        if item["original_path"] in indexed or item["gzip_path"] not in indexed:
            raise ValueError("A compressed original was included or its gzip copy is absent")
    return manifest


def restore():
    manifest = verify()
    outcomes = []
    for item in manifest["compressed_originals"]:
        target = resolve(item["original_path"])
        if target.exists():
            check(target, item["original_bytes"], item["original_sha256"])
            outcomes.append(dict(path=item["original_path"], status="EXISTING_MATCH_UNCHANGED"))
            continue
        data = gzip.decompress(resolve(item["gzip_path"]).read_bytes())
        if len(data) != item["original_bytes"] or hashlib.sha256(data).hexdigest() != item["original_sha256"]:
            raise ValueError(f"Decompressed original mismatch: {target}")
        # Exclusive creation also prevents overwriting a file created concurrently.
        with target.open("xb") as output:
            output.write(data)
        check(target, item["original_bytes"], item["original_sha256"])
        outcomes.append(dict(path=item["original_path"], status="RESTORED_BYTE_IDENTICAL"))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "restore"))
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify()
        result = dict(status="PASS", published_files=len(manifest["files"]), compressed_originals=len(manifest["compressed_originals"]))
    else:
        result = dict(status="PASS", originals=restore())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
