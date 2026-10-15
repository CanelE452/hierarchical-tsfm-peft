"""Create byte-preserving gzip copies for large new JSON evidence."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", BASE.relative_to(REPO).as_posix()],
        cwd=REPO, text=True).splitlines()
    records = []
    for relative in untracked:
        source = REPO / relative
        if source.suffix != ".json" or source.stat().st_size <= 10 * 1024 * 1024 or HERE in source.parents:
            continue
        target = HERE / "evidence" / (source.relative_to(BASE).as_posix() + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=9) as packed:
                packed.write(source.read_bytes())
        with gzip.open(target, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == sha(source)
        records.append({"original_repo_path": relative, "original_sha256": sha(source),
                        "original_bytes": source.stat().st_size,
                        "published_repo_path": target.relative_to(REPO).as_posix(),
                        "published_relative_path": target.relative_to(HERE).as_posix(),
                        "archive_sha256": sha(target), "archive_bytes": target.stat().st_size,
                        "decompressed_byte_identity_verified": True})
    summary = {"status": "PASS", "original_files_modified": False,
               "rule": "Previously untracked campaign JSON larger than 10 MiB; gzip copies only; originals remain local.",
               "records": records}
    with (HERE / "EVIDENCE_ARCHIVES.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps({"status": "PASS", "archives": len(records),
                      "original_bytes": sum(p["original_bytes"] for p in records),
                      "archive_bytes": sum(p["archive_bytes"] for p in records)}))


if __name__ == "__main__":
    main()
