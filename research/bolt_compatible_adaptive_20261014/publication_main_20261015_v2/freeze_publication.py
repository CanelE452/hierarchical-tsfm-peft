"""Freeze the named campaign files and produce an explicit Git staging list."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent


def main():
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def names(*args):
        raw = subprocess.check_output(["git", args[0], "-z", *args[1:]], cwd=ROOT)
        return {name.decode("utf-8") for name in raw.split(b"\0") if name}

    def rel(path):
        return path.relative_to(ROOT).as_posix()

    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).strip() == b"main"
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    assert parent == "6f7e312a3b1f6eecb15b134239f2893816864aa4"
    assert not names("diff", "--cached", "--name-only"), "STAGING_ALREADY_EXISTS"
    manifest_path = HERE / "FILES.json"
    assert not manifest_path.exists(), "FRESH_MANIFEST_REQUIRED"
    archives = read(HERE / "EVIDENCE_ARCHIVES.json")["records"]
    archived = {row["original_repo_path"] for row in archives}
    paths = names("ls-files", "--", rel(BASE)) | names("ls-files", "--others", "--exclude-standard", "--", rel(BASE))
    paths |= {"README.md", "_docs/history/2026-10-15.md"}
    exclusions = {rel(manifest_path), rel(HERE / "LINK_AUDIT.json")}
    paths -= archived | exclusions
    forbidden = {".npz", ".npy", ".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".pkl", ".pickle", ".parquet", ".arrow", ".log", ".zip"}
    records = []
    for name in sorted(paths):
        path = ROOT / name
        assert path.is_file() and path.suffix.lower() not in forbidden, name
        assert path.stat().st_size <= 32 * 1024 ** 2, name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"repo_path": name, "bytes": path.stat().st_size, "sha256": digest})
    old = read(BASE / "publication_main_20261015_v1/FILES.json")
    result = {
        "status": "FROZEN_PUBLICATION_PAYLOAD",
        "created_unix": time.time(),
        "parent_HEAD": parent,
        "authorization": "Latest human: current work wrap-up and push everything completed to main.",
        "files": records,
        "payload_files": len(records),
        "payload_bytes": sum(row["bytes"] for row in records),
        "self_hash_exclusions": sorted(exclusions),
        "expected_invalid_JSON_artifacts": old["expected_invalid_JSON_artifacts"],
        "large_JSON_originals_kept_local_and_archived": sorted(archived),
        "maximum_file_mib": 32,
        "maximum_file_reason": "Two byte-identical 18.1 MiB per-channel CSV tables remain directly readable; original and publication copy. JSON above 10 MiB uses verified lossless gzip.",
        "raw_data_models_weights_checkpoints_NPZ_included": False,
        "other_campaign_dirty_files_included": False,
        "scientific_reexecution_during_publication": False,
        "publication_copy_map": rel(HERE / "REPORT_COPY_MAP_v2.json"),
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    staging = ROOT / ".cache" / "publication_main_20261015_v2_paths.nul"
    staging.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("xb") as handle:
        handle.write(b"\0".join(name.encode("utf-8") for name in sorted(paths | {rel(manifest_path)})) + b"\0")
    print(json.dumps({"status": "FROZEN", "files": len(records), "bytes": result["payload_bytes"], "staging_list": str(staging)}))


if __name__ == "__main__":
    main()
