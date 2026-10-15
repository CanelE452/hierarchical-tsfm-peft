"""Preserve the first snapshot and bind the final publication after receipt creation."""
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

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def rel(path):
        return path.relative_to(ROOT).as_posix()

    def names(*args):
        value = subprocess.check_output(["git", args[0], "-z", *args[1:]], cwd=ROOT)
        return {item.decode("utf-8") for item in value.split(b"\0") if item}

    initial_path = HERE / "FILES.json"
    initial = read(initial_path)
    first = read(HERE / "LINK_AUDIT.json")
    passing = read(HERE / "LINK_AUDIT_v2.json")
    assert first["status"] == "FAILED" and len(first["errors"]) == 1
    assert first["errors"][0]["kind"] == "PUBLICATION_MARKDOWN_LOCAL_TARGET_MISSING"
    assert first["errors"][0]["detail"] == rel(HERE / "LINK_AUDIT.json")
    assert passing["status"] == "PASS" and not passing["errors"]
    assert passing["manifest"]["sha256"] == sha(initial_path)
    allowed_updates = {"README.md", "_docs/history/2026-10-15.md", rel(HERE / "README_KO.md"), rel(HERE / "CODE_AND_REPRODUCTION_KO.md")}
    changed = []
    for row in initial["files"]:
        if sha(ROOT / row["repo_path"]) != row["sha256"]:
            assert row["repo_path"] in allowed_updates, row["repo_path"]
            changed.append(row["repo_path"])
    paths = names("ls-files", "--", rel(BASE)) | names("ls-files", "--others", "--exclude-standard", "--", rel(BASE))
    paths |= {"README.md", "_docs/history/2026-10-15.md"}
    target = HERE / "FILES_v2.json"
    exclusions = {rel(target), rel(HERE / "FINAL_INDEX_CHECKS.json")}
    paths -= set(initial["large_JSON_originals_kept_local_and_archived"]) | exclusions
    records = []
    for name in sorted(paths):
        path = ROOT / name
        assert path.is_file() and path.stat().st_size <= 32 * 1024 ** 2
        assert path.suffix.lower() not in {".npz", ".npy", ".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".pkl", ".pickle", ".parquet", ".arrow", ".log", ".zip"}
        records.append({"repo_path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    result = {**initial, "created_unix": time.time(), "files": records,
              "payload_files": len(records), "payload_bytes": sum(row["bytes"] for row in records),
              "self_hash_exclusions": sorted(exclusions),
              "initial_manifest": {"repo_path": rel(initial_path), "sha256": sha(initial_path)},
              "first_link_check_preserved": {"repo_path": rel(HERE / "LINK_AUDIT.json"), "sha256": sha(HERE / "LINK_AUDIT.json"), "reason": "Only the check's own output link did not exist before receipt creation."},
              "passing_initial_payload_check": {"repo_path": rel(HERE / "LINK_AUDIT_v2.json"), "sha256": sha(HERE / "LINK_AUDIT_v2.json")},
              "publication_documentation_updates_after_initial_manifest": changed,
              "scientific_originals_changed_since_initial_manifest": False,
              "final_index_check_output": rel(HERE / "FINAL_INDEX_CHECKS.json")}
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    staging = ROOT / ".cache/publication_main_20261015_v2_final_paths.nul"
    with staging.open("xb") as handle:
        handle.write(b"\0".join(item.encode("utf-8") for item in sorted(paths | {rel(target)})) + b"\0")
    print(json.dumps({"status": "FINAL_FROZEN", "files": len(records), "bytes": result["payload_bytes"], "documentation_updates": changed, "staging_list": str(staging)}))


if __name__ == "__main__":
    main()
