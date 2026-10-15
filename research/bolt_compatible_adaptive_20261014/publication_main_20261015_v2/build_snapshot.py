"""Publish existing research evidence; never load data/model arrays or run science."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
SCIENCE = BASE / "projection_followup_v1"
PREVIOUS = BASE / "publication_main_20261015_v1"
REPORTS = ("professor_method_brief_v1", "purged_readout_stability_v1",
           "horizon_decoder_diagnostic_v1", "horizon_error_terms_v1",
           "new_direction_comparison_v1", "ridge_regularization_extension_v1",
           "methodology_diagnostics_v1", "fixed_l7_latent_calibration_v1")
FORBIDDEN = {".npz", ".npy", ".pt", ".pth", ".safetensors", ".ckpt", ".bin",
             ".pkl", ".pickle", ".parquet", ".arrow", ".zip", ".log"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").splitlines()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        if isinstance(value, str):
            handle.write(value)
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")


def main():
    assert not (HERE / "REPORT_COPY_MAP.json").exists(), "FRESH_PUBLICATION_REQUIRED"
    parent = git("rev-parse", "HEAD")[0]
    assert parent == "6f7e312a3b1f6eecb15b134239f2893816864aa4"
    old_archives = read(PREVIOUS / "EVIDENCE_ARCHIVES.json")["records"]
    archives = list(old_archives)
    archived = {row["original_repo_path"]: ROOT / row["published_repo_path"] for row in archives}
    untracked = git("ls-files", "--others", "--exclude-standard", "--", rel(SCIENCE))
    for name in untracked:
        source = ROOT / name
        if source.suffix != ".json" or source.stat().st_size <= 10 * 1024 ** 2:
            continue
        if name in archived:
            row = next(row for row in archives if row["original_repo_path"] == name)
            assert sha(source) == row["original_sha256"] and sha(archived[name]) == row["archive_sha256"]
            continue
        target = HERE / "evidence" / (source.relative_to(BASE).as_posix() + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as raw, target.open("xb") as handle:
            with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0, compresslevel=9) as packed:
                shutil.copyfileobj(raw, packed, 1024 * 1024)
        digest = hashlib.sha256()
        size = 0
        with gzip.open(target, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        assert digest.hexdigest() == sha(source) and size == source.stat().st_size
        archives.append({"original_repo_path": name, "original_sha256": sha(source),
                         "original_bytes": size, "published_repo_path": rel(target),
                         "published_relative_path": os.path.relpath(target, HERE).replace("\\", "/"),
                         "archive_sha256": sha(target), "archive_bytes": target.stat().st_size,
                         "decompressed_byte_identity_verified": True})
        archived[name] = target
    write(HERE / "EVIDENCE_ARCHIVES.json", {"status": "PASS", "original_files_modified": False,
          "rule": "Reuse six verified earlier archives; archive new JSON above 10 MiB without changing originals.",
          "records": archives})
    copies = {}
    for name in REPORTS:
        folder = SCIENCE / "reports" / name
        assert (folder / "MANIFEST.json").is_file()
        for source in sorted(folder.rglob("*")):
            if source.is_file():
                assert source.suffix not in FORBIDDEN
                copies[source.resolve()] = HERE / "reports" / source.relative_to(SCIENCE / "reports")
    for name in untracked:
        source = ROOT / name
        if source.parent == SCIENCE and source.suffix == ".md":
            copies[source.resolve()] = HERE / "notes" / source.name
    old_copies = {str((ROOT / row["original_path"]).resolve()): ROOT / row["copy_path"]
                  for row in read(PREVIOUS / "REPORT_COPY_MAP.json")["files"]}
    tracked = set(git("ls-files"))
    eligible = {name for name in untracked if name not in archived}
    rewrites, records = [], []
    def rewrite(source, target, link):
        parsed = urlsplit(link)
        if parsed.scheme.lower() in ("http", "https", "mailto", "data") or not parsed.path:
            return link
        local = Path(unquote(parsed.path))
        local = (source.parent / local).resolve() if not local.is_absolute() else local.resolve()
        reason = None
        if local.is_relative_to(ROOT) and rel(local) in archived:
            destination, reason = archived[rel(local)], "lossless_gzip_JSON"
        elif local in copies:
            destination, reason = copies[local], "new_publication_copy"
        elif str(local) in old_copies:
            destination, reason = old_copies[str(local)], "previous_publication_copy"
        elif local.is_relative_to(ROOT) and (rel(local) in tracked or rel(local) in eligible):
            destination, reason = local, "repository_source_or_evidence"
        else:
            destination, reason = HERE / "CODE_AND_REPRODUCTION_KO.md", "local_asset_not_published"
        output = os.path.relpath(destination, target.parent).replace("\\", "/")
        if reason == "local_asset_not_published":
            output += "#local-assets"
        else:
            output += ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
        rewrites.append({"source_report": rel(source), "copy_report": rel(target),
                         "original_target": link, "published_target": output, "reason": reason})
        return output
    import verify_publication as checker
    header = "> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인]({index})을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.\n\n"
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".md":
            text = source.read_text(encoding="utf-8-sig")
            for link in dict.fromkeys(checker.markdown_links(text)):
                updated = rewrite(source, target, link)
                if updated == link:
                    continue
                old = "](" + link + ")"
                if old in text:
                    text = text.replace(old, "](" + updated + ")")
                else:
                    pattern = r"(?m)^(\s*\[[^\]]+\]:\s*)" + re.escape(link) + r"(?=\s|$)"
                    text, count = re.subn(pattern, lambda m: m.group(1) + updated, text)
                    assert count, (source, link, "UNHANDLED_MARKDOWN_DESTINATION")
            index = os.path.relpath(HERE / "README_KO.md", target.parent).replace("\\", "/")
            write(target, header.format(index=index) + text)
        else:
            with source.open("rb") as raw, target.open("xb") as handle:
                shutil.copyfileobj(raw, handle)
        records.append({"original_path": rel(source), "original_sha256": sha(source),
                        "copy_path": rel(target), "copy_sha256": sha(target),
                        "bytes": target.stat().st_size, "original_bytes": source.stat().st_size,
                        "byte_identical": sha(source) == sha(target)})
    assert all(row["byte_identical"] for row in records if Path(row["copy_path"]).suffix != ".md")
    write(HERE / "REPORT_COPY_MAP.json", {"status": "COPIED_WITH_LINK_ONLY_MD_TRANSFORM",
          "original_scientific_files_modified": False, "numeric_CSV_PNG_SVG_modified": False,
          "NPZ_model_or_raw_read": False, "files": records, "markdown_link_rewrites": rewrites,
          "original_manifests_retained_byte_exact": True,
          "copied_MD_has_new_publication_hash_not_original_scientific_manifest_hash": True})
    print(json.dumps({"status": "BUILT", "copied_files": len(records),
          "copied_MD": sum(Path(row["copy_path"]).suffix == ".md" for row in records),
          "archives_total": len(archives), "new_archives": len(archives) - len(old_archives)}))


if __name__ == "__main__":
    main()
