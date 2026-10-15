"""Publish only eight named legacy text files and repair nine existing report links."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib
import json
import os
import re
import subprocess

import verify_publication as checker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "publication_main_20261015_v1"
TARGETS = {
    "research/level_compressor_fullscreen_20261011/REPORT_KO.md",
    "research/level_compressor_fullscreen_20261011/DATA_CONTRACT.json",
    "research/level_compressor_fullscreen_20261011/data_contract.py",
    "research/level_compressor_fullscreen_20261011/train.py",
    "research/level_compressor_fullscreen_20261011/models.py",
    "research/level_compressor_fullscreen_20261011/ROTATION_PROTOCOL.json",
    "research/level_compressor_fullscreen_20261011/PROTECTED_ACCESS_INCIDENT.json",
    "research/smallk_failure_followup_20261014/FORECAST_COORDINATE_COMPATIBILITY_REVIEW_KO_v1.md",
}
HEADER = "> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인]({index})을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.\n\n"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def destination(source, link):
    parsed = urlsplit(link)
    if parsed.scheme.lower() in ("http", "https", "mailto", "data") or not parsed.path:
        return None
    local = Path(unquote(parsed.path))
    return (source.parent / local).resolve() if not local.is_absolute() else local.resolve()


def link_to(path, target, original=""):
    parsed = urlsplit(original)
    return os.path.relpath(path, target.parent).replace("\\", "/") + ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")


def replace_link(text, old, new):
    if old == new:
        return text
    token = "](" + old + ")"
    if token in text:
        return text.replace(token, "](" + new + ")")
    pattern = r"(?m)^(\s*\[[^\]]+\]:\s*)" + re.escape(old) + r"(?=\s|$)"
    result, count = re.subn(pattern, lambda match: match.group(1) + new, text)
    assert count, (old, "UNHANDLED_MARKDOWN_DESTINATION")
    return result


def main():
    map_path = HERE / "REPORT_COPY_MAP.json"
    original_map_bytes = map_path.read_bytes()
    old_map = json.loads(original_map_bytes.decode("utf-8"))
    assert not (HERE / "references").exists() and not (HERE / "REPORT_COPY_MAP_v2.json").exists(), "FRESH_SUPPLEMENT_REQUIRED"
    missing = [row for row in old_map["markdown_link_rewrites"] if row["reason"] == "local_asset_not_published"]
    assert len(missing) == 9
    sources = {destination(ROOT / row["source_report"], row["original_target"]) for row in missing}
    assert {relative(path) for path in sources} == TARGETS
    references = {source: HERE / "references" / source.relative_to(ROOT) for source in sources}
    for source in sources:
        assert source.is_file() and source.suffix in (".md", ".py", ".json")
    tracked = {name.decode("utf-8") for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0") if name}
    existing = {(ROOT / row["original_path"]).resolve(): ROOT / row["copy_path"] for row in old_map["files"]}
    previous = {(ROOT / row["original_path"]).resolve(): ROOT / row["copy_path"] for row in read(PREVIOUS / "REPORT_COPY_MAP.json")["files"]}
    archives = {(ROOT / row["original_repo_path"]).resolve(): ROOT / row["published_repo_path"] for row in read(HERE / "EVIDENCE_ARCHIVES.json")["records"]}
    outputs, reference_rewrites, new_rewrites, original_hashes = {}, [], [], {}
    for row in old_map["files"]:
        source, target = ROOT / row["original_path"], ROOT / row["copy_path"]
        assert source.suffix not in (".npz", ".pt", ".bin")
        original_hashes[relative(source)] = sha(source.read_bytes())
        assert original_hashes[relative(source)] == row["original_sha256"] and sha(target.read_bytes()) == row["copy_sha256"]
        if source.suffix != ".md":
            assert source.read_bytes() == target.read_bytes()
    for row in old_map["markdown_link_rewrites"]:
        changed = dict(row)
        if row["reason"] == "local_asset_not_published":
            local = destination(ROOT / row["source_report"], row["original_target"])
            changed.update(previous_published_target=row["published_target"], previous_reason=row["reason"], published_target=link_to(references[local], ROOT / row["copy_report"], row["original_target"]), reason="legacy_text_reference_copy")
        new_rewrites.append(changed)
    for row in old_map["files"]:
        source, target = ROOT / row["original_path"], ROOT / row["copy_path"]
        if source.suffix != ".md":
            continue
        original = source.read_text(encoding="utf-8-sig")
        old_text, new_text = original, original
        for before, after in zip(old_map["markdown_link_rewrites"], new_rewrites):
            if before["copy_report"] == row["copy_path"]:
                old_text = replace_link(old_text, before["original_target"], before["published_target"])
                new_text = replace_link(new_text, after["original_target"], after["published_target"])
        header = HEADER.format(index=link_to(HERE / "README_KO.md", target))
        assert (header + old_text).encode("utf-8") == target.read_bytes(), "CURRENT_DRAFT_MD_MUST_MATCH_ORIGINAL_TRANSFORM"
        outputs[target] = (header + new_text).encode("utf-8")
    for source, target in sorted(references.items()):
        payload = source.read_bytes()
        original_hashes[relative(source)] = sha(payload)
        if source.suffix != ".md":
            outputs[target] = payload
            continue
        text = source.read_text(encoding="utf-8-sig")
        for link in dict.fromkeys(checker.markdown_links(text)):
            local = destination(source, link)
            if local is None:
                continue
            if local in references:
                chosen, reason = references[local], "selected_legacy_text_reference"
            elif local in existing:
                chosen, reason = existing[local], "current_publication_copy"
            elif local in previous:
                chosen, reason = previous[local], "previous_publication_copy"
            elif local in archives:
                chosen, reason = archives[local], "lossless_gzip_JSON"
            elif local.is_relative_to(ROOT) and relative(local) in tracked:
                chosen, reason = local, "already_tracked_repository_evidence"
            else:
                chosen, reason = HERE / "references/README_KO.md", "unpublished_historical_dependency_not_recursively_copied"
            updated = link_to(chosen, target, link) if reason != "unpublished_historical_dependency_not_recursively_copied" else link_to(chosen, target) + "#unpublished-historical-dependencies"
            text = replace_link(text, link, updated)
            reference_rewrites.append({"source_report": relative(source), "copy_report": relative(target), "original_target": link, "published_target": updated, "reason": reason})
        header = "> 과거 캠페인 텍스트의 게시 사본입니다. 당시의 수치·상태·내용을 유지했으며 링크와 이 안내만 바꿨습니다. 현재 연구 완료 상태는 [최신 색인]({index}), 추가 게시 범위와 미게시 역사자료는 [참고 안내]({reference})를 확인하세요.\n\n"
        outputs[target] = (header.format(index=link_to(HERE / "README_KO.md", target), reference=link_to(HERE / "references/README_KO.md", target)) + text).encode("utf-8")
    descriptions = ["# 과거 캠페인 텍스트 참고 자료", "", "현재 보고서의 9개 참조 링크에 대응하는 기존 텍스트 8파일만 추가했다. 과거 캠페인 전체를 재귀 복사하거나 다시 실행하지 않았다. Python·JSON 사본은 byte 동일하며 Markdown은 링크와 게시 안내만 변환했다. 원본은 수정하지 않았다.", "", "이 자료의 과거 완료·미완료·누출/접근 incident·성능 수치는 작성 당시의 기록이다. 최신 상태는 [게시 색인](../README_KO.md)을 따른다. 파일별 원본/사본 SHA와 원래 참조 경로는 [수정 복사 기록](../REPORT_COPY_MAP_v2.json)에 남겼다.", ""]
    descriptions += ["- [" + relative(source) + "](" + source.relative_to(ROOT).as_posix() + ")" for source in sorted(sources)]
    descriptions += ["", '<a id="unpublished-historical-dependencies"></a>', "", "## 미게시 역사자료 링크", "", "이 절로 온 링크의 원 대상은 과거 캠페인의 추가 문서·코드·JSON·수치표 등으로, 이번 8파일 추가 게시 범위에 포함하지 않은 역사자료다. 파일이 로컬에 없다는 뜻이나 NPZ·모델 가중치라는 뜻이 아니다. 이미 Git에 게시된 의존성은 그 파일로 연결하고, 나머지는 여기서 범위를 설명한다. 원 링크와 대상은 수정 복사 기록의 reference_markdown_link_rewrites에 보존했다. 전체 의존성을 다시 가져오거나 실행하지 않았다.", "", "원자료·NPZ·모델·checkpoint가 필요한 실제 실험 재현은 별도의 [로컬 전용 자산 안내](../CODE_AND_REPRODUCTION_KO.md#local-assets)를 따른다. 공개 참고 코드만으로 과거 캠페인 전체가 독립 재현됐다고 주장하지 않는다.", ""]
    outputs[HERE / "references/README_KO.md"] = "\n".join(descriptions).encode("utf-8")
    before = {relative(target): sha(target.read_bytes()) for target in outputs if target.exists()}
    for target, payload in outputs.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.is_relative_to(HERE) and sha(target.read_bytes()) == before[relative(target)]
            target.write_bytes(payload)
        else:
            with target.open("xb") as handle:
                handle.write(payload)
    records = []
    for row in old_map["files"]:
        target = ROOT / row["copy_path"]
        records.append({**row, "copy_sha256": sha(target.read_bytes()), "bytes": target.stat().st_size, "byte_identical": sha(target.read_bytes()) == row["original_sha256"], "previous_copy_sha256": row["copy_sha256"]})
    for source, target in sorted(references.items()):
        records.append({"original_path": relative(source), "original_sha256": original_hashes[relative(source)], "original_bytes": source.stat().st_size, "copy_path": relative(target), "copy_sha256": sha(target.read_bytes()), "bytes": target.stat().st_size, "byte_identical": source.read_bytes() == target.read_bytes(), "copy_role": "selected_legacy_text_reference"})
    for path, digest in original_hashes.items():
        assert sha((ROOT / path).read_bytes()) == digest, "ORIGINAL_FILE_CHANGED"
    assert map_path.read_bytes() == original_map_bytes and sum(Path(row["copy_path"]).suffix == ".md" for row in old_map["files"]) == 29
    assert all(row["byte_identical"] for row in records if Path(row["copy_path"]).suffix != ".md")
    result = {**old_map, "status": "COPIED_WITH_LINK_ONLY_MD_TRANSFORM_AND_NINE_LEGACY_LINK_REPAIRS", "previous_copy_map": {"repo_path": relative(map_path), "sha256": sha(original_map_bytes)}, "files": records, "markdown_link_rewrites": new_rewrites, "reference_markdown_link_rewrites": reference_rewrites, "legacy_reference_link_repairs": 9, "unique_legacy_text_files_added": 8, "existing_publication_MD_hashes_rechecked": 29, "publication_MD_files_changed": sorted(path for path, digest in before.items() if sha((ROOT / path).read_bytes()) != digest), "original_files_rechecked": original_hashes, "recursive_dependency_copy_performed": False, "new_scientific_execution": False}
    with (HERE / "REPORT_COPY_MAP_v2.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "SUPPLEMENTED", "reference_files": 8, "repaired_links": 9, "original_MD_hashes_rechecked": 29, "publication_MD_files_changed": result["publication_MD_files_changed"], "non_MD_copies_byte_identical": True, "original_files_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
