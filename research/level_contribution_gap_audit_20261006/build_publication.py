"""Export redacted evidence snapshots without modifying the local audit originals."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
PUBLIC = AUDIT / "public"
BASE_COMMIT = "2154e7f2b0e9878164e77536b8deaa9f372ac760"
BASE_URL = f"https://github.com/CanelE452/hierarchical-tsfm-peft/blob/{BASE_COMMIT}/"
AUDIT_NAMES = (
    "STATUS.md", "CRITICAL_REVIEW_KO.md", "NEXT_ACTIONS.md", "evidence_index.csv",
    "early_artifact_checks.json", "later_artifact_checks.json",
    "chronos_artifact_checks.json", "verification.json",
)
EXTRA_C2 = (
    "FINAL_REPORT_KO.md", "STATUS.md", "ACCURACY_INTERPRETATION.md", "ledger.json",
    "cost_binding.json", "cost_cpu_grid.json", "cost_cuda_grid.json",
    "cost_cpu_rows.json", "cost_cuda_rows.json",
    "cost_cpu_summary.json", "cost_cuda_summary.json",
)
TRACKED = set(subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", BASE_COMMIT], cwd=ROOT, text=True,
).splitlines())
SOURCE_PATTERN = re.compile(
    r"research/(?:level_chronos2_controls_v1|tsfm_peft_peacock_matched_raw_v17_20261002)/[A-Za-z0-9_./-]+"
)
PRIVATE_IDS = [value for key, value in json.loads(
    (AUDIT / "verification.json").read_text(encoding="utf-8-sig")
)["prior_authorization"].items() if key.endswith("_id") or "turn" in key]


def path_forms(path):
    forward = path.as_posix()
    back = str(path)
    return sorted({forward, back, back.replace("\\", "\\\\")}, key=len, reverse=True)


def sanitize(value):
    for prefix in path_forms(ROOT):
        value = value.replace(prefix + "/", "").replace(prefix + "\\\\", "").replace(prefix + "\\", "")
        value = value.replace(prefix, "<repository-root>")
    for prefix in path_forms(Path.home()):
        value = value.replace(prefix, "<user>")
    value = re.sub(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>|)\],;]+", "<local-path>", value)
    for identifier in PRIVATE_IDS:
        if isinstance(identifier, str):
            value = value.replace(identifier, "<private-chat-record>")
    return value


def sanitize_tree(value):
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [sanitize_tree(item) for item in value]
    if isinstance(value, dict):
        return {sanitize(key): sanitize_tree(item) for key, item in value.items()}
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


sources = {AUDIT / name for name in AUDIT_NAMES}
for name in AUDIT_NAMES:
    for match in SOURCE_PATTERN.findall((AUDIT / name).read_text(encoding="utf-8-sig")):
        path = ROOT / match.rstrip(".")
        if path.is_file() and path.suffix in {".md", ".json", ".csv", ".py", ".txt"}:
            sources.add(path)
sources.update(ROOT / "research/level_chronos2_controls_v1" / name for name in EXTRA_C2)
assert all(path.is_file() for path in sources)
destinations = {
    source: PUBLIC / source.name if source.parent == AUDIT
    else PUBLIC / "source_snapshots" / source.relative_to(ROOT)
    for source in sources
}
omitted_links = []


def fix_links(text, source, destination):
    def replace(match):
        label, target = match.group(1), match.group(2)
        if target.startswith(("https://", "http://", "mailto:", "#")):
            return match.group(0)
        path_part, separator, anchor = target.partition("#")
        candidate = ROOT / path_part if path_part.startswith(("research/", "_docs/", "docs/")) else source.parent / path_part
        candidate = candidate.resolve()
        suffix = separator + anchor
        if candidate in destinations:
            relative = Path(os.path.relpath(destinations[candidate], destination.parent)).as_posix()
            return f"[{label}]({relative}{suffix})"
        if candidate.is_relative_to(ROOT) and candidate.relative_to(ROOT).as_posix() in TRACKED:
            return f"[{label}]({BASE_URL}{candidate.relative_to(ROOT).as_posix()}{suffix})"
        omitted_links.append({"source": source.relative_to(ROOT).as_posix(), "target": sanitize(target)})
        return f"{label} (로컬 보존, 공개 사본 미포함: `{sanitize(target)}`)"
    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", replace, text)


records = []
for source in sorted(sources):
    original = source.read_bytes()
    destination = destinations[source]
    original_text = original.decode("utf-8-sig")
    if source.suffix == ".json":
        exported = json.dumps(sanitize_tree(json.loads(original_text)), ensure_ascii=False, indent=2) + "\n"
    else:
        exported = sanitize(original_text)
        if source.suffix == ".md":
            exported = fix_links(exported, source, destination)
            exported = (
                "> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 "
                + ("[publication_manifest.json](publication_manifest.json)" if source.parent == AUDIT
                   else "감사 공개 폴더의 `publication_manifest.json`")
                + "에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.\n\n"
                + exported
            )
            exported = "\n".join(line.rstrip() for line in exported.splitlines()).rstrip() + "\n"
    published = exported.encode("utf-8")
    if original.startswith(b"\xef\xbb\xbf") and source.suffix == ".csv":
        published = b"\xef\xbb\xbf" + published
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(published)
    records.append({
        "original_repository_path": source.relative_to(ROOT).as_posix(),
        "original_sha256": digest(original), "original_bytes": len(original),
        "published_path": destination.relative_to(AUDIT).as_posix(),
        "published_sha256": digest(published), "published_bytes": len(published),
        "byte_identical": original == published,
    })

manifest = {
    "purpose": "GitHub publication requested after the local evidence audit was complete",
    "base_commit_for_previously_tracked_evidence": BASE_COMMIT,
    "original_audit_receipts_are_historical_not_publication_checks": True,
    "source_hashes_in_snapshots_bind_original_bytes_not_redacted_bytes": True,
    "transforms": [
        "repository absolute paths become repository-relative",
        "personal/system paths and private chat identifiers are redacted",
        "JSON reserialized as UTF-8; Markdown note, links and trailing whitespace adapted for GitHub",
        "numeric values and scientific dispositions are preserved",
    ],
    "new_work": {"fits": 0, "optimizer_updates": 0, "model_forwards": 0, "gpu_measurements": 0,
                 "score_recomputations": 0, "bootstrap_runs": 0, "model_downloads": 0},
    "scope": "audit eight files plus selected existing Chronos-2 and Peacock RAW text evidence",
    "excluded": ["raw data", "model weights", "checkpoints", "prediction arrays", "unrelated untracked files"],
    "cost_row_storage": "36 CPU and 279 CUDA row dictionaries are included in aggregate JSONs; duplicate individual row files omitted",
    "source_snapshots_are_static_evidence_not_a_complete_executable_campaign": True,
    "files": records,
    "omitted_markdown_links": omitted_links,
}
(PUBLIC / "publication_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)
print(json.dumps({"published_snapshots": len(records), "bytes": sum(r["published_bytes"] for r in records),
                  "omitted_links": len(omitted_links)}, ensure_ascii=False))
