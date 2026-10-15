"""Check the retrospective decision ledger's six fields, links and source bytes."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PUBLICATION = HERE.parent / "publication_main_20261015_v2"
DOCUMENT = HERE / "DECISION_LEDGER_CONTINUATION_KO_v1.md"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path):
    return {"repo_path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}


def main(output):
    checker_path = PUBLICATION / "verify_publication.py"
    spec = importlib.util.spec_from_file_location("ledger_publication_checker", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    payload = DOCUMENT.read_bytes()
    text = payload.decode("utf-8-sig")
    fields = ("질문", "현 증거", "경쟁 설명", "선택한 검사", "실제 결과", "다음 결정")
    entries = list(re.finditer(r"(?m)^## C(\d{2})\s+[—–-]\s+(.+)$", text))
    assert [int(item.group(1)) for item in entries] == list(range(1, 26)), "ALL_TWENTY_FIVE_CONTIGUOUS_DECISION_IDS"
    indexed = {name.decode("utf-8") for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0") if name}
    for path in (DOCUMENT, Path(__file__).resolve()):
        name = path.relative_to(ROOT).as_posix()
        assert name in indexed and subprocess.check_output(["git", "show", ":" + name], cwd=ROOT) == path.read_bytes(), "EXACT_INDEXED_DOCUMENT_AND_AUDITOR"
    verified, decisions = {}, []
    link_count = 0
    allowed_suffixes = {".md", ".py", ".json", ".jsonl", ".csv", ".png", ".svg", ".gz", ".txt"}
    archives = {row["published_repo_path"]: row["archive_sha256"] for row in json.loads((PUBLICATION / "EVIDENCE_ARCHIVES.json").read_text(encoding="utf-8"))["records"]}
    for index, match in enumerate(entries):
        section = text[match.end():entries[index + 1].start() if index + 1 < len(entries) else len(text)]
        found = list(re.finditer(r"(?m)^([1-6])\. \*\*(.+?)\*\*:\s*", section))
        assert [(int(item.group(1)), item.group(2)) for item in found] == list(enumerate(fields, 1)), (match.group(0), "EXACT_SIX_FIELDS")
        for number, item in enumerate(found):
            content = section[item.end():found[number + 1].start() if number + 1 < len(found) else len(section)].strip()
            assert len(content) >= 20, (match.group(0), item.group(2), "NONEMPTY_FIELD")
        targets = checker.markdown_links(section)
        assert len(targets) >= 2, (match.group(0), "TRACEABLE_SOURCES")
        decisions.append({"id": "C" + match.group(1), "title": match.group(2), "six_fields_checked": list(fields), "source_links": len(targets)})
    for target in checker.markdown_links(text):
        resolved = checker.resolved_link(DOCUMENT, target)
        if resolved is None:
            continue
        path = (ROOT / resolved).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), (target, "EXISTING_REPOSITORY_FILE")
        assert path.suffix.lower() in allowed_suffixes, (target, "ONLY_DOCUMENT_CODE_AND_PUBLISHED_EVIDENCE_LINKS")
        assert path.suffix.lower() != ".gz" or resolved in archives, (target, "ONLY_REGISTERED_JSON_ARCHIVES")
        assert resolved in indexed, (target, "SOURCE_ALREADY_IN_GIT_INDEX")
        if resolved not in verified:
            binding = record(path)
            assert path.suffix.lower() != ".gz" or binding["sha256"] == archives[resolved], (target, "REGISTERED_ARCHIVE_SHA")
            indexed_bytes = subprocess.check_output(["git", "show", ":" + resolved], cwd=ROOT)
            assert len(indexed_bytes) == binding["bytes"] and hashlib.sha256(indexed_bytes).hexdigest() == binding["sha256"], (target, "EXACT_INDEXED_SOURCE_BYTES")
            verified[resolved] = binding
        link_count += 1
    findings = {name: len(pattern.findall(payload)) for name, pattern in checker.SECRET.items() if pattern.search(payload)}
    assert not findings, findings
    result = {"status": "PASS", "scope": "Retrospective decision-document structure, local indexed links and byte bindings only; ROOT separately reviews the six-field substance against original plans/results/reports. This is not a new scientific execution, independent optimization replay or generalization validation.",
              "document": record(DOCUMENT), "source": record(Path(__file__).resolve()), "reused_checker": record(checker_path),
              "original_requirement_sources": [record(HERE.parent / name) for name in ("INPUT_PLAN_20261014.md", "PHASE1_PLAN_KO.md", "DECISION_LEDGER_FINAL_KO.md")],
              "prior_publication_index_check": record(PUBLICATION / "FINAL_INDEX_CHECKS.json"),
              "decision_entries_checked": len(decisions), "required_fields_per_entry": 6, "decisions": decisions,
              "local_indexed_links_checked": link_count, "unique_source_files_checked": len(verified), "source_files": list(verified.values()),
              "all_linked_source_bytes_match_index": True,
              "sensitive_pattern_findings": findings, "scientific_array_model_or_F_opened": False,
              "new_training_inference_or_cost_measurements": False,
              "small_K_Raw_accuracy_gap_resolved": False,
              "unexecuted_TRAIN_expansion_or_A_only_design_counted_as_execution": False,
              "HEAD_at_audit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()}
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "decisions": len(decisions), "fields": 6, "links": link_count, "source_files": len(verified), "document_sha256": result["document"]["sha256"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "audits/DECISION_LEDGER_CONTINUATION_AUDIT_v1.json")
    main(parser.parse_args().output)
