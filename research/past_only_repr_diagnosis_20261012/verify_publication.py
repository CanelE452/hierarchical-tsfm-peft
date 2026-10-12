"""Check only published code, reports and metadata; never open model/data arrays."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALLOWED = HERE / "PUBLICATION_FILES_v1.txt"
AUDIT = HERE / "publication_receipts/PRE_PUSH_AUDIT_v2.json"
FORBIDDEN = {".pt", ".pth", ".safetensors", ".ckpt", ".npz", ".npy", ".parquet"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def saved_path(value):
    """Map recorded repo paths for metadata inspection on a relocated clone."""
    value = value.replace("\\", "/")
    for marker in ("research/", "README.md"):
        index = value.find(marker)
        if index >= 0:
            return ROOT / value[index:]
    raise ValueError("Not a published repository path: " + value)


def png_check(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", str(path)
    offset, dimensions, ended = 8, None, False
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        stored_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == stored_crc, str(path)
        if kind == b"IHDR":
            dimensions = struct.unpack_from(">II", payload)
        offset += length + 12
        if kind == b"IEND":
            ended = True
            break
    assert ended and offset == len(data) and dimensions, str(path)
    return {"path": path.relative_to(ROOT).as_posix(), "width": dimensions[0],
            "height": dimensions[1], "crc": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-audit", action="store_true")
    args = parser.parse_args()
    paths = [ROOT / row for row in ALLOWED.read_text(encoding="utf-8-sig").splitlines() if row]
    assert len(paths) == len(set(paths)), "Duplicate publication files"
    allowed = set(paths)
    tracked = {ROOT / row for row in subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT).decode("utf-8").splitlines()}
    assert all(path.is_relative_to(ROOT) for path in paths)
    assert all(path.suffix.lower() not in FORBIDDEN and ".cache" not in path.parts for path in paths)
    assert all(path.exists() for path in paths if path != AUDIT)
    assert all(path.stat().st_size < 100 * 1024 ** 2 for path in paths if path.exists())

    evidence = read(HERE / "EVIDENCE_FINAL_v5.json")
    seal_checks = []
    for row in evidence["canonical_bindings"]:
        path = saved_path(row["path"])
        assert path in allowed
        actual = digest(path)
        assert actual == row["sha256"], str(path)
        seal_checks.append({"path": path.relative_to(ROOT).as_posix(), "sha256": actual, "status": "PASS"})
    manifest = read(HERE / "final_report_v5_20261012/MANIFEST_v5.json")
    decision = manifest["verification_decision"]
    assert decision["status"] == decision["binding_status"] == "PASS"
    assert decision["binding_summary"] == {"count": 3429, "pass": 3429, "fail": 0, "unchecked": 0}
    assert len(decision["excluded"]) == 0
    proof_path = saved_path(decision["proof_path"])
    assert digest(proof_path) == decision["proof_sha256"]
    proof = read(proof_path)
    assert proof["overall_status"] == "PASS"
    primary = {key: proof[key]["status"] for key in ("E3_E4", "E5", "E5_BOLT", "E6", "COST_CLOSURE")}
    assert set(primary.values()) == {"PASS"}

    metadata_bindings = []
    for row in manifest["manifest"]:
        relative = row.get("relative_path", "")
        if not relative or relative.startswith(".cache/") or Path(relative).suffix.lower() in FORBIDDEN:
            continue
        path = ROOT / relative
        assert path in allowed or path in tracked, relative
        assert path.exists() and digest(path) == row["sha256"], relative
        metadata_bindings.append(relative)
    outputs = manifest["outputs"]
    for key in ("raw_metrics_csv", "cost_closure_timing_csv", "report_ko", "professor_summary_ko"):
        path = saved_path(outputs[key])
        assert path in allowed and digest(path) == outputs[key + "_sha256"], key
    figures = list(outputs["figures"])
    supplement = read(HERE / "final_report_v5_20261012/COVERAGE_READABILITY_RECEIPT_v1.json")
    figures.append({"path": supplement["figure"], "sha256": supplement["figure_sha256"]})
    pngs = []
    for figure in figures:
        path = saved_path(figure["path"])
        assert path in allowed and digest(path) == figure["sha256"]
        pngs.append(png_check(path))
    assert len(pngs) == 8
    with saved_path(outputs["raw_metrics_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        counts = Counter(row["experiment"] for row in csv.DictReader(handle))
    assert dict(counts) == supplement["counts"] and sum(counts.values()) == 668

    docs = [ROOT / "README.md", HERE / "README.md", HERE / "PUBLICATION_KO.md", HERE / "INDEX_FINAL_v5.md",
            HERE / "final_report_v5_20261012/REPORT_KO_v5.md",
            HERE / "final_report_v5_20261012/PROFESSOR_SUMMARY_KO_v5.md"]
    links = []
    for doc in docs:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8-sig")):
            target = target.strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            if doc == ROOT / "README.md" and HERE.name not in target:
                continue
            path = (doc.parent / unquote(target.split("#", 1)[0])).resolve()
            assert path in allowed or path in tracked, (doc.name, target, "not published")
            assert path.exists() or path == AUDIT, (doc.name, target, "missing")
            links.append({"source": doc.relative_to(ROOT).as_posix(), "target": target, "status": "PASS"})
    syntax = []
    hashes = []
    secret = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,})")
    for path in paths:
        if path == AUDIT:
            continue
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            syntax.append(path.relative_to(ROOT).as_posix())
        if path.suffix != ".png":
            assert not secret.search(path.read_bytes()), "Credential-like token in " + str(path)
        hashes.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
    result = {"status": "PASS", "scope": "Publication metadata only; not a rerun of scientific array verification",
              "protected_TEST_access": 0, "model_inference": 0, "new_training": 0,
              "sealed_evidence": seal_checks, "historical_science_primary_sections": primary,
              "historical_report_binding_summary": decision["binding_summary"],
              "historical_completed_science_exclusions": 0,
              "fresh_published_metadata_bindings_checked": len(metadata_bindings),
              "raw_csv_rows_not_independent_fits": sum(counts.values()), "csv_coverage_counts": dict(counts),
              "figures": pngs, "relative_links": links, "syntax_checked": syntax,
              "publication_files": len(paths), "file_hashes_excluding_audit_itself": hashes,
              "self_hash_exclusion_reason": "Avoid circular audit receipt hashing"}
    if args.write_audit:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    else:
        sealed = read(AUDIT)
        assert result == sealed, "Published files changed after pre-push audit"
    print(json.dumps({key: value for key, value in result.items() if key not in
                      {"file_hashes_excluding_audit_itself", "relative_links", "syntax_checked", "sealed_evidence"}}, ensure_ascii=True))
    print("relative_links", len(links), "syntax_files", len(syntax))


if __name__ == "__main__":
    main()
