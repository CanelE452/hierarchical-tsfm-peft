"""Create a non-overwriting safe metadata derivative and verify this delta audit."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--diagnostic", required=True)
    args = parser.parse_args()
    public, cache, diagnostic = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.diagnostic))
    before_files = {p.name: sha(p) for p in diagnostic.iterdir() if p.is_file()}
    original = read(diagnostic / "RESULT.json")
    safe = copy.deepcopy(original)
    changed = []
    for index, value in enumerate(safe.get("threadpools", [])):
        if "filepath" in value:
            old = value["filepath"]
            value["filepath"] = Path(old).name
            changed.append({"field": f"threadpools[{index}].filepath", "replacement": value["filepath"], "original_value_sha256": hashlib.sha256(old.encode()).hexdigest()})
    new(diagnostic / "RESULT_PUBLIC.json", safe)
    restored = copy.deepcopy(safe)
    for index, value in enumerate(restored.get("threadpools", [])):
        if "filepath" in value:
            value["filepath"] = original["threadpools"][index]["filepath"]
    if restored != original:
        raise AssertionError("Projection changed a field outside the declared paths")
    projection = {"status": "PASS_METADATA_PATH_ONLY_PROJECTION", "original_file": "RESULT.json", "original_sha256": before_files["RESULT.json"],
        "public_file": "RESULT_PUBLIC.json", "public_sha256": sha(diagnostic / "RESULT_PUBLIC.json"), "changed_fields": changed,
        "all_scientific_and_resource_values_identical": True, "restored_JSON_equals_original": True, "original_file_preserved": True,
        "original_file_excluded_from_publication": diagnostic.relative_to(public).as_posix() + "/RESULT.json"}
    new(diagnostic / "PUBLIC_PROJECTION_PROOF.json", projection)
    records = read(diagnostic / "AUDIT_RESULTS.json")["records"]
    with (diagnostic / "SAVED_V_RESCORING.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(records) != 13 or len(csv_rows) != 13 or {row["run_id"] for row in records} != {row["run_id"] for row in csv_rows}:
        raise AssertionError("Actual audit record/CSV identity count differs")
    for row, record in zip(csv_rows, records):
        for key, value in row.items():
            if value != str(record[key]):
                raise AssertionError("Actual audit record/CSV value differs")
    before = read(diagnostic / "INPUTS_SHA256_BEFORE.json")
    after = read(diagnostic / "INPUTS_SHA256_AFTER.json")
    if before != after or len(before) != 549:
        raise AssertionError("Original immutable input before/after changed")
    rechecked = 0
    for identity, expected in after.items():
        scope, relative = identity.split(":", 1)
        path = (public if scope == "public" else cache) / relative
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise AssertionError("Original input changed after audit")
        rechecked += 1
    for name, expected in original["sources_after"].items():
        if sha(public / name) != expected:
            raise AssertionError("Audited source changed")
    union = []
    seen_keys, seen_archives = set(), set()
    for identity in before:
        scope, relative = identity.split(":", 1)
        if scope != "public" or not relative.endswith("/PREDICTION_MANIFEST.json"):
            continue
        manifest = read(public / relative)
        key = (manifest["run_id"], manifest["split"], manifest["cohort"])
        archive = cache / manifest["cache_relative_path"]
        if key in seen_keys or manifest["cache_relative_path"] in seen_archives:
            raise AssertionError("Union has duplicated bundle identity/archive path")
        if not archive.is_file():
            raise AssertionError("Union declared archive no longer exists")
        seen_keys.add(key)
        seen_archives.add(manifest["cache_relative_path"])
        union.append({"run_id": key[0], "split": key[1], "cohort": key[2], "manifest_public_relative_path": relative,
            "manifest_sha256": sha(public / relative), "archive_cache_relative_path": manifest["cache_relative_path"],
            "archive_sha256_as_declared": manifest["archive_sha256"], "archive_exists": True, "archive_bytes": archive.stat().st_size,
            "new13_actual_FP64_rescore": key[0] in {row["run_id"] for row in records}, "old395_arrays_rerescored_in_this_delta": False})
    if len(union) != 408 or sum(row["split"] == "V_select" for row in union) != 372 or sum(row["split"] == "M_cal" for row in union) != 36:
        raise AssertionError("Verified archive union dimensions differ")
    with (diagnostic / "UNION_BUNDLES.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(union[0]))
        writer.writeheader()
        writer.writerows(union)
    sys.path.insert(0, str(public))
    import publication_check
    failures = []
    excluded = {"RESULT.json"}
    candidates = sorted(path for path in diagnostic.iterdir() if path.is_file() and path.name not in excluded)
    for path in candidates:
        publication_check._scan_text(path.relative_to(public).as_posix(), path.read_text(encoding="utf-8"), failures)
    original_findings = []
    publication_check._scan_text("RESULT.json", (diagnostic / "RESULT.json").read_text(encoding="utf-8"), original_findings)
    if failures or not any(item["rule"] == "PRIVATE_ABSOLUTE_PATH" for item in original_findings):
        raise AssertionError("Publication scan did not match expected safe/original scope")
    if before_files != {name: sha(diagnostic / name) for name in before_files}:
        raise AssertionError("An existing evidence file was overwritten")
    result = {"status": "PASS_READBACK_UNION_AND_PUBLIC_PROJECTION", "actual_scoring_process_exit_code": 0,
        "exit_evidence_scope": "ROOT tool session61289 returned actual exit_code0 after all13 PASS", "audited_records": len(records), "CSV_rows_and_values_exact": True,
        "max_abs_MSE_difference": max(abs(row["mse_difference"]) for row in records), "max_abs_MAE_difference": max(abs(row["mae_difference"]) for row in records),
        "immutable_input_files_rechecked": rechecked, "source_files_rechecked": len(original["sources_after"]), "original_files_preserved": True,
        "unique_bundles_with_existing_archives": len(union), "unique_V": 372, "unique_dense_M": 36,
        "previous395_numeric_audits_reused_not_rerun": True, "new13_actual_rescores": 13, "arrays_or_model_forward_in_this_readback": False,
        "publication_scan_module_sha256": sha(public / "publication_check.py"), "publication_scan_scope": "Existing _scan_text privacy/token/phone/path rules on this explicit delta allowlist; no Git operations",
        "scanned_safe_files": [path.name for path in candidates], "safe_scan_findings": failures,
        "original_excluded_findings": original_findings, "exact_public_exclusions": [diagnostic.relative_to(public).as_posix() + "/RESULT.json"],
        "created_unix": time.time()}
    new(diagnostic / "VERIFICATION.json", result)
    print(json.dumps({"status": result["status"], "records": len(records), "existing_unique_bundles": len(union), "privacy_findings": len(failures)}))


if __name__ == "__main__":
    main()
