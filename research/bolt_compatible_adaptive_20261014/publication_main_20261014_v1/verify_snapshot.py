"""Verify a publication cohort using text, CSV, and image files only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
LATEST = BASE / "reports/first_and_dense_v3"
DEFAULT_OUTPUT = ROOT / ".cache/bolt_compatible_adaptive_20261014/publication_main_20261014_v1/LOCAL_VERIFICATION.json"
FORBIDDEN = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".pkl", ".pickle"}
TEXT = {".md", ".json", ".csv", ".txt", ".py", ".svg", ".log", ".ps1", ".sh", ".yaml", ".yml", ".toml"}
SECRET_PATTERNS = {
    "github_token": r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b",
    "hf_token": r"\bhf_[A-Za-z0-9]{20,}\b",
    "openai_token": r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b",
    "aws_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "phone_number": r"\b010[- ]?\d{4}[- ]?\d{4}\b",
    "quoted_credential_assignment": r"(?im)\b(?:api_key|password|passwd|access_token|secret_key)\s*[:=]\s*[\"'][^\"']+[\"']",
}


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path):
    assert path.suffix.lower() == ".json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    assert path.suffix.lower() not in FORBIDDEN, "Forbidden payload hashing: " + relative(path)
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def csv_rows(name):
    with (LATEST / "tables" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_csv_values():
    # Arithmetic follows the pure checks inside verify_report_artifacts_v1.py.
    # Its guarded main is deliberately neither imported nor invoked.
    producers = [read_json(BASE / "runs" / name / "RESULT.json")
                 for name in ("first_robin_v1", "first_beijing_v1")]
    dense = read_json(BASE / "runs/expanded_robin_v2/RESULT.json")
    counts = {}
    cells = 0
    for name, cohort, expected in (("COMPARISON.csv", producers, 72),
                                   ("EXPANDED_COMPARISON.csv", [dense], 12)):
        rows = csv_rows(name)
        assert len(rows) == expected, name
        unique = set()
        for row in rows:
            producer = next(p for p in cohort if p["dataset"] == row["dataset"])
            method, split, scale = row["method"], row["split"], row["metric_scale"]
            identity = (producer["dataset"], method, split, scale)
            assert identity not in unique, str(identity)
            unique.add(identity)
            if method == "RAW_BOLT":
                values = producer["references"][split]["RAW_BOLT"][scale]
            elif method in ("PERSIST", "MEAN"):
                values = producer["reference_shared_controls"][method][split][scale]
            else:
                values = producer["families"][method]["splits"][split][scale]
            for key in ("mse", "mae"):
                assert math.isclose(float(row[key]), values[key], rel_tol=2e-12, abs_tol=2e-12), str(identity)
                cells += 1
            for key in ("per_channel_mse", "per_channel_mae", "observed_counts"):
                saved = json.loads(row[key])
                assert len(saved) == len(values[key])
                assert all(math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12)
                           for a, b in zip(saved, values[key])), str(identity)
                cells += len(saved)
            assert int(row["active_channels"]) == values["active_channels"]
            assert row["independent_holdout"] == "False"
            cells += 2
        counts[name] = len(rows)
    for name, cohort, expected in (("POOL_SELECTION.csv", producers, 54),
                                   ("EXPANDED_POOL_SELECTION.csv", [dense], 18)):
        rows = csv_rows(name)
        assert len(rows) == expected, name
        for row in rows:
            producer = next(p for p in cohort if p["dataset"] == row["dataset"])
            family = producer["families"][row["method"]]
            index = int(row["candidate_index"])
            entry = family["pool"][index]
            assert row["candidate_kind"] == entry["kind"]
            assert (row["selected"] == "True") == (index == family["selected_index"])
            for split in ("fit", "tune"):
                for key in ("mse", "mae"):
                    assert math.isclose(float(row[split + "_" + key]),
                                        entry[split + "_scores"]["normalized"][key],
                                        rel_tol=2e-12, abs_tol=2e-12)
                    cells += 1
        counts[name] = len(rows)
    for name, cohort, expected in (("CAPACITY.csv", producers, 6),
                                   ("EXPANDED_CAPACITY.csv", [dense], 2)):
        rows = csv_rows(name)
        assert len(rows) == expected, name
        for row in rows:
            producer = next(p for p in cohort if p["dataset"] == row["dataset"])
            state = producer["feature_states"][row["method"]]
            assert int(row["channels"]) == state["input_channels"]
            assert int(row["features_per_output_channel"]) == state["feature_count"]
            assert int(row["potential_registered_coefficients"]) == state["feature_count"] * 48 * state["input_channels"]
        counts[name] = len(rows)
    for name, tag, expected in (("COST.csv", "first_readout24_v2", 24),
                                ("EXPANDED_COST.csv", "expanded_robin4_v1", 4)):
        cost = read_json(BASE / "cost_runs" / tag / "RESULT.json")
        rows = csv_rows(name)
        assert len(rows) == expected, name
        for row in rows:
            source = next(r for r in cost["rows"]
                          if (r["dataset"], r["route"], r["batch"]) ==
                          (row["dataset"], row["method"], int(row["batch"])))
            assert math.isclose(float(row["whole_median_ms_per_window"]),
                                1000 * source["median_seconds_per_window"],
                                rel_tol=2e-12, abs_tol=2e-12)
            cells += 1
        counts[name] = len(rows)
    attempts = csv_rows("EXECUTION_ATTEMPTS.csv")
    assert attempts, "Empty execution-attempt ledger"
    counts["EXECUTION_ATTEMPTS.csv"] = len(attempts)
    return {"csv_rows": counts, "metric_cells_checked": cells,
            "arithmetic_source_sha256": digest(BASE / "verify_report_artifacts_v1.py")}


def verify(output):
    manifest_path = HERE / "FILES.json"
    manifest = read_json(manifest_path)
    assert isinstance(manifest.get("base_commit"), str) and len(manifest["base_commit"]) == 40
    assert manifest.get("status")
    entries = manifest["files"]
    cohort = {}
    for entry in entries:
        name = entry["path"]
        assert isinstance(name, str) and "\\" not in name
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and relative(path) == name
        assert name not in cohort, "Duplicate cohort path: " + name
        assert not name.startswith("_docs/history/"), "Dirty historical file in cohort"
        assert path.suffix.lower() not in FORBIDDEN, "Forbidden publication extension: " + name
        assert path.is_file() and path.stat().st_size == entry["bytes"], name
        assert path.stat().st_size <= 20 * 1024 * 1024, "Unexpected large publication file: " + name
        assert digest(path) == entry["sha256"], "Publication SHA mismatch: " + name
        cohort[name] = path
    assert relative(manifest_path) not in cohort, "FILES.json must not hash itself"
    allowed = set(cohort) | {relative(manifest_path)}
    assert relative(output) not in allowed, "Verification receipt would create a SHA cycle"
    tracked_output = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    tracked = set(tracked_output.decode("utf-8").split("\0"))
    links = []
    for name, path in cohort.items():
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8-sig")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip().split(' "', 1)[0].strip("<>")
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            assert not urlsplit(target).scheme and not target.startswith(("/", "\\")), "Local absolute Markdown target in " + name
            local = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            assert local.is_relative_to(ROOT) and local.exists(), "Missing Markdown target in " + name + ": " + target
            local_name = relative(local)
            if local.is_dir():
                assert any(p.startswith(local_name + "/") for p in allowed | tracked), "Unpublished linked directory: " + local_name
            else:
                assert local_name in allowed | tracked, "Unpublished linked file: " + local_name
            links.append({"document": name, "target": target, "resolved_path": local_name})
    secret_counts = {name: 0 for name in SECRET_PATTERNS}
    secret_files = {}
    for name, path in list(cohort.items()) + [(relative(manifest_path), manifest_path)]:
        if path.suffix.lower() not in TEXT:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for pattern_name, pattern in SECRET_PATTERNS.items():
            count = len(re.findall(pattern, text))
            secret_counts[pattern_name] += count
            if count:
                secret_files.setdefault(pattern_name, []).append(name)
    assert not any(secret_counts.values()), "Credential/phone patterns detected; inspect filenames only: " + json.dumps(secret_files)
    bindings = []
    def check_bindings(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path = Path(value["path"])
                if not path.is_absolute():
                    path = ROOT / path
                path = path.resolve()
                assert path.is_relative_to(ROOT), "External source SHA record"
                assert path.suffix.lower() not in FORBIDDEN, "Scientific payload in report manifest; not opened"
                assert path.is_file() and digest(path) == value["sha256"], "Scientific source SHA mismatch: " + relative(path)
                if isinstance(value.get("snapshot_path"), str):
                    snapshot = Path(value["snapshot_path"])
                    if not snapshot.is_absolute():
                        snapshot = ROOT / snapshot
                    assert snapshot.resolve().is_relative_to(ROOT)
                    assert snapshot.is_file() and digest(snapshot) == value["sha256"], "Scientific snapshot SHA mismatch"
                bindings.append(relative(path))
            for child in value.values():
                check_bindings(child)
        elif isinstance(value, list):
            for child in value:
                check_bindings(child)
    scientific_manifest = read_json(LATEST / "MANIFEST.json")
    assert scientific_manifest["status"] == "COMPLETE"
    check_bindings(scientific_manifest)
    from PIL import Image
    pngs = sorted((LATEST / "figures").glob("*.png"))
    svgs = sorted((LATEST / "figures").glob("*.svg"))
    assert len(pngs) == len(svgs) == 8
    assert sum(p.suffix.lower() == ".png" for p in cohort.values()) == 8
    assert sum(p.suffix.lower() == ".svg" for p in cohort.values()) == 8
    image_records = []
    for path in pngs:
        assert relative(path) in allowed
        with Image.open(path) as image:
            width, height = image.size
            assert width >= 1000 and height >= 400
            image.verify()
        image_records.append({"path": relative(path), "width": width, "height": height, "sha256": digest(path)})
    for path in svgs:
        assert relative(path) in allowed
        tree = ET.parse(path)
        assert tree.getroot().tag.endswith("}svg") or tree.getroot().tag == "svg"
    csv_result = verify_csv_values()
    return {"status": "PASS", "scope": "CPU filesystem/text/CSV/image publication verification only",
            "base_commit": manifest["base_commit"], "files_manifest_sha256": digest(manifest_path),
            "cohort_file_count": len(cohort), "cohort_bytes": sum(p.stat().st_size for p in cohort.values()),
            "relative_links_checked": len(links), "links": links,
            "scientific_manifest_sha256": digest(LATEST / "MANIFEST.json"),
            "scientific_sha_bindings_checked": len(bindings), "scientific_bound_paths": bindings,
            "PNG": image_records, "SVG_count": len(svgs), "secret_pattern_counts": secret_counts,
            "forbidden_payload_opened": False, "TEST_opened": False,
            "new_fit_or_model_forward": False, "scientific_runtime_imported": False,
            "git_mutated": False, **csv_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert output.is_relative_to(ROOT / ".cache"), "Receipt must stay in local .cache"
    assert not output.exists(), "Preserve old receipt; choose a fresh output filename"
    try:
        result = verify(output)
    except Exception as failure:
        result = {"status": "FAIL", "error_type": type(failure).__name__, "error": str(failure),
                  "scope": "CPU filesystem/text/CSV/image publication verification only",
                  "TEST_opened": False, "new_fit_or_model_forward": False,
                  "scientific_runtime_imported": False, "git_mutated": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "receipt": relative(output),
                      "files": result.get("cohort_file_count"),
                      "links": result.get("relative_links_checked"),
                      "metric_cells": result.get("metric_cells_checked"),
                      "error_type": result.get("error_type")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
