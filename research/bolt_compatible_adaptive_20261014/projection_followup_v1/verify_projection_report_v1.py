"""Verify report bytes, local links, and primary CSV scores against terminal metadata."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_score(result, row):
    data = result["datasets"][row["dataset"]]
    if row["method"] == "RAW_BOLT":
        metrics = data["raw_scores"][row["split"]]
    else:
        condition = data["projections"][row["projection"]]
        if row["method"].endswith("_NATIVE"):
            metrics = condition["native"][row["seed"]]["scores"][row["split"]]
        else:
            metrics = condition["bolt"]["scores"][row["split"]]
    return metrics[row["metric_scale"]]


def check_comparison(result, rows):
    keys, fields = set(), 0
    for row in rows:
        key = tuple(row[name] for name in ("dataset", "method", "seed", "split", "metric_scale"))
        if key in keys:
            raise ValueError("DUPLICATE_COMPARISON_ROW")
        keys.add(key)
        expected = expected_score(result, row)
        for name in ("mse", "mae"):
            if float(row[name]) != expected[name]:
                raise ValueError("CSV_SCORE_MISMATCH:" + repr(key) + ":" + name)
            fields += 1
        for name in ("active_channels",):
            if int(row[name]) != expected[name]:
                raise ValueError("CSV_CHANNEL_COUNT_MISMATCH")
            fields += 1
        for name in ("observed_counts", "per_channel_mse", "per_channel_mae"):
            if json.loads(row[name]) != expected[name]:
                raise ValueError("CSV_CHANNEL_VECTOR_MISMATCH:" + name)
            fields += len(expected[name])
    return fields


def check_links(report):
    checked = []
    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", report.read_text(encoding="utf-8-sig")):
        target = match.group(1).strip().strip("<>")
        parts = urlsplit(target)
        if parts.scheme or not parts.path:
            continue
        path = (report.parent / unquote(parts.path)).resolve()
        if not path.is_file():
            raise ValueError("MISSING_LOCAL_REPORT_LINK:" + target)
        checked.append(target)
    return checked


def verify(folder):
    folder = Path(folder).resolve()
    manifest, provenance = read(folder / "MANIFEST.json"), read(folder / "PROVENANCE.json")
    producer = provenance["producer"]
    result = read(producer["path"])
    audit = read(provenance["independent_audit"]["path"])
    if result["status"] != "COMPLETE" or audit["status"] != "PASS":
        raise ValueError("VERIFICATION_REQUIRES_ACCEPTED_PRODUCER")
    if sha(producer["path"]) != producer["sha256"] or audit["producer"]["sha256"] != producer["sha256"]:
        raise ValueError("REPORT_PRODUCER_BINDING")
    for field in ("producer", "independent_audit", "generator"):
        if sha(provenance[field]["path"]) != provenance[field]["sha256"]:
            raise ValueError("PROVENANCE_FILE_CHANGED:" + field)
    diagnostic = None
    if provenance.get("additional_diagnostic") is not None:
        diagnostic_record = provenance["additional_diagnostic"]
        diagnostic_audit_record = provenance["additional_diagnostic_audit"]
        for item in (diagnostic_record, diagnostic_audit_record):
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("ADDITIONAL_DIAGNOSTIC_PROVENANCE_CHANGED")
        diagnostic = read(diagnostic_record["path"])
        diagnostic_audit = read(diagnostic_audit_record["path"])
        if (diagnostic["status"] != "COMPLETE" or diagnostic_audit["status"] != "PASS"
                or diagnostic_audit["diagnostic"]["sha256"] != diagnostic_record["sha256"]
                or diagnostic["producer"]["sha256"] != producer["sha256"]):
            raise ValueError("ADDITIONAL_DIAGNOSTIC_ACCEPTED_BINDING")
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder) or sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("REPORT_ARTIFACT_MANIFEST_CHANGED:" + item["path"])
    row_counts = {}
    for table, count in manifest["csv_rows"].items():
        rows = csv_rows(folder / "tables" / (table + ".csv"))
        if len(rows) != count:
            raise ValueError("CSV_ROW_COUNT_MISMATCH:" + table)
        row_counts[table] = len(rows)
    rows = csv_rows(folder / "tables/COMPARISON.csv")
    if len(rows) != 84:
        raise ValueError("EXACT_DEPLOYMENT_COMPARISON_COUNT")
    score_fields = check_comparison(result, rows)
    fixed_d_fields = 0
    if diagnostic is not None:
        for row in csv_rows(folder / "tables/FIXED_D_MATCHED_MASK.csv"):
            cell = diagnostic["datasets"][row["dataset"]]["projections"][row["projection"]][row["split"]]
            scores = (cell["fixed_D_bound_scores_same_mask"] if row["method"] == "FIXED_D_OPTIMAL_REAL_LATENT"
                      else cell["comparisons"][row["method"]]["saved_prediction_scores_same_mask"])
            expected = scores[row["metric_scale"]]
            for name in ("mse", "mae"):
                if float(row[name]) != expected[name]:
                    raise ValueError("FIXED_D_CSV_SCORE_MISMATCH:" + row["method"])
                fixed_d_fields += 1
            for name in ("observed_counts", "per_channel_mse", "per_channel_mae"):
                if json.loads(row[name]) != expected[name]:
                    raise ValueError("FIXED_D_CSV_CHANNEL_VECTOR_MISMATCH")
                fixed_d_fields += len(expected[name])
            if (row["this_MAE_is_a_minimum"] != "False" or row["this_physical_MSE_is_a_minimum"] != "False"
                    or row["Raw_bound_inequality_applied"] != "False"):
                raise ValueError("FIXED_D_CSV_MINIMUM_SCOPE_CHANGED")
    links = check_links(folder / "REPORT_KO.md")
    png = len(list((folder / "figures").glob("*.png")))
    svg = len(list((folder / "figures").glob("*.svg")))
    if png != manifest["PNG_count"] or svg != manifest["SVG_count"]:
        raise ValueError("FIGURE_COUNT_MISMATCH")
    return {"status": "PASS", "report": str(folder / "REPORT_KO.md"),
            "manifest_sha256": sha(folder / "MANIFEST.json"),
            "producer_sha256": producer["sha256"], "files_checked": len(manifest["files"]),
            "comparison_score_fields_checked": score_fields,
            "fixed_D_score_fields_checked": fixed_d_fields,
            "CSV_rows": row_counts, "local_links_checked": links,
            "PNG_count": png, "SVG_count": svg,
            "scope": "Artifact bytes, CSV primary scores, metadata bindings, and local links; visual layout and external HTTP are separate checks",
            "new_training_or_F_forward": False, "NPZ_or_model_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verification = verify(args.report)
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(verification, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: verification[key] for key in ("status", "files_checked", "comparison_score_fields_checked", "PNG_count", "SVG_count")}))
