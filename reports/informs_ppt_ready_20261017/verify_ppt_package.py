"""Verify existing PPT package sources, exact table values and missing-data boundaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_ref(entry, checked):
    if not entry:
        return
    path = (REPO / entry["path"]).resolve()
    if not path.is_relative_to(REPO) or sha(path) != entry["sha256"]:
        raise ValueError("Original source file changed: " + entry["path"])
    checked.add(entry["path"])


def exact_case(path):
    relative = path.relative_to(REPO)
    parent = REPO
    for part in relative.parts:
        if part not in {child.name for child in parent.iterdir()}:
            raise ValueError("Link filename case differs: " + str(relative))
        parent = parent / part


def execute(args):
    if sha(args.evidence) != args.evidence_sha or sha(args.scope) != args.scope_sha:
        raise ValueError("Private ROOT input SHA differs")
    evidence, scope = load(args.evidence), load(args.scope)
    if time.time() >= scope["package_deadline_unix"]:
        raise TimeoutError("Package boundary")
    required = ["PPT_RESULT_MAP_KO.md", "PPT_MAIN_TABLES.csv", "PPT_ABLATION_TABLES.csv", "PPT_PENDING_CELLS.csv",
                "PPT_CLAIM_EVIDENCE_MATRIX.csv", "PPT_SPEAKER_NOTES_KO.md", "SOURCE_BINDINGS.json",
                "QUALITATIVE_SELECTION.json", "FIGURE_BUILD_VALIDATION.json", "README_KO.md"]
    figures = ["JENA_B0_B2_TWO_SEEDS", "BOLT_VS_NO_BOLT_TWO_DATASETS", "MAIN_ACCURACY_TABLE",
               "ACCURACY_COST_TRADEOFF", "QUALITATIVE_FORECASTS"]
    required += ["figures/" + name + ".svg" for name in figures]
    if any(not (ROOT / name).is_file() for name in required):
        raise FileNotFoundError("Requested package artifact missing")
    rows = {row["row_id"]: row for row in evidence["rows"]}
    checked, source_csvs = set(), {}
    for row in rows.values():
        for key in ("score_source", "manifest_source", "training_receipt_source"):
            source_ref(row.get(key), checked)
        filename = row["score_source"]["path"]
        if filename not in source_csvs:
            source_csvs[filename] = csv_rows(REPO / filename)
        original = row["raw_csv_row"]
        if not any(all(candidate.get(k) == v for k, v in original.items()) for candidate in source_csvs[filename]):
            raise ValueError("Original CSV row no longer matches the bound evidence")
        for batch in (1, 8):
            cost = row.get("cost_b" + str(batch)) or {}
            for key in ("source", "independent_cost_arithmetic_source", "alias_independent_review_source"):
                source_ref(cost.get(key), checked)
    main, appendix = csv_rows(ROOT / "PPT_MAIN_TABLES.csv"), csv_rows(ROOT / "PPT_ABLATION_TABLES.csv")
    score_cells = cost_cells = pending_rows = 0
    for row in main + appendix:
        if not row["row_id"].startswith("legacy_"):
            if row["MSE"] != "NA" or row["MAE"] != "NA":
                raise ValueError("Unexecuted experiment was filled with a score")
            pending_rows += 1
            continue
        original = rows[row["row_id"]]
        if row["MSE"] != original["mse_text"] or row["MAE"] != original["mae_text"]:
            raise ValueError("Full precision score text changed")
        score_cells += 2
        for key in ("run_id", "scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "channel_sha256", "accuracy_device", "training_device"):
            expected = "NA" if original.get(key) is None or original.get(key) == "" else str(original[key])
            if row[key] != expected:
                raise ValueError("Table provenance differs: " + key)
        for batch in (1, 8):
            cost = original.get("cost_b" + str(batch)) or {}
            present = cost.get("seconds") is not None and cost.get("device") == "cpu" and cost.get("threads") == 2
            actual = row["B" + str(batch) + "_seconds"]
            if present:
                if float(actual) != cost["seconds"] or float(actual) <= 0:
                    raise ValueError("Measured cost changed")
                cost_cells += 1
            elif actual != "NA":
                raise ValueError("Missing cost was replaced by a number")
    pems = [r["method"] for r in main if r["table_id"] == "Overall_PeMS_C170"]
    if pems[-1] != "B2" or pems.index("PCA") > pems.index("B0"):
        raise ValueError("Overall method order differs")
    pending = csv_rows(ROOT / "PPT_PENDING_CELLS.csv")
    if not {"slide", "table", "row", "column", "required_experiment", "status", "reason"} <= set(pending[0]):
        raise ValueError("Pending-cell schema differs")
    if len({tuple(r[k] for k in ("slide", "table", "row", "column")) for r in pending}) != len(pending):
        raise ValueError("Duplicate pending cell")
    claims = csv_rows(ROOT / "PPT_CLAIM_EVIDENCE_MATRIX.csv")
    if not {"claim_id", "claim_text_ko", "status", "dataset", "seed", "split", "metric", "supporting_run_ids", "supporting_files", "main_or_appendix"} <= set(claims[0]):
        raise ValueError("Claim schema differs")
    if not all(r["status"] in {"CONFIRMED", "CONDITIONAL", "PENDING", "REJECTED"} for r in claims):
        raise ValueError("Unknown claim status")
    links = []
    output = Path(args.output).resolve()
    for path in ROOT.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).strip("<>")
            if target.startswith(("https://", "http://", "#")):
                continue
            dest = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            if dest != output:
                if not dest.is_file() or not dest.is_relative_to(REPO):
                    raise FileNotFoundError("Local Markdown link target: " + target)
                exact_case(dest)
            links.append(dict(file=path.relative_to(ROOT).as_posix(), target=target))
    svg_text = {}
    for name in figures:
        tree = ET.parse(ROOT / "figures" / (name + ".svg"))
        svg_text[name] = " ".join(tree.getroot().itertext())
    for idx in (4, 10, 11, 12):
        for metric in ("mse", "mae"):
            if f"{rows['legacy_'+str(idx)][metric]:.6f}" not in svg_text["JENA_B0_B2_TWO_SEEDS"]:
                raise ValueError("Displayed temporal score differs")
    for idx in (10, 7, 8, 20, 21):
        if f"{rows['legacy_'+str(idx)]['mse']:.6f}" not in svg_text["BOLT_VS_NO_BOLT_TWO_DATASETS"]:
            raise ValueError("Displayed no-Bolt score differs")
    qualitative = load(ROOT / "QUALITATIVE_SELECTION.json")
    if qualitative["scope_sha256"] != args.scope_sha or len(qualitative["existing_V_FP64_rescores"]) != 8:
        raise ValueError("Qualitative extraction proof differs")
    if any(r["max_score_delta"] > 1e-12 for r in qualitative["existing_V_FP64_rescores"]):
        raise ValueError("Existing V FP64 recheck failed")
    for data in qualitative["datasets"].values():
        if len(data["panels"]) != 9:
            raise ValueError("Three-by-three fixed panels required")
        for panel in data["panels"]:
            if str(panel["origin_id"]) not in svg_text["QUALITATIVE_FORECASTS"] or panel["channel_id"] not in svg_text["QUALITATIVE_FORECASTS"]:
                raise ValueError("Fixed qualitative panel missing")
    if "Normalized curves" not in svg_text["QUALITATIVE_FORECASTS"] or "C170 panels do not complete P1-1 C32" not in svg_text["QUALITATIVE_FORECASTS"]:
        raise ValueError("Required qualitative comparison boundary missing")
    inventory = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".csv", ".json", ".py", ".svg"}:
            raise ValueError("Unrequested binary or checkpoint in public package")
        text = path.read_text(encoding="utf-8-sig")
        private_keys = ("absolute" + "_private", "checkpoint_" + "path_declared")
        if re.search(r"[A-Za-z]:[\\/](?:Users[\\/]|CODING[\\/])", text) or any(value in text for value in private_keys):
            raise ValueError("Personal absolute path or private raw metadata in public package")
        inventory.append(dict(file=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    build = load(ROOT / "FIGURE_BUILD_VALIDATION.json")
    if sum(r["bytes"] for r in inventory) >= scope["maximum_new_storage_bytes"] or build["OS_peak_RSS_bytes"] >= scope["maximum_RSS_bytes"]:
        raise ValueError("Package resource cap failed")
    result = dict(schema="PPT_READY_PACKAGE_VALIDATION_v1", status="PASS_RESULT_PACKAGE_WITH_EXPLICIT_INCOMPLETE_EXPERIMENTS",
                  checked_unix=time.time(), base_commit=scope["base_commit"], input_evidence_sha256=args.evidence_sha,
                  scope_sha256=args.scope_sha, original_score_sources_verified=len(source_csvs), original_source_files_verified=len(checked),
                  main_rows=len(main), appendix_rows=len(appendix), original_score_cells_exact=score_cells,
                  measured_cost_cells_exact=cost_cells, pending_result_rows=pending_rows, pending_cells=len(pending),
                  claims=len(claims), local_links_case_checked=len(links), svg_files=5, fixed_qualitative_panels=18,
                  existing_V_FP64_rescores=8, model_training=0, new_model_inference=0, new_cost_measurement=0,
                  protected_T_numeric_access=0, P1_1="PENDING_INSUFFICIENT_LEARNING", P1_2="NOT_ENTERED_GATE_UNMET",
                  P1_3="COMPLETE_SAVED_V_FIGURES", P1_4="EXISTING_MEASUREMENTS_RECONCILED_UNMEASURED_NA",
                  Priority2="NOT_ENTERED", MASTER="paused/incomplete", scientific_experiment_set_complete=False,
                  publication="LOCAL_ONLY_NO_COMMIT_PUSH_THIS_REQUEST", original_experiment_files_overwritten=False,
                  PPT_PDF_created=False, renderer_OS_peak_RSS_bytes=build["OS_peak_RSS_bytes"], files_excluding_this_receipt=inventory)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files_excluding_this_receipt"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for key in ("evidence", "evidence-sha", "scope", "scope-sha", "output"):
        parser.add_argument("--" + key)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_NO_INPUT_READS", writes=0))); return
    if any(v is None for k, v in vars(args).items() if k != "execute"):
        parser.error("Pinned existing evidence, scope and fresh output required")
    execute(args)


if __name__ == "__main__":
    main()
