"""Read completed original-auditor results; make an explicit safe text allowlist."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--public-root", required=True)
parser.add_argument("--actual-exitcode", type=int, required=True)
args = parser.parse_args()
public, here = Path(args.public_root).resolve(), Path(__file__).resolve().parent
result = read(here / "RESULT.json")
audit = public / result["original_audit_public_relative_path"]
report = read(audit / "AUDIT_REPORT.json")
records = read(audit / "PREDICTION_AUDITS.json")
assert args.actual_exitcode == 0
assert result["status"] == "PASS_ORIGINAL_AUDITOR_EXACT_DISCOVERY_SCOPE"
assert len(records) == report["prediction_rescoring_passed"] == 453
assert report["public_stats_reaggregation_passed"] == 405
assert report["dense_private_rescoring_passed"] == 48
assert report["global_scores_checked"] is False
assert report["errors"] == []
assert report["source_sha256"] == read(here / "SOURCE_FREEZE.json")["source39"]["audit_campaign.py"]
assert result["source39_input_and_registry_science_preserved"]
assert result["registry_before_byte_sha256"] == result["registry_after_byte_sha256"]
assert len({(r["run_id"], r["split"], r["manifest_public_relative_path"]) for r in records}) == 453
new(here / "EXECUTION_RECEIPT.json", dict(
    actual_wrapper_exitcode=args.actual_exitcode,
    actual_FP64_saved_array_rescores=453,
    split_counts=dict(Counter(r["split"] for r in records)),
    public_stats_reaggregations=405, original_dense_M_private_rescores=48,
    audit_errors=0, global_scores_checked=False,
    original_report_status=report["status"],
    original_audit_public_relative_path=audit.relative_to(public).as_posix(),
    original_report_sha256=sha(audit / "AUDIT_REPORT.json"),
    original_records_sha256=sha(audit / "PREDICTION_AUDITS.json"),
    exact_four_posthoc_M_remain_separate=True,
    original_auditor_source_mutated=False,
    source39_and_inputs_unchanged=True, registry_bytes_unchanged=True,
    wrapper_sha256=sha(here / "discovery_wrapper.py"),
    checkpoint_replay=0, training_rerun=0, new_model_inference=0,
    GPU_acquisition=0, T_access=0,
    audit_pass_is_not_convergence_campaign_completion_or_generalization=True,
))
files = sorted(path for path in here.iterdir() if path.is_file())
files += sorted(path for path in audit.iterdir() if path.is_file())
patterns = [re.compile(r"[A-Za-z]:[\\/]"), re.compile(r"/(?:home|Users)/[^/\s\"]+")]
findings = []
for path in files:
    text = path.read_text(encoding="utf-8-sig")
    for pattern in patterns:
        if pattern.search(text):
            findings.append(dict(path=path.relative_to(public).as_posix(), kind="absolute_personal_or_machine_path"))
assert not findings, findings
allowlist = here / "PUBLICATION_ALLOWLIST.txt"
relative = [path.relative_to(public).as_posix() for path in files]
relative += [(here / "PUBLICATION_SCAN.json").relative_to(public).as_posix(), allowlist.relative_to(public).as_posix()]
with allowlist.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write("\n".join(sorted(relative)) + "\n")
new(here / "PUBLICATION_SCAN.json", dict(
    scanned_text_files=len(files), exact_allowlist_files=len(relative), findings=findings,
    file_sha256={path.relative_to(public).as_posix(): sha(path) for path in files},
    allowlist_sha256=sha(allowlist),
    scan_self_contains_no_absolute_root=True,
    preserves_originals=True, safe_derivatives_created=0,
))
print(json.dumps(dict(status="PASS_COMPLETED_453_AND_SAFE_ALLOWLIST", actual_exitcode=args.actual_exitcode,
    rescores=453, V=405, original_dense_M=48, scan_findings=0, allowlist_files=len(relative))))
