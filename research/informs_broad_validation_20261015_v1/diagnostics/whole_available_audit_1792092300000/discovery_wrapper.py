"""Existing whole auditor with exactly four SHA-bound posthoc-M contracts split from discovery.

Default is metadata only. --execute runs audit_campaign.audit_all without changing its
scoring code. --expect-global-scores is only for a later post-T stored-result audit.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PREFIX = "diagnostics/raw_deep_g_saved_rescore_1792090799000/DERIVED_M_CONTRACTS/"
EXCLUDED = {
    PREFIX+"primary_beijing_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_6a1e0351808b/PREDICTION_MANIFEST.json": "1032877be1ac002fbb76e33f5f0bac0ef4779263239d5476b15f697ae4b80687",
    PREFIX+"primary_jena_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_7943f4e9d13b/PREDICTION_MANIFEST.json": "73b5a258ab2c7cec0936ca805dfd3485f81d2a31ddd35f8f0016b96e31f7f311",
    PREFIX+"primary_electricity_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_2a8965c95f86/PREDICTION_MANIFEST.json": "f25213d6c70daee42f4dabab03cee2c5766df9090583d3fc163dc4574a6b060f",
    PREFIX+"primary_pems08_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_6d69e6bb44c5/PREDICTION_MANIFEST.json": "0d184d8bef3e570df6f74efbc3bb52f581c7daf24978ba6bb6fe5ec535f1c447",
}
UNION = "diagnostics/raw_deep_g_saved_finalize_1792091230000/UNION_BUNDLES.csv"
UNION_SHA = "12b46d151739edb970862763a222387d7056e758787dabefc264b4364d3770f0"
RAW_FINAL = "diagnostics/raw_deep_g_saved_finalize_1792091230000/RESULT.json"
RAW_FINAL_SHA = "fec1ceb551799e5c25f5e7d0f6b95e710341ca1e3a0c4a0db9beed3fd1dd3ec4"
MUTABLE = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write("\n")


def fingerprint(value):return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def registry(public):
    path = public/"EXPERIMENT_REGISTRY.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:rows=list(csv.DictReader(stream))
    identities={row["experiment_id"]:row for row in rows}
    if len(identities)!=len(rows):raise ValueError("Duplicate registry IDs")
    scientific=[{key:value for key,value in row.items() if key not in MUTABLE} for row in rows]
    return dict(byte_sha256=sha(path),scientific_sha256=fingerprint(sorted(scientific,key=lambda row:row["experiment_id"])),rows=len(rows))


def metadata(public, expect_global_scores=False):
    public=Path(public).resolve()
    if sha(public/UNION)!=UNION_SHA or sha(public/RAW_FINAL)!=RAW_FINAL_SHA:raise ValueError("Pinned Raw8 proof/union changed")
    raw=read(public/RAW_FINAL)
    if raw["actual_new_rescores"]!=8 or raw["original_numeric_process_exitcode"]!=1 or raw["new_M_manifest_is_audit_only_derived_from_original_parent_truth"] is not True:raise ValueError("Posthoc-M scientific scope differs")
    sources={name:sha(public/name) for name in raw["source39_after"]}
    if sources!=raw["source39_after"] or len(sources)!=39:raise ValueError("Frozen source39 changed")
    with (public/UNION).open(encoding="utf-8",newline="") as stream:known=list(csv.DictReader(stream))
    if len(known)!=457:raise ValueError("Original457 bundle inventory differs")
    original={row["manifest_public_relative_path"]:row["manifest_sha256"] for row in known if row["manifest_public_relative_path"] not in EXCLUDED}
    if len(original)!=453:raise ValueError("Original canonical453 differs")
    all_paths={path.relative_to(public).as_posix():path for path in public.rglob("PREDICTION_MANIFEST.json")}
    if not (set(original)|set(EXCLUDED))<=set(all_paths):raise ValueError("An original or excluded manifest is missing")
    for name,digest in {**original,**EXCLUDED}.items():
        if sha(all_paths[name])!=digest:raise ValueError("Pinned manifest bytes changed: "+name)
    for name in EXCLUDED:
        m=read(all_paths[name])
        if m["split"]!="M_cal" or m.get("metadata",{}).get("audit_only_derived_manifest") is not True:raise ValueError("Exact exclusion is not an audit-only posthoc M contract")
    additions=set(all_paths)-set(original)-set(EXCLUDED)
    scores=public/"SCORES.csv";selection=public/"SELECTION_LOCK.json"
    if not expect_global_scores:
        if scores.exists() or selection.exists() or additions:raise ValueError("Pre-final scope requires no global scores, no selection, no added manifests")
    else:
        if not scores.is_file() or not selection.is_file():raise ValueError("Post-final binding requires actual global scores and selection lock")
        lock=read(selection)
        if lock.get("locked") is not True:raise ValueError("Post-final selection is not locked")
        for name in additions:
            if "/T_final/" not in name:raise ValueError("Only actual new final-T manifests may extend this frozen inventory")
            m=read(all_paths[name])
            if m["split"]!="T_final" or m["run_id"] not in lock["allowed_run_ids"]:raise ValueError("Added final manifest is outside locked selection")
    canonical={name:sha(path) for name,path in all_paths.items() if name not in EXCLUDED}
    splits={}
    for name in canonical:
        split=read(all_paths[name])["split"];splits[split]=splits.get(split,0)+1
    return dict(status="METADATA_ONLY_NO_FP64_EXECUTION",all_manifest_count=len(all_paths),exact_posthoc_M_exclusions=EXCLUDED,canonical_manifest_count=len(canonical),canonical_split_counts=splits,
        original_canonical453_preserved=True,additional_final_manifest_paths=sorted(additions),canonical_manifest_sha256=canonical,source39=sources,registry=registry(public),
        global_scores_expected=expect_global_scores,global_scores_sha256=sha(scores) if scores.is_file() else None,selection_sha256=sha(selection) if selection.is_file() else None,
        original_scoring_source_sha256=sources["audit_campaign.py"],wrapper_sha256=sha(__file__),original_auditor_mutated=False,NPZ_weights_or_gzip_deserialized=False,FP64_executed=False)


@contextmanager
def discovery(public, calls):
    public=Path(public).resolve();original=Path.rglob
    def exact(path,pattern):
        if path.resolve()!=public or pattern!="PREDICTION_MANIFEST.json":
            yield from original(path,pattern);return
        call=dict(root_scope="exact_public_root_only",pattern=pattern,excluded=[],yielded=[]);calls.append(call)
        for candidate in original(path,pattern):
            relative=candidate.relative_to(public).as_posix()
            if relative in EXCLUDED:
                if sha(candidate)!=EXCLUDED[relative]:raise ValueError("Excluded bytes changed during discovery")
                call["excluded"].append(relative)
            else:
                call["yielded"].append(relative);yield candidate
        if set(call["excluded"])!=set(EXCLUDED):raise ValueError("Exact four excluded contracts were not all discovered")
    Path.rglob=exact
    try:yield
    finally:Path.rglob=original


def input_hashes(public,private,info):
    paths={}
    def add(path,root,scope):
        path=Path(path).resolve();root=Path(root).resolve()
        if not path.is_relative_to(root):raise ValueError("Input escaped supplied root")
        paths[scope+":"+path.relative_to(root).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    for name in info["canonical_manifest_sha256"]:
        path=public/name;add(path,public,"public")
        reference=path.with_name("REFERENCE_CONTRACT.json");add(reference,public,"public")
        m,ref=read(path),read(reference)
        add(private/m["cache_relative_path"],private,"private")
        if m.get("public_stats_relative_path"):add(public/m["public_stats_relative_path"],public,"public")
        if ref.get("pre_target_contract_public_relative_path"):add(public/ref["pre_target_contract_public_relative_path"],public,"public")
        artifact=ref.get("realized_model_binding",{}).get("selected_artifact_cache_relative_path")
        if artifact:add(private/artifact,private,"private")
    for name in ("EXPERIMENT_REGISTRY_INITIAL.csv","DATA_CONTRACT.json","CONFIG_LOCK.json"):
        add(public/name,public,"public")
    for path in public.glob("EXPERIMENT_REGISTRY_EXTENSION_*.csv"):
        add(path,public,"public");add(path.with_suffix(".json"),public,"public")
    for name in EXCLUDED:add(public/name,public,"public")
    if info["global_scores_expected"]:
        add(public/"SCORES.csv",public,"public");add(public/"SELECTION_LOCK.json",public,"public")
    return paths


def execute(public,private,output,expect_global_scores):
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONDONTWRITEBYTECODE="1")
    if any(key.startswith("INFORMS_GPU_") for key in os.environ):raise ValueError("No borrowed GPU owner environment allowed")
    output=Path(output).resolve()
    if not output.is_relative_to(public/"diagnostics"):raise ValueError("New wrapper output must be under diagnostics")
    output.mkdir(parents=True,exist_ok=True)
    new(output/"RUN_CLAIM.json",dict(started_unix=time.time(),wrapper_sha256=sha(__file__),actual_GPU_training_T_inference_requested=False))
    before=metadata(public,expect_global_scores)
    immutable=input_hashes(public,private,before)
    new(output/"BEFORE.json",dict(metadata=before,immutable_inputs=immutable))
    existing={path.name for path in (public/"audits").glob("audit_*") if path.is_dir()}
    sys.path.insert(0,str(public))
    import audit_campaign
    calls=[];started=time.time()
    with discovery(public,calls):report=audit_campaign.audit_all(public,private)
    after=metadata(public,expect_global_scores)
    immutable_after=input_hashes(public,private,after)
    new(output/"AFTER.json",dict(metadata=after,immutable_inputs=immutable_after))
    auditdir=public/"audits"/report["audit_batch"]
    expected=set(before["canonical_manifest_sha256"])
    preserved=before["source39"]==after["source39"] and before["canonical_manifest_sha256"]==after["canonical_manifest_sha256"] and immutable==immutable_after and before["registry"]["scientific_sha256"]==after["registry"]["scientific_sha256"]
    dispatch_exact=len(calls)==1 and set(calls[0]["yielded"])==expected and set(calls[0]["excluded"])==set(EXCLUDED)
    fresh=auditdir.name not in existing and auditdir.is_dir()
    completed=preserved and dispatch_exact and fresh and report["global_scores_checked"]==expect_global_scores and not report["errors"] and report["manifests_found"]==len(expected)
    new(output/"RESULT.json",dict(status="PASS_ORIGINAL_AUDITOR_EXACT_DISCOVERY_SCOPE" if completed else "FAILED_OR_PARTIAL_AUDIT",original_audit_report_status=report["status"],
        original_audit_public_relative_path=auditdir.relative_to(public).as_posix(),original_audit_report_sha256=sha(auditdir/"AUDIT_REPORT.json"),original_prediction_audits_sha256=sha(auditdir/"PREDICTION_AUDITS.json"),
        manifests_actually_audited=report["manifests_found"],actual_prediction_rescores=report["prediction_rescoring_passed"],actual_public_stats_reaggregations=report["public_stats_reaggregation_passed"],actual_dense_private_rescores=report["dense_private_rescoring_passed"],
        error_count=len(report["errors"]),source39_input_and_registry_science_preserved=preserved,registry_before_byte_sha256=before["registry"]["byte_sha256"],registry_after_byte_sha256=after["registry"]["byte_sha256"],
        exact_discovery_calls=calls,discovery_scope_verified=dispatch_exact,new_actual_auditor_output_directory_verified=fresh,global_scores_checked=report["global_scores_checked"],canonical_count=len(expected),posthoc4_kept_in_separate_independent_audit=True,
        mandatory_coverage_complete=report["mandatory_coverage_complete"],scientific_or_campaign_success_claimed=False,checkpoint_replay=0,training_rerun=0,new_model_inference=0,GPU_acquired=False,
        immutable_input_count=len(immutable),wrapper_sha256=sha(__file__),existing_auditor_source_unchanged=True,wall_seconds=time.time()-started,finished_unix=time.time()))
    print(json.dumps(dict(status="PASS_ORIGINAL_AUDITOR_EXACT_DISCOVERY_SCOPE" if completed else "FAILED_OR_PARTIAL_AUDIT",original_report_status=report["status"],actual_rescores=report["prediction_rescoring_passed"],errors=len(report["errors"]),original_audit_relative_path=auditdir.relative_to(public).as_posix())),flush=True)
    if not completed:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root",required=True);parser.add_argument("--private-root",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--expect-global-scores",action="store_true")
    args=parser.parse_args();public,private=Path(args.public_root).resolve(),Path(args.private_root).resolve()
    if args.execute:execute(public,private,args.output,args.expect_global_scores)
    else:
        output=Path(args.output).resolve()
        if not output.is_relative_to(public/"diagnostics"):raise ValueError("Proof output outside diagnostics")
        output.mkdir(parents=True,exist_ok=True);info=metadata(public,args.expect_global_scores)
        new(output/"METADATA_PROOF.json",info)
        new(output/"SOURCE_FREEZE.json",dict(wrapper_sha256=sha(__file__),source39=info["source39"],metadata_proof_sha256=sha(output/"METADATA_PROOF.json"),exact_four_exclusion_hashes=EXCLUDED,FP64_GPU_T_executed=False))
        print(json.dumps(dict(status=info["status"],all_manifest_count=info["all_manifest_count"],canonical_count=info["canonical_manifest_count"],split_counts=info["canonical_split_counts"],exact_exclusions=4,FP64_executed=False)),flush=True)


if __name__=="__main__":main()
