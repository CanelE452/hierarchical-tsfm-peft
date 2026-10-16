"""원 감사기의 정확한 사후 M4 발견 분리. 기본은 준비 metadata, 실제 감사는 --execute."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import sys
import time

BASE = "diagnostics/whole_available_audit_1792092300000/"
BASE_HELPER_SHA = "a7c92b52077057307b2526927b35c4a7ac8ada3a235f0ec462797ffdb17413f6"
BASE_PROOF_SHA = "6925a01799f170e3cb557547f59f0234aab689a4adc1bb98ff80a4fb606d0991"
BASE_RESULT_SHA = "345352d33f3c460a64fa377ebc7d75bde68a144564d8558179a550e399c3dec1"
E0 = "diagnostics/e0_b_execution_ready_20261016T012806/"
E0_PINS_SHA = "0a4bdd3870fc938b79e0ab5895620d273a921bd0f88e14d646c9668825d33f18"
E0_INVENTORY_SHA = "e47e179972d5758f49319c6882005f76e8226b333a6ccade13feb5de40ebc5a5"
LEGACY_SPLIT = "legacy_B_development"
FINAL_SOURCES = ("final_evaluation.py", "route_predictions.py", "route_catalog.py", "deployment.py",
    "data_contract.py", "evaluation.py", "training.py", "networks.py", "gpu_runtime.py",
    "ridge_baseline.py", "combination.py", "raw_calibration.py", "fit_base.py", "fit_control.py",
    "prediction_cache.py", "ordered_pilot.py", "audit_predictions.py", "audit_campaign.py")


def load_base(public):
    import hashlib
    path = public / BASE / "discovery_wrapper.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_HELPER_SHA:
        raise ValueError("기존 wrapper 봉인이 변경됨")
    spec = importlib.util.spec_from_file_location("original_exact4_discovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(public, base):
    for name, expected in ((BASE+"METADATA_PROOF.json", BASE_PROOF_SHA),
            (BASE+"RESULT.json", BASE_RESULT_SHA), (E0+"INPUT_PINS.json", E0_PINS_SHA),
            (E0+"PINNED_INVENTORY.json", E0_INVENTORY_SHA)):
        if base.sha(public/name) != expected:
            raise ValueError("준비 근거의 봉인 변경: "+name)
    baseline = base.read(public/BASE/"METADATA_PROOF.json")
    result = base.read(public/BASE/"RESULT.json")
    if result["actual_prediction_rescores"] != 453 or result["error_count"] != 0:
        raise ValueError("기존 원본 453 감사 근거 불일치")
    sources = {name:base.sha(public/name) for name in baseline["source39"]}
    if len(sources) != 39 or sources != baseline["source39"]:
        raise ValueError("원 소스 39개 변경")
    if base.registry(public)["scientific_sha256"] != baseline["registry"]["scientific_sha256"]:
        raise ValueError("registry 과학 조건 변경")
    pins, inventory = base.read(public/E0/"INPUT_PINS.json"), base.read(public/E0/"PINNED_INVENTORY.json")
    if pins["legacy_split"] != LEGACY_SPLIT or pins["checkpoint_count"] != 55:
        raise ValueError("봉인 historical B 범위 변경")
    eligible = {}
    for route in inventory["routes"]:
        if route["status"] != "READY" or route["id"] not in pins["legacy_ids"]:
            raise ValueError("봉인 historical B route 불일치")
        for checkpoint in route["checkpoints"]:
            key = (route["id"], checkpoint["update"])
            if checkpoint["status"] != "READY" or key in eligible:
                raise ValueError("봉인 checkpoint 중복/미준비")
            eligible[key] = dict(route=route, checkpoint=checkpoint)
    if len(eligible) != 55 or set(r["id"] for r in inventory["routes"]) != set(pins["legacy_ids"]):
        raise ValueError("봉인 9ID/55CP 불일치")
    return baseline, sources, pins, inventory, eligible


def validate_lock(lock, sources, registry_scientific_sha, base):
    if lock.get("locked") is not True or lock.get("selection_sha256") != base.fingerprint({k:v for k,v in lock.items() if k != "selection_sha256"}):
        raise ValueError("실제 selection lock 봉인 불일치")
    ids = lock["allowed_run_ids"]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) or x in {".", ".."} for x in ids):
        raise ValueError("선택 ID 누락/중복/경로 이탈")
    if lock["source_bindings"] != {name:sources[name] for name in FINAL_SOURCES} or lock["registry_scientific_sha"] != registry_scientific_sha:
        raise ValueError("선택 source/registry 과학 조건 불일치")
    for identity in ids:
        if lock["allowed_route_sha"][identity] != base.fingerprint(lock["route_bindings"][identity]["route_spec"]):
            raise ValueError("선택 route 봉인 불일치")
    return set(ids)


def classify_added(name, manifest, lock, eligible, inventory, sources, base):
    parts = Path(name).parts
    if len(parts) == 4 and parts[0] == "route_predictions" and parts[2:] == ("T_final", "PREDICTION_MANIFEST.json"):
        identity = parts[1]
        if identity not in lock["allowed_run_ids"] or manifest.get("run_id") != identity or manifest.get("split") != "T_final":
            raise ValueError("추가 T가 locked ID/공식 경로와 다름")
        binding = lock["route_bindings"][identity]
        route_sha = lock["allowed_route_sha"][identity]
        meta = manifest.get("metadata", {})
        if (manifest.get("model_sha") != route_sha or meta.get("route_sha") != route_sha
                or meta.get("route_spec") != binding["route_spec"] or meta.get("T_gate_checked") is not True
                or manifest.get("data_sha") != binding["data_sha"] or manifest.get("scaler_sha") != binding["scaler_sha"]):
            raise ValueError("추가 T의 실제 route/data/scaler가 선택과 다름")
        return "T_final"
    if len(parts) != 5 or parts[:2] != ("legacy_e0", "evaluation") or parts[-1] != "PREDICTION_MANIFEST.json" or re.fullmatch(r"cp\d+", parts[3]) is None:
        raise ValueError("허용되지 않은 추가 manifest 경로")
    key = (parts[2], int(parts[3][2:]))
    if key not in eligible or parts[3] != "cp"+str(key[1]):
        raise ValueError("봉인 inventory 밖의 legacy ID/CP")
    item = eligible[key]; route, cp = item["route"], item["checkpoint"]
    identity = manifest.get("identity", {})
    expected = dict(legacy_id=key[0], checkpoint_update=key[1], spec=route["spec"],
        model_file_sha256=cp["file_sha256"], model_state_sha256=cp["state_sha256"],
        checkpoint_relative_path=cp["checkpoint_relative_path"], model_path_scope=cp["scope"],
        source_hashes=inventory["source_hashes"], evaluation_source_sha256=sources["legacy_evaluation.py"],
        continuation_source_sha256=sources["legacy_continuation.py"], statistics_sha256=inventory["statistics_sha256"],
        foundation_sha256=inventory["foundation_sha256"], legacy_receipt_sha256=route["legacy_receipt_sha256"],
        curve_evidence_status=cp["curve_evidence_status"], recorded_curve_sha256=base.fingerprint(cp["expected_B"]),
        length=512, horizon=48, fold="primary")
    if (manifest.get("run_id") is not None or manifest.get("split") != LEGACY_SPLIT
            or manifest.get("historically_unseen") is not False or manifest.get("protected_TEST_accessed") is not False
            or manifest.get("C_scored") is not False or manifest.get("private_archive_name") != "predictions.npz"
            or any(identity.get(k) != v for k,v in expected.items()) or manifest.get("expected_B") != cp["expected_B"]):
        raise ValueError("legacy 공식 schema/봉인 ID·CP·모델·소스가 다름")
    if set(manifest.get("arrays", {})) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
        raise ValueError("legacy 공식 배열 schema가 다름")
    return LEGACY_SPLIT


def final_metadata(public, base):
    baseline, sources, pins, inventory, eligible = prepare(public, base)
    lock = base.read(public/"SELECTION_LOCK.json")
    selected = validate_lock(lock, sources, base.registry(public)["scientific_sha256"], base)
    if not (public/"SCORES.csv").is_file():
        raise ValueError("최종 전역 SCORES 실존 필요")
    paths = {p.relative_to(public).as_posix():p for p in public.rglob("PREDICTION_MANIFEST.json")}
    original = baseline["canonical_manifest_sha256"]
    for name, expected in {**original, **base.EXCLUDED}.items():
        if name not in paths or base.sha(paths[name]) != expected:
            raise ValueError("원 453 또는 정확 제외 M4 봉인 변경: "+name)
    additions = set(paths)-set(original)-set(base.EXCLUDED)
    kinds = {name:classify_added(name, base.read(paths[name]), lock, eligible, inventory, sources, base) for name in additions}
    canonical = {name:base.sha(path) for name,path in paths.items() if name not in base.EXCLUDED}
    t_ids = {Path(name).parts[1] for name,kind in kinds.items() if kind == "T_final"}
    b_keys = {(Path(name).parts[2], int(Path(name).parts[3][2:])) for name,kind in kinds.items() if kind == LEGACY_SPLIT}
    return dict(canonical_manifest_sha256=canonical, global_scores_expected=True, source39=sources,
        registry=base.registry(public), added_manifest_kinds=kinds, actual_T_count=len(t_ids), actual_legacy_B_count=len(b_keys),
        selected_T_missing=sorted(selected-t_ids), sealed_legacy_B_missing=[list(x) for x in sorted(set(eligible)-b_keys)],
        selection_sha256=base.sha(public/"SELECTION_LOCK.json"), global_scores_sha256=base.sha(public/"SCORES.csv")), lock, pins


def inputs(public, private, asset, info, lock, pins, base):
    standard = dict(info, canonical_manifest_sha256={name:digest for name,digest in info["canonical_manifest_sha256"].items()
        if info["added_manifest_kinds"].get(name) != LEGACY_SPLIT})
    result = base.input_hashes(public, private, standard)
    def add(root, relative, scope, expected=None):
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError("입력 경로가 공급한 root 밖임")
        digest=base.sha(path)
        if expected is not None and digest != expected:raise ValueError("봉인 입력 bytes 변경: "+relative)
        result[scope+":"+path.relative_to(root).as_posix()]=dict(sha256=digest,bytes=path.stat().st_size)
    for name in (E0+"INPUT_PINS.json", E0+"PINNED_INVENTORY.json", BASE+"METADATA_PROOF.json", BASE+"RESULT.json", BASE+"discovery_wrapper.py"):
        add(public,name,"public")
    for item in pins["asset_metadata"]:add(asset,item["relative_path"],"asset",item["sha256"])
    for item in pins["public_bindings"]:add(public,item["relative_path"],"public",item["sha256"])
    for item in pins["checkpoints"]:
        root,scope=(asset,"asset") if item["scope"]=="asset" else (private,"private")
        add(root,item["checkpoint_relative_path"],scope,item["file_sha256"])
    for name, kind in info["added_manifest_kinds"].items():
        if kind != LEGACY_SPLIT:continue
        manifest=base.read(public/name);folder=Path(name).parent
        for filename in ("REFERENCE_CONTRACT.json", "PUBLIC_ERROR_STATS.csv.gz", "INDEPENDENT_RESCORE.json"):
            add(public,(folder/filename).as_posix(),"public",manifest["public_stats_sha256"] if filename == "PUBLIC_ERROR_STATS.csv.gz" else None)
        add(private,(folder/"predictions.npz").as_posix(),"private",manifest["archive_sha256"])
    for binding in lock["route_bindings"].values():
        for item in binding["artifact_bindings"].values():add(private,item["cache_relative_path"],"private",item["sha256"])
    for prefix in ("report", "records"):
        item=lock["independent_audit"];add(public,item[prefix+"_public_relative_path"],"public",item[prefix+"_sha256"])
    return result


def expected_counts(t, b):
    return dict(manifests_found=453+t+b,prediction_rescoring_passed=453+t,
        public_stats_reaggregation_passed=405+t,dense_private_rescoring_passed=48,legacy_B_rescoring_passed=b)


def execute(public, private, asset, output, base):
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if any(key.startswith("INFORMS_GPU_") for key in os.environ):raise ValueError("빌린 GPU 소유자 환경을 허용하지 않음")
    if not output.is_relative_to(public/"diagnostics"):raise ValueError("새 출력은 공개 diagnostics 안이어야 함")
    output.mkdir(parents=True,exist_ok=True)
    base.new(output/"RUN_CLAIM.json",dict(started_unix=time.time(),wrapper_v2_sha256=base.sha(__file__),new_model_or_T_inference=False))
    before,lock,pins=final_metadata(public,base)
    immutable=inputs(public,private,asset,before,lock,pins,base)
    base.new(output/"BEFORE.json",dict(metadata=before,immutable_inputs=immutable))
    existing={p.name for p in (public/"audits").glob("audit_*") if p.is_dir()}
    sys.path.insert(0,str(public));import audit_campaign
    calls=[];started=time.time()
    with base.discovery(public,calls):report=audit_campaign.audit_all(public,private)
    after,lock_after,pins_after=final_metadata(public,base)
    immutable_after=inputs(public,private,asset,after,lock_after,pins_after,base)
    base.new(output/"AFTER.json",dict(metadata=after,immutable_inputs=immutable_after))
    t,b=before["actual_T_count"],before["actual_legacy_B_count"]
    expected=expected_counts(t,b)
    numeric=not report["errors"] and report["global_scores_checked"] is True and all(report[k]==v for k,v in expected.items())
    preserved=(before==after and immutable==immutable_after and prepare(public,base)[1]==before["source39"])
    dispatch=(len(calls)==1 and set(calls[0]["yielded"])==set(before["canonical_manifest_sha256"]) and set(calls[0]["excluded"])==set(base.EXCLUDED))
    auditdir=public/"audits"/report["audit_batch"]
    passed=numeric and preserved and dispatch and auditdir.name not in existing and auditdir.is_dir()
    base.new(output/"RESULT.json",dict(status="PASS_AVAILABLE_POST_FINAL_ORIGINAL_AUDITOR" if passed else "FAILED_OR_PARTIAL_AUDIT",
        original_audit_public_relative_path=auditdir.relative_to(public).as_posix(),original_audit_report_sha256=base.sha(auditdir/"AUDIT_REPORT.json"),
        original_prediction_audits_sha256=base.sha(auditdir/"PREDICTION_AUDITS.json"),original_legacy_B_audits_sha256=base.sha(auditdir/"LEGACY_B_AUDITS.json"),
        original_report_status=report["status"],actual_counts={key:report[key] for key in expected},expected_count_relationships=expected,
        error_count=len(report["errors"]),global_scores_checked=report["global_scores_checked"],source39_inputs_registry_bytes_preserved=preserved,
        immutable_input_count=len(immutable),discovery_scope_verified=dispatch,selected_T_missing=before["selected_T_missing"],sealed_legacy_B_missing=before["sealed_legacy_B_missing"],
        selected_T_and_E0_inventory_coverage_complete=not before["selected_T_missing"] and not before["sealed_legacy_B_missing"],
        mandatory_coverage_complete=report["mandatory_coverage_complete"],scientific_or_campaign_completion_claimed=False,
        historical_B_is_new_V_T=False,posthoc_M4_separately_preserved=True,checkpoint_replay=0,training=0,new_model_inference=0,GPU_acquired=False,
        protected_TEST_accessed=False,stored_T_arrays_rescored=t,source39=before["source39"],base_wrapper_sha256=BASE_HELPER_SHA,wrapper_v2_sha256=base.sha(__file__),wall_seconds=time.time()-started))
    print(base.fingerprint(expected), "PASS_AVAILABLE_POST_FINAL_ORIGINAL_AUDITOR" if passed else "FAILED_OR_PARTIAL_AUDIT", flush=True)
    if not passed:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root",default=os.environ.get("INFORMS_PUBLIC_ROOT"))
    parser.add_argument("--private-root",default=os.environ.get("INFORMS_CACHE_ROOT"))
    parser.add_argument("--asset-root",default=os.environ.get("INFORMS_ASSET_ROOT"))
    parser.add_argument("--output",required=True);parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.public_root:parser.error("--public-root 또는 INFORMS_PUBLIC_ROOT 필요")
    public,output=Path(args.public_root).resolve(),Path(args.output).resolve();base=load_base(public)
    if args.execute:
        if not args.private_root or not args.asset_root:parser.error("실행에는 private/asset root 필요")
        execute(public,Path(args.private_root).resolve(),Path(args.asset_root).resolve(),output,base)
    else:
        baseline,sources,pins,inventory,eligible=prepare(public,base)
        if not output.is_relative_to(public/"diagnostics"):raise ValueError("준비 출력 root 이탈")
        output.mkdir(parents=True,exist_ok=True)
        base.new(output/"PREPARATION.json",dict(status="PREPARATION_ONLY_NO_FINAL_EXECUTION",source39=sources,base_wrapper_sha256=BASE_HELPER_SHA,
            wrapper_v2_sha256=base.sha(__file__),original_453_manifest_inventory_sha256=BASE_PROOF_SHA,sealed_E0_inventory_sha256=E0_INVENTORY_SHA,
            original_standard_count=453,original_V_count=405,original_dense_M_count=48,exact_posthoc_M_exclusions=base.EXCLUDED,
            eligible_legacy_B_count=len(eligible),eligible_legacy_ids=pins["legacy_ids"],actual_T_or_B_outputs_inspected=False,
            existing_453_manifest_files_read=False,NPZ_weights_gzip_deserialized=False,FP64_executed=False,GPU_T_execution=False,
            numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules))
        print("PREPARATION_ONLY; 원453 재독0; NPZ/weights/GPU/T/실제채점0",flush=True)


if __name__ == "__main__":main()
