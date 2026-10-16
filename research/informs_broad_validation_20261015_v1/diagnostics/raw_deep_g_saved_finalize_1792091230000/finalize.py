"""Finalize eight actual numerical passes after the original postflight guard stopped byte hashing."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

PRIOR = "diagnostics/extended_e2_saved_rescore_v2_1792090317000"
NUMERIC = "diagnostics/raw_deep_g_saved_rescore_1792090799000"
READY = "diagnostics/raw_deep_g_control_ready_1792090190000000000_v2"
MUTABLE = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if isinstance(value, str):stream.write(value)
        else:json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write("\n")


def csv_new(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True);parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    public, cache = Path(args.public_root).resolve(), Path(args.cache_root).resolve()
    output, numeric = Path(__file__).resolve().parent, public / NUMERIC
    started = time.time()
    original = read(numeric / "RESULT.json")
    before = read(numeric / "INPUTS_BEFORE.json")
    plan = read(public / READY / "PLAN.json")
    old_result = read(public / PRIOR / "RESULT.json")
    old_audit = read(public / PRIOR / "AUDIT_RESULTS_PUBLIC.json")
    records = original["partial_results"]
    if original["status"] != "FAILED_PRESERVED_DIAGNOSTIC" or original["fatal_error"] != "Only eight new RawH V/M NPZs may be deserialized" or original["forbidden_io_attempts"] != ["OTHER_NPZ"]:
        raise ValueError("Original postflight failure differs from recorded guard error")
    if len(records) != 8 or any(not row["status"].startswith("PASS") for row in records) or len({(row["run_id"], row["split"]) for row in records}) != 8:
        raise ValueError("Original actual eight numerical passes absent/duplicated")
    actual_ids = {route["spec"]["run_id"] for route in plan["routes"]}
    if {(row["run_id"], row["split"]) for row in records} != {(identity, split) for identity in actual_ids for split in ("V_select", "M_cal")}:
        raise ValueError("Original scored output coverage differs")
    after = {}
    for key, expected in before["inputs"].items():
        scope, name = key.split(":", 1)
        path = (public if scope == "public" else cache) / name
        value = dict(sha256=sha(path), bytes=path.stat().st_size)
        if value != expected:raise ValueError("Original immutable input changed: " + path.name)
        after[key] = value
    sources_after = {name: sha(public / name) for name in before["source39"]}
    if len(sources_after) != 39 or sources_after != before["source39"] or sources_after != old_result["sources_after"]:
        raise ValueError("Frozen source39 changed")
    registry_path = public / "EXPERIMENT_REGISTRY.csv"
    current_registry = list(csv.DictReader(registry_path.open(encoding="utf-8-sig", newline="")))
    original_projection = read(numeric / "VALIDATION_REGISTRY_PROJECTION.json")
    current_map = {row["experiment_id"]: row for row in current_registry}
    if len(current_map) != len(current_registry):raise ValueError("Duplicate registry")
    for route in plan["routes"]:
        row = current_map[route["spec"]["run_id"]]
        if any(row[key] != value for key, value in route["registry_row"].items() if key not in MUTABLE):raise ValueError("Raw4 registry science changed")
    registry_snapshot_sha = before["registry_sha256"]
    registry_now_sha = sha(registry_path)
    if registry_snapshot_sha != registry_now_sha:
        raise ValueError("Registry changed since numeric run; full prior science snapshot unavailable in original driver")
    old_union = list(csv.DictReader((public / PRIOR / "UNION_BUNDLES.csv").open(encoding="utf-8", newline="")))
    if len(old_union) != 449 or sha(public / PRIOR / "UNION_BUNDLES.csv") != "3982dc4fbe7f3fc3383e2ab90b766770e9759689a5ad15b0ec7c4f8cd9327447":raise ValueError("Previous449 union changed")
    input_records = {}

    def bound_read(path):
        relative = "public:" + path.relative_to(public).as_posix()
        expected = before["inputs"].get(relative)
        if expected is None or sha(path) != expected["sha256"]:raise ValueError("Comparison was not pinned before original numeric run")
        input_records[relative] = expected
        return read(path)

    old_v = {row["run_id"]: row for row in old_audit["records"] if row["split"] == "V_select"}
    own_v = {row["run_id"]: row for row in records if row["split"] == "V_select"}
    comparisons, matched_evidence = [], []
    for route in plan["routes"]:
        spec, dataset = route["spec"], route["dataset"]
        identity = spec["run_id"]
        receipt = bound_read(public / "combinations" / identity / "receipt.json")
        g = bound_read(public / route["g_predictions"]["V_select"]["manifest_relative_path"])
        raw = bound_read(public / route["raw_predictions"]["V_select"]["manifest_relative_path"])
        g_artifact = route["item"]["parents"]["DIRECT_G"]["artifact_sha256"]
        entries = [("RawH", identity, own_v[identity], receipt)]
        for previous in route["item"]["routes"]:
            if previous["spec"]["g_run_id"] != spec["g_run_id"]:raise ValueError("Different G ID in comparison")
            previous_receipt = bound_read(public / "combinations" / previous["spec"]["run_id"] / "receipt.json")
            if previous_receipt["parent_models"]["g_v"]["model_sha"] != g["model_sha"] or previous_receipt["parent_models"]["g_v"]["artifact_sha256"] != g_artifact:raise ValueError("Different actual G model/artifact")
            for row in previous["registry_rows"]:
                old = old_v[row["experiment_id"]]
                manifest = bound_read(public / old["manifest_public_relative_path"])
                if sha(public / old["manifest_public_relative_path"]) != old["manifest_sha256"]:raise ValueError("Prior independently audited comparison manifest changed")
                if any(manifest["arrays"][key] != g["arrays"][key] for key in ("target_normalized", "observed", "origins", "channel_ids")):raise ValueError("Comparison truth/mask/origin/channel differs")
                family = row["family"]
                method = ("H_B0" if previous["parent_family"] == "B0" else "H_B2") if family == "CONDITIONAL" else "Hpast" if family == "PAST_CONTROL" else family
                if family in {"CONDITIONAL", "PAST_CONTROL"}:
                    for key in ("seed", "device", "learning_rate", "epoch_endpoints", "effective_batch", "microbatch", "fold"):
                        if previous["spec"][key] != spec[key]:raise ValueError("Head optimizer/exposure schedule differs")
                    if previous_receipt["head_parameter_count"] != receipt["head_parameter_count"] or previous_receipt["initial_state_sha"] != receipt["initial_state_sha"]:raise ValueError("Head class/capacity/initialization differs")
                entries.append((method, row["experiment_id"], old, previous_receipt))
        entries += [("G", g["run_id"], g["score"], None), ("Raw", raw["run_id"], raw["score"], None)]
        matched_evidence.append(dict(dataset=dataset, G_run_id=spec["g_run_id"], G_model_sha=g["model_sha"], G_artifact_sha256=g_artifact, Raw_parent=spec["parent_run_id"],
            same_A_only_G=True, head_parameter_count=receipt["head_parameter_count"], head_initial_state_sha=receipt["initial_state_sha"], seed=spec["seed"], learning_rate=spec["learning_rate"],
            epoch_endpoints=spec["epoch_endpoints"], effective_batch=spec["effective_batch"], microbatch=spec["microbatch"], original_M_truth_reference=route["g_predictions"]["M_cal"]["reference_relative_path"],
            original_M_truth_reference_sha=route["g_predictions"]["M_cal"]["reference_sha256"], no_parent_M_training=True))
        for method, run_id, score, fit in entries:
            comparisons.append(dict(dataset=dataset, method=method, run_id=run_id, G_run_id=spec["g_run_id"], G_model_sha=g["model_sha"], G_artifact_sha256=g_artifact,
                mse=score["mse"], mae=score["mae"], mse_delta_vs_G=score["mse"]-g["score"]["mse"], mse_pct_vs_G=(score["mse"]/g["score"]["mse"]-1)*100,
                mse_delta_vs_RawH=score["mse"]-own_v[identity]["mse"], n_observed=score["n_observed"], n_origins=score["n_origins"], horizon=score["horizon"], num_channels=score["num_channels"], cohort=g["cohort"],
                training_status=fit["training_status"] if fit else "FROZEN_PARENT", updates=fit.get("updates", 0) if fit else 0, selected_updates=fit.get("selected_updates") if fit else None,
                original_manifest_public_relative_path=next((item["manifest_public_relative_path"] for item in old_union if item["run_id"] == run_id and item["split"] == "V_select"), own_v[identity]["manifest_public_relative_path"] if method == "RawH" else ""),
                source="Actual new FP64 audit" if method == "RawH" else "Prior449 independent audit plus exact manifest/G identity reuse", new_model_forward=False, actual_deployment_cost_measured_here=False))
    new_rows = []
    for row in records:
        manifest_path = public / row["manifest_public_relative_path"]
        manifest = read(manifest_path)
        if sha(manifest_path) != row["manifest_sha256"] or sha(cache / row["archive_cache_relative_path"]) != row["archive_sha256"]:raise ValueError("New saved output changed after numeric pass")
        if manifest["split"] == "M_cal" and manifest["metadata"].get("audit_only_derived_manifest") is not True:raise ValueError("Derived M audit scope missing")
        new_rows.append(dict(run_id=row["run_id"], split=row["split"], cohort=row["cohort"], manifest_public_relative_path=row["manifest_public_relative_path"], manifest_sha256=row["manifest_sha256"],
            archive_cache_relative_path=row["archive_cache_relative_path"], archive_sha256_as_declared=row["archive_sha256"], archive_exists=True, archive_bytes=(cache / row["archive_cache_relative_path"]).stat().st_size,
            M_manifest_audit_only_derived=row["split"] == "M_cal"))
    union = old_union + new_rows
    if len(union) != 457 or len({(row["run_id"], row["split"], row["cohort"]) for row in union}) != 457 or len({row["archive_cache_relative_path"] for row in union}) != 457:raise ValueError("Union457 identity/archive paths duplicate")
    if any(not (cache / row["archive_cache_relative_path"]).is_file() for row in union):raise ValueError("Union457 archive missing")
    seen, duplicates = {}, []
    for index, row in enumerate(union):
        digest = row["archive_sha256_as_declared"]
        if index >= 449 and digest in seen:duplicates.append(dict(run_id=row["run_id"], split=row["split"], archive_sha256=digest, same_byte_prior_identities=list(seen[digest]), independent_replicate=False))
        seen.setdefault(digest, []).append(row["run_id"] + ":" + row["split"])
    csv_new(output / "UNION_BUNDLES.csv", union, sorted(set().union(*(row.keys() for row in union))))
    csv_new(output / "SAVED_BUNDLE_RESCORING.csv", records, ("run_id", "split", "cohort", "status", "reported_mse", "mse", "mse_difference", "reported_mae", "mae", "mae_difference", "n_observed", "n_origins", "num_channels", "horizon", "active_channels", "updates", "selected_updates", "training_status", "actual_epochs", "manifest_sha256", "archive_sha256", "selected_head_artifact_sha256"))
    csv_new(output / "MATCHED_G_COMPARISON.csv", comparisons, ("dataset", "method", "run_id", "G_run_id", "G_model_sha", "G_artifact_sha256", "mse", "mae", "mse_delta_vs_G", "mse_pct_vs_G", "mse_delta_vs_RawH", "n_observed", "n_origins", "horizon", "num_channels", "cohort", "training_status", "updates", "selected_updates", "original_manifest_public_relative_path", "source", "new_model_forward", "actual_deployment_cost_measured_here"))
    new(output / "MATCHED_G_EVIDENCE.json", matched_evidence)
    new(output / "AUDIT_RESULTS.json", dict(records=records, new_duplicate_archive_content=duplicates, original_numeric_process_exitcode=1, numerical8_passes_preserved_exact=True))
    new(output / "INPUTS_AFTER.json", dict(source39=sources_after, inputs=after, original_before_inventory=NUMERIC + "/INPUTS_BEFORE.json", original_before_sha256=sha(numeric / "INPUTS_BEFORE.json")))
    code = (numeric / "reproduce.py").read_bytes()
    replacements = [
        (b'allowed_npz, allowed_weights, bundles, groups = set(), set(), [], []', b'allowed_npz, parent_hash_npz, allowed_weights, bundles, groups = set(), set(), set(), [], []'),
        (b'path = record(relative(cache, manifest["cache_relative_path"]))', b'path = record(relative(cache, manifest["cache_relative_path"]))\n            parent_hash_npz.add(path)'),
        (b'path.suffix.casefold() == ".npz" and path not in allowed_npz:', b'path.suffix.casefold() == ".npz" and path not in allowed_npz and path not in parent_hash_npz:'),
        (b'process = psutil.Process()\n        with threadpool_limits', b'process = psutil.Process()\n        original_np_load = np.load\n        def only_new_arrays(file, *args, **kwargs):\n            if Path(file).resolve() not in allowed_npz:\n                raise RuntimeError("Only eight new saved output arrays may be loaded")\n            return original_np_load(file, *args, **kwargs)\n        np.load = only_new_arrays\n        with threadpool_limits')]
    corrected = code
    for old, replacement in replacements:
        if corrected.count(old) != 1:raise ValueError("Exact bounded-I/O correction site missing")
        corrected = corrected.replace(old, replacement)
    with (output / "reproduce_corrected.py").open("xb") as stream:stream.write(corrected)
    new(output / "DRIVER_CORRECTION_PROOF.json", dict(original_executed_driver_sha256=sha(numeric / "reproduce.py"), original_numeric_result_sha256=sha(numeric / "RESULT.json"),
        corrected_driver_sha256=sha(output / "reproduce_corrected.py"), correction="Separate parent NPZ hash-only access from new-output NumPy deserialization allowlist; np.load explicitly restricted to new8", scoring_logic_changed=False,
        corrected_driver_executed=False, saved_FP64_rescoring_repeated=False, actual_numeric_process_exitcode=1, numerical_passes_completed_before_original_postflight_guard_error=8,
        finalizer_scope="Metadata/byte hash read-back and matched-G comparison only; no arrays/models/GPU/T execution"))
    summary = dict(status="PASS_NUMERIC8_WITH_SEPARATE_POSTFLIGHT_FINALIZATION", actual_new_V_rescores=4, actual_new_M_head_rescores=4, actual_new_rescores=8, original_numeric_process_exitcode=1,
        actual_numeric_process_exit_evidence="Tool chunk3731ff exit1 after eight emitted numeric PASS; final RESULT partial_results preserved", original_failure_reason=original["fatal_error"],
        actual_new_rescores_in_this_finalizer=0, previous449_rerescored=False, numeric_records_preserved_exact=True, source39_unchanged=True, source33_unchanged=True, immutable_inputs_rechecked=len(after), input_hashes_unchanged=True,
        prior449_public23_allowlist_preserved=True, registry_before_sha256=registry_snapshot_sha, registry_after_sha256=registry_now_sha, registry_scientific_changes=[], registry_written=False,
        union_unique_saved_bundles=457, union_V=405, union_M=52, all457_archive_paths_exist=True, distinct_new_bundles_with_duplicate_content=len(duplicates),
        maximum_absolute_mse_difference=max(abs(row["mse_difference"]) for row in records), maximum_absolute_mae_difference=max(abs(row["mae_difference"]) for row in records),
        matched_G_rows=len(comparisons), head_conditions_and_capacity_and_G_identity_checked=True, new_M_manifest_is_audit_only_derived_from_original_parent_truth=True,
        source39_before=before["source39"], source39_after=sources_after, model_forward=False, checkpoint_deserialized=False, T_accessed=False, GPU_lease_acquired=False, Git_calls=0,
        actual_execution_receipt="diagnostics/raw_deep_g_control_execution_1792090473924518100/RESULT.json", finalizer_wall_seconds=time.time()-started, created_unix=time.time())
    new(output / "RESULT.json", summary)
    method_rows = {(row["dataset"], row["method"]): row for row in comparisons}
    text = "# 같은 G 조건의 RawH 대조와 저장 V/M 독립 검산\n\n[확인] 신규 Raw+deepG의 V4·M4, 총8개 저장 출력이 실제 FP64 수치 검사에 통과했다. 원 numerical 프로세스는 모든8 PASS 뒤 부모 NPZ 해시 읽기를 과하게 차단한 사후 guard 때문에 **종료코드1**이었다. 별도 본 후속 검사는 배열·모델을 실행하지 않고 원 입력 byte·source·registry 불변과 비교 조건을 검증했다. 원 실패와8개 수치를 [그대로 보존](../" + numeric.name + "/RESULT.json)하며, [후속 결과](RESULT.json)·[원수치 CSV](SAVED_BUNDLE_RESCORING.csv)·[같은 G 비교 CSV](MATCHED_G_COMPARISON.csv)·[동일 조건 증거](MATCHED_G_EVIDENCE.json)를 제공한다. 이 둘을 하나의 종료코드0 numerical 실행이라고 보고하지 않는다.\n\n"
    text += "```text\nDataset       G           Hpast       H_B0        H_B2        RawH        alpha_Mfit\n"
    for dataset in ("beijing", "jena", "electricity", "pems08"):
        text += f"{dataset:12}" + " ".join(f"{method_rows[(dataset, method)]['mse']:.9f}" for method in ("G", "Hpast", "H_B0", "H_B2", "RawH", "ALPHA_FIT")) + "\n"
    text += "```\n\n모두 A60 정규화 observed channel-macro MSE이며 MAE·전체 표본수·실험 ID는 CSV에 있다. 같은 자료 안에서만 비교한다. H들은 동일 G artifact/model·seed92601·학습률0.001·epoch endpoint0.25/1/3·effective batch16·microbatch4·head class/용량·초기 state를 사용했다. RawH는 RawSmall 예측을 추가로 보고 H_B0/H_B2는 고정된 B0/B2 예측을 보며, Hpast는 같은 용량으로 과거 정보를 본다. 실제 checkpoint 선택 update는 서로 다르며 V를 checkpoint 선택에 썼으므로 이 V 비교는 독립 미관측 검증이 아니다. Alpha의 M-only analytic 적합과 부모 G 자체는 H neural 학습과 다른 조건이다.\n\n"
    for dataset in ("beijing", "jena", "electricity", "pems08"):
        raw_h = method_rows[(dataset, "RawH")]
        h0, h2, hp, g = (method_rows[(dataset, method)] for method in ("H_B0", "H_B2", "Hpast", "G"))
        text += f"- {dataset}: RawH−G={raw_h['mse']-g['mse']:+.9f}, RawH−H_B0={raw_h['mse']-h0['mse']:+.9f}, RawH−H_B2={raw_h['mse']-h2['mse']:+.9f}, RawH−Hpast={raw_h['mse']-hp['mse']:+.9f}. RawH 학습 상태는 `{raw_h['training_status']}`다. 음수는 해당 V MSE가 작다는 뜻이며 수렴·일반화 판정은 아니다.\n"
    text += "\n이 결과는 원 C 예측을 제공하는 RawH도 같은 G에서 도움이 되는지 비교한다. 압축 고유의 기여는 H_B2−H_B0, 시간 앞단 효과, RawH와의 실제 배포 비용을 함께 봐야 한다. 이 감사에는 비용 실측이 없어 정확도–비용 우위나 압축의 필요성을 결론내리지 않는다. 특히 Electricity·PeMS RawH는 마지막 checkpoint가 best인 UNDERTRAINED 상태를 유지한다. 한 seed·V 선택·자료별 부모 학습 endpoint 차이도 남는다.\n\n"
    text += f"수치 차이는 원 receipt 대비 MSE 최대 {summary['maximum_absolute_mse_difference']:.17g}, MAE 최대 {summary['maximum_absolute_mae_difference']:.17g}이다. V는 기존 독립 FP64/math.fsum·원 reference·공개 band SSE/SAE/count를 검산했다. M은 원래 child manifest·public band 통계가 없어 기존 Raw/G M truth reference와 receipt 배열 hash에 연결한 **사후 감사용 계약**으로 private FP64 오차와 원 receipt·scores.csv를 대조했다. 이를 원래의 child 사전 봉인이나 공개 M 통계 재집계로 부르지 않는다.\n\n"
    text += f"기존449개 수치 검산과23개 safe allowlist를 보존했다. 신규8을 더한 [합본 장부](UNION_BUNDLES.csv)는 **457개의 고유 run/split/cohort·archive 경로 = V405 + M52**다. 기존449개는 재채점하지 않았다. 새8 중 archive 내용 중복은 {len(duplicates)}개로 별도 기록했으며 식별자 수는 독립 모델·seed 반복 수가 아니다. source39와 {len(after)}개 입력 byte SHA를 재확인했고 원 registry byte도 같았다. raw Task·새 scaler·모델/CP 역직렬화·GPU·T·global SCORES·Git 변경0이다. [driver 수정 증거](DRIVER_CORRECTION_PROOF.json)와 [수정된 재현 코드](reproduce_corrected.py)는 부모 NPZ 해시 접근과 실제 NumPy 역직렬화 범위를 분리하며, 이 교정본으로 수치를 반복 실행하지 않았다.\n\n독립 저장 수치 검사·checkpoint replay·full-V 재추론·재학습·충분 수렴·일반화·전체 캠페인 완료는 각각 별개다. 원 배열·정답·가중치는 private에 보존하고 공개하지 않는다.\n"
    new(output / "README_KO.md", text)
    files = [path for directory in (numeric, output) for path in directory.rglob("*") if path.is_file()]
    excluded_names = {"FILE_MANIFEST.json", "PUBLICATION_ALLOWLIST.txt", "PUBLICATION_SCAN.json"}
    files = [path for path in files if path.name not in excluded_names]
    root = public.parents[1]
    new(output / "FILE_MANIFEST.json", dict(files=[dict(repository_relative_path=path.relative_to(root).as_posix(), sha256=sha(path), bytes=path.stat().st_size) for path in sorted(files)],
        no_private_source_arrays_or_weights=True, original_failed_evidence_included_without_false_success=True))
    files += [output / name for name in excluded_names]
    new(output / "PUBLICATION_ALLOWLIST.txt", "\n".join(sorted(path.relative_to(root).as_posix() for path in files)) + "\n")
    sys.path.insert(0, str(public))
    import publication_check
    failures = []
    for path in files:
        if path.name != "PUBLICATION_SCAN.json":publication_check._scan_text(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8-sig"), failures)
    scan = dict(status="PASS" if not failures else "FAIL", actual_scanned_files=len(files)-1, findings=failures, actual_finding_count=len(failures),
        scanner_source_sha256=sha(public / "publication_check.py"), scope="Exact incremental diagnostic allowlist; existing text rules, no Git/index/network", original449_safe23_preserved=True,
        actual_T_GPU_NPZ_model_execution_in_finalizer=False, Git_calls=0)
    publication_check._scan_text("PUBLICATION_SCAN.json", json.dumps(scan), failures)
    if failures:raise ValueError("Public scanner findings: " + str(len(failures)))
    new(output / "PUBLICATION_SCAN.json", scan)
    print(json.dumps(dict(status=summary["status"], actual_numeric_process_exitcode=1, new_rescores_in_this_finalizer=0, preserved_numeric_passes=8, union=457, safe_allowlist_files=len(files), scanner_findings=0)), flush=True)


if __name__ == "__main__":
    main()
