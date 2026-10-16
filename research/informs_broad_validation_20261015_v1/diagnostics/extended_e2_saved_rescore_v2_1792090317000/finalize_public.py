"""Read back the completed delta, preserve original evidence, and seal safe public files."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    public, cache = Path(args.public_root).resolve(), Path(args.cache_root).resolve()
    output = Path(__file__).resolve().parent
    failed = public / "diagnostics/extended_e2_saved_rescore_1792089998000"
    root = public.parents[1]
    result, audit = read(output / "RESULT.json"), read(output / "AUDIT_RESULTS.json")
    if result["status"] != "PASS_SAVED_E2_AND_NEW_PARENT_M" or len(audit["records"]) != 40 or audit["errors"]:
        raise ValueError("Actual scoring did not finish successfully")
    if (result["actual_new_V_rescores"], result["actual_new_dense_M_rescores"], result["union_unique_saved_bundles"]) != (28, 12, 449):
        raise ValueError("Actual scope count differs")
    rows = list(csv.DictReader((output / "SAVED_BUNDLE_RESCORING.csv").open(encoding="utf-8", newline="")))
    by_key = {(row["run_id"], row["split"], row["cohort"]): row for row in audit["records"]}
    if len(rows) != len(by_key) or len(rows) != 40:
        raise ValueError("CSV/JSON unique actual result count differs")
    for row in rows:
        expected = by_key[(row["run_id"], row["split"], row["cohort"])]
        for key, value in row.items():
            if key not in expected or expected[key] is None:
                if value:
                    raise ValueError("Unexpected populated score field")
            elif isinstance(expected[key], (int, float)):
                if float(value) != float(expected[key]):
                    raise ValueError("CSV numeric read-back differs")
            elif value != str(expected[key]):
                raise ValueError("CSV string read-back differs")
    before, after = read(output / "INPUTS_BEFORE.json"), read(output / "INPUTS_AFTER.json")
    if before["source39"] != after["source39"] or before["inputs"] != after["inputs"]:
        raise ValueError("Before/after immutable input inventory differs")
    for name, digest in after["source39"].items():
        if sha(public / name) != digest:
            raise ValueError("Source changed after actual scoring")
    for key, expected in after["inputs"].items():
        scope, relative = key.split(":", 1)
        path = (public if scope == "public" else cache) / relative
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Immutable input changed after scoring")
    union = list(csv.DictReader((output / "UNION_BUNDLES.csv").open(encoding="utf-8", newline="")))
    if len(union) != 449 or len({(row["run_id"], row["split"], row["cohort"]) for row in union}) != 449 or len({row["archive_cache_relative_path"] for row in union}) != 449:
        raise ValueError("Union identity/archive path count differs")
    if any(not (cache / row["archive_cache_relative_path"]).is_file() for row in union):
        raise ValueError("Union archive path absent")
    seen, corrected_duplicates = {}, []
    for index, row in enumerate(union):
        digest = row["archive_sha256_as_declared"]
        if index >= 409 and digest in seen:
            corrected_duplicates.append(dict(run_id=row["run_id"], split=row["split"], archive_sha256=digest,
                same_byte_prior_identities=list(seen[digest]), distinct_identity_not_independent_replication=True))
        seen.setdefault(digest, []).append(row["run_id"] + ":" + row["split"])
    if len(corrected_duplicates) != 14:
        raise ValueError("Duplicate-content count differs")
    projected_audit = copy.deepcopy(audit)
    projected_audit["duplicate_archive_content"] = corrected_duplicates
    restored = copy.deepcopy(projected_audit)
    restored["duplicate_archive_content"] = audit["duplicate_archive_content"]
    if restored != audit:
        raise ValueError("Scientific audit values changed during projection")
    new(output / "AUDIT_RESULTS_PUBLIC.json", projected_audit)
    original_code = (output / "reproduce.py").read_bytes()
    old = b'same_byte_prior_identities=content[manifest["archive_sha256"]]'
    replacement = b'same_byte_prior_identities=list(content[manifest["archive_sha256"]])'
    if original_code.count(old) != 1:
        raise ValueError("Exact single-line duplicate-list correction not found")
    corrected_code = original_code.replace(old, replacement)
    with (output / "reproduce_corrected.py").open("xb") as stream:
        stream.write(corrected_code)
    new(output / "PUBLIC_PROJECTION_PROOF.json", dict(original_audit_sha256=sha(output / "AUDIT_RESULTS.json"), public_audit_sha256=sha(output / "AUDIT_RESULTS_PUBLIC.json"),
        only_changed_field="duplicate_archive_content; snapshot prior identities rather than mutable list references", all40_numeric_records_and8_alias_checks_exact=True,
        original_duplicate_records=14, corrected_duplicate_records=14, original_executed_driver_sha256=sha(output / "reproduce.py"), corrected_reproduction_driver_sha256=sha(output / "reproduce_corrected.py"),
        reproduction_driver_change="One list(...) copy at duplicate-content evidence construction; no scoring/training logic changed", corrected_driver_executed=False, FP64_rescoring_repeated=False,
        original_evidence_preserved=True))
    failed_raw = read(failed / "RESULT.json")
    failed_safe = copy.deepcopy(failed_raw)
    failed_safe["fatal_error"] = "Optional per-output SCORES.csv absent in E2 wrapper; metadata preflight stopped before NumPy import and FP64 rescoring. Original failure preserved privately."
    failed_restored = copy.deepcopy(failed_safe)
    failed_restored["fatal_error"] = failed_raw["fatal_error"]
    if failed_restored != failed_raw:
        raise ValueError("Failed-run scientific fields changed")
    new(failed / "RESULT_PUBLIC.json", failed_safe)
    new(failed / "PUBLIC_PROJECTION_PROOF.json", dict(original_sha256=sha(failed / "RESULT.json"), public_sha256=sha(failed / "RESULT_PUBLIC.json"), only_changed_field="fatal_error",
        reason="Remove absolute private workspace path from missing optional CSV error", all_other_values_exact=True, original_evidence_preserved=True,
        excluded_public_relative_path=(failed / "RESULT.json").relative_to(public).as_posix()))
    new(failed / "README_KO.md", "# 첫 E2 검산 시도의 형식 가정 오류\n\n[확인] 첫 실행은 E2 wrapper가 만들지 않는 개별 `SCORES.csv`를 필수로 요구해 metadata preflight에서 중단됐다. FP64 재채점과 모델·GPU 실행은 0이다. 원 실행 driver와 실패 영수증을 보존했다. [공개용 실패 결과](RESULT_PUBLIC.json)는 오류 문자열의 개인 절대경로만 의미가 같은 설명으로 바꿨으며 [파생 증거](PUBLIC_PROJECTION_PROOF.json)에 원본·파생 SHA와 변경 필드를 남겼다. 원 `RESULT.json`은 공개 allowlist에서 제외한다.\n\n후속 검산은 원 manifest 점수·public band 통계·combination receipt의 점수를 실제 대조했다. 이 첫 실패는 모델이나 기존 E2 예측 결과의 오류 판정이 아니다.\n")
    maximum_mse = max(abs(row["mse_difference"]) for row in audit["records"] if row["split"] == "V_select")
    maximum_mae = max(abs(row["mae_difference"]) for row in audit["records"] if row["split"] == "V_select")
    new(output / "VERIFICATION.json", dict(status="PASS_ACTUAL_READBACK", actual_scoring_process_exit_code=0, exit_evidence="Tool session5700 / final chunk2472ac returned exit_code0",
        actual_new_scored_bundles=40, actual_new_standard_V=28, actual_new_parent_dense_M=12, all40_CSV_JSON_values_exact=True,
        maximum_absolute_V_mse_difference=maximum_mse, maximum_absolute_V_mae_difference=maximum_mae, actual_alias_checks=8, same_archive_bytes_aliases=8,
        source39_rechecked_unchanged=True, immutable_inputs_rechecked=len(after["inputs"]), previous409_not_rerescored=True,
        union_unique_bundle_identities_and_archive_paths=449, union_V=401, union_M=48, all449_archive_paths_exist=True,
        distinct_new_identities_with_duplicate_content=14, duplicate_identity_is_not_independent_model_or_replicate=True,
        raw_model_or_array_execution_in_readback=False, registry_written=False, T_accessed=False, GPU_acquired=False, Git_calls=0, created_unix=time.time()))
    report = f"""# 고정된 연장 부모 E2 저장 결과 독립 검산

[확인] 원 계획의 표준 V 28개와 새 부모 dense M 12개를 실제 저장 배열에서 FP64로 재채점했다. **40/40 PASS, 실제 종료코드 0**이다. [실행 영수증](RESULT.json), [원수치·차이·표본수](SAVED_BUNDLE_RESCORING.csv), [검산 상세](AUDIT_RESULTS_PUBLIC.json), [종료 후 read-back](VERIFICATION.json)을 제공한다. 실제 scoring wall은 {result['CPU_wall_seconds']:.6f}초, CPU process {result['CPU_process_seconds']:.5f}초, peak RSS {result['RSS_peak_bytes']} byte다.

V 점수의 원 manifest 대비 최대 절대 차이는 MSE {maximum_mse:.17g}, MAE {maximum_mae:.17g}다. 이는 허용된 FP64 집계 오차 안이며 성능 개선 판정이 아니다. `audit_predictions.audit_prediction_bundle`의 별도 FP64/math.fsum 계산, 원 reference의 target/mask/origin/channel fingerprint, 공개 band SSE/SAE/count와 원 manifest·combination receipt 점수를 대조했다. E2 wrapper에는 개별 `SCORES.csv`가 없어 없는 파일을 만들어 맞추지 않았다.

새 부모 M 12개는 원 `score=null/public_stats=null` dense-cache 계약이다. 원 reference·배열 SHA·cohort·전체 stride1 origin·채널·F stream 장부를 검사하고 관측 채널별 FP64 오차를 계산했다. 공개 통계 재집계나 원 M 점수와의 일치 검증을 했다고 주장하지 않는다. 실제 학습·부모·selected head의 정확한 spec/source/data/scaler/artifact SHA도 대조했으나 checkpoint tensor/optimizer를 역직렬화하거나 모델을 실행하지 않았다.

12개 신경망 H fit과 4개 analytic alpha-family 적합을 구분한다. alpha 4출력씩 16개 조건은 16개 신경망 학습이 아니며, analytic 가족의 optimizer update는 0이다. Alpha0/1 8개는 각각 실제 활성 G/B2 부모의 5개 배열과 archive bytes가 정확히 같고, realized model binding도 그 부모만 요구함을 확인했다. wrapper model SHA는 별도 식별용이므로 부모 model SHA와 같다고 바꾸지 않는다. 저장 부모 출력으로 결합한 F stream0을 실제 배포 F 비용0으로 해석하지 않는다.

[bundle 장부](UNION_BUNDLES.csv)는 기존409 + 신규40 = **449개의 서로 다른 run/split/cohort 및 archive 경로**, 그중 V401·dense M48이다. 이전409개는 재채점하지 않았다. 새40개 중 14개는 이전 또는 앞선 새 출력과 archive 내용이 같다. 이는 별도 ID의 저장 파일 수이며 독립 모델·seed·통계 반복 수가 아니다. Jena 신경망 H3개가 G 출력과 같거나 Electricity B0/B2 조건부 출력이 같은 사실도 숨기지 않고 상세 중복 목록에 남긴다.

원 `AUDIT_RESULTS.json`의 보조 중복 장부에서 mutable list를 참조해 prior-ID 목록에 자기/후속 ID가 포함되는 오류를 발견했다. 원 파일·실행 driver를 보존하고, 저장된449개 metadata 순서에서 prior 목록만 복사해 바로잡은 `AUDIT_RESULTS_PUBLIC.json`을 공개한다. 40개 모든 수치와 8개 alias 검사, 중복 수14는 그대로다. [변경 증거](PUBLIC_PROJECTION_PROOF.json)와 단일행을 고친 [재현 driver](reproduce_corrected.py)를 제공하며, 교정본으로 FP64를 반복 실행하지 않았다. 원 실행 코드는 `reproduce.py`다.

source39와 321개 입력 byte SHA의 전후·추가 read-back 불변을 확인했다. 감사 중 registry 과학·상태 변경0이고 검증 호출용 상태/reason projection만 별도 파일에 남겼다. CUDA=-1/BLAS thread2, torch·Chronos import0, 보호 T·GPU lease·raw Task·새 scaler·global SCORES·Git 변경0이다. 추가 Raw-conditioned H4는 이 범위에 포함하지 않았다. 학습 상태의 UNDERTRAINED/SCREENED 표시는 원 receipt 그대로 유지한다.

이 감사는 저장 예측의 수치 무결성을 확인했다. checkpoint replay, full-V 모델 재추론, 독립 재학습, 학습 수렴, 방법의 효과·신규성, 독립 일반화 또는 전체 캠페인 완료를 입증하지 않는다. [첫 실패 시도](../{failed.name}/README_KO.md)는 보존했고 잘못된 개별 CSV 가정을 바로잡았다.

재현 시 새로운 output 폴더와 고정된 E2 terminal receipt/SHA를 지정한다. 최종 public allowlist는 별도 파일이며 원자료·target/prediction 배열·가중치는 공개하지 않는다.

```powershell
& <original_python> -X utf8 -u -B <this_folder>/reproduce_corrected.py --public-root <campaign> --cache-root <private_cache> --output <campaign>/diagnostics/<new_unique_output> --execution-receipt execution_batches/extended_parent_e2_1792089082149853700_451af576.json --execution-sha 60253e28d982af3b30399ea850f41defe60d8a239cdf72930970bf34af202565 --terminal-evidence <actual_terminal_evidence>
```
"""
    new(output / "README_KO.md", report)
    excluded = [failed / "RESULT.json", output / "AUDIT_RESULTS.json"]
    new(output / "PUBLIC_EXCLUSIONS.json", dict(excluded_repository_relative_paths=[path.relative_to(root).as_posix() for path in excluded],
        reasons=["Original failed-run error contains private absolute path; safe RESULT_PUBLIC derivative supplied", "Original auxiliary prior-ID list includes mutable-list self/future references; corrected AUDIT_RESULTS_PUBLIC supplied"],
        original_bytes_preserved=True, scored_numeric_results_not_excluded=True))
    self_metadata = {"PUBLICATION_ALLOWLIST.txt", "FILE_MANIFEST.json", "PUBLICATION_SCAN.json"}
    files = [path for directory in (failed, output) for path in directory.iterdir() if path.is_file() and path not in excluded and path.name not in self_metadata]
    new(output / "FILE_MANIFEST.json", dict(files=[dict(repository_relative_path=path.relative_to(root).as_posix(), sha256=sha(path), bytes=path.stat().st_size) for path in sorted(files)],
        metadata_self_references_excluded=sorted(self_metadata)))
    files += [output / name for name in self_metadata]
    new(output / "PUBLICATION_ALLOWLIST.txt", "\n".join(sorted(path.relative_to(root).as_posix() for path in files)) + "\n")
    sys.path.insert(0, str(public))
    import publication_check
    failures = []
    for path in files:
        if path.name != "PUBLICATION_SCAN.json":
            publication_check._scan_text(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8-sig"), failures)
    scan = dict(status="PASS" if not failures else "FAIL", actual_scanned_files=len(files)-1, actual_failure_count=len(failures), failures=failures,
        scope="Exact new diagnostic allowlist only; existing publication_check._scan_text; no Git/network/index check", scanner_source_sha256=sha(public / "publication_check.py"),
        all_scoring_numbers_preserved=True, absolute_DLL_fields_in_public_result=False, original_failed_path_file_and_auxiliary_bad_list_excluded=True, Git_calls=0)
    publication_check._scan_text((output / "PUBLICATION_SCAN.json").relative_to(root).as_posix(), json.dumps(scan), failures)
    if failures:
        raise ValueError("Public diagnostic text scanner findings: " + str(len(failures)))
    new(output / "PUBLICATION_SCAN.json", scan)
    print(json.dumps(dict(status="PASS_READBACK_AND_PUBLICATION_DELTA", new_rescores=40, union=449, scanned_files=len(files), findings=0, public_allowlist=(output / "PUBLICATION_ALLOWLIST.txt").relative_to(public).as_posix())), flush=True)


if __name__ == "__main__":
    main()
