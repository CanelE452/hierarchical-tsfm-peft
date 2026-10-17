"""CreateNew Korean Raw/G reports from actual independent C32 evidence.

Default zero I/O. This builder reads aggregate audit metadata, not trial rows,
arrays, checkpoints or models. Honest partial outcomes remain partial.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import posixpath

NAME = "priority_pems_subset_20261016_v1"
OUTPUTS = ("RAW_G_RESULTS_v1.csv", "RAW_G_KO_v1.md", "RAW_G_BINDINGS_v1.json")
RUNS = {family: "postT_cpu_pems08_E1_SUBSET_" + family + "_k0_s92601_C32_priority_v1" for family in ("RAW", "DIRECT_G")}
INPUT_HELPER = "priority_valid_minima_20261016_v1/build_beijing_report.py"
INPUT_HELPER_SHA = "8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
FIELDS = ("family", "run_id", "accuracy_status", "C", "L", "H", "seed", "MSE", "MAE", "observed_labels", "V_origins", "search_updates", "search_exposures",
    "selected_refit_epoch", "refit_updates", "refit_exposures", "training_status", "F_width", "F_identity", "artifact_sha256", "manifest_sha256",
    "parameter_numel", "parameter_numel_scope", "F_only_resident_bytes", "F_only_resident_scope") + tuple("B" + str(b) + "_" + k for b in (1, 8) for k in
    ("status", "median_seconds", "p90_seconds", "resident_CPU_parameter_buffer_MiB", "foundation_instances_created", "guard_peak_RSS_bytes", "model_load_seconds",
     "cold_load_in_whole_trial", "native_F_counters_per_trial", "derived_50trial_F_counters", "cost_scope"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Finite nonnegative actual aggregate value required")
    return result


def validate_audit(audit):
    if (audit.get("schema") != "INDEPENDENT_PEMS_RAW_G_AUDIT_v1" or audit.get("status") not in ("PASS", "PARTIAL_VALID_EVIDENCE_PRESERVED")
            or audit.get("error") is not None or audit.get("input_SHA_before_after_equal") is not True or audit.get("source48_preserved") is not True
            or audit.get("whole_MASTER_complete") is not False or audit.get("independent_generalization_claimed") is not False
            or audit.get("full_C170_or_B2_or_Bolt_necessity_proven") is not False or audit.get("actual_current_owner_absence_bound") is not True
            or audit.get("costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals") is not True):
        raise ValueError("Actual independently audited complete or honest partial C32 evidence required")
    rows = audit.get("rows", []); by_id = {r["run_id"]: r for r in rows}
    if not rows or len(rows) != len(by_id) or not set(by_id) <= set(RUNS.values()):
        raise ValueError("Only actually verified Raw/G rows, never a missing peer invented as complete")
    costs = 0
    for row in rows:
        if row.get("run_id") != RUNS.get(row.get("family")) or any(row.get(k) != v for k, v in dict(actual_C=32, L=512, H=12, seed=92601, full_V_origins=148,
                accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6", full_C170_or_B2_comparison=False, independent_generalization_claimed=False).items()):
            raise ValueError("Exact new subset and complete per-row accuracy proof required")
        number(row["mse"]); number(row["mae"])
        if not 0 < row["n_observed"] <= 56832 or row.get("actual_parameter_numel") is not None or row.get("F_only_resident_bytes") is not None:
            raise ValueError("Actual observed count and unavailable isolated model quantities must remain distinct")
        fit = row["fit_receipt"]; epoch = fit.get("selected_epoch")
        if row["family"] == "RAW":
            if epoch is not None or any(fit.get(k) != 0 for k in ("search_updates", "search_exposures", "refit_updates", "refit_exposures")) or fit.get("training_status") != "NO_TRAINING":
                raise ValueError("Raw optimizer zero identity must remain explicit")
        elif epoch not in (0., .25, 1., 3.) or fit.get("search_updates") != 1577 or fit.get("search_exposures") != 25215 or fit.get("refit_updates") != math.ceil(math.ceil(epoch * 10190) / 16) or fit.get("refit_exposures") != math.ceil(epoch * 10190):
            raise ValueError("Original G search and fresh selected refit activity required")
        for batch in (1, 8):
            status = row["cost_status"].get(str(batch)); cost = row["costs"].get(str(batch))
            if status == "PASS_ACTUAL_PHYSICAL_TRIALS":
                if cost is None or cost.get("trial_rows") != 50 or cost.get("derived_trial_F_scope") != "INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS" or cost.get("cold_model_load_in_whole_trial") is not False:
                    raise ValueError("Actual independently summed fifty-trial cost required")
                groups = cost.get("summaries", [])
                if len(groups) != 2 or {r.get("phase") for r in groups} != {"timed", "stage_probe"} or any(r.get("trials") != 20 for r in groups):
                    raise ValueError("Exact ordinary20 and separate stage20 summaries required")
                for group in groups:
                    for key in ("median_seconds", "p90_seconds"):
                        number(group[key])
                costs += 1
            elif status != "NOT_COMPLETE" or cost is not None:
                raise ValueError("Missing/failed cost remains unavailable, not zero or a completed native row")
    if audit["status"] == "PASS" and (len(rows) != 2 or costs != 4 or audit.get("failures") != [] or audit.get("skipped") != []):
        raise ValueError("Complete status requires both actual accuracy rows and four costs")
    if audit["status"] == "PARTIAL_VALID_EVIDENCE_PRESERVED" and not (audit.get("failures") or audit.get("skipped")):
        raise ValueError("A partial report requires explicit preserved failure/skipped reasons")


def rows_from_audit(audit):
    validate_audit(audit); rows = []
    for original in audit["rows"]:
        fit = original["fit_receipt"]; family = original["family"]
        row = dict(family=family, run_id=original["run_id"], accuracy_status=original["accuracy_status"], C=32, L=512, H=12, seed=92601,
            MSE=original["mse"], MAE=original["mae"], observed_labels=original["n_observed"], V_origins=148,
            search_updates=fit["search_updates"], search_exposures=fit["search_exposures"], selected_refit_epoch=fit["selected_epoch"],
            refit_updates=fit["refit_updates"], refit_exposures=fit["refit_exposures"], training_status=fit["training_status"],
            F_width=32 if family == "RAW" else 0, F_identity=original["source_identity"], artifact_sha256=original["artifact_sha256"], manifest_sha256=original["manifest_sha256"],
            parameter_numel=None, parameter_numel_scope=original["parameter_numel_scope"], F_only_resident_bytes=None, F_only_resident_scope=original["F_only_resident_scope"])
        for batch in (1, 8):
            prefix = "B" + str(batch) + "_"; row[prefix + "status"] = original["cost_status"][str(batch)]
            cost = original["costs"].get(str(batch))
            if cost is None:
                continue
            timed = next(r for r in cost["summaries"] if r["phase"] == "timed"); memory = cost["memory_scope"]
            row.update({prefix + "median_seconds": timed["median_seconds"], prefix + "p90_seconds": timed["p90_seconds"],
                prefix + "resident_CPU_parameter_buffer_MiB": memory["resident_weights_cpu_MiB"], prefix + "foundation_instances_created": memory["foundation_instances_created"],
                prefix + "guard_peak_RSS_bytes": cost["guard_sampled_peak_RSS_bytes"], prefix + "model_load_seconds": cost["model_load_seconds"],
                prefix + "cold_load_in_whole_trial": False, prefix + "native_F_counters_per_trial": cost["native_F_counters_per_trial"],
                prefix + "derived_50trial_F_counters": {k: cost[k] for k in ("F_logical_streams", "F_physical_streams", "F_calls")}, prefix + "cost_scope": cost["recorded_scope"]})
        rows.append(row)
    return rows


def markdown(audit, rows, audit_relative, audit_sha):
    link = posixpath.relpath(audit_relative, NAME)
    text = ["# PeMS08 C32 Raw·직접 G 검증 결과", "", "상태: `" + audit["status"] + "`. 전체 MASTER와 원 subset 전체 실험은 미완료다.", "",
        "[독립 감사](" + link + ") · [원정밀도 CSV](RAW_G_RESULTS_v1.csv) · [입출력 SHA](RAW_G_BINDINGS_v1.json)", "",
        "이번 비교는 ID만으로 정한 C32와 동일 scaler·V148·mask·origin·seed92601의 Raw와 직접 G다. L512/H12이며 주지표는 A 정규화 채널 평균 MSE, 보조지표는 MAE다.", "",
        "Raw는 압축 없이 C32를 동결 Chronos-Bolt에 입력한다. G는 foundation 없이 채널 공유 512→32→12 GELU innovation과 현재값 anchor를 학습한다. 이 두 경로만으로 B2·압축의 가치나 Bolt 필요성을 확정할 수 없다.", "", "## 실제 정확도·학습·비용", ""]
    for row in rows:
        text += ["### " + row["family"], "", "```text", "정규화 MSE  " + repr(row["MSE"]), "정규화 MAE  " + repr(row["MAE"]),
            "관측 label  " + str(row["observed_labels"]) + " / V origin 148", "search      " + str(row["search_updates"]) + " updates / " + str(row["search_exposures"]) + " exposures",
            "선택 refit  epoch=" + str(row["selected_refit_epoch"]) + " / " + str(row["refit_updates"]) + " updates / " + str(row["refit_exposures"]) + " exposures",
            "학습 상태   " + row["training_status"]]
        for batch in (1, 8):
            p = "B" + str(batch) + "_"
            text += ["B" + str(batch) + " 비용    " + row[p + "status"] + (" / median=" + repr(row[p + "median_seconds"]) + " s / p90=" + repr(row[p + "p90_seconds"]) + " s" if p + "median_seconds" in row else " / 미측정·미검증을 0으로 표시하지 않음")]
        text += ["```", ""]
    if len(rows) == 2:
        pair = {r["family"]: r for r in rows}
        text += ["같은 반복 개발 V에서 G−Raw MSE 차이는 `" + repr(pair["DIRECT_G"]["MSE"] - pair["RAW"]["MSE"]) + "`다. 구조와 학습량이 달라 이 차이를 foundation 단독의 인과 효과로 해석하지 않는다.", ""]
    text += ["## 검산 범위와 남은 판단", "",
        "별도 saved full-V FP64 재채점과 실제 새 CPU6(4+2) 증거를 결속하고, 공개 SSE·SAE·count 14,208행/route를 채널별로 math.fsum 재집계했다. 검사 통과는 독립 일반화나 전체 V 재추론을 의미하지 않는다.", "",
        "Raw B1/B8의 실제 full 대 row128 출력 parity는 별도 증거다. B8은 full 256streams/1call, row128 256streams/2calls다. parity가 실패하면 그 Raw batch 비용만 미완료로 남기며 G의 독립 경로를 취소하지 않는다.", "",
        "비용은 CPU2의 실제 physical 입력→채움·정규화→F 또는 G→FP64 inverse·CPU 반환이다. warm10/timed20/separate-stage20을 사용하며, cold model load는 whole trial 밖에서 별도로 기록한다. stage 합이 whole과 같다고 주장하지 않는다.", "",
        "CPU parameter+buffer resident bytes, 실제 guard RSS, 학습 peak, F 단독 resident는 다른 수치다. native memory API가 기록하지 않은 parameter numel·F 단독 resident는 NA로 남긴다. G에는 F hash를 만들어 넣지 않는다.", "",
        "단일 seed·반복 개발 V·제한 search와 선택 refit으로 수렴은 미확정이다. C170의 기존 1epoch와 C32의 새 3epoch search는 C만의 공정한 인과 대조가 아니다. upstream 미래 보간·전처리 계보는 UNKNOWN이며 새 prefix 접근만으로 이를 해결했다고 말하지 않는다.", "",
        "다음 판단에는 원 승인 범위의 같은 C32 B0/B2 등 실제 추가 대조가 필요하다. 이번 Raw/G 결과를 full C170·다른 자료 전체 순위나 B2 주 방법 검증으로 확대하지 않는다.", ""]
    if audit.get("failures") or audit.get("skipped"):
        text += ["## 실제 미완료·실패·건너뜀", "", "```json", json.dumps(dict(failures=audit.get("failures", []), skipped=audit.get("skipped", []), native_incomplete_outcomes=audit.get("native_incomplete_outcomes", [])), ensure_ascii=False, indent=2), "```", "",
            "원 결과·checkpoint·실행 영수증은 보존한다. native summary에 기록된 진행량만 보고하며 마지막 checkpoint나 손실된 진행량은 추정하지 않는다. 현재 고정 경로 CLI는 기존 attempt의 재시작·덮기를 거부한다. ROOT가 원 범위·남은 자원 안에서 별도 continuation과 새 출력 경로를 구체화해 검산해야 한다. 이 보고는 자동 재실행이나 추가 자원 승인이 아니다.", ""]
    text += ["감사 SHA: `" + audit_sha + "`. CSV/MD는 이 실제 감사에 결속되며 입력 목록 전체는 감사의 `inputs[]`를 원본으로 참조한다.", ""]
    return "\n".join(text)


def execute(public, cache, audit_relative, audit_sha, auditor_sha, builder_sha):
    if sha(__file__) != builder_sha:
        raise ValueError("Exact reviewed builder SHA required")
    helper_path = Path(public) / INPUT_HELPER
    if sha(helper_path) != INPUT_HELPER_SHA:
        raise ValueError("Frozen stdlib input helper changed")
    spec = importlib.util.spec_from_file_location("raw_g_report_inputs", helper_path); helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    inputs = helper.Inputs(public, cache)
    audit = inputs.json("public", audit_relative, audit_sha)
    inputs.capture("public", NAME + "/audit_raw_g_v2.py", auditor_sha)
    inputs.capture("public", NAME + "/build_raw_g_report_v2.py", builder_sha)
    inputs.capture("public", INPUT_HELPER, INPUT_HELPER_SHA)
    if audit.get("audit_source_sha256") != auditor_sha or not audit_relative.startswith(NAME + "/"):
        raise ValueError("Exact same-namespace actual independent audit/source binding required")
    for ref in audit["inputs"]:
        inputs.capture(ref["root"], ref["relative_path"], ref["sha256"])
    if any(inputs.path("public", name).exists() for name in audit.get("missing_public_native_paths", [])):
        raise ValueError("An audit-observed missing native receipt changed after the actual audit snapshot")
    rows = rows_from_audit(audit)
    folder = inputs.path("public", NAME)
    if any((folder / name).exists() for name in OUTPUTS):
        raise FileExistsError("Existing report outputs are preserved; CreateNew only")
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n"); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    blobs = {OUTPUTS[0]: stream.getvalue().encode(), OUTPUTS[1]: markdown(audit, rows, audit_relative, audit_sha).encode()}
    if not inputs.preserved():
        raise ValueError("Actual audit/transitive inputs changed before report creation")
    for name, blob in blobs.items():
        with (folder / name).open("xb") as file:
            file.write(blob)
    binding = dict(schema="PEMS_RAW_G_REPORT_BINDINGS_v1", status="FROM_ACTUAL_PEMS_RAW_G_AUDIT_" + audit["status"], audit_relative_path=audit_relative,
        audit_sha256=audit_sha, auditor_source_sha256=auditor_sha, builder_source_sha256=builder_sha, input_SHA_before_after_equal=inputs.preserved(), source48_preserved=True,
        inputs=[dict(root="public", relative_path=audit_relative, sha256=audit_sha), dict(root="public", relative_path=NAME + "/audit_raw_g_v2.py", sha256=auditor_sha),
            dict(root="public", relative_path=NAME + "/build_raw_g_report_v2.py", sha256=builder_sha), dict(root="public", relative_path=INPUT_HELPER, sha256=INPUT_HELPER_SHA)],
        transitive_input_manifest=dict(root="public", relative_path=audit_relative, sha256=audit_sha, field="inputs"),
        outputs=[dict(relative_path=NAME + "/" + name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()) for name, blob in blobs.items()],
        self_hash_policy="Bindings JSON is excluded from its own output hash list; final publication inventory binds its exact bytes",
        completed_accuracy_rows=len(rows), completed_cost_rows=len(audit["cost_conditions"]), full_subset_experiments_complete=False,
        whole_MASTER_complete=False, independent_generalization_claimed=False)
    if binding["input_SHA_before_after_equal"] is not True:
        raise ValueError("Inputs changed; partial CSV/MD preserved without a completed bindings claim")
    with (folder / OUTPUTS[2]).open("x", encoding="utf-8", newline="\n") as file:
        json.dump(binding, file, ensure_ascii=False, indent=2, allow_nan=False); file.write("\n")
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute", action="store_true"); parser.add_argument("--root-triggered", action="store_true")
    for name in ("public-root", "cache-root", "audit-relative", "audit-sha", "auditor-sha", "builder-sha"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED", input_reads=0, files_written=0))); return 0
    if not args.root_triggered or any(getattr(args, k.replace("-", "_")) is None for k in ("public-root", "cache-root", "audit-relative", "audit-sha", "auditor-sha", "builder-sha")):
        parser.error("Explicit actual audit/source/root bindings required")
    value = execute(args.public_root, args.cache_root, args.audit_relative, args.audit_sha, args.auditor_sha, args.builder_sha)
    print(json.dumps(dict(status=value["status"], completed_accuracy_rows=value["completed_accuracy_rows"]))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
