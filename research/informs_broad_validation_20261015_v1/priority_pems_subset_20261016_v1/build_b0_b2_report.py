"""CreateNew C32 B0/B2 reports from complete or honest partial audit metadata.

Default zero I/O; no scientific imports, native statistics, arrays or models.
Raw/G references are optional already-audited rows, never new measurements.
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
import sys

NAME = "priority_pems_subset_20261016_v1"
OUTPUTS = ("B0_B2_RESULTS_v1.csv", "B0_B2_KO_v1.md", "B0_B2_BINDINGS_v1.json")
RUNS = {family: f"postT_cpu_pems08_E1_SUBSET_{family}_k8_s92601_C32_priority_v1" for family in ("B0", "B2")}
REFERENCES = {family: f"postT_cpu_pems08_E1_SUBSET_{family}_k0_s92601_C32_priority_v1" for family in ("RAW", "DIRECT_G")}
INPUT_HELPER = "priority_valid_minima_20261016_v1/build_beijing_report.py"
INPUT_HELPER_SHA = "8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
FIELDS = ("row_scope", "family", "run_id", "accuracy_reused", "cost_reused", "accuracy_status", "C", "L", "H", "seed", "K", "F_width",
    "MSE", "MAE", "observed_labels", "V_origins", "device", "CPU_threads", "search_updates", "search_exposures", "selected_refit_epoch",
    "refit_updates", "refit_exposures", "training_status", "F_identity", "artifact_sha256", "manifest_sha256", "band_count_binding_sha256",
    "parameter_numel", "parameter_numel_scope", "F_only_resident_bytes", "F_only_resident_scope") + tuple(f"B{batch}_" + key for batch in (1, 8) for key in
    ("status", "median_seconds", "p90_seconds", "resident_CPU_parameter_buffer_MiB", "foundation_instances_created", "guard_peak_RSS_bytes",
     "model_load_seconds", "cold_load_in_whole_trial", "native_F_counters_per_trial", "derived_50trial_F_counters", "cost_scope"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative audited aggregate required")
    return value


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_row(row, reference=False):
    family = row.get("family"); identifiers = REFERENCES if reference else RUNS
    expected = dict(actual_C=32, L=512, H=12, seed=92601, full_V_origins=148, accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6")
    if row.get("run_id") != identifiers.get(family) or any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Only the exact audited C32/H12/full-V route is reportable")
    number(row["mse"]); number(row["mae"])
    if type(row.get("n_observed")) is not int or not 0 < row["n_observed"] <= 56832:
        raise ValueError("Actual observed count required; missing targets are not labels")
    if (row.get("actual_parameter_numel") is not None or row.get("F_only_resident_bytes") is not None
            or row.get("F_only_resident_scope") != "NOT_SEPARATELY_RECORDED"):
        raise ValueError("Unavailable F-only resident bytes and parameter numel must stay NA")
    if not all(digest(row.get(key)) for key in ("artifact_sha256", "manifest_sha256", "band_count_binding_sha256")) or not row.get("reference_axis_fingerprints"):
        raise ValueError("Exact artifact, manifest and same-cohort evidence required")
    if not reference and (row.get("latent_k") != 8 or row.get("full_C170_comparison") is not False or row.get("B0_B2_C32_structural_comparison") is not True):
        raise ValueError("New comparison is original C32/K8 B0/B2, not full C170")
    fit = row["fit_receipt"]; epoch = fit.get("selected_epoch")
    if fit.get("status") != "COMPLETE_VALID" or fit.get("run_id") != row["run_id"] or fit.get("family") != family or not all(digest(fit.get(key)) for key in ("data_sha256", "refit_scaler_sha")):
        raise ValueError("Actual same-route completed fit receipt required")
    if family == "RAW":
        if epoch is not None or fit.get("training_status") != "NO_TRAINING" or any(fit.get(key) != 0 for key in ("search_updates", "search_exposures", "refit_updates", "refit_exposures")):
            raise ValueError("Reused Raw remains an optimizer-zero identity")
    else:
        exposures = math.ceil(epoch * 10190) if epoch in (0., .25, 1., 3.) else None
        if (exposures is None or fit.get("search_updates") != 1577 or fit.get("search_exposures") != 25215
                or fit.get("refit_exposures") != exposures or fit.get("refit_updates") != math.ceil(exposures / 16)
                or fit.get("training_status") != ("UNDERTRAINED_OR_BUDGET_LIMIT" if epoch == 3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")):
            raise ValueError("Original segmented search and actual fresh selected refit, including zero, required")
    passed = 0
    for batch in (1, 8):
        status = row["cost_status"].get(str(batch)); cost = row["costs"].get(str(batch))
        if status == "NOT_COMPLETE" and cost is None:
            continue
        if status != "PASS_ACTUAL_PHYSICAL_TRIALS" or cost is None or cost.get("trial_rows") != 50:
            raise ValueError("Unavailable costs remain NA, not zero or complete")
        groups = cost.get("summaries", [])
        if len(groups) != 2 or {value.get("phase") for value in groups} != {"timed", "stage_probe"}:
            raise ValueError("Original ordinary20 and separate-stage20 summaries required")
        for group in groups:
            if group.get("batch_size") != batch or group.get("trials") != 20:
                raise ValueError("Exact actual batch and trial count required")
            number(group["median_seconds"]); number(group["p90_seconds"])
        if cost.get("derived_trial_F_scope") != "INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS" or cost.get("cold_model_load_in_whole_trial") not in (None, False):
            raise ValueError("Audited public trial totals and separate loading scope required")
        width = 32 if family == "RAW" else (0 if family == "DIRECT_G" else 8)
        calls = math.ceil(batch * width / 128)
        native = dict(streams=batch * width, physical_streams=batch * width, physical_calls=calls)
        if (cost.get("native_F_counters_per_trial") != native or any(cost.get(key) != value for key, value in
                dict(F_logical_streams=50 * batch * width, F_physical_streams=50 * batch * width, F_calls=50 * calls).items())):
            raise ValueError("Family-specific actual native and derived stream counters required")
        memory = cost["memory_scope"]; number(memory["resident_weights_cpu_MiB"])
        if memory.get("foundation_instances_created") != int(family != "DIRECT_G"):
            raise ValueError("G has zero foundation instances; frozen-F routes have one")
        number(cost["guard_sampled_peak_RSS_bytes"]); number(cost["model_load_seconds"])
        passed += 1
    return passed


def validate_audit(audit):
    if (audit.get("schema") != "INDEPENDENT_PEMS_B0_B2_AUDIT_v1" or audit.get("status") not in ("PASS", "PARTIAL_VALID_EVIDENCE_PRESERVED")
            or audit.get("error") is not None or audit.get("input_SHA_before_after_equal") is not True or audit.get("source48_preserved") is not True
            or audit.get("actual_current_owner_absence_bound") is not True
            or audit.get("costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals") is not True
            or audit.get("whole_MASTER_complete") is not False or audit.get("independent_generalization_claimed") is not False):
        raise ValueError("Actual complete or honest partial independent B0/B2 audit required")
    rows = audit.get("rows", [])
    if not rows or len({row["run_id"] for row in rows}) != len(rows) or not {row["run_id"] for row in rows} <= set(RUNS.values()):
        raise ValueError("At least one valid new peer required; missing peers must not be invented")
    costs = sum(validate_row(row) for row in rows)
    if audit["status"] == "PASS" and (len(rows) != 2 or costs != 4 or audit.get("failures") != [] or audit.get("skipped") != []):
        raise ValueError("PASS requires both fresh accuracy rows and all four costs")
    if audit["status"] == "PARTIAL_VALID_EVIDENCE_PRESERVED" and not (audit.get("failures") or audit.get("skipped") or audit.get("native_incomplete_outcomes")):
        raise ValueError("Partial evidence needs explicit actual failure or skipped reasons")
    references = audit.get("reference_rows") or []
    if references:
        if (audit.get("reference_status") != "MATCHED_ACTUAL_RAW_G_C32_REFERENCE" or not audit.get("reference_inputs")
                or len({row["run_id"] for row in references}) != len(references) or not {row["run_id"] for row in references} <= set(REFERENCES.values())):
            raise ValueError("Only exact already-audited Raw/G references with bound input SHA are allowed")
        for ref in audit["reference_inputs"]:
            if ref.get("root") != "public" or not ref.get("relative_path", "").startswith(NAME + "/") or not digest(ref.get("sha256")):
                raise ValueError("Reference audit/report/CSV descriptors must be explicit public SHA bindings")
        for row in references:
            validate_row(row, reference=True)
    elif audit.get("reference_status") != "NOT_AVAILABLE_OR_NOT_MATCHED":
        raise ValueError("Unavailable baseline must remain explicitly unavailable")
    first = rows[0]
    for row in [*rows, *references]:
        if any(row.get(key) != first.get(key) for key in ("reference_axis_fingerprints", "band_count_binding_sha256", "n_observed")):
            raise ValueError("Exact same C32 scaler/target/mask/origin/channel cohort required")
        if any(row["fit_receipt"][key] != first["fit_receipt"][key] for key in ("data_sha256", "refit_scaler_sha")):
            raise ValueError("Exact same derived subset identity and A60 scaler required")


def rows_from_audit(audit):
    validate_audit(audit); result = []
    references = {row["family"]: row for row in audit.get("reference_rows") or []}
    fresh = {row["family"]: row for row in audit["rows"]}
    for family in ("RAW", "DIRECT_G", "B0", "B2"):
        original = references.get(family) if family in REFERENCES else fresh.get(family)
        if original is None:
            continue
        reused = family in REFERENCES; fit = original["fit_receipt"]
        row = dict(row_scope="REUSED_REFERENCE" if reused else "NEW_B0_B2", family=family, run_id=original["run_id"],
            accuracy_reused=reused, cost_reused=reused, accuracy_status=original["accuracy_status"], C=32, L=512, H=12, seed=92601,
            K=8 if family in RUNS else 0, F_width=32 if family == "RAW" else (0 if family == "DIRECT_G" else 8),
            MSE=original["mse"], MAE=original["mae"], observed_labels=original["n_observed"], V_origins=148, device="cpu", CPU_threads=2,
            selected_refit_epoch=fit["selected_epoch"], training_status=fit["training_status"], F_identity=original["source_identity"],
            artifact_sha256=original["artifact_sha256"], manifest_sha256=original["manifest_sha256"], band_count_binding_sha256=original["band_count_binding_sha256"],
            parameter_numel=None, parameter_numel_scope=original["parameter_numel_scope"], F_only_resident_bytes=None, F_only_resident_scope=original["F_only_resident_scope"])
        for key in ("search_updates", "search_exposures", "refit_updates", "refit_exposures"):
            row[key] = fit[key]
        for batch in (1, 8):
            prefix = f"B{batch}_"; row[prefix + "status"] = original["cost_status"][str(batch)]
            cost = original["costs"].get(str(batch))
            if cost is None:
                continue
            timed = next(group for group in cost["summaries"] if group["phase"] == "timed"); memory = cost["memory_scope"]
            row.update({prefix + "median_seconds": timed["median_seconds"], prefix + "p90_seconds": timed["p90_seconds"],
                prefix + "resident_CPU_parameter_buffer_MiB": memory["resident_weights_cpu_MiB"], prefix + "foundation_instances_created": memory["foundation_instances_created"],
                prefix + "guard_peak_RSS_bytes": cost["guard_sampled_peak_RSS_bytes"], prefix + "model_load_seconds": cost["model_load_seconds"],
                prefix + "cold_load_in_whole_trial": cost.get("cold_model_load_in_whole_trial"), prefix + "native_F_counters_per_trial": cost["native_F_counters_per_trial"],
                prefix + "derived_50trial_F_counters": {key: cost[key] for key in ("F_logical_streams", "F_physical_streams", "F_calls")}, prefix + "cost_scope": cost.get("recorded_scope")})
        result.append(row)
    return result


def compose(audit, audit_relative, audit_sha):
    rows = rows_from_audit(audit)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n"); writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    fresh = {row["family"]: row for row in rows if row["row_scope"] == "NEW_B0_B2"}
    question = "B0→B2 구조 추가가치는 같은 C32/K8의 실제 오차와 비용으로 판단한다. "
    if len(fresh) == 2:
        question += "반복 개발 V에서 B2−B0 MSE 차이는 `" + repr(fresh["B2"]["MSE"] - fresh["B0"]["MSE"]) + "`다. "
    else:
        question += "한쪽만 유효하므로 B0와 B2의 수치 차이는 아직 계산하지 않는다. "
    text = ["# PeMS08 C32 B0·B2: 구조 추가가치와 압축 대조", "", question + "선택 refit 길이·노출과 용량이 달라 이 차이만으로 구조의 인과 효과나 충분 학습을 확정하지 않는다.", "",
        "상태: `" + audit["status"] + "`. [독립 감사](" + posixpath.relpath(audit_relative, NAME) + ") · [원정밀도 CSV](B0_B2_RESULTS_v1.csv) · [보고 결속](B0_B2_BINDINGS_v1.json)", "",
        "Raw/G 참고 상태: `" + audit.get("reference_status", "NOT_AVAILABLE_OR_NOT_MATCHED") + "`. 기존 참고 행은 재사용이며 새 학습·추론·비용·독립 표본으로 세지 않는다. 참고가 없으면 baseline을 만들지 않는다. 압축의 도움과 Bolt 필요성은 Raw/G와의 실제 조건까지 함께 보아야 하며 없는 비교는 NA다.", "",
        "같은 ID 해시 선택 C32/L512/H12·CPU2·seed92601·V148의 A 정규화 채널 평균 MSE가 주지표이고 MAE는 보조지표다. 같은 target/mask/origin/channel/scaler의 관측 라벨을 반복 사용하므로 독립 관측 수를 더하지 않는다.", ""]
    for row in rows:
        text += ["## " + row["family"] + " / " + row["row_scope"], "", "```text", "정규화 MSE  " + repr(row["MSE"]), "정규화 MAE  " + repr(row["MAE"]),
            "관측 label  " + str(row["observed_labels"]) + " / V148", "search      " + str(row["search_updates"]) + " updates / " + str(row["search_exposures"]) + " exposures",
            "선택 refit  epoch=" + str(row["selected_refit_epoch"]) + " / " + str(row["refit_updates"]) + " updates / " + str(row["refit_exposures"]) + " exposures", "학습 상태   " + row["training_status"]]
        for batch in (1, 8):
            prefix = f"B{batch}_"
            text.append(f"B{batch} 비용    " + row[prefix + "status"] + (" / median=" + repr(row[prefix + "median_seconds"]) + " s / p90=" + repr(row[prefix + "p90_seconds"]) + " s" if prefix + "median_seconds" in row else " / NA: 미완료를0으로 쓰지 않음"))
        text += ["```", ""]
    text += ["B0/B2는 원 PCA 초기화의 K8 표현·미분 가능한 frozen F·학습 lag decoder/current anchor를 쓰는 새 독립 학습이다. 원 search0/.25/1/3과 A_stop 선택을 보존하고 실제 선택0도 유효하다. search3와1577 updates가 선택 refit3나 수렴을 뜻하지 않는다. Raw는 C32 frozen F, 직접 G는 F 없이 학습하는 별도 참고다.", "",
        "비용은 CPU2 physical 입력→채움·정규화→P/E/F/D 또는 Raw/G→FP64 inverse·CPU 반환의 초/batch다. B8은8개 입력 batch 전체이며8로나누지 않는다. warm10·ordinary timed20·별도 intrusive stage20을 분리한다. cold load 범위는 원 기록을 보존하며 stage 합을 whole과 같다고 하지 않는다.", "",
        "Route parameter+buffer CPU resident MiB·관측된 guard RSS·학습 peak·F 단독 resident bytes는 다른 범위다. F-only resident와 parameter numel은 별도 미기록 NA이며 전체 Route 메모리로 채우지 않는다. saved full-V FP64·fresh CPU6·공개 충분통계/trial 검산은 원 감사 증거이고 이 builder가 다시 수행한 실험이 아니다.", "",
        "단일 seed·반복 개발 V·제한 search와 선택 refit이다. 충분 학습·독립 일반화는 미확정이며 C170의 기존 예산과 C32의 새 search는 C만의 공정한 인과 대조가 아니다. upstream 미래 보간·전처리 계보는 UNKNOWN으로 남는다. 전체 subset·MASTER는 미완료다.", "",
        "## 미완료·참조 한계", "", "```json", json.dumps({key: audit.get(key, []) for key in ("failures", "skipped", "native_incomplete_outcomes", "reference_failures")}, ensure_ascii=False, indent=2), "```", "",
        "기존 결과와 checkpoint는 보존한다. 진행량·중단 이유는 원 감사/native 기록만 인용하고 잃은 진행량·재개 성공을 추정하지 않는다. ROOT가 기존 승인 범위·남은 자원·새 출력 경로와 continuation 방법을 확인해야 하며 이 보고는 자동 재개나 추가 자원 승인이 아니다.", "",
        "감사 SHA: `" + audit_sha + "`. 이 builder의 새 Task/NPZ/CP/모델·통계/trial 계산·학습·추론·비용·프로세스·Git·network 작업은0이다.", ""]
    return {OUTPUTS[0]: stream.getvalue(), OUTPUTS[1]: "\n".join(text)}


def execute(public, cache, audit_relative, audit_sha, auditor_sha, builder_sha):
    public, cache = Path(public).resolve(), Path(cache).resolve(); folder = public / NAME
    if any((folder / name).exists() for name in OUTPUTS):
        raise FileExistsError("Existing report outputs preserved before input reads")
    if Path(__file__).resolve() != folder / "build_b0_b2_report.py" or sha(__file__) != builder_sha:
        raise ValueError("Exact reviewed same-namespace builder SHA required")
    helper_path = public / INPUT_HELPER
    if sha(helper_path) != INPUT_HELPER_SHA:
        raise ValueError("Frozen stdlib input helper changed")
    spec = importlib.util.spec_from_file_location("b0_b2_report_inputs", helper_path); helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    inputs = helper.Inputs(public, cache)
    if not audit_relative.startswith(NAME + "/") or Path(audit_relative).suffix != ".json":
        raise ValueError("Same-namespace actual independent audit required")
    audit = inputs.json("public", audit_relative, audit_sha)
    if audit.get("audit_source_sha256") != auditor_sha:
        raise ValueError("Actual audit must bind the current reviewed auditor source")
    for name, expected in ((NAME + "/build_b0_b2_report.py", builder_sha), (NAME + "/audit_b0_b2.py", auditor_sha), (INPUT_HELPER, INPUT_HELPER_SHA)):
        inputs.capture("public", name, expected)
    for ref in [*audit["inputs"], *audit.get("reference_inputs", [])]:
        inputs.capture(ref["root"], ref["relative_path"], ref["sha256"])
    if any(inputs.path("public", name).exists() for name in audit.get("missing_public_native_paths", [])):
        raise ValueError("An audit-observed missing native file appeared after the snapshot")
    texts = compose(audit, audit_relative, audit_sha)
    if not inputs.preserved():
        raise ValueError("Captured audit and reference input bytes changed")
    blobs = {name: text.encode("utf-8") for name, text in texts.items()}
    for name, blob in blobs.items():
        with (folder / name).open("xb") as stream:
            stream.write(blob)
    if not inputs.preserved():
        raise ValueError("Inputs changed; partial CSV/MD preserved without completed bindings")
    binding = dict(schema="PEMS_B0_B2_REPORT_BINDINGS_v1", status="FROM_ACTUAL_PEMS_B0_B2_AUDIT_" + audit["status"],
        audit_relative_path=audit_relative, audit_sha256=audit_sha, auditor_source_sha256=auditor_sha, builder_source_sha256=builder_sha,
        input_SHA_before_after_equal=True, source48_preserved=True,
        inputs=[dict(root=root, relative_path=name, sha256=value) for (root, name), value in sorted(inputs.files.items())],
        outputs=[dict(relative_path=NAME + "/" + name, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)) for name, blob in blobs.items()],
        completed_new_accuracy_rows=len(audit["rows"]), completed_new_cost_rows=sum(status == "PASS_ACTUAL_PHYSICAL_TRIALS" for row in audit["rows"] for status in row["cost_status"].values()),
        reference_status=audit.get("reference_status"), reference_inputs=audit.get("reference_inputs", []), reference_failures=audit.get("reference_failures", []),
        reused_reference_rows=len(audit.get("reference_rows") or []), failures=audit.get("failures", []), skipped=audit.get("skipped", []),
        native_incomplete_outcomes=audit.get("native_incomplete_outcomes", []), full_subset_experiments_complete=False, whole_MASTER_complete=False,
        independent_generalization_claimed=False, new_model_array_statistic_trial_training_inference_cost_process_Git_network_operations=0,
        self_hash_policy="Bindings excludes itself; publication inventory binds its exact bytes")
    with (folder / OUTPUTS[2]).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(binding, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute", action="store_true"); parser.add_argument("--root-triggered", action="store_true")
    for key in ("public-root", "cache-root", "audit-relative", "audit-sha", "auditor-sha", "builder-sha"):
        parser.add_argument("--" + key)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED", input_reads=0, files_written=0,
            numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules, actual_report_written=0))); return 0
    if not args.root_triggered or any(getattr(args, key) is None for key in ("public_root", "cache_root", "audit_relative", "audit_sha", "auditor_sha", "builder_sha")):
        parser.error("Explicit actual audit/current source SHA, roots and ROOT trigger required")
    value = execute(args.public_root, args.cache_root, args.audit_relative, args.audit_sha, args.auditor_sha, args.builder_sha)
    print(json.dumps(dict(status=value["status"], completed_new_accuracy_rows=value["completed_new_accuracy_rows"]))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
