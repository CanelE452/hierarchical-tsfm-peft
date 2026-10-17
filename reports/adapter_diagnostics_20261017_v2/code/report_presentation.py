"""Presentation metadata from existing contracts and bound results; never writes files."""
import hashlib
import json
import math
from pathlib import Path
import re

from research.adapter_diagnostics_20261017_v2.core import read, sha


BROAD = "research/informs_broad_validation_20261015_v1"
REPORT = "reports/adapter_diagnostics_20261017_v2"
UNKNOWN = "UNKNOWN: absent from the pinned local source metadata; no external inference"
SUCCESS = {"COMPLETE", "REUSED_VERIFIED"}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _canonical(method):
    return {"Bolt_Raw": "Raw", "Bolt_B0": "B0", "Bolt_B2": "B2", "B2_CAP_POINTWISE": "CAP",
            "RAW": "Raw", "RAW_RIDGE": "Ridge", "RIDGE": "Ridge"}.get(method, method)


def _source(path, repo):
    path, repo = Path(path).resolve(), Path(repo).resolve()
    if not path.is_relative_to(repo):
        raise ValueError("Presentation source escapes repository")
    return dict(reference="repo://"+path.relative_to(repo).as_posix(), sha256=sha(path))


def _m5_available(data):
    return data.get("m5_local_assets", {}).get("status") == "LOCAL_ASSETS_SCHEMA_VERIFIED"


def _m5_metadata_note():
    return ("M5: 로컬 파일의 헤더 schema·SHA를 확인했다. "
            "[M5_LOCAL_ASSET_DISCOVERY.json](M5_LOCAL_ASSET_DISCOVERY.json)과 "
            "[M5_LOCAL_ASSETS_KO.md](M5_LOCAL_ASSETS_KO.md)에 근거와 범위를 보존한다. "
            "숫자 행 파싱=0, fit/evaluation=NOT_RUN, C/L/H·split=NOT_REGISTERED다. "
            "성능·자료 품질·단위·첫 사용·사전학습 미노출을 확인한 결과가 아니다. "
            "원 감사의 2026-10-16 HTTP 401·등록 inventory 부재 기록은 과거 접근 기록으로 보존하며, "
            "현재 전체 PC의 파일 부재나 라이선스 거부를 뜻하지 않는다.")


def data_source_details(audit, data, repo):
    repo = Path(repo)
    contract_path = repo/BROAD/"DATA_CONTRACT.json"
    contracts = read(contract_path)["assets"]
    sources, datasets = [_source(contract_path, repo)], {}
    for name, row in audit["datasets"].items():
        if name == "m5":
            if _m5_available(data):
                local = data["m5_local_assets"]
                datasets[name] = dict(status=local["status"], data_access="HEADER_SCHEMA_HASH_ONLY",
                    checked_csv_files=local["checked_csv_files"], checked_archives=local["checked_archives"],
                    inventory_sha256=local["inventory_binding"]["sha256"], source_sha256=None,
                    source_reference="M5_LOCAL_ASSET_DISCOVERY.json", original_url=UNKNOWN, units=UNKNOWN,
                    C="NOT_REGISTERED", L="NOT_REGISTERED", H="NOT_REGISTERED", split="NOT_REGISTERED",
                    exposure="HEADER_HASH_METADATA_ONLY; numeric row values parsed=0; M5 fit/evaluation=NOT_RUN",
                    historically_unseen_to_researcher="UNKNOWN", new_to_campaign="UNKNOWN",
                    current_protected_T="NOT_REGISTERED: no M5 split/evaluation; numeric row values parsed=0",
                    new_numeric_values_parsed=local["new_numeric_values_parsed"], fit_status="NOT_RUN", evaluation_status="NOT_RUN",
                    historical_access_record=dict(date="2026-10-16", status=row["status"], reason=row.get("reason"),
                        scope="Historical HTTP/registered-inventory access record; no current whole-PC absence or license-denial conclusion"),
                    raw_publication_allowed=False,
                    claim_limit="Verified local asset headers/schema/hash only; no M5 performance or data-quality evaluation",
                    source_details="M5_LOCAL_ASSET_DISCOVERY.json", scope_note="M5_LOCAL_ASSETS_KO.md")
                continue
            datasets[name] = dict(status=row["status"], data_access="NOT_ACCESSED", original_url="UNKNOWN: approved assets not available",
                                  units="NA: not accessed", source_sha256=None, split=None, exposure="NOT_ACCESSED",
                                  historically_unseen_to_researcher="NA: not accessed", new_to_campaign="NA: not accessed",
                                  current_protected_T="NOT_ACCESSED", reason=row.get("reason"))
            continue
        contract = contracts[name]
        metadata = contract.get("metadata", {})
        access = row.get("access", {})
        parsed = access.get("new_T_numeric_values_parsed")
        datasets[name] = dict(status=row["status"], data_access="AUDITED_PRE_T_PREFIX" if row["status"] == "COMPLETE" else row["status"],
            source_sha256=metadata.get("source_sha256", access.get("source_sha256")),
            source_reference=metadata.get("source_reference", UNKNOWN), original_url=metadata.get("source_url", UNKNOWN),
            units=metadata.get("units", UNKNOWN), target_semantics=metadata.get("target_semantics", UNKNOWN),
            ancestry=metadata.get("ancestry", metadata.get("preprocessing_ancestry", UNKNOWN)),
            permitted_source_bounds=metadata.get("permitted_source_bounds", [0, contract["shape"][0]]),
            materialized_pre_T_rows=access.get("materialized_rows"), split=contract["primary_split_ranges"],
            C=contract["shape"][1], L=contract["L"], H=contract["H"], interval_seconds=contract.get("interval_seconds"),
            historically_unseen_to_researcher=metadata.get("historically_unseen_to_researcher", "UNKNOWN"),
            new_to_campaign=metadata.get("new_to_campaign", "UNKNOWN"),
            novelty_flag_scope="Inherited DATA_CONTRACT source-campaign flags; not a claim of first use in the current v2 campaign",
            historical_T_exposure="UNKNOWN per-dataset count; the historical global 46 conditions are not apportioned to every dataset",
            current_protected_T="LOCKED; new numeric T values parsed=0" if parsed == 0 else "UNKNOWN: no zero-parse access evidence",
            protected_tail_policy=metadata.get("protected_tail_policy", "Current pre-T access only; original protected tail remains locked"),
            pretraining_overlap=metadata.get("pretraining_overlap_known", "UNKNOWN"),
            raw_publication_allowed=False)
    if any(run.get("cohort") == "pems32" for run in data.get("runs", [])) or any(job.get("dataset_cohort") == "pems32" for job in data.get("jobs", [])):
        plan_path = repo/BROAD/"priority_pems_subset_20261016_v1/PLAN.json"
        plan = read(plan_path)
        ids = plan["column_selection"]["selected_ids"]
        parent = contracts["pems08"]
        if (len(ids) != 32 or len(set(ids)) != 32 or any(identifier not in parent["columns"] for identifier in ids)
                or plan["parent_task_contract"] != parent
                or plan["original_primary_split_ranges"] != parent["primary_split_ranges"]):
            raise ValueError("PeMS32 sealed selection or parent contract mismatch")
        sources.append(_source(plan_path, repo))
        datasets["pems32"] = dict(datasets["pems08"], C=len(ids), parent_dataset="pems08", selected_channel_ids=ids,
            subset_plan_sha256=sha(plan_path), selected_ids_metadata_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            channel_selection="Exactly the metadata-selected ordered PLAN.selected_ids; distinct from C170",
            historical_T_exposure="C32 post-T development subset; no numeric T evaluation in this campaign; historical parent exposure does not imply C32 T scoring")
    return dict(schema="DATA_SOURCE_DETAILS_v1", datasets=datasets, source_bindings=sources,
                units_note="Accuracy is normalized observed-channel macro MSE/MAE; source physical units are not inferred from channel names",
                original_url_policy="UNKNOWN when absent locally; no new web investigation or guessed upstream URL")


def dataset_protocol_rows(audit, data, repo):
    details = data_source_details(audit, data, repo)
    rows = []
    for name, value in details["datasets"].items():
        if name == "m5":
            if _m5_available(data):
                rows.append(dict(dataset=name, version="inventory_sha256="+value["inventory_sha256"]+"; local header/schema/hash verified",
                    C=value["C"], L=value["L"], H=value["H"], input_information="NOT_REGISTERED: header/hash metadata only; numeric row values parsed=0",
                    fit_selection_evaluation="NOT_REGISTERED; M5 fit/evaluation NOT_RUN", origins="NOT_REGISTERED", exposure=value["exposure"],
                    status=value["status"], original_url=value["original_url"], units=value["units"],
                    historically_unseen_to_researcher=value["historically_unseen_to_researcher"], new_to_campaign=value["new_to_campaign"],
                    current_protected_T=value["current_protected_T"], historical_access_record=value["historical_access_record"],
                    source_details="M5_LOCAL_ASSET_DISCOVERY.json", scope_note="M5_LOCAL_ASSETS_KO.md", claim_limit=value["claim_limit"]))
                continue
            rows.append(dict(dataset=name, version="NOT_ACCESSED", C=None, L=None, H=None, input_information="NA: not accessed",
                             fit_selection_evaluation=None, origins=None, exposure="NOT_ACCESSED", status=value["status"],
                             original_url=value["original_url"], units=value["units"], historically_unseen_to_researcher=value["historically_unseen_to_researcher"],
                             new_to_campaign=value["new_to_campaign"], current_protected_T=value["current_protected_T"], source_details="DATA_SOURCE_DETAILS.json#m5"))
            continue
        parent = value.get("parent_dataset", name)
        source = audit["datasets"][parent]
        version = "source_sha256="+str(value["source_sha256"])+"; bounds="+json.dumps(value["permitted_source_bounds"])
        if value.get("subset_plan_sha256"):
            version += "; subset_plan_sha256="+value["subset_plan_sha256"]
        rows.append(dict(dataset=name, version=version, C=value["C"], L=value["L"], H=value["H"],
            input_information="Past observed history of this cohort only; causal fill; no future covariates",
            fit_selection_evaluation=value["split"], origins=source.get("checks", {}).get("origins"),
            exposure="Repeated development V; "+value["historical_T_exposure"]+"; "+value["current_protected_T"], status=value["status"],
            original_url=value["original_url"], units=value["units"], source_sha256=value["source_sha256"],
            historically_unseen_to_researcher=value["historically_unseen_to_researcher"], new_to_campaign=value["new_to_campaign"],
            novelty_flag_scope=value["novelty_flag_scope"], current_protected_T=value["current_protected_T"],
            source_details="DATA_SOURCE_DETAILS.json#"+name))
    return rows


def data_audit_supplement(audit, data, repo):
    details = data_source_details(audit, data, repo)
    paragraphs = ["## 자료 출처·단위와 노출 상태 보완", "[DATA_SOURCE_DETAILS.json](DATA_SOURCE_DETAILS.json)에 실제 로컬 source SHA·허용 범위·URL·단위·노출 근거를 보존한다. 채널 이름에서 물리 단위를 추정하지 않는다."]
    for name, value in details["datasets"].items():
        if name == "m5":
            paragraphs.append(_m5_metadata_note() if _m5_available(data) else
                              "M5: NOT_ACCESSED. 입력·split·단위·노출은 NA이며 다른 자료의 과거 T 노출을 적용하지 않는다.")
        else:
            paragraphs.append(name+": source SHA="+str(value["source_sha256"])+"; original URL="+value["original_url"]+
                "; units="+value["units"]+". historically_unseen="+str(value["historically_unseen_to_researcher"])+
                "; new_to_source_campaign="+str(value["new_to_campaign"])+"; "+value["current_protected_T"]+". "+value["historical_T_exposure"]+".")
    paragraphs.append("new_to_campaign은 원 DATA_CONTRACT의 과거 캠페인 플래그다. 이번 v2의 신규 독립 자료 또는 사전학습 미노출을 뜻하지 않는다. 정규화 MSE/MAE를 물리 단위 오차로 부르지 않는다.")
    return "\n\n".join(paragraphs)+"\n"


def enrich_accuracy_cost_rows(rows, data, hardware):
    runs = {run["run_id"]: run for run in data["runs"]}
    output = []
    for row in rows:
        run = runs[row["run_id"]]
        value = dict(row)
        device = row.get("device") or run.get("device")
        gpu = str(device).startswith("cuda")
        value.update(hardware=hardware.get("device", "UNKNOWN GPU model") if gpu else
                     "CPU model UNKNOWN; physical cores="+str(hardware.get("cpu_physical", "UNKNOWN"))+"; logical="+str(hardware.get("cpu_logical", "UNKNOWN")),
                     hardware_source="HARDWARE.json", hardware_observed_utc=hardware.get("observed_utc"),
                     input_info_scope=run.get("input_info_scope", "UNKNOWN: input-information scope absent from source"),
                     supervision_scope=run.get("supervision_scope", "UNKNOWN: supervision scope absent from source"),
                     point_statistic=run.get("point_statistic", "UNKNOWN"), metric_scale=run.get("metric_scale", "UNKNOWN"),
                     accuracy_units="Normalized observed-channel macro MSE/MAE; not physical units", latency_units="ms/origin",
                     seed_scope="Source/evaluation seed retained; Raw has no target-training seed" if _canonical(run["method"]) in ("Raw", "Moirai_Raw", "Chronos2_nativeRaw") else
                     "Deterministic supervised regression; source/evaluation seed retained, no random optimizer seed" if _canonical(run["method"]) == "Ridge" else "Supervised adapter seed as recorded",
                     fit_time_status="MEASURED_STAGE_SCOPE" if _number(row.get("fit_stage_total_seconds")) is not None else "UNKNOWN: no bound search+selection/refit stage timing for this run",
                     cold_load_status="MEASURED_SHARED_SCOPE" if _number(row.get("cold_shared_load_seconds")) is not None else
                     "MEASURED_METHOD_SCOPE" if _number(row.get("cold_load_seconds")) is not None else "UNKNOWN: no bound cold/model-load timer for this row")
        if not value.get("fit_time_scope"):
            value["fit_time_scope"] = "UNKNOWN: no source-bound training/preparation timer; missing is not zero"
        if not value.get("cold_load_scope"):
            value["cold_load_scope"] = "UNKNOWN: per-method versus shared cold-load scope not supplied"
        output.append(value)
    return output


def _cost_matches(cost, run):
    if (cost.get("cohort") != run.get("cohort") or _canonical(cost.get("method")) != _canonical(run.get("method"))
            or any(_number(cost.get(key)) is None or _number(cost[key]) != _number(run.get(key)) for key in ("C", "L", "H"))):
        return False
    seed = cost.get("supervision_seed")
    if seed is not None and _number(seed) != _number(run.get("seed")):
        return False
    if cost.get("accuracy_archive_sha256") and cost["accuracy_archive_sha256"] != run.get("independent_private_score_sha256"):
        return False
    for key in ("MSE", "MAE"):
        if _number(cost.get("accuracy_"+key)) is not None and abs(float(cost["accuracy_"+key])-float(run[key])) > 1e-10:
            return False
    return not str(cost.get("accuracy_source", "")).startswith("NA:")


def _facet(cost):
    if (cost.get("exclusive") is not True or cost.get("parity_state") != "PASS"
            or _number(cost.get("median_ms_per_origin")) is None or float(cost["median_ms_per_origin"]) <= 0
            or not re.fullmatch("[a-f0-9]{64}", str(cost.get("source_sha256", "")))
            or any(not re.fullmatch("[a-f0-9]{64}", str(cost.get(field, ""))) for field in ("physical_input_sha256", "observation_mask_sha256"))):
        return None
    numeric = tuple(_number(cost.get(field)) for field in ("B", "C", "L", "H", "threads"))
    text = tuple(cost.get(field) for field in ("cohort", "device", "dtype", "physical_input_sha256", "observation_mask_sha256", "scope"))
    if any(value is None for value in numeric) or any(value in (None, "", "source_bound", "UNKNOWN") for value in text):
        return None
    return (*numeric, *text)


def _parameter_bytes(run):
    if _number(run.get("parameter_bytes")) is not None:
        return float(run["parameter_bytes"])
    widths = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
    width = widths.get(run.get("parameter_dtype"))
    count = _number(run.get("trainable_params"))
    return width*count if width is not None and count is not None else None


def mechanism_rows(pairs, data):
    runs = {run["run_id"]: run for run in data["runs"]+data.get("endpoints", [])}
    output = []
    for pair in pairs:
        left, right = (runs.get(pair[key]) for key in ("reference_run_id", "method_run_id"))
        row = dict(pair)
        if left is None or right is None:
            row.update(additional_warm_cost_status="NOT_MEASURED", additional_warm_cost_reason="Comparison run record unavailable")
            output.append(row); continue
        for field, label in (("trainable_params", "additional_trainable_params"), ("deployed_trainable_params", "additional_deployed_trainable_params")):
            a, b = _number(left.get(field)), _number(right.get(field))
            row[label] = b-a if a is not None and b is not None else None
        row["parameter_delta_scope"] = "Same source field subtraction; B03 trainable_params is D block; deployed totals include retained P/E"
        a, b = _parameter_bytes(left), _parameter_bytes(right)
        row["additional_parameter_bytes"] = b-a if a is not None and b is not None else None
        row["parameter_byte_status"] = "SOURCE_BOUND_PARAMETER_STORAGE" if a is not None and b is not None else "UNKNOWN: parameter storage dtype/bytes unrecorded; prediction dtype does not determine parameter bytes"
        costs = [[cost for cost in data["costs"] if _cost_matches(cost, run) and _facet(cost) is not None] for run in (left, right)]
        matched = [(a, b) for a in costs[0] for b in costs[1] if _facet(a) == _facet(b)]
        if not matched:
            row.update(additional_warm_ms_per_origin=None, additional_warm_cost_status="NOT_MEASURED",
                       additional_warm_cost_reason="No pair of exclusive parity-PASS measured costs with identical device/B/dtype/physical input/mask/scope/threads/cohort/C/L/H")
            output.append(row)
        else:
            for a, b in matched:
                output.append(dict(row, additional_warm_ms_per_origin=float(b["median_ms_per_origin"])-float(a["median_ms_per_origin"]),
                    additional_warm_cost_status="MEASURED_SAME_FACET", additional_warm_cost_reason=None,
                    cost_B=a["B"], cost_device=a["device"], cost_dtype=a["dtype"], cost_threads=a["threads"], cost_scope=a["scope"],
                    physical_input_sha256=a["physical_input_sha256"], observation_mask_sha256=a["observation_mask_sha256"],
                    reference_cost_source_sha256=a.get("source_sha256"), method_cost_source_sha256=b.get("source_sha256")))
    return output


def coverage_rows(questions, data, audit=None):
    families = {"Q1": {"A01", "A04"}, "Q2": {"A02", "A03", "B01", "B02", "B03", "C00", "C01"},
                "Q3": {"C02"}, "Q4": {"D01", "D02"}, "Q5": {"D01", "D03"}, "Q6": {"E01", "E02", "E03", "E04"}}
    output = []
    successful = {job["job_id"] for job in data["jobs"] if job["status"] in SUCCESS}
    corrected = {job["correction_lineage"]["parent_job_id"]: job["job_id"] for job in data.get("plan", {}).get("jobs", [])
                 if job["job_id"] in successful and job.get("correction_lineage", {}).get("parent_job_id")}
    for question, title, body in questions:
        jobs = [job for job in data["jobs"] if job["question_id"] in families[question]
                and not (job["question_id"] == "D01" and ("moirai" not in job["job_id"] if question == "Q4" else "chronos" not in job["job_id"]))]
        blocked = [dict(job_id=job["job_id"], status=job["status"], reason=job.get("reason") or "Registered comparison has no successful terminal result") for job in jobs if job["status"] not in SUCCESS and job["job_id"] not in corrected]
        superseded = [dict(job_id=job["job_id"], status=job["status"], resolved_by=corrected[job["job_id"]]) for job in jobs if job["job_id"] in corrected]
        if question == "Q1" and not _m5_available(data) and audit and audit["datasets"].get("m5", {}).get("status") == "BLOCKED_ACCESS_OR_LICENSE":
            blocked.append(dict(job_id="M5_access", status="BLOCKED_ACCESS_OR_LICENSE", reason=audit["datasets"]["m5"].get("reason")))
        ids = [job["job_id"] for job in jobs]
        scored = [run for run in data["runs"]+data.get("endpoints", []) if all(_number(run.get(key)) is not None for key in ("MSE", "MAE"))]
        evidence = [run["run_id"] for run in scored if any(run["run_id"] == identifier or run["run_id"].startswith(identifier+"_") for identifier in ids)]
        evidence_costs = [cost.get("job_id") for cost in data["costs"] if cost.get("job_id") in ids and _facet(cost) is not None]
        required = {"C02": {"B2_PCA", "B2_QR"}, "D02": {"Moirai_Raw", "Moirai_B0", "Moirai_B2"}, "D03": {"Chronos2_nativeRaw"}}
        for job in jobs:
            if job["status"] not in SUCCESS:
                continue
            expected = required.get(job["question_id"])
            if expected:
                actual = {run.get("method") for run in scored if run["run_id"] == job["job_id"] or run["run_id"].startswith(job["job_id"]+"_")}
                missing = expected-actual
                if missing:
                    blocked.append(dict(job_id=job["job_id"], status="MISSING_SCORED_EVIDENCE", reason="Required scored methods absent: "+", ".join(sorted(missing))))
            if question == "Q6" and job["job_id"] not in evidence_costs:
                blocked.append(dict(job_id=job["job_id"], status="MISSING_VERIFIED_COST_METADATA", reason="No exclusive parity-PASS source-bound cost with explicit measured facet metadata"))
        has_evidence = bool(evidence or evidence_costs or (question == "Q1" and audit and audit.get("status") == "PASS_WITH_M5_BLOCK"))
        status = "SATISFIED_REGISTERED_SCOPE" if jobs and not blocked and has_evidence else "PARTIAL" if has_evidence else "NOT_VERIFIED"
        output.append(dict(question=question, goal=title, evidence_status=body, satisfaction_status=status,
            scientific_status="SCIENTIFIC_COMPLETE_REGISTERED_SCOPE" if status == "SATISFIED_REGISTERED_SCOPE" else "SCIENTIFIC_PARTIAL",
            completed_jobs=sum(job["status"] in SUCCESS for job in jobs), registered_jobs=len(jobs), evidence_run_ids=evidence,
            metadata_only_datasets=["m5"] if question == "Q1" and _m5_available(data) else [],
            question_scope="A01/A04 registered global data/solver audit; M5 local headers/schema/hash only" if question == "Q1" and _m5_available(data) else
                           "Registered question jobs and their measured/scored evidence",
            superseded_attempts=superseded,
            measured_cost_job_ids=sorted(set(evidence_costs)), blocked_comparison=blocked,
            next_execution=[("Reconcile existing bound saved results/metadata for "+item["job_id"]+"; no new training or measurement" if item["status"].startswith("MISSING_") else
                             "Supply approved original assets/access; no mirror or new download inferred" if item["job_id"] == "M5_access" else
                             "Separate approved budget and bound assets required before "+item["job_id"]+"; retain existing deadline and consult NEXT_STEPS_KO.md") for item in blocked]
                           or ["No uncompleted registered comparison; any new experiment requires a separately approved scope"],
            claim_limit="Completion of this diagnostic scope does not prove the hypothesis, convergence, or independent generalization"+
                        ("; M5 fit/evaluation NOT_RUN; no M5 data-quality or performance validation" if question == "Q1" else "")))
    return output


def readme_supplement(data, publication=None):
    publication = publication or {}
    paragraphs = []
    if _m5_available(data):
        paragraphs.append(_m5_metadata_note())
    for device in ("cpu", "cuda"):
        costs = [cost for cost in data["costs"] if cost.get("cohort") == "jena" and cost.get("device") == device
                 and _number(cost.get("B")) == 1 and _canonical(cost.get("method")) in ("Raw", "B2", "Ridge") and _facet(cost) is not None]
        representatives = {}
        for cost in costs:
            representatives.setdefault(_canonical(cost["method"]), cost)
        costs = list(representatives.values())
        if costs:
            paragraphs.append("[확인] Jena "+device+" B1 실측 warm ms/origin: "+"; ".join(_canonical(cost["method"])+"="+repr(float(cost["median_ms_per_origin"]))+" ("+str(cost["dtype"])+")" for cost in costs)+". 장치·dtype·scope가 같은 행만 공정 비교하며, B1을 다른 batch로 일반화하지 않는다.")
    incomplete = [job for job in data["jobs"] if job["status"] not in SUCCESS]
    paragraphs.append("미완료: "+("; ".join(job["job_id"]+"="+job["status"]+" ("+str(job.get("reason") or "Reason in JOB_STATUS.csv")+")" for job in incomplete) if incomplete else "등록 job 모두 성공 상태; 과학적 주장 범위는 질문별 판단을 따른다.") )
    successful = {job["job_id"] for job in data["jobs"] if job["status"] in SUCCESS}
    corrections = [job for job in data.get("plan", {}).get("jobs", []) if job["job_id"] in successful and job.get("correction_lineage")]
    if corrections:
        paragraphs.append("실패 시도는 그대로 보존하며 교정 실행으로 해결했다: "+"; ".join(job["correction_lineage"]["parent_job_id"]+" → "+job["job_id"]+" COMPLETE" for job in corrections)+". T3의 superseded_attempts와 원 receipt를 함께 확인한다.")
    base = publication.get("repository_url", "https://github.com/CanelE452/hierarchical-tsfm-peft").rstrip("/")
    paragraphs.append("[원격 main 보고서]("+base+"/tree/main/"+REPORT+"). "+
                      ("현재 payload 원격 확인 SHA="+str(publication.get("remote_main"))+"." if publication.get("current_payload_verified") is True else
                       "이번 추가 산출물의 main 포함 여부는 정상 push 후 별도 원격 SHA 조회로 확인한다."))
    paragraphs.append("[T1 정확도·비용](PPT_T1_ACCURACY_COST.csv) · [T2 메커니즘](PPT_T2_MECHANISM_ABLATION.csv) · [T3 근거·미완료](PPT_T3_COVERAGE_CLAIMS.csv) · [T4 자료·프로토콜](PPT_T4_DATASET_PROTOCOL.csv)")
    return "\n\n".join(paragraphs)+"\n"
