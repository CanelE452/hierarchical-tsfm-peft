"""Material presentation contracts: cohorts, exposure and cost matching."""
import importlib.util
from pathlib import Path

import pytest

from research.adapter_diagnostics_20261017_v2.core import atomic_json


PATH = Path(__file__).resolve().parents[1]/"reports/adapter_diagnostics_20261017_v2/code/report_presentation.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_report_presentation", PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def fixture(tmp_path):
    parent = dict(shape=[100, 170], columns=["s"+str(i) for i in range(170)], L=512, H=12,
                  primary_split_ranges={"A_fit": [0, 50], "A_stop": [50, 60], "V_select": [70, 80]},
                  metadata=dict(source_sha256="a"*64, source_url="https://example.test/source", units="Exact counting units UNKNOWN",
                                historically_unseen_to_researcher="unknown", new_to_campaign=True))
    jena = dict(shape=[100, 21], columns=[str(i) for i in range(21)], L=512, H=48, primary_split_ranges=parent["primary_split_ranges"],
                metadata=dict(source_sha256="b"*64, historically_unseen_to_researcher=False, new_to_campaign=False))
    broad = tmp_path/helper.BROAD
    atomic_json(broad/"DATA_CONTRACT.json", dict(assets=dict(pems08=parent, jena=jena)))
    atomic_json(broad/"priority_pems_subset_20261016_v1/PLAN.json", dict(parent_task_contract=parent,
                original_primary_split_ranges=parent["primary_split_ranges"], column_selection=dict(selected_ids=parent["columns"][:32])))
    audit = dict(status="PASS_WITH_M5_BLOCK", datasets={name: dict(status="COMPLETE", access=dict(new_T_numeric_values_parsed=0), checks=dict(origins={"V_select": 2})) for name in ("jena", "pems08")})
    audit["datasets"]["m5"] = dict(status="BLOCKED_ACCESS_OR_LICENSE", reason="Approved assets absent")
    return audit, dict(runs=[dict(run_id="pems_run", cohort="pems32")], jobs=[], costs=[])


def test_pems32_uses_sealed_selected_ids_and_m5_is_not_accessed(tmp_path):
    audit, data = fixture(tmp_path)
    rows = helper.dataset_protocol_rows(audit, data, tmp_path)
    subset = next(row for row in rows if row["dataset"] == "pems32")
    assert subset["C"] == 32
    assert "subset_plan_sha256=" in subset["version"]
    m5 = next(row for row in rows if row["dataset"] == "m5")
    assert m5["exposure"] == "NOT_ACCESSED" and m5["fit_selection_evaluation"] is None
    assert m5["input_information"] == "NA: not accessed"
    jena = next(row for row in rows if row["dataset"] == "jena")
    assert jena["units"].startswith("UNKNOWN") and jena["original_url"].startswith("UNKNOWN")
    assert jena["historically_unseen_to_researcher"] is False
    assert "LOCKED" in jena["current_protected_T"]
    assert not any(row["exposure"] == "historical T exposed" for row in rows)
    details = helper.data_source_details(audit, data, tmp_path)
    assert details["datasets"]["pems32"]["selected_channel_ids"] == ["s"+str(i) for i in range(32)]


def test_subset_parent_contract_mismatch_is_rejected(tmp_path):
    audit, data = fixture(tmp_path)
    plan = tmp_path/helper.BROAD/"priority_pems_subset_20261016_v1/PLAN.json"
    value = helper.read(plan); value["column_selection"]["selected_ids"][0] = "not-parent"
    atomic_json(plan, value)
    with pytest.raises(ValueError, match="selection"):
        helper.dataset_protocol_rows(audit, data, tmp_path)


def cost_fixture():
    shared = dict(cohort="jena", C=21, L=512, H=48, seed=1, MSE=1., MAE=1., input_info_scope="cohort past only", supervision_scope="A_fit/A_stop/A60")
    a, b = dict(shared, run_id="a", method="B0", trainable_params=10), dict(shared, run_id="b", method="B2", trainable_params=15)
    common = dict(cohort="jena", C=21, L=512, H=48, B=1, threads=2, device="cpu", dtype="float32", scope="physical-to-CPU",
                  exclusive=True, parity_state="PASS", physical_input_sha256="a"*64, observation_mask_sha256="b"*64, supervision_seed=1,
                  source_sha256="c"*64)
    costs = [dict(common, method="B0", median_ms_per_origin=2.), dict(common, method="B2", median_ms_per_origin=3.)]
    pair = dict(reference_run_id="a", method_run_id="b", reference="B0", method="B2")
    return dict(runs=[a, b], endpoints=[], costs=costs), pair


def test_additional_cost_requires_same_facet_and_parameter_bytes_are_not_guessed():
    data, pair = cost_fixture()
    row = helper.mechanism_rows([pair], data)[0]
    assert row["additional_warm_cost_status"] == "MEASURED_SAME_FACET"
    assert row["additional_warm_ms_per_origin"] == 1.
    assert row["additional_trainable_params"] == 5.
    assert row["additional_parameter_bytes"] is None
    for field, replacement in (("dtype", "float64"), ("device", "cuda"), ("B", 8), ("threads", 1),
                               ("scope", "F-only"), ("physical_input_sha256", "c"*64), ("observation_mask_sha256", "d"*64)):
        data, pair = cost_fixture(); data["costs"][1][field] = replacement
        assert helper.mechanism_rows([pair], data)[0]["additional_warm_cost_status"] == "NOT_MEASURED"


def test_explicit_parameter_dtype_can_supply_parameter_bytes():
    data, pair = cost_fixture()
    for run in data["runs"]:
        run["parameter_dtype"] = "float32"
    assert helper.mechanism_rows([pair], data)[0]["additional_parameter_bytes"] == 20.


def test_t1_preserves_accuracy_and_fit_devices_and_missing_timer_is_unknown():
    data, _ = cost_fixture()
    data["runs"][0]["device"] = "cuda"
    rows = [dict(run_id="a", device="cpu", MSE=1., MAE=1., fit_device="cuda", fit_stage_total_seconds=None)]
    enriched = helper.enrich_accuracy_cost_rows(rows, data, dict(cpu_physical=4, cpu_logical=8, device="GPU model"))[0]
    assert enriched["fit_device"] == "cuda" and enriched["device"] == "cpu"
    assert enriched["input_info_scope"] == "cohort past only"
    assert enriched["fit_time_status"].startswith("UNKNOWN")
    assert "not zero" in enriched["fit_time_scope"]
    assert rows[0].get("hardware") is None


def test_coverage_is_question_specific_and_keeps_blocked_comparisons():
    data, _ = cost_fixture()
    data["jobs"] = [dict(job_id="C02_s1", question_id="C02", status="COMPLETE"),
                    dict(job_id="D03_jena", question_id="D03", status="SKIPPED_BUDGET", reason="Reserve")]
    data["runs"] += [dict(run_id="C02_s1_0", method="B2_PCA", MSE=1., MAE=1.), dict(run_id="C02_s1_1", method="B2_QR", MSE=1., MAE=1.)]
    rows = helper.coverage_rows([("Q3", "init", "observed"), ("Q5", "native", "pending")], data)
    assert rows[0]["satisfaction_status"] == "SATISFIED_REGISTERED_SCOPE"
    assert rows[1]["satisfaction_status"] == "NOT_VERIFIED"
    assert rows[1]["blocked_comparison"][0]["reason"] == "Reserve"
    assert "Separate approved budget" in rows[1]["next_execution"][0]


def test_green_job_without_scored_pair_is_not_satisfied():
    data = dict(runs=[dict(run_id="C02_s1_0", method="B2_PCA", MSE=1., MAE=1.)], costs=[],
                jobs=[dict(job_id="C02_s1", question_id="C02", status="COMPLETE")])
    row = helper.coverage_rows([("Q3", "init", "observed")], data)[0]
    assert row["satisfaction_status"] == "PARTIAL"
    assert row["blocked_comparison"][0]["status"] == "MISSING_SCORED_EVIDENCE"
    assert "no new training or measurement" in row["next_execution"][0]


def test_unbound_or_unverified_costs_never_supply_additional_latency():
    for field, replacement in (("source_sha256", None), ("exclusive", False), ("parity_state", "FAILED")):
        data, pair = cost_fixture(); data["costs"][1][field] = replacement
        assert helper.mechanism_rows([pair], data)[0]["additional_warm_cost_status"] == "NOT_MEASURED"


def test_q6_includes_both_registered_e04_jobs_and_requires_actual_cost_metadata():
    data, _ = cost_fixture()
    data["jobs"] = [dict(job_id="E04_first", question_id="E04", status="COMPLETE"),
                    dict(job_id="E04_second", question_id="E04", status="COMPLETE")]
    data["costs"][0]["job_id"] = "E04_first"
    data["costs"][1]["job_id"] = "E04_second"
    row = helper.coverage_rows([("Q6", "cost", "measured")], data)[0]
    assert row["registered_jobs"] == 2 and row["completed_jobs"] == 2
    assert row["satisfaction_status"] == "SATISFIED_REGISTERED_SCOPE"
    assert row["measured_cost_job_ids"] == ["E04_first", "E04_second"]
    data["costs"].pop()
    row = helper.coverage_rows([("Q6", "cost", "missing")], data)[0]
    assert row["satisfaction_status"] == "PARTIAL"
    assert row["blocked_comparison"][0]["job_id"] == "E04_second"
    assert row["blocked_comparison"][0]["status"] == "MISSING_VERIFIED_COST_METADATA"


def test_corrected_cost_preserves_failed_attempt_without_suggesting_it_is_still_blocked():
    data, _ = cost_fixture()
    data["jobs"] = [dict(job_id="E04_old", question_id="E04", status="FAILED", reason="Parent mismatch"),
                    dict(job_id="E04_corrected", question_id="E04", status="COMPLETE")]
    data["plan"] = dict(jobs=[dict(job_id="E04_corrected", correction_lineage=dict(parent_job_id="E04_old"))])
    data["costs"][0]["job_id"] = "E04_corrected"
    row = helper.coverage_rows([("Q6", "cost", "measured")], data)[0]
    assert row["satisfaction_status"] == "SATISFIED_REGISTERED_SCOPE"
    assert row["completed_jobs"] == 1 and row["registered_jobs"] == 2
    assert row["superseded_attempts"] == [dict(job_id="E04_old", status="FAILED", resolved_by="E04_corrected")]
    assert data["jobs"][0]["status"] == "FAILED"


def local_m5(data):
    data["m5_local_assets"] = dict(status="LOCAL_ASSETS_SCHEMA_VERIFIED", checked_csv_files=4, checked_archives=1,
        inventory_binding=dict(reference="cache://m5_local_assets/INVENTORY.json", sha256="f"*64),
        source_files=[dict(filename="sales_train_validation.csv", absolute_path="PRIVATE_PATH_MUST_NOT_APPEAR",
                           day_column_range=dict(first="d_1", last="d_1913", count=1913))],
        new_numeric_values_parsed=0, no_M5_evaluation=True, no_fitting=True)


def test_verified_m5_local_assets_replace_current_absence_without_inventing_protocol_or_scoring(tmp_path):
    audit, data = fixture(tmp_path)
    local_m5(data)
    details = helper.data_source_details(audit, data, tmp_path)["datasets"]["m5"]
    assert details["status"] == "LOCAL_ASSETS_SCHEMA_VERIFIED"
    assert details["data_access"] == "HEADER_SCHEMA_HASH_ONLY"
    assert details["checked_csv_files"] == 4 and details["checked_archives"] == 1
    assert details["inventory_sha256"] == "f"*64
    assert details["new_numeric_values_parsed"] == 0 and details["fit_status"] == details["evaluation_status"] == "NOT_RUN"
    assert details["historical_access_record"]["status"] == "BLOCKED_ACCESS_OR_LICENSE"
    row = next(item for item in helper.dataset_protocol_rows(audit, data, tmp_path) if item["dataset"] == "m5")
    assert row["C"] == row["L"] == row["H"] == "NOT_REGISTERED"
    assert "NOT_REGISTERED" in row["fit_selection_evaluation"] and "NOT_RUN" in row["fit_selection_evaluation"]
    assert row["historically_unseen_to_researcher"] == row["new_to_campaign"] == "UNKNOWN"
    text = helper.data_audit_supplement(audit, data, tmp_path)+helper.readme_supplement(data)
    assert "M5_LOCAL_ASSET_DISCOVERY.json" in text and "M5_LOCAL_ASSETS_KO.md" in text
    assert "PRIVATE_PATH_MUST_NOT_APPEAR" not in repr(details)+repr(row)+text
    assert audit["datasets"]["m5"]["status"] == "BLOCKED_ACCESS_OR_LICENSE"


def test_q1_metadata_only_m5_availability_removes_current_access_block_and_limits_completion(tmp_path):
    audit, data = fixture(tmp_path)
    data["jobs"] = [dict(job_id="A01_data", question_id="A01", status="COMPLETE"),
                    dict(job_id="A04_solver", question_id="A04", status="COMPLETE")]
    before = helper.coverage_rows([("Q1", "data", "global audit")], data, audit)[0]
    assert before["blocked_comparison"][0]["job_id"] == "M5_access"
    local_m5(data)
    after = helper.coverage_rows([("Q1", "data", "global audit")], data, audit)[0]
    assert after["blocked_comparison"] == []
    assert after["satisfaction_status"] == "SATISFIED_REGISTERED_SCOPE"
    assert "M5" in after["claim_limit"] and "NOT_RUN" in after["claim_limit"]
    assert after["metadata_only_datasets"] == ["m5"]
    assert "A01/A04" in after["question_scope"]
