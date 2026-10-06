"""Audit the saved resource stop without executing model, fit, TEST, or bootstrap code."""
import argparse
import csv
import json
import re
import subprocess
import time

import numpy as np

from runtime_v18 import DATASETS, HERE, ROOT, artifact, array_hash, digest, read_json, resolve, save_json, score_arrays


class Checks:
    def __init__(self):
        self.rows = []
        self.sources = {}

    def add(self, name, condition, details=None):
        status = "incomplete" if condition is None else "pass" if condition else "fail"
        self.rows.append({"name": name, "status": status, "details": details})

    def load(self, name):
        path = HERE / name
        if not path.exists():
            self.add("artifact_present:" + name, None)
            return None
        self.sources[name] = artifact(path)
        return read_json(path)

    def receipt(self, receipt, name):
        try:
            actual = artifact(receipt["path"], receipt["sha256"])
            self.add(name, actual["bytes"] == receipt["bytes"], actual)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self.add(name, False, str(error))


def provenance(checks):
    initial, source, protocol, phase0 = (checks.load(n) for n in
        ("initial_manifest.json", "source_binding.json", "protocol.json", "phase0_audit.json"))
    if source and protocol:
        pin = source["repository"]["commit"]
        checks.add("official_AdaPTS_revision_and_primary_publication", len(pin) == 40 and pin != "main",
                   {"revision": pin, "publication": source["publication"]["primary_url"],
                    "pin_scope": source["repository"]["pin_scope"]})
        checks.add("matched_control_not_full_AdaPTS_reproduction", protocol["full_AdaPTS_reproduction"] is False
                   and source["matched_control_distinctions"]["encoder_decoder_bias"] is False)
    if phase0:
        checks.add("duplicate_execution_audit_preserved", phase0["status"] == "complete_read_only_audit",
                   {"decision": phase0["decision"], "activity": phase0["activity"]})
    try:
        result = subprocess.run(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30, check=True)
        fields = result.stdout.split()
        checks.add("live_remote_main_SHA", len(fields) == 2 and len(fields[0]) == 40,
                   {"live_origin_main": fields[0] if fields else None,
                    "initial_origin_main": initial.get("live_origin_main") if initial else None})
    except (OSError, subprocess.SubprocessError) as error:
        checks.add("live_remote_main_SHA", None, str(error))
    if initial:
        receipt = initial["protected_receipt"]
        checks.receipt(receipt, "protected_before_receipt_bound")
        protected = read_json(receipt["path"])
        changed = [name for name, old in protected.items() if not resolve(name).is_file()
                   or resolve(name).stat().st_size != old["bytes"] or digest(name) != old["sha256"]]
        checks.add("unrelated_files_and_original_PPT_PDF_unchanged", not changed,
                   {"protected_count": len(protected), "changed": changed})


def preflight_checks(checks, preflight):
    parents = checks.load("parent_reuse_manifest.json")
    initial = checks.load("preflight_initial_models.json")
    for receipt in preflight["source_artifacts"]:
        checks.receipt(receipt, "scientific_source_unchanged:" + receipt["path"])
    checks.receipt(preflight["initial_models"], "preflight_initial_models_bound")
    checks.add("preflight_is_not_fit_TEST_or_optimizer_update", preflight["new_fits"] == 0
               and preflight["optimizer_updates"] == 0 and preflight["test_accessed"] is False)
    checks.add("FP32_TF32_off", preflight["environment"]["dtype"] == "float32" and preflight["environment"]["tf32"] is False)
    for dataset in DATASETS:
        row = preflight["units"][dataset]
        contract = checks.load(f"data_contract_{dataset}.json")
        if not parents or not initial or not contract:
            continue
        for key in ("data", "contract", "step0_prediction"):
            checks.receipt(row[key], key + "_bound:" + dataset)
        receipt = row["model_receipt"]
        for key in ("checkpoint", "restore_module_artifact"):
            checks.receipt(receipt["backbone_source"][key], "parent_" + key + "_bound:" + dataset)
        basis_hash = contract["array_contract"]["trainval"]["value_hashes"]["basis"]
        checks.add("literal_parent_PCA_E_D_initialization:" + dataset,
                   row["literal_PCA"]["sha256"] == basis_hash == parents["units"][dataset]["basis"]["sha256"]
                   == receipt["basis_values_sha256"] == initial["units"][dataset]["literal_PCA"]["sha256"]
                   and row["adapter_initial_sha256"] == initial["units"][dataset]["adapter_initial_sha256"])
        checks.add("only_E_D_trainable_frozen_backbone_eval_autograd:" + dataset,
                   set(receipt["trainable_parameter_names"]) == {"encoder.weight", "decoder.weight"}
                   and receipt["trainable_parameter_count"] == 2 * contract["c"] * contract["k"]
                   and receipt["bias"] is False and receipt["backbone_weights_trainable"] is False
                   and receipt["backbone_eval_mode"] is True and receipt["backbone_autograd_operations"] is True
                   and receipt["G"] is None and receipt["level_persistence"] is False)
        state = row["state_preservation"]
        checks.add("backbone_and_E_D_state_unchanged_zero_updates:" + dataset,
                   state["unchanged"] is True and state["before_sha256"] == state["after_sha256"]
                   and state["backbone_before_sha256"] == state["backbone_after_sha256"]
                   == receipt["backbone_state_sha256"] and state["optimizer_updates"] == 0)
        probe = row["autograd_probe"]
        checks.add("E_D_finite_nonzero_gradients_microbatch4_2_1:" + dataset,
                   {r["microbatch"] for r in probe["microbatch_records"]} == {1, 2, 4}
                   and all(r["optimizer_updates"] == 0 and r["effective_batch_origins"] == 4
                           and set(r["gradients"]) == {"encoder.weight", "decoder.weight"}
                           and all(g["present"] and g["finite"] and g["nonzero"] > 0 for g in r["gradients"].values())
                           for r in probe["microbatch_records"])
                   and probe["optimizer_updates"] == 0 and probe["backbone_gradients_absent"] is True
                   and probe["full_effective_count_denominator"] is True)
        checks.add("recorded_microbatch_gradient_parity:" + dataset,
                   len(probe["gradient_parity"]) == 6 and all(r["max_absolute"] <= r["atol"] for r in probe["gradient_parity"]),
                   "Execution receipt; this audit does not re-execute gradients")
        sampling = row["sampling"]
        checks.receipt(sampling["source"], "literal_parent_sampler_bound:" + dataset)
        checks.add("all_240_parent_seed_epoch_schedules:" + dataset,
                   sampling["all_240_seed_epoch_schedules_literal_parent_identity"] is True
                   and {(r["seed"], r["epoch"]) for r in sampling["checks"]}
                   == {(seed, epoch) for seed in (92601, 92602) for epoch in range(1, 121)},
                   {"historical_saved_epoch_hashes": sum(r["recorded_parent_hash_checked"] for r in sampling["checks"]),
                    "other_epochs": "Compared with the literal parent sampler during executed preflight"})
        with np.load(resolve(row["step0_prediction"]["path"]), allow_pickle=False) as saved:
            prediction, reference, origins = saved["prediction"], saved["compress"], saved["origins"]
        with np.load(resolve(row["data"]["path"]), allow_pickle=False) as data:
            canonical, basis = data["val_origins"], data["basis"]
            target = np.stack([data["x"][int(o):int(o) + 48] for o in origins])
            mask = np.stack([data["finite"][int(o):int(o) + 48] for o in origins])
            first = np.asarray(probe["origins"], dtype=canonical.dtype)
            x_probe = np.stack([data["x"][int(o) - 512:int(o)] for o in first]).astype(np.float32)
            y_probe = np.stack([data["x"][int(o):int(o) + 48] for o in first])
            mask_probe = np.stack([data["finite"][int(o):int(o) + 48] for o in first])
        checks.add("stored_VAL_origins_shape_dtype_finite_PCA:" + dataset,
                   np.array_equal(origins, canonical) and origins.tolist() == contract["origins"]["val"]["values"]
                   and array_hash(origins) == row["VAL_origin_values_sha256"]
                   and prediction.shape == reference.shape == (len(origins), 48, contract["c"])
                   and prediction.dtype == reference.dtype == np.float32 and np.isfinite(prediction).all()
                   and np.isfinite(reference).all() and array_hash(basis) == basis_hash)
        checks.add("preflight_TRAIN_probe_input_target_mask_hashes:" + dataset,
                   array_hash(x_probe) == probe["input_values_sha256"] and array_hash(y_probe) == probe["target_values_sha256"]
                   and array_hash(mask_probe) == probe["mask_values_sha256"])
        parity = row["step0_COMPRESS_parity"]
        delta = np.abs(prediction.astype(np.float64) - reference.astype(np.float64))
        checks.add("step0_COMPRESS_saved_parity_replay:" + dataset, parity["pass"] is True
                   and np.all(delta <= parity["atol"] + parity["rtol"] * np.abs(reference))
                   and float(delta.max()) == parity["max_absolute"],
                   {"max_absolute": float(delta.max()), "atol": parity["atol"], "rtol": parity["rtol"]})
        score = score_arrays(prediction, target, mask)
        checks.add("stored_step0_VAL_macro_sums_counts_replay:" + dataset,
                   all(np.allclose(score[k], row["step0_full_VAL_score"][k], atol=1e-12, rtol=1e-12) for k in score),
                   "Verification of saved VAL scores only; no new scientific metrics or TEST")
        checks.add("recorded_same_path_replay_and_B1_B4_origin_order:" + dataset,
                   row["replay_parity"]["pass"] is True and row["single_B4_origin_order_parity"]["pass"] is True)


def resource_checks(checks, preflight, stop, ledger):
    estimate = preflight["estimate"]
    summed = estimate["measured_preflight_gpu_job_s"] + sum(r["four_fits_upper_s"] + r["post_training_upper_s"]
                                                           for r in estimate["units"].values())
    checks.add("resource_gate_failure_and_honest_incomplete_execution", preflight["status"] == "STOP_RESOURCE"
               and preflight["all_checks_passed"] is False and stop["status"] == "BLOCKED_STOP_RESOURCE"
               and stop["resource_gate_passed"] is False and stop["full_campaign_complete"] is False
               and stop["scientific_verdict"] is None)
    checks.add("full_120_epoch_projection_not_early_stop_or_reduced_scope", estimate["assumed_early_stopping"] is False
               and estimate["scope_reduced"] is False and all(r["full_epochs_without_early_stopping"] == 120
               and r["optimizer_updates_per_epoch"] == 128 and r["VAL_passes_including_step0"] == 121 for r in estimate["units"].values()))
    checks.add("projection_sum_cap_excess_replay", abs(summed - estimate["projected_full_campaign_upper_s"]) < 1e-8
               and summed > estimate["hard_cap_s"] == 10800 and abs(stop["projected_excess_s"] - (summed - 10800)) < 1e-8,
               {"projected_s": summed, "cap_s": 10800, "excess_s": summed - 10800})
    checks.add("genuine_executed_fit_TEST_optimizer_counts_zero", len(ledger["fits"]) == len(ledger["test_predictions"]) == 0
               and stop["new_fits"] == stop["new_test_predictions"] == stop["optimizer_updates"] == 0)
    jobs = ledger["jobs"]
    checks.add("failed_preflight_job_traceback_and_actual_time_preserved", len(jobs) == 1 and jobs[0]["status"] == "failed"
               and "STOP_RESOURCE" in jobs[0].get("error", "") and "Traceback" in preflight.get("error", "")
               and jobs[0].get("optimizer_updates") == 0 and abs(jobs[0]["elapsed_s"] - stop["preflight_actual_gpu_job_wall_s"]) < 1e-8
               and jobs[0]["elapsed_s"] < 10800)
    checks.receipt(stop["failed_attempt"], "failed_partial_preflight_attempt_preserved")
    checks.add("partial_attempt_not_replaced_with_success", digest(HERE / "preflight_attempt01.json") == digest(HERE / "preflight.json"))
    absent = ("selection.json", "selection_seal.json", "test_exposure.json", "prediction_manifest.json", "evaluation.json",
              "accuracy_models.csv", "accuracy_channels.csv", "paired_comparisons.csv", "training_cost_probe.json",
              "training_summary.csv", "inference_cost_rows.json", "inference_cost_summary.json", "mechanism_diagnostics.json")
    checks.add("gate_prevented_scientific_outputs_not_fabricated", not any((HERE / n).exists() for n in absent), list(absent))
    for name in ("only_E_D_updated_in_fit", "selected_non_step0_actual_updates", "VAL_only_selection",
                 "TEST_after_joint_selection_seal", "TEST_origins_masks_exact", "A_B_combined_aggregation",
                 "no_prediction_ensemble", "paired_bootstrap_pairing_exact", "matched_cost_input_hash_exact",
                 "training_probe_zero_updates_state_unchanged", "all_cost_raw_blocks_complete", "selected_checkpoint_diagnostics"):
        checks.rows.append({"name": name, "status": "gate_prevented", "details": "No trained or selected ED model exists"})
    checks.add("next_budget_plan_not_granted_or_executed", stop["minimum_next_plan"]["new_permission_granted"] is False
               and stop["minimum_next_plan"]["recipe_or_scientific_scope_changes"] is False)


def report_checks(checks, preflight):
    for name in ("FAIRNESS_TABLE.md", "FINAL_REPORT_KO.md", "PRESENTATION_PATCH.md", "README.md"):
        path = HERE / name
        checks.add("report_present:" + name, True if path.exists() else None)
        if path.exists():
            checks.sources[name] = artifact(path)
    path = HERE / "FINAL_REPORT_KO.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        checks.add("blocked_no_scientific_verdict_independent_novelty_claim",
                   all(s in text for s in ("BLOCKED_STOP_RESOURCE", "자체 검산", "독립 재현", "full AdaPTS", "unavailable", "keep:", "shrink:", "hold:")))
    summary = checks.load("report_summary.json")
    if summary:
        checks.add("report_status_values_replay", summary["status"] == "BLOCKED_STOP_RESOURCE"
                   and summary["full_campaign_complete"] is False and summary["scientific_verdict"] is None
                   and summary["new_fits"] == summary["new_test_predictions"] == 0
                   and summary["projected_full_campaign_upper_s"] == preflight["estimate"]["projected_full_campaign_upper_s"])
        source = summary["reused_LEVEL_source"]
        checks.receipt(source, "historical_LEVEL_table_source_bound")
        with resolve(source["path"]).open(encoding="utf-8", newline="") as stream:
            old = list(csv.DictReader(stream))
        for r in summary["reused_LEVEL_rows"]:
            old_row = old[r["csv_record"]]
            checks.add("historical_LEVEL_exact_table_row:" + r["dataset"] + ":" + r["period"], old_row["dataset"] == r["dataset"]
                       and old_row["method"] == "BOLT_SMALL_LEVEL" and old_row["aggregation"] != "individual"
                       and old_row["period"] == r["period"] and float(old_row["mse"]) == r["mse"] and float(old_row["mae"]) == r["mae"])
    manifest = checks.load("figures/manifest.json")
    if manifest:
        checks.add("two_stored_evidence_figures_no_model_fit_TEST", manifest["model_forwards"] == manifest["fits"]
                   == manifest["new_test_predictions"] == 0 and len(manifest["figures"]) == 2)
        for receipt in manifest["sources"].values():
            checks.receipt(receipt, "figure_source_bound:" + receipt["path"])
        for figure in manifest["figures"]:
            for receipt in figure["outputs"]:
                checks.receipt(receipt, "figure_output_bound:" + receipt["path"])
        with (HERE / "preflight_summary.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        checks.add("preflight_plot_table_values_replay", len(rows) == 3 and all(
                   float(r["max_absolute_error"]) == preflight["units"][r["dataset"]]["step0_COMPRESS_parity"]["max_absolute"]
                   and int(r["ED_trainable_parameters"]) == preflight["units"][r["dataset"]]["model_receipt"]["trainable_parameter_count"] for r in rows))
        with (HERE / "resource_projection.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        checks.add("resource_plot_table_sum_replay", len(rows) == 5 and abs(sum(float(r["projected_seconds"]) for r in rows)
                   - preflight["estimate"]["projected_full_campaign_upper_s"]) < 1e-8)


def source_review_checks(checks):
    if not (HERE / "source_review_checks.json").exists():
        return
    review = checks.load("source_review_checks.json")
    checks.add("post_stop_source_review_not_scientific_execution", review["status"] == "PASS_SOURCE_REVIEW"
               and review["execution_status"] == "BLOCKED_STOP_RESOURCE"
               and review["full_campaign_complete"] is False and review["scientific_verdict"] is None
               and all(value == 0 for value in review["additional_scientific_execution"].values()),
               "Static review and isolated synthetic CPU checks; no matched experiment executed")
    checks.add("post_stop_original_budget_and_scientific_contract_preserved", review["limits"]
               == {"gpu_seconds": 10800, "new_fits": 12, "new_test_predictions": 6}
               and review["budget_amendment_approved"] is False)
    for receipt in review["sources_after"].values():
        checks.receipt(receipt, "reviewed_source_bound:" + receipt["path"])
    for name, expected in review["scientific_and_figure_sha256_unchanged"].items():
        checks.add("post_stop_existing_evidence_unchanged:" + name, digest(HERE / name) == expected)
    for row in review["isolated_CPU_checks"]:
        checks.receipt(row["receipt"], "isolated_CPU_test_receipt_bound:" + row["scope"])
        for receipt in row["test_sources"]:
            checks.receipt(receipt, "isolated_CPU_test_source_bound:" + receipt["path"])
        checks.add("isolated_CPU_checks_passed:" + row["scope"], row["passed"] is True,
                   "Fake fixtures only; does not validate real learned-model predictions or timing")


def request_coverage():
    lines = (HERE / "REQUEST.txt").read_text(encoding="utf-8").splitlines()
    starts = [(0, "preamble")] + [(i, line) for i, line in enumerate(lines) if re.match(r"^\d+\. ", line)]
    deferred = {5, 6, 7, 8, 9, 10, 11, 12, 18}
    coverage = []
    for position, (start, title) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        number = int(title.split(".", 1)[0]) if title != "preamble" else -1
        coverage.append({"section": title, "lines": [start + 1, end],
                         "status": "gate_prevented_or_partial" if number in deferred else "audited_stop_scope",
                         "meaning": "Coverage index, not a claim that every requested scientific result executed"})
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    checks = Checks()
    provenance(checks)
    preflight, stop, ledger = (checks.load(n) for n in ("preflight.json", "resource_stop.json", "ledger.json"))
    if preflight and stop and ledger:
        if preflight["status"] != "STOP_RESOURCE":
            checks.add("supported_resource_stop_state", False, "A future complete experiment needs its full result verifier")
        else:
            preflight_checks(checks, preflight)
            resource_checks(checks, preflight, stop, ledger)
            if not args.data_only:
                report_checks(checks, preflight)
                source_review_checks(checks)
    counts = {s: sum(r["status"] == s for r in checks.rows) for s in ("pass", "fail", "incomplete", "gate_prevented")}
    status = "FAIL" if counts["fail"] else "INCOMPLETE" if counts["incomplete"] else "PASS"
    save_json(HERE / "final_checks.json", {"schema": "matched_learned_ed_resource_stop_checks_v18", "status": status,
              "audit_status": status, "execution_status": "BLOCKED_STOP_RESOURCE", "full_campaign_complete": False,
              "scientific_verdict": None, "verified_utc": time.time(), "counts": counts, "checks": checks.rows,
              "sources": checks.sources, "request_coverage": request_coverage(),
              "scope": "Saved preflight and honest stop self-check, not full completion, independent reproduction, or novelty certification",
              "model_forwards": 0, "fits": 0, "new_test_predictions": 0, "new_bootstrap_draws": 0,
              "saved_VAL_score_replay_only": True, "data_only": args.data_only})
    print(json.dumps({"audit_status": status, "execution_status": "BLOCKED_STOP_RESOURCE", "counts": counts}))
    raise SystemExit(0 if status == "PASS" else 3)


if __name__ == "__main__":
    main()
