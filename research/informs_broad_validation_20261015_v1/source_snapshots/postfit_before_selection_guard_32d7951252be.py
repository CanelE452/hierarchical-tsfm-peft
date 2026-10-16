"""Registered validation predictions and out-of-time combination execution.

No T_final route is exposed by this driver. Final selection and T scoring are
separate commands after the immutable selection lock has been written.
"""
import argparse
import csv
import gc
import gzip
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import campaign
import combination
import data_contract as data
import fit_control
from gpu_runtime import FrozenBolt
import prediction_cache
import raw_calibration
import ordered_pilot
import registry_extension
import audit_campaign
from resources import BudgetExceeded, GPUReservation, check_storage_budget

HERE, CACHE, ASSET_ROOT = campaign.HERE, campaign.CACHE, campaign.ASSET_ROOT


def parent_id(ds, family, k=0, fold="primary"):
    return f"{fold}_{ds}_E1_{family}_k{k}_s92601"


class ParentPool:
    def __init__(self, task, fold="primary", deadline=None):
        self.task, self.fold = task, fold
        self.deadline = deadline
        self.foundation = None
        self.model_size = None

    def get_foundation(self, size="small"):
        if self.foundation is None or self.model_size != size:
            self.close()
            if size == "small":
                path = ASSET_ROOT / ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots" / campaign.SMALL_REV
            else:
                assets = json.loads((HERE / "MODEL_ASSETS.json").read_text(encoding="utf-8"))
                # Asset entries are deliberately resolved from their verified receipt.
                entries = assets.get("models", assets)
                if isinstance(entries, list):
                    entry = next(v for v in entries if v.get("size") == size or v.get("model_id", "").endswith("-" + size))
                else:
                    entry = entries.get(size, entries.get("chronos-bolt-" + size))
                if entry is None:
                    raise ValueError("Pinned official model asset receipt absent")
                path = CACHE / entry["cache_relative_path"]
            self.foundation = FrozenBolt(path, self.task["H"], 128, False)
            self.model_size = size
        return self.foundation

    def predict(self, run_id, split="V_select", dense=False):
        rows = {r["experiment_id"]: r for r in campaign.registry_rows()}
        row = rows[run_id]
        family = row["family"]
        baseline, role = None, "base_parent"
        needs_f, size = False, "small"
        if family.startswith("RAW_BOLT_"):
            baseline = "RAW"
            size = family.removeprefix("RAW_BOLT_").lower()
            needs_f = True
        elif family == "CURRENT":
            baseline = "CURRENT"
        elif family == "SEASONAL_NAIVE":
            baseline = dict(kind="SEASONAL", period=int(self.task["period"]))
        elif family == "RAW_RIDGE":
            pass
        else:
            needs_f = family not in {"DIRECT_G", "DIRECT_B2", "DIRECT_B4", "G_REFIT_APLUSM"}
            if family in {"G_REFIT_APLUSM", "LORA_REFIT_APLUSM"}:
                role = "AplusM_control"
        f = self.get_foundation(size) if needs_f else None
        result = prediction_cache.predict_parent(run_id, self.task, split, CACHE, HERE, f,
                                                 dense=dense, baseline=baseline, fold=self.fold, role=role, deadline=self.deadline)
        if baseline is not None or family == "RAW_RIDGE":
            campaign.record_status(run_id, execution_status="COMPLETE_VALID", training_status="CLOSED_FORM_A_ONLY" if family == "RAW_RIDGE" else "NO_TRAINING",
                                   validity_status="PENDING_INDEPENDENT_VALIDATION", reason="Fixed baseline V prediction; no T accessed")
        return result

    def close(self):
        if self.foundation is not None:
            self.foundation.assert_frozen_unchanged()
            self.foundation = None
            gc.collect()
            torch.cuda.empty_cache()
        self.model_size = None


def predict_validation(ds, lease, include_small=True):
    task = data.load_task(ds)
    pool = ParentPool(task, deadline=lease.deadline - 5)
    try:
        for row in campaign.registry_rows():
            if row["dataset"] != ds or row["fold"] != "primary" or row["block"] not in {"E1", "E3"}:
                continue
            family = row["family"]
            ready = (HERE / "runs" / row["experiment_id"] / "BASE_FIT_RECEIPT.json").is_file()
            ready |= (HERE / "runs" / row["experiment_id"] / "RIDGE_FIT_RECEIPT.json").is_file()
            ready |= family in {"RAW_BOLT_SMALL", "CURRENT", "SEASONAL_NAIVE"}
            ready |= include_small and family in {"RAW_BOLT_MINI", "RAW_BOLT_TINY"}
            if not ready:
                continue
            lease.check()
            check_storage_budget(CACHE, pending_bytes=128 * 1024 ** 2)
            print(json.dumps(dict(event="V_PREDICT_START", run_id=row["experiment_id"])), flush=True)
            result = pool.predict(row["experiment_id"])
            print(json.dumps(dict(event="V_PREDICT_DONE", run_id=row["experiment_id"], score=result["score"])), flush=True)
    finally:
        pool.close()


def _validate_cpu_baseline_identity(row, manifest):
    metadata = manifest["metadata"]
    # Registry names the method; fixed-baseline caches name its execution kind.
    expected_family = {"CURRENT": "CURRENT", "SEASONAL_NAIVE": "SEASONAL", "RAW_RIDGE": "RAW_RIDGE"}[row["family"]]
    expected_cohort = f"{row['dataset']}:{row['fold']}:V_select:stride{int(row['horizon'])}"
    if (manifest.get("run_id") != row["experiment_id"]
            or metadata.get("run_id") != row["experiment_id"]
            or manifest.get("split") != "V_select" or metadata.get("split") != "V_select"
            or manifest.get("cohort") != expected_cohort or metadata.get("cohort") != expected_cohort
            or metadata.get("family") != expected_family or metadata.get("fold") != row["fold"]
            or metadata.get("dense") is not False or metadata.get("role") != "base_parent"
            or not manifest.get("data_sha") or metadata.get("data_sha") != manifest["data_sha"]
            or metadata.get("data_sha256") != manifest["data_sha"]):
        raise ValueError("CPU baseline manifest family/split/data identity differs")


def reconcile_cpu_baselines(lease):
    """Audit already-written CPU baselines; registry writes require owner lock."""
    lease.check()
    if not getattr(lease, "_entered", False) or getattr(getattr(lease, "lock", None), "file", None) is None:
        raise ValueError("CPU baseline reconciliation requires the active exclusive owner lease")
    rows = campaign.registry_rows()
    registry = {row["experiment_id"]: row for row in rows}
    allowed = {parent_id(ds, family) for ds in ("beijing", "jena", "electricity", "pems08")
               for family in ("CURRENT", "SEASONAL_NAIVE", "RAW_RIDGE")}
    candidates = {}
    source_sha = data.sha256_file(Path(__file__).with_name("cpu_baselines.py"))
    for path in sorted((HERE / "execution_batches").glob("cpu_baselines*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if batch.get("status") != "ACTUAL_V_INFERENCE_COMPLETE" or batch.get("GPU_calls") != 0 or batch.get("T_final_accessed") is not False or batch.get("source_sha256") != source_sha:
            raise ValueError("CPU baseline execution receipt is not the pinned complete CPU-only batch")
        for output in batch["outputs"]:
            run_id = output["run_id"]
            if run_id not in allowed or run_id not in registry:
                raise ValueError("CPU baseline reconciliation cannot change another route")
            row = registry[run_id]
            if row["family"] not in {"CURRENT", "SEASONAL_NAIVE", "RAW_RIDGE"} or row["fold"] != "primary" or row["block"] != "E1":
                raise ValueError("CPU baseline registered scientific identity differs")
            relative = f"predictions/{run_id}/V_select/PREDICTION_MANIFEST.json"
            if output.get("prediction_manifest_relative_path") != relative or output.get("actual_F_streams") != 0:
                raise ValueError("CPU baseline receipt references another route or F execution")
            manifest_path = HERE / relative
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reference = json.loads(manifest_path.with_name("REFERENCE_CONTRACT.json").read_text(encoding="utf-8"))
            cost = manifest["cost_record"]
            _validate_cpu_baseline_identity(row, manifest)
            if cost.get("actual_F_streams") != 0 or cost.get("expected_F_streams") != 0 or reference.get("expected_F_streams") != 0:
                raise ValueError("CPU baseline inference must have zero foundation streams")
            if manifest["score"] != output["score"]:
                raise ValueError("CPU batch score no longer matches its saved manifest")
            score = manifest["score"]
            if int(score["horizon"]) != int(row["horizon"]) or int(score["num_channels"]) != int(row["num_channels"]):
                raise ValueError("CPU baseline dimensions differ from registered task")
            stats_path = audit_campaign._inside(HERE, manifest["public_stats_relative_path"])
            if stats_path != manifest_path.with_name("PUBLIC_ERROR_STATS.csv.gz").resolve():
                raise ValueError("CPU baseline statistics must belong to the same route")
            with gzip.open(stats_path, "rt", encoding="utf-8", newline="") as stream:
                statistics = list(csv.DictReader(stream))
            with manifest_path.with_name("SCORES.csv").open(encoding="utf-8", newline="") as stream:
                public_scores = list(csv.DictReader(stream))
            if len(public_scores) != 1:
                raise ValueError("CPU baseline requires exactly one saved score row")
            local_registry = [dict(run_id=r["experiment_id"], execution_status="COMPLETE_VALID" if r["experiment_id"] == run_id else r["execution_status"],
                                   reason=r.get("reason") or "Outside this CPU-baseline reconciliation") for r in rows]
            audit = audit_campaign.independent.audit_prediction_bundle(manifest, reference, public_scores[0], statistics,
                        local_registry, list(registry), [cost], CACHE)
            previous = candidates.get(run_id)
            signature = (data.sha256_file(manifest_path), manifest["archive_sha256"])
            if previous is not None and previous["signature"] != signature:
                raise ValueError("CPU batch receipts disagree on the same route")
            candidates[run_id] = dict(row=row, signature=signature, audit=audit)
    # Validate every candidate before the first registry mutation.
    for run_id, candidate in candidates.items():
        lease.check()
        if candidate["row"]["execution_status"] != "COMPLETE_VALID":
            family = candidate["row"]["family"]
            campaign.record_status(run_id, execution_status="COMPLETE_VALID",
                                   training_status="CLOSED_FORM_A_ONLY" if family == "RAW_RIDGE" else "NO_TRAINING",
                                   validity_status="PENDING_INDEPENDENT_VALIDATION",
                                   reason="CPU V cache SHA, independent saved-reference rescoring and public-statistics reaggregation passed; zero F streams; T untouched")
    result = dict(event="CPU_BASELINES_RECONCILED", reconciled_run_ids=sorted(candidates),
                  prediction_rescoring_passed=len(candidates), saved_reference_audit=True, checkpoint_replay=False)
    print(json.dumps(result), flush=True)
    return result


def _combine_row(row, task, pool, gid, g_m, g_v, mean, std, lease, endpoints):
    family = row["family"]
    suffix = row["experiment_id"].split("s92601_", 1)[1]
    if family == "PAST_CONTROL":
        b_m, b_v, bid = g_m, g_v, gid
    else:
        parent_family = "RAW_BOLT_SMALL" if suffix in {"RAW", "RAW_BOLT_SMALL"} else suffix
        bid = parent_id(task["name"], parent_family, 0 if parent_family.startswith("RAW_BOLT") else int(row["latent_k"]))
        b_m, b_v = pool.predict(bid, "M_cal", True), pool.predict(bid)
    if family == "RAW_CALIBRATED":
        refs = audit_campaign.prepare_combination_reference(row["experiment_id"], task, b_v, b_v, mean, std,
                                                              CACHE, HERE, spec=dict(variant="raw_calibration"), kind="raw_calibration")
        receipt = raw_calibration.fit_raw_calibration(row["experiment_id"], task, b_m, b_v, mean, std, CACHE, HERE)
        audit_campaign.wrap_combination_output(receipt, refs, CACHE, HERE, kind="raw_calibration")
        campaign.record_status(row["experiment_id"], execution_status=receipt["status"], training_status=receipt["training_status"],
                               validity_status="PENDING_INDEPENDENT_VALIDATION", reason="M-only per-channel Raw bias correction; no V/T parameter fit")
        return
    variant = {"ALPHA_FIT": "alpha_family", "CONDITIONAL": "conditional", "PAST_CONTROL": "past_control", "LEGACY_RESIDUAL": "legacy_residual"}[family]
    spec = dict(run_id=row["experiment_id"], variant=variant, seed=int(row["seed"]), device="cuda",
                learning_rate=.001, epoch_endpoints=list(endpoints), effective_batch=16, microbatch=4,
                parent_run_id=bid, g_run_id=gid, fold=row["fold"])
    refs = audit_campaign.prepare_combination_reference(row["experiment_id"], task, b_v, g_v, mean, std,
                                                          CACHE, HERE, spec=spec)
    print(json.dumps(dict(event="COMBINATION_START", run_id=row["experiment_id"], variant=variant)), flush=True)
    receipt_path = HERE / "combinations" / row["experiment_id"] / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        combination.validate_completed_receipt(receipt, spec, task, b_m, b_v, g_m, g_v, mean, std, CACHE)
    else:
        receipt = combination.fit_combination(spec, task, b_m, b_v, g_m, g_v, mean, std, CACHE,
                                               HERE / "combinations", lease.deadline - 5)
    status = receipt["execution_status"]
    if status == "COMPLETE_VALID":
        audit_campaign.wrap_combination_output(receipt, refs, CACHE, HERE)
    campaign.record_status(row["experiment_id"], execution_status=status, training_status=receipt["training_status"],
                           validity_status="PENDING_INDEPENDENT_VALIDATION", reason="M-only calibration with frozen A parents; V-only selection; T untouched")
    if family == "ALPHA_FIT" and status == "COMPLETE_VALID":
        for label in ("ALPHA0", "ALPHA05", "ALPHA1"):
            alias = row["experiment_id"].replace("_ALPHA_FIT_", "_" + label + "_")
            campaign.record_status(alias, execution_status="COMPLETE_VALID", training_status="NO_TRAINING",
                                   validity_status="PENDING_INDEPENDENT_VALIDATION", reason="Fixed alpha from shared frozen-parent family receipt " + row["experiment_id"])
    print(json.dumps(dict(event="COMBINATION_DONE", run_id=row["experiment_id"], status=status,
                          updates=receipt["updates"], training_status=receipt["training_status"])), flush=True)


def combine(ds, lease, endpoints):
    task = data.load_task(ds)
    task = dict(task, split_ranges=data.split_ranges(task))
    mean, std, _ = data.fit_statistics(task, task["split_ranges"]["A_stop"][1])
    rows = [r for r in campaign.registry_rows() if r["dataset"] == ds and r["fold"] == "primary" and r["block"] == "E2"]
    pool = ParentPool(task, deadline=lease.deadline - 5)
    g_m = g_v = g_error = None
    g_attempted = False
    gid = parent_id(ds, "DIRECT_G")
    try:
        for row in rows:
            family = row["family"]
            if row["execution_status"] != "PENDING_QUEUE" or family not in {"ALPHA_FIT", "CONDITIONAL", "PAST_CONTROL", "LEGACY_RESIDUAL", "RAW_CALIBRATED"}:
                continue
            lease.check()
            check_storage_budget(CACHE, pending_bytes=512 * 1024 ** 2)
            try:
                if family != "RAW_CALIBRATED":
                    if not g_attempted:
                        g_attempted = True
                        try:
                            g_m, g_v = pool.predict(gid, "M_cal", True), pool.predict(gid)
                        except (BudgetExceeded, TimeoutError):
                            raise
                        except Exception as error:
                            g_error = error
                    if g_error is not None:
                        raise RuntimeError("MISSING_OR_INVALID_G_PARENT") from g_error
                _combine_row(row, task, pool, gid, g_m, g_v, mean, std, lease, endpoints)
            except (BudgetExceeded, TimeoutError):
                raise
            except Exception as error:
                directory = CACHE / "failures"
                directory.mkdir(parents=True, exist_ok=True)
                campaign.new_json(directory / (row["experiment_id"] + "_" + str(time.time_ns()) + ".json"),
                                  dict(run_id=row["experiment_id"], error_type=type(error).__name__, error=str(error),
                                       traceback=traceback.format_exc(), independent_E2_rows_continue=True))
                campaign.record_status(row["experiment_id"], execution_status="EXECUTION_FAILED",
                                       validity_status="INVALID_OR_NOT_COMPLETED", reason=type(error).__name__ +
                                       ("; MISSING_OR_INVALID_G_PARENT" if g_error is not None and family != "RAW_CALIBRATED" else "") +
                                       "; private traceback preserved; independent E2 rows continue")
                print(json.dumps(dict(event="COMBINATION_FAILED", run_id=row["experiment_id"], error_type=type(error).__name__)), flush=True)
            finally:
                gc.collect()
    finally:
        pool.close()


def controls(ds, lease, endpoints):
    task = data.load_task(ds)
    pool = ParentPool(task, deadline=lease.deadline - 5)
    try:
        for row in campaign.registry_rows():
            if row["dataset"] != ds or row["fold"] != "primary" or row["family"] not in {"G_REFIT_APLUSM", "LORA_REFIT_APLUSM"} or row["execution_status"] != "PENDING_QUEUE":
                continue
            lease.check()
            family = "DIRECT_G" if row["family"] == "G_REFIT_APLUSM" else "LORA"
            spec = dict(run_id=row["experiment_id"], family=family, seed=int(row["seed"]), fold="primary", role="AplusM_control",
                        epochs=list(endpoints), lr=.0001 if family == "LORA" else .001, device="cuda", microbatch=4,
                        effective_batch=16, F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16)
            f = None if family == "DIRECT_G" else pool.get_foundation()
            print(json.dumps(dict(event="CONTROL_START", run_id=row["experiment_id"])), flush=True)
            receipt = fit_control.fit_control(spec, task, CACHE, HERE, f, lease.deadline - 5)
            campaign.record_status(row["experiment_id"], execution_status=receipt["status"], training_status=receipt["training_status"],
                                   validity_status="PENDING_INDEPENDENT_VALIDATION", reason="Fresh A+M training; normalization A60; V checkpoint selection")
            if receipt["status"] == "COMPLETE_VALID":
                result = pool.predict(row["experiment_id"])
                print(json.dumps(dict(event="CONTROL_DONE", run_id=row["experiment_id"], score=result["score"])), flush=True)
    finally:
        pool.close()


def ordered(ds, lease, endpoints):
    task = data.load_task(ds)
    pool = ParentPool(task, deadline=lease.deadline - 5)
    grid = ordered_pilot.k_grid(task)
    try:
        candidates = [pool.predict(parent_id(ds, family, k)) for family in ("B0", "B2") for k in grid]
        parent_run = f"primary_{ds}_E4_ORDERED_MULTI_K_k{len(task['columns'])}_s92601"
        template = next(row for row in campaign.registry_rows() if row["experiment_id"] == parent_run)
        aliases = [dict(template, experiment_id=parent_run + f"__prefix_k{k}", latent_k=k,
                        execution_status="PENDING_QUEUE", training_status="NOT_STARTED", validity_status="NOT_CHECKED",
                        requirement="MANDATORY_DERIVED_PREFIX", reason="Predeclared shared checkpoint prefix at registered K",
                        resume_command=f"python postfit.py --phase ordered --dataset {ds} --epochs " + " ".join(map(str, endpoints)))
                   for k in grid if k != len(task["columns"])]
        registry_extension.register(HERE, aliases, "E4 already-authorized ordered prefix outputs; no new independent training family")
        lease.check()
        print(json.dumps(dict(event="ORDERED_PILOT_START", dataset=ds, k_grid=grid)), flush=True)
        result = ordered_pilot.run_ordered_pilot(task, candidates, CACHE, HERE, pool.get_foundation(), lease.deadline - 5,
                                                epochs=list(endpoints))
        for action in result.get("registry_actions", []):
            campaign.record_status(action["experiment_id"], execution_status="PARTIAL_SCREEN025" if max(endpoints) < 3 and action["family"] != "nested_pca" else action["execution_status"],
                                   training_status="NO_TRAINING" if action["family"] == "nested_pca" else result["training_status"],
                                   validity_status="PENDING_INDEPENDENT_VALIDATION", reason="Shared/max/PCA prefix actual K-row V inference; no T accessed")
        print(json.dumps(dict(event="ORDERED_PILOT_DONE", dataset=ds, status=result["status"],
                              training_status=result.get("training_status"), prediction_aliases=result.get("prediction_aliases"))), flush=True)
        return result
    finally:
        pool.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["predict", "combine", "controls", "ordered"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs", nargs="+", type=float, default=[.25, 1, 3])
    p.add_argument("--skip-small", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    with GPUReservation(CACHE, total_gpu_seconds=43200, reserve_seconds=3600,
                        wall_deadline=1792050350 + 172800, purpose="training", kind="postfit_" + a.phase) as lease:
        reconcile_cpu_baselines(lease)
        if a.phase == "predict":
            predict_validation(a.dataset, lease, not a.skip_small)
        elif a.phase == "combine":
            combine(a.dataset, lease, a.epochs)
        elif a.phase == "controls":
            controls(a.dataset, lease, a.epochs)
        else:
            ordered(a.dataset, lease, a.epochs)


if __name__ == "__main__":
    main()
