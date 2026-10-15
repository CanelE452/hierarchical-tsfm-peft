"""Checked actual whole inference costs for a deterministic decoder-only intervention."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / "research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
DECODERS = {"PAST_LS": "ORIGINAL_PAST_LS", "ENCODER_PINV": "ENCODER_PINV"}
ROUTES = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT",
          "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_binding(record, path, label):
    recorded_path = Path(record["path"])
    resolved = recorded_path.resolve() if recorded_path.is_absolute() else (ROOT / recorded_path).resolve()
    require(resolved == Path(path).resolve() and record["sha256"] == sha(path),
            "EXACT_METADATA_BINDING_REQUIRED:" + label)


def load_gates(producer_path, projection_audit_path, decoder_path, decoder_audit_path):
    """Read only JSON and hashes before any scientific import, asset, or GPU work."""
    producer, projection_audit = read(producer_path), read(projection_audit_path)
    decoder, decoder_audit = read(decoder_path), read(decoder_audit_path)
    require(producer.get("status") == decoder.get("status") == "COMPLETE", "BOTH_COMPLETE_PRODUCERS_REQUIRED")
    require(projection_audit.get("status") == decoder_audit.get("status") == "PASS", "BOTH_INDEPENDENT_PASS_AUDITS_REQUIRED")
    verify_binding(projection_audit["producer"], producer_path, "projection_audit.producer")
    verify_binding(decoder["producer"], producer_path, "decoder.producer")
    verify_binding(decoder["projection_audit"], projection_audit_path, "decoder.projection_audit")
    verify_binding(decoder_audit["diagnostic"], decoder_path, "decoder_audit.diagnostic")
    verify_binding(decoder_audit["producer"], producer_path, "decoder_audit.producer")
    verify_binding(decoder_audit["projection_audit"], projection_audit_path, "decoder_audit.projection_audit")
    for key in ("span_diagnostic", "span_audit"):
        verify_binding(decoder[key], decoder_audit[key]["path"], "decoder." + key)
        verify_binding(decoder_audit[key], decoder[key]["path"], "decoder_audit." + key)
    totals = projection_audit["totals"]
    require(all(totals.get(key) == value for key, value in {
        "projection_states_checked": 4, "native_fits_checked": 8,
        "checkpoint_states_checked": 40, "cost_cells_checked": 20,
        "recorded_timer_samples_checked": 120}.items()), "COMPLETE_PROJECTION_AUDIT_COVERAGE_REQUIRED")
    require(decoder.get("completed_cells") == 12 and not decoder.get("failed_paths"), "COMPLETE_TWELVE_DECODER_CELLS_REQUIRED")
    require(all(decoder_audit["totals"].get(key) == value for key, value in {
        "base_cells_checked": 12, "decoder_cells_checked": 24,
        "cached_forecast_arrays_checked": 96, "original_predictions_byte_exact_checked": 48}.items()),
        "COMPLETE_DECODER_AUDIT_COVERAGE_REQUIRED")
    require(set(producer["datasets"]) == set(decoder["datasets"]) == set(DATASETS), "EXACT_TWO_DATASETS_REQUIRED")
    require(decoder.get("new_training") is False
            and decoder.get("future_labels_used_to_fit_alternative_D") is False
            and decoder.get("method_checkpoint_branch_selection_used") is False
            and decoder.get("original_selected_forecasts_and_locks_unchanged") is True,
            "LABEL_FREE_FIXED_MODEL_DECODER_INTERVENTION_REQUIRED")
    for data in (producer, projection_audit, decoder, decoder_audit):
        require(data.get("protected_TEST_accessed") is False, "PROTECTED_TEST_SCOPE_REQUIRED")
    for dataset, (channels, k, development) in DATASETS.items():
        source, probe = producer["datasets"][dataset], decoder["datasets"][dataset]
        require(source["status"] == "COMPLETE" and source["K"] == probe["K"] == k
                and probe["development_split"] == development, "SAME_K_AND_DEVELOPMENT_SPLIT:" + dataset)
        require(source["data_contract"][development]["raw_channels"] == channels, "SAME_C:" + dataset)
        for route in ("PCA", "FORECA"):
            condition = probe["projections"][route]
            require(condition["status"] == "COMPLETE" and condition["splits"][development]["status"] == "COMPLETE",
                    "COMPLETE_DEVELOPMENT_DECODER_REFERENCES:" + dataset + ":" + route)
            require(condition["source_state"] == source["projections"][route]["state_artifact"], "SAME_FROZEN_E_D_SOURCE")
            verify_binding(condition["splits"][development]["cached_future_sources"]["BOLT"],
                           source["projections"][route]["bolt"]["artifacts"][development]["path"], "SAME_ORIGINAL_FUTURE")
    return producer, projection_audit, decoder, decoder_audit


def make_predictor(center, encoder, decoder, frozen_F):
    import numpy as np
    center, encoder, decoder = (np.ascontiguousarray(value, dtype=np.float32) for value in (center, encoder, decoder))
    require(center.ndim == 1 and encoder.shape[0] == len(center)
            and decoder.shape == (encoder.shape[1], len(center))
            and all(np.isfinite(value).all() for value in (center, encoder, decoder)), "FINITE_MATCHED_FROZEN_E_D")
    center, encoder, decoder = center.copy(), encoder.copy(), decoder.copy()

    def predict(history):
        latent = (history - center) @ encoder
        future = frozen_F(latent)
        return np.ascontiguousarray(history[:, -1:, :] + (future - latent[:, -1:, :]) @ decoder, dtype=np.float32)
    return predict


def execute(tag, producer_path, projection_audit_path, decoder_path, decoder_audit_path):
    producer, projection_audit, decoder, decoder_audit = load_gates(
        producer_path, projection_audit_path, decoder_path, decoder_audit_path)
    require(Path(tag).name == tag and tag not in ("", ".", ".."), "FRESH_SIMPLE_RUN_TAG_REQUIRED")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import whole_decoder_metric_cost_v1 as wc
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import matched_readout_contract_v1 as contract
    import cost_readout_v1 as old_cost

    require(tuple(wc.ROUTES) == ROUTES, "AUTHENTIC_FIVE_BOLT_COST_ROUTES_REQUIRED")
    output = HERE / "runs" / tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    meter = rt.Meter(tag, gpu=True)
    inputs.install_protected_test_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = {"status": "RUNNING", "scientific_execution": True,
              "producer": file_record(producer_path), "projection_audit": file_record(projection_audit_path),
              "decoder_diagnostic": file_record(decoder_path), "decoder_audit": file_record(decoder_audit_path),
              "protected_TEST_accessed": False, "independent_holdout": False,
              "new_backbone_training": False, "new_training_updates": 0, "projection_fits": 0,
              "alternative_decoder_refits": 0, "method_checkpoint_branch_selection_used": False,
              "forecast_contract": "FP32 raw_last + (latent_future - encoded_history_last) @ decoder",
              "routes": list(ROUTES), "cost": {}, "datasets": {}, "source_snapshots": [], "source_records": [],
              "hard_elapsed_or_cumulative_resource_caps": None,
              "authorization": file_record(rt.AUTH), "continuation_manifest": file_record(rt.MANIFEST),
              "timing_helper_copy": {"original": file_record(HERE / "whole_cost_v1.py"),
                                     "only_change": "Five authentic decoder-only BOLT route names"}}
    failure = None
    started = time.perf_counter()
    guarded = []

    def checked_load(record):
        values, checked = probes.load_checked_npz(record)
        result["source_records"].append(checked)
        guarded.append(dict(record))
        return {key: np.ascontiguousarray(value) for key, value in values.items()}

    def exact_array(actual, expected, label):
        require(actual.dtype == expected.dtype and actual.shape == expected.shape
                and probes.array_sha(actual) == probes.array_sha(expected), "BYTE_EXACT_CONDITION_REQUIRED:" + label)

    try:
        meter.protected_check()
        for record in (producer["selection_lock"], decoder["source"], decoder["checked_math_helper"]):
            probes.verified_file(record)
            guarded.append(dict(record))
        for record in producer["source_snapshots"]:
            require(sha(record["path"]) == sha(record["snapshot_path"]) == record["sha256"], "ORIGINAL_PRODUCER_SOURCE_CHANGED")
            guarded.extend((dict(record), {"path": record["snapshot_path"], "sha256": record["sha256"]}))
        source_folder = output / "sources"
        source_folder.mkdir()
        for path in (Path(__file__), HERE / "whole_decoder_metric_cost_v1.py", HERE / "continuation_runtime_v1.py",
                     BASE / "cost_readout_v1.py", OLD / "run_condition_probes.py", OLD / "prepare_inputs.py",
                     OLD / "matched_readout_contract_v1.py"):
            snapshot = source_folder / path.name
            with path.open("rb") as source, snapshot.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            result["source_snapshots"].append({**file_record(path), "snapshot_path": str(snapshot)})
        old_cost_result = probes.read(BASE / "cost_runs/first_readout24_v2/RESULT.json")
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(producer[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt()
        result["F_before"] = probes.model_hashes(bolt)
        require(result["F_before"] == producer["F_before"] == producer["F_after"] == old_cost_result["F_before"],
                "EXACT_ORIGINAL_FROZEN_BOLT_DUAL_HASH_REQUIRED")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        bolt.model.eval().requires_grad_(False)

        def raw_predict(history):
            with torch.no_grad():
                return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(), dtype=np.float32)

        with threadpool_limits(limits=2):
            for dataset, (channels, k, development) in DATASETS.items():
                meter.check({"stage": "load_cost_development", "dataset": dataset})
                source = producer["datasets"][dataset]
                dev = checked_load(source["inputs"][development])
                fit = checked_load(source["inputs"]["fit"])
                exact_array(dev["mean"], fit["mean"], "FIT_mean")
                exact_array(dev["std"], fit["std"], "FIT_std")
                observed_contract = contract.validate_raw_split(dataset, development, dev, fit["mean"], fit["std"])
                require(dev["history"].shape[-1] == channels, "SOURCE_C_REQUIRED")
                physical_record = source["physical_bridge"]
                physical = checked_load(physical_record)
                old_cost.validate_physical_inputs(physical["history_physical"], physical["history_observed"],
                    dev["history"], fit["mean"], fit["std"], dev["origins"])
                predictions = {"RAW_BOLT": dev["raw_forecast"]}
                predictors = {"RAW_BOLT": raw_predict}
                metadata = {"RAW_BOLT": {"source": "original frozen Bolt", "shared_resident_model": True,
                    "source_input": source["inputs"][development], "prediction_array": "raw_forecast"}}
                row = {"status": "RUNNING", "K": k, "development_split": development,
                       "inputs": {"fit": source["inputs"]["fit"], development: source["inputs"][development]},
                       "data_contract": observed_contract, "physical_bridge": physical_record, "decoders": {}}
                result["datasets"][dataset] = row
                for projection in ("PCA", "FORECA"):
                    source_condition = source["projections"][projection]
                    state = checked_load(source_condition["state_artifact"])
                    probe = decoder["datasets"][dataset]["projections"][projection]["splits"][development]
                    refs = checked_load(probe["artifact"])
                    exact_array(refs["origins"], dev["origins"], "origins")
                    exact_array(refs["raw_last"], dev["raw_last"], "raw_last")
                    exact_array(refs["FIT_mean"], fit["mean"], "probe_FIT_mean")
                    exact_array(refs["FIT_std"], fit["std"], "probe_FIT_std")
                    exact_array(refs["encoder_deployed_FP32"], state["encoder"].astype(np.float32), "frozen_encoder")
                    exact_array(refs["ORIGINAL_PAST_LS_decoder_deployed_FP32"], state["decoder"].astype(np.float32), "original_decoder")
                    for variant, key in DECODERS.items():
                        route = projection + "_" + variant + "_BOLT"
                        d_key, p_key = key + "_decoder_deployed_FP32", key + "_BOLT_prediction"
                        matrix = refs[d_key]
                        predictors[route] = make_predictor(state["center"], refs["encoder_deployed_FP32"], matrix, raw_predict)
                        predictions[route] = refs[p_key]
                        entry = {"projection_route": projection, "decoder_variant": key, "E_F_frozen": True,
                            "D_frozen": True, "encoder_decoder_refit_in_cost": False, "shared_resident_model": True,
                            "source_state": source_condition["state_artifact"], "saved_reference": probe["artifact"],
                            "reference_prediction_array": p_key, "deployed_decoder_array": d_key,
                            "decoder_array_metadata": wc.array_record(matrix),
                            "encoder_array_metadata": wc.array_record(refs["encoder_deployed_FP32"]),
                            "center_array_metadata": wc.array_record(state["center"].astype(np.float32))}
                        metadata[route], row["decoders"][route] = entry, entry
                memory = lambda: {"resident_rss_bytes": psutil.Process().memory_info().rss,
                                  "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                                  "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
                result["cost"][dataset] = wc.measure_cost(dataset, physical["history_physical"], physical["history_observed"],
                    fit["mean"], fit["std"], dev["origins"], predictors, predictions, metadata,
                    output / ("cost_" + dataset), meter.check, torch.cuda.synchronize, memory)
                row["status"] = "COMPLETE"
                print(json.dumps({"event": "DECODER_COST_COMPLETE", "dataset": dataset, "cells": 10}), flush=True)
        result["F_after"] = probes.model_hashes(bolt)
        require(result["F_before"] == result["F_after"], "FROZEN_BOLT_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for record in result["source_snapshots"]:
            require(sha(record["path"]) == sha(record["snapshot_path"]) == record["sha256"], "EXECUTED_SOURCE_CHANGED")
        for record in guarded:
            probes.verified_file(record)
        load_gates(producer_path, projection_audit_path, decoder_path, decoder_audit_path)
        meter.protected_check()
        result.update(status="COMPLETE", source_before_after_SHA_pass=True,
                      expected_cost_cells=20, recorded_timer_samples=120,
                      original_selected_forecasts_and_locks_unchanged=True)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        result.update(status="FAILED_PRESERVED", failure=failure)
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["resource_terminal"] = meter.close(result["status"], failure)
        rt.write_new(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "output": str(output / "RESULT.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--producer", required=True, type=Path)
    parser.add_argument("--projection-audit", required=True, type=Path)
    parser.add_argument("--decoder-diagnostic", required=True, type=Path)
    parser.add_argument("--decoder-audit", required=True, type=Path)
    args = parser.parse_args()
    execute(args.tag, args.producer, args.projection_audit, args.decoder_diagnostic, args.decoder_audit)
