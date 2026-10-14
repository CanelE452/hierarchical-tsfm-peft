"""Guarded evidence for dense FIT feature/reduction layout reproducibility."""
from pathlib import Path
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.execute:
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    sys.path[:0] = [str(HERE), str(ROOT / "research/smallk_failure_followup_20261014"), str(ROOT)]
    from adaptive_runtime_v1 import ContinuationLimit
    limit = ContinuationLimit(args.tag, gpu=False)
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    report = {"status": "RUNNING", "scope": "diagnostic_not_a_successful_scientific_audit", "scientific_execution": True}
    error = None
    try:
        import numpy as np
        import audit_readout_v2 as audit
        audit.initialize_numpy()
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        audit.require(output.is_relative_to(HERE) and not output.exists(), "NEW_CAMPAIGN_DIAGNOSTIC_OUTPUT_ONLY")
        check = limit.shared.checkfast
        record = {"path": str(audit.resolve(args.input_result)), "sha256": audit.file_sha(audit.resolve(args.input_result))}
        producer = audit.verified_json(record, check)
        audit.require(producer["status"] == "COMPLETE" and producer["dataset"] == "robin"
                      and producer["phase"] == "MATCHED_FIT512_ONE_DIAGNOSTIC", "EXACT_DENSE_ROBIN_DIAGNOSTIC_REQUIRED")
        original = audit.verified_json(producer["first256_result"], check)
        contract_path = ROOT / "research/smallk_predictive_alignment_20261013/run_dev_v1/robin_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        path = audit.resolve(contract["source"])
        audit.require(audit.file_sha(path) == contract["source_sha256"], "ORIGINAL_SOURCE_SHA_REQUIRED")
        lower, upper = contract["bounds"]["fit"]
        with np.load(path, allow_pickle=False) as source:
            physical = (source["x"][:upper] * source["std"] + source["mean"]).astype(np.float32)
            mask = source["finite"][:upper].copy()
        physical[~mask] = np.nan
        fit = audit.verified_arrays(producer["inputs"]["fit"], check)
        mean, std = fit["mean"], fit["std"]
        origins = fit["origins"]
        past = np.stack([physical[origin - 512:origin] for origin in origins]).astype(np.float32)
        past_mask = np.stack([mask[origin - 512:origin] for origin in origins]).astype(bool)
        # Reproduce the documented producer layout, without loading its helpers.
        filled = np.empty_like(past, dtype=np.float32)
        last = np.broadcast_to(mean, (len(past), len(mean))).copy()
        for time_index in range(512):
            last = np.where(past_mask[:, time_index], past[:, time_index], last)
            filled[:, time_index] = last
        generated = ((filled - mean) / std).astype(np.float32)
        audit.exact(generated, fit["history"], "generated_vs_saved_history_values")
        variants = {"generated_original_layout": generated,
                    "generated_C_contiguous": np.ascontiguousarray(generated),
                    "generated_F_contiguous": np.asfortranarray(generated),
                    "saved_NPZ": fit["history"]}

        def layout(array):
            return {"shape": list(array.shape), "dtype": str(array.dtype), "strides": list(array.strides),
                    "C_contiguous": bool(array.flags.c_contiguous), "F_contiguous": bool(array.flags.f_contiguous),
                    "array_sha256": audit.array_sha(array)}

        def difference(actual, expected):
            error64 = actual.astype(np.float64) - expected.astype(np.float64)
            return {"exact_equal": bool(np.array_equal(actual, expected, equal_nan=True)),
                    "unequal_elements": int(np.count_nonzero(actual != expected)),
                    "maximum_absolute_difference": float(np.abs(error64).max()),
                    "root_mean_square_difference": float(np.sqrt(np.square(error64).mean()))}

        report.update(producer_record=record, original_source_record={"path": str(path), "sha256": contract["source_sha256"]},
                      raw_history_layouts={name: layout(value) for name, value in variants.items()},
                      raw_history_values_exact=True, routes={})
        with threadpool_limits(limits=2):
            for route in ("SELF33", "JOINT33"):
                limit.checkfull()
                metadata = producer["feature_states"][route]
                stored_state = audit.verified_arrays(metadata["artifact"], check)
                summaries = {}
                for name, history in variants.items():
                    features = audit.raw_features(route, history)
                    center, scale = audit.expected_feature_state(route, history)
                    stored_sha = audit.feature_state_digest(route, metadata, stored_state)
                    summaries[name] = {"raw_features_layout": layout(features),
                        "raw_feature_SHA_matches_producer_declared": audit.array_sha(features) == metadata["fit_raw_features_sha256"],
                        "center_vs_saved": difference(center, stored_state["center"]),
                        "scale_vs_saved": difference(scale, stored_state["scale"]),
                        "stored_feature_state_SHA_matches_metadata": stored_sha == metadata["sha256"],
                        "FP64_center_gamma_n_roundoff_bound_max": float((len(features) * np.finfo(np.float64).eps /
                            (1 - len(features) * np.finfo(np.float64).eps)) * np.abs(features.astype(np.float64)).mean(axis=0).max())}
                predictions = []
                for entry in producer["families"][route]["alpha_grid"]:
                    check()
                    coefficients = audit.verified_arrays(entry["coefficient_artifact"], check)
                    saved = audit.verified_arrays(entry["fit_artifact"], check)
                    row = {"alpha": entry["alpha"], "variants": {}}
                    for name, history in variants.items():
                        predicted = audit.prediction_from_saved(route, history, history[:, -1],
                            stored_state["center"], stored_state["scale"], coefficients["coefficients"], coefficients["intercept"])
                        row["variants"][name] = {**difference(predicted, saved["prediction"]),
                            "published_prediction_parity_pass": bool(np.allclose(predicted, saved["prediction"], rtol=2e-6, atol=1e-5))}
                    predictions.append(row)
                report["routes"][route] = {"producer_declared_raw_feature_sha256": metadata["fit_raw_features_sha256"],
                                           "variant_statistics": summaries, "saved_FIT_prediction_parity": predictions}
        report["status"] = "DIAGNOSTIC_COMPLETE"
        limit.checkfull()
    except Exception as failure:
        import traceback
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
        report.update(status="FAILED_PRESERVED", failure=error)
    finally:
        terminal = limit.close("COMPLETE" if report["status"] == "DIAGNOSTIC_COMPLETE" else "FAILED_PRESERVED", error)
        report["resource_terminal"] = terminal
        if terminal["terminal_status"] != "COMPLETE":
            report["status"] = "FAILED_PRESERVED"
        if output.is_relative_to(HERE) and not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"diagnostic": str(output), "status": report["status"], "failure": error}), flush=True)
    return 0 if report["status"] == "DIAGNOSTIC_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
