"""Read-only independent CPU replay of saved checkpoints; never repair producer v1."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new, sha
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    meter = Meter("diagnose_all40_checkpoint_batches_v1", gpu=False)
    report, failure = None, None
    try:
        meter.protected_check()
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(2)
        source = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
        producer_record = {"path": str(Path(result_path).resolve()), "sha256": sha(result_path)}
        result = json.loads(Path(result_path).read_text(encoding="utf-8-sig"))
        if result["status"] != "COMPLETE":
            raise ValueError("TERMINAL_COMPLETE_ORIGINAL_PRODUCER_REQUIRED")
        input_records = []

        def load(record):
            meter.check()
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("ARTIFACT_FILE_DIGEST_CHANGED")
            with np.load(record["path"], allow_pickle=False) as opened:
                values = {key: opened[key].copy() for key in opened.files}
            if set(values) != set(record["arrays"]):
                raise ValueError("ARTIFACT_ARRAY_KEYS_CHANGED")
            for key, value in values.items():
                declared = record["arrays"][key]
                if (str(value.dtype) != declared["dtype"] or list(value.shape) != declared["shape"]
                        or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != declared["sha256"]):
                    raise ValueError("ARTIFACT_ARRAY_CONTENT_CHANGED:" + key)
            input_records.append({"path": record["path"], "sha256": record["sha256"]})
            return values

        def scores(prediction, target, mask):
            counts = mask.sum(axis=(0, 1))
            active = counts > 0
            error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
            per_mse = (error * error).sum(axis=(0, 1)) / np.maximum(counts, 1)
            per_mae = np.abs(error).sum(axis=(0, 1)) / np.maximum(counts, 1)
            return {"mse": float(per_mse[active].mean()), "mae": float(per_mae[active].mean()),
                    "observed_counts": counts.tolist(), "active_channels": int(active.sum()),
                    "per_channel_mse": [float(value) if ok else None for value, ok in zip(per_mse, active)],
                    "per_channel_mae": [float(value) if ok else None for value, ok in zip(per_mae, active)]}

        def difference(actual, reference):
            error = actual.astype(np.float64) - reference.astype(np.float64)
            changed = actual != reference
            first = np.argwhere(changed)[0].tolist() if changed.any() else None
            return {"exact": bool(np.array_equal(actual, reference)), "different_values": int(changed.sum()),
                    "max_abs": float(np.abs(error).max()), "first_mismatch_index": first}

        records, selections = [], []
        with threadpool_limits(limits=2), torch.no_grad():
            for dataset, row in result["datasets"].items():
                values = load(row["inputs"]["tune"])
                history = torch.from_numpy(values["history"])
                raw_last = torch.from_numpy(values["raw_last"])[:, None]
                for route, condition in row["projections"].items():
                    for seed, entry in condition["native"].items():
                        condition_rows = []
                        for checkpoint in entry["checkpoints"]:
                            arrays = load(checkpoint["state_artifact"])
                            state = {key: torch.from_numpy(value) for key, value in arrays.items()}
                            saved = load(checkpoint["tune_prediction_artifact"])["prediction"]

                            def encode(batch):
                                return torch.cat([(history[index:index + batch] - state["projection.center"])
                                                  @ state["projection.encoder"] for index in range(0, len(history), batch)])

                            def forward(latent, batch):
                                outputs = []
                                for index in range(0, len(latent), batch):
                                    z = latent[index:index + batch]
                                    last = z[:, -1:]
                                    innovation = torch.einsum("nlk,khl->nhk", z - last, state["head.weight"])
                                    innovation = innovation + state["head.bias"].T[None]
                                    future = last + innovation
                                    prediction = raw_last[index:index + batch] + (future - last) @ state["projection.decoder"]
                                    outputs.append(prediction)
                                return torch.cat(outputs).numpy()

                            z32, z64 = encode(32), encode(64)
                            p32, p64 = forward(z32, 32), forward(z64, 64)
                            s32, s64 = (scores(prediction, values["target"], values["observed"]) for prediction in (p32, p64))
                            observed = checkpoint["tune"]
                            item = {"dataset": dataset, "route": route, "seed": int(seed), "update": checkpoint["update"],
                                    "encoder_E32_vs_E64": difference(z32.numpy(), z64.numpy()),
                                    "head_forecast32_vs_saved_whole64": difference(p32, saved),
                                    "whole64_vs_saved_whole64": difference(p64, saved),
                                    "recorded_metrics": observed, "replayed_batch32_metrics": s32,
                                    "saved_whole64_metrics": s64, "recorded_matches_replayed32_exact": observed == s32,
                                    "saved_mse_minus_recorded": s64["mse"] - observed["mse"],
                                    "saved_mae_minus_recorded": s64["mae"] - observed["mae"]}
                            records.append(item)
                            condition_rows.append(item)
                        chosen32 = min(condition_rows, key=lambda item: (item["replayed_batch32_metrics"]["mse"], item["update"]))
                        chosen64 = min(condition_rows, key=lambda item: (item["saved_whole64_metrics"]["mse"], item["update"]))
                        ordered = sorted(item["replayed_batch32_metrics"]["mse"] for item in condition_rows)
                        selections.append({"dataset": dataset, "route": route, "seed": int(seed),
                                           "original_selected_update": entry["metadata"]["selected_update"],
                                           "batch32_selected_update": chosen32["update"], "whole64_selected_update": chosen64["update"],
                                           "unchanged": entry["metadata"]["selected_update"] == chosen32["update"] == chosen64["update"],
                                           "best_to_second_mse_gap_batch32": ordered[1] - ordered[0]})
        if (len(records) != 40 or len(selections) != 8
                or not all(item["encoder_E32_vs_E64"]["exact"] for item in records)
                or not all(item["whole64_vs_saved_whole64"]["exact"] for item in records)
                or not all(item["recorded_matches_replayed32_exact"] for item in records)
                or not all(item["unchanged"] for item in selections)):
            raise ValueError("EXACT_BATCH_CAUSE_REPRODUCTION_OR_SELECTION_STABILITY_FAILED")
        meter.protected_check()
        if sha(result_path) != producer_record["sha256"] or sha(__file__) != source["sha256"]:
            raise ValueError("PRODUCER_OR_DIAGNOSIS_SOURCE_CHANGED")
        for record in input_records:
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("DIAGNOSTIC_INPUT_CHANGED_DURING_READ_ONLY_REPLAY")
        report = {"status": "CONFIRMED_BATCH_ARITHMETIC_PROVENANCE_MISMATCH", "producer": producer_record,
                  "source": source, "input_file_records": input_records, "records": records, "selections": selections,
                  "summary": {"checkpoint_replays": 40, "encoder32_64_exact": True,
                              "recorded_all_metrics_exactly_reproduced_by_batch32": True,
                              "saved_callback_predictions_exactly_reproduced_by_whole64": True,
                              "selection_unchanged_all_8": True,
                              "max_abs_mse_metric_mismatch": max(abs(item["saved_mse_minus_recorded"]) for item in records),
                              "max_abs_mae_metric_mismatch": max(abs(item["saved_mae_minus_recorded"]) for item in records),
                              "max_abs_FP32_prediction_batch_mismatch": max(item["head_forecast32_vs_saved_whole64"]["max_abs"] for item in records)},
                  "cause": "Encoder coordinates identical; latent temporal head/forecast batch32 vs whole64 FP32 arithmetic differs",
                  "producer_v1_repaired_or_PASS_claimed": False, "metric_tolerance_relaxed": False,
                  "new_training_or_GPU_execution": False, "protected_TEST_accessed": False,
                  "recommendation": "Preserve v1; produce fresh canonical32 checkpoint and selected predictions and scores from identical saved arrays."}
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure}
    finally:
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output, report)
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "output": str(output),
                      "failure": failure}, ensure_ascii=False), flush=True)
    return 0 if failure is None else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.result, arguments.output))
