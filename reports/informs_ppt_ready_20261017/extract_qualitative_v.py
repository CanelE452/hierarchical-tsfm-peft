"""Extract fixed, presealed cases from explicitly bound existing V-only archives."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(args):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    if sha(args.inventory) != args.inventory_sha or sha(args.scope) != args.scope_sha:
        raise ValueError("Inventory/scope SHA changed")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    scope = json.loads(Path(args.scope).read_text(encoding="utf-8"))
    if inventory["scope_descriptor"]["sha256"] != args.scope_sha or time.time() >= scope["package_deadline_unix"]:
        raise ValueError("Presealed selection or package wall boundary invalid")
    if Path(args.curves).exists() or Path(args.metadata).exists() or Path(args.public_metadata).exists():
        raise FileExistsError("Preserve previous extraction outputs")
    import numpy as np
    assets = {a["id"]: a for a in inventory["assets"]}
    chosen = {"jena": [("JENA_B0", "B0"), ("JENA_B2", "B2"), ("JENA_R5B", "Joint no-Bolt"), ("JENA_LORA", "LoRA")],
              "pems08": [("PEMS_B0", "B0"), ("PEMS_B2_CPU", "B2"), ("PEMS_FIXED_HEAD", "Fixed no-Bolt"), ("PEMS_G_CPU3", "G")]}
    archive_root = Path(args.asset_root).resolve() / ".cache/informs_broad_validation_20261015_v1"
    curves, groups, scores, bindings = [], {}, [], []
    for dataset, methods in chosen.items():
        reference = None
        group = next(g for g in inventory["groups"] if g["dataset"] == dataset)
        panels = []
        for asset_id, method in methods:
            asset = assets[asset_id]
            path = Path(asset["private_V_archive"]["path"]).resolve()
            if not path.is_relative_to(archive_root) or path.name != "V_select.npz" or asset["split"] != "V_select":
                raise ValueError("Only the declared V-only cache files are allowed")
            if sha(path) != asset["private_V_archive"]["sha256"] or sha(asset["manifest"]["path"]) != asset["manifest"]["sha256"]:
                raise ValueError("Existing archive/manifest changed")
            manifest = json.loads(Path(asset["manifest"]["path"]).read_text(encoding="utf-8"))
            if manifest["split"] != "V_select" or manifest["archive_sha256"] != sha(path):
                raise ValueError("Exact V-only manifest binding required")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
                    raise ValueError("Unexpected archive members")
                arrays = {name: archive[name] for name in archive.files}
            for name, array in arrays.items():
                expected = asset["array_fingerprints_from_manifest"][name]
                actual = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
                if actual != expected:
                    raise ValueError("Exact saved array fingerprint mismatch: " + name)
            if reference is None:
                reference = arrays
                origin_order = np.argsort(arrays["origins"])
                selected_origins = [int(origin_order[i]) for i in (0, len(origin_order) // 2, len(origin_order) - 1)]
                selected_channels = [0, asset["C"] // 2, asset["C"] - 1]
                for oi, origin_index in enumerate(selected_origins):
                    for ci, channel_index in enumerate(selected_channels):
                        panels.append(dict(origin_slot=oi, channel_slot=ci, origin_array_index=origin_index,
                                           origin_id=int(arrays["origins"][origin_index]), channel_array_index=channel_index,
                                           channel_id=str(arrays["channel_ids"][channel_index])))
            else:
                for name in ("target_normalized", "observed", "origins", "channel_ids"):
                    if asset["array_fingerprints_from_manifest"][name] != group["common_array_fingerprints"][name]:
                        raise ValueError("Method-specific cohort differs")
                    equal = np.array_equal(arrays[name], reference[name], equal_nan=arrays[name].dtype.kind in "fc")
                    if not equal:
                        raise ValueError("Shared target/mask/origin/channel values differ")
            if asset["scaler_sha256"] != group["common_scaler_sha256"]:
                raise ValueError("Different scaler cannot share an axis")
            pred, target, observed = arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"]
            if observed.dtype != np.bool_ or not np.isfinite(pred[observed]).all() or not np.isfinite(target[observed]).all():
                raise ValueError("Observed target/prediction must be finite with original bool mask")
            per_channel_mse, per_channel_mae = [], []
            for c in range(asset["C"]):
                errors = pred[:, :, c][observed[:, :, c]].astype(np.float64) - target[:, :, c][observed[:, :, c]].astype(np.float64)
                if errors.size:
                    per_channel_mse.append(math.fsum(float(x * x) for x in errors) / errors.size)
                    per_channel_mae.append(math.fsum(abs(float(x)) for x in errors) / errors.size)
            mse, mae = math.fsum(per_channel_mse) / len(per_channel_mse), math.fsum(per_channel_mae) / len(per_channel_mae)
            if abs(mse - manifest["score"]["mse"]) > 1e-12 or abs(mae - manifest["score"]["mae"]) > 1e-12:
                raise ValueError("Independent FP64 archive score disagrees with existing V manifest")
            scores.append(dict(asset_id=asset_id, method=method, dataset=dataset, run_id=asset["run_id"],
                               mse=mse, mae=mae, max_score_delta=max(abs(mse-manifest["score"]["mse"]), abs(mae-manifest["score"]["mae"])),
                               tolerance=1e-12, status="PASS_SAVED_V_FP64_RESCORE_ONLY"))
            for panel in panels:
                oi, ci = panel["origin_array_index"], panel["channel_array_index"]
                for h in range(asset["H"]):
                    mask = bool(observed[oi, h, ci])
                    curves.append(dict(dataset=dataset, method=method, origin_id=panel["origin_id"], channel_id=panel["channel_id"],
                                       horizon_step=h+1, observed=int(mask), value=format(float(pred[oi, h, ci]), ".17g") if mask else ""))
                    if method == "B0":
                        curves.append(dict(dataset=dataset, method="Truth", origin_id=panel["origin_id"], channel_id=panel["channel_id"],
                                           horizon_step=h+1, observed=int(mask), value=format(float(target[oi, h, ci]), ".17g") if mask else ""))
            bindings.append(dict(asset_id=asset_id, method=method, run_id=asset["run_id"], dataset=dataset,
                                 archive_sha256=asset["private_V_archive"]["sha256"], checkpoint_sha256=asset["actual_checkpoint"]["sha256"],
                                 manifest_file=asset["manifest"]["public_relative_path"], manifest_sha256=asset["manifest"]["sha256"],
                                 prediction_device=asset["prediction_device"], source_fit_device=asset["source_fit_device"],
                                 scaler_sha256=asset["scaler_sha256"], array_fingerprints=asset["array_fingerprints_from_manifest"]))
            if sha(path) != asset["private_V_archive"]["sha256"]:
                raise ValueError("Original V archive changed while reading")
        groups[dataset] = dict(C=asset["C"], H=asset["H"], V_origins=len(reference["origins"]), panels=panels,
                               normalization="A60/refit channel z-score; original units not recovered; no scaler values opened",
                               common_scaler_sha256=group["common_scaler_sha256"],
                               same_prediction_device_all_methods=group["same_device_all_methods"],
                               same_training_budget_claimed=False)
    with Path(args.curves).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    result = dict(schema="PRESEALED_V_ONLY_QUALITATIVE_SELECTION_v1", status="PASS_FIXED_SELECTION_AND_SAVED_V_RESCORE",
                  scope_sha256=args.scope_sha, inventory_sha256=args.inventory_sha,
                  selection_rule=scope["qualitative_selection"], datasets=groups, source_bindings=bindings,
                  existing_V_FP64_rescores=scores, curves_csv_sha256=sha(args.curves), curve_rows=len(curves),
                  new_training=0, new_model_inference=0, new_cost_measurement=0, protected_T_numeric_access=0,
                  checkpoint_tensor_opened=False, independent_generalization_proven=False, whole_MASTER_complete=False)
    if time.time() >= scope["package_deadline_unix"]:
        raise TimeoutError("Package wall boundary")
    put(args.metadata, result); put(args.public_metadata, result)
    print(json.dumps(dict(status=result["status"], curve_rows=len(curves), existing_V_rescores=len(scores),
                         selected_panels=sum(len(g["panels"]) for g in groups.values()), metadata_sha256=sha(args.metadata))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("inventory", "inventory-sha", "scope", "scope-sha", "asset-root", "curves", "metadata", "public-metadata"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_NO_INPUT_READS", writes=0, model_calls=0))); return
    if any(v is None for k, v in vars(args).items() if k != "execute"):
        parser.error("All pinned V inventory, scope, ROOT output paths are required")
    execute(args)


if __name__ == "__main__":
    main()
