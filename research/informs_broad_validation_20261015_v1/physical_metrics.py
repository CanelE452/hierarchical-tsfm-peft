"""CPU native-unit auxiliary errors from sealed predictions and frozen A60 scales."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import audit_predictions as audit
import channel_patterns as patterns
import data_contract as data


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def scaler_fingerprint(mean, std):
    values = {"mean": audit._fingerprint(np.asarray(mean, np.float32)), "std": audit._fingerprint(np.asarray(std, np.float32))}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def physical_statistics(prediction, target, observed, std):
    p, y, mask = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(observed)
    std = np.asarray(std, np.float64)
    if p.ndim != 3 or p.shape != y.shape or p.shape != mask.shape or mask.dtype != np.bool_ or min(p.shape) < 1:
        raise ValueError("Matching [N,H,C] arrays and original bool mask required")
    if std.shape != (p.shape[2],) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Positive finite frozen channel scales matching C required")
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Observed prediction/target must be finite")
    channels = []
    for c in range(p.shape[2]):
        good = mask[:, :, c]
        error = (p[:, :, c][good] - y[:, :, c][good]) * std[c]
        count = int(good.sum()); sse = float(np.sum(error * error, dtype=np.float64)); sae = float(np.sum(abs(error), dtype=np.float64))
        channels.append(dict(channel_index=c, count=count, physical_sse=sse, physical_sae=sae,
                             physical_mse=sse / count if count else None, physical_mae=sae / count if count else None,
                             physical_rmse=math.sqrt(sse / count) if count else None))
    active = [row for row in channels if row["count"]]
    macro_mse = math.fsum(row["physical_mse"] for row in active) / len(active) if active else None
    count = sum(row["count"] for row in channels); sse = math.fsum(row["physical_sse"] for row in channels); sae = math.fsum(row["physical_sae"] for row in channels)
    return dict(physical_macro_mse=macro_mse, physical_macro_mae=math.fsum(row["physical_mae"] for row in active) / len(active) if active else None,
                physical_macro_rmse=math.sqrt(macro_mse) if active else None,
                physical_mean_channel_rmse=math.fsum(row["physical_rmse"] for row in active) / len(active) if active else None,
                physical_pooled_mse=sse / count if count else None, physical_pooled_mae=sae / count if count else None,
                physical_pooled_rmse=math.sqrt(sse / count) if count else None, physical_total_sse=sse, physical_total_sae=sae,
                observed_target_cells=count, active_channels=len(active), zero_observed_channels=len(channels)-len(active)), channels


def _inside(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Proof pointer escapes its declared root")
    return path


def _input(root, path, scope):
    return dict(scope=scope, file=path.relative_to(root).as_posix(), sha256=data.sha256_file(path))


def _normalization(public, cache, registry, registered, manifest, expected_end):
    dataset = registered["dataset"]
    candidates = list((public / "runs").glob("*/RIDGE_FIT_RECEIPT.json")) + list((public / "runs").glob("*/BASE_FIT_RECEIPT.json"))
    for receipt_path in sorted(candidates, key=lambda p: (p.name != "RIDGE_FIT_RECEIPT.json", p.as_posix())):
        receipt = _json(receipt_path); rid = receipt.get("run_id")
        if rid not in registry or registry[rid]["dataset"] != dataset or receipt.get("status") != "COMPLETE_VALID":
            continue
        artifact = _inside(cache, receipt["artifact_cache_relative_path"])
        blob = artifact.read_bytes()
        if hashlib.sha256(blob).hexdigest() != receipt["artifact_sha256"]:
            raise ValueError("Frozen normalization artifact SHA mismatch")
        if artifact.suffix == ".npz":
            with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                mean, std = archive["mean"].copy(), archive["std"].copy()
                own_data, fit_end = str(archive["data_sha"].item()), int(archive["canonical_A60_end"].item())
        else:
            import torch
            torch.set_num_threads(2)
            saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            mean, std = np.asarray(saved["normalization"]["mean"]), np.asarray(saved["normalization"]["std"])
            own_data, fit_end = saved["contract"]["data_sha"], int(saved["normalization"]["fit_end"])
        if own_data != manifest["data_sha"] or fit_end != expected_end or scaler_fingerprint(mean, std) != manifest["scaler_sha"]:
            continue
        return dict(mean=mean, std=std, fit_end=fit_end, data_sha=own_data,
                    input_files=[_input(public, receipt_path, "public"), _input(cache, artifact, "private")])
    raise ValueError("No pinned matching A60 normalization artifact; raw Task/targets are not reopened")


def generate(public_root, cache_root, split="V_select", normalizations=None):
    public, cache = Path(public_root).resolve(), Path(cache_root).resolve()
    if split not in {"V_select", "T_final"}:
        raise ValueError("Only V_select or locked T_final existing caches supported")
    lock = {}
    if split == "T_final":
        if not (public / "SELECTION_LOCK.json").is_file() or _json(public / "SELECTION_LOCK.json").get("locked") is not True:
            raise ValueError("T requires locked SELECTION_LOCK.json before any target cache read")
        lock = _json(public / "SELECTION_LOCK.json")
    started, cpu_started = time.perf_counter(), time.process_time()
    registry_path = public / "EXPERIMENT_REGISTRY.csv"; registry_blob = registry_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(registry_blob.decode("utf-8-sig"))))
    registry = {row["experiment_id"]: row for row in rows}
    if len(registry) != len(rows):
        raise ValueError("Duplicate registered IDs")
    contract_path = public / "DATA_CONTRACT.json"; contract_blob = contract_path.read_bytes()
    assets = json.loads(contract_blob)["assets"]
    output = public / "diagnostics" / ("physical_metrics_" + str(time.time_ns())); output.mkdir(parents=True, exist_ok=False)
    headers, excluded, providers, inputs = [], [], {}, [dict(scope="public", file=registry_path.name, sha256=hashlib.sha256(registry_blob).hexdigest()),
                                                        dict(scope="public", file=contract_path.name, sha256=hashlib.sha256(contract_blob).hexdigest())]
    for path in sorted(public.rglob("PREDICTION_MANIFEST.json")):
        try:
            blob = path.read_bytes(); manifest = json.loads(blob)
            if manifest.get("split") != split or manifest.get("score") is None:
                continue
            rid = manifest["run_id"]; registered = registry[rid]
            if split == "T_final" and not patterns._t_allowed(lock, manifest):
                raise ValueError("T route/artifact not in SELECTION_LOCK")
            asset = assets[registered["dataset"]]; fold = registered.get("fold", "primary")
            if manifest["data_sha"] != asset["data_sha256"]:
                raise ValueError("Manifest differs from frozen public data contract")
            expected_end = int(asset[fold + "_split_ranges"]["A_stop"][1])
            key = (registered["dataset"], fold, manifest["scaler_sha"])
            if key not in providers:
                own = dict(normalizations[(key[0], key[1])]) if normalizations is not None else _normalization(public, cache, registry, registered, manifest, expected_end)
                if own["data_sha"] != manifest["data_sha"] or int(own["fit_end"]) != expected_end or scaler_fingerprint(own["mean"], own["std"]) != manifest["scaler_sha"]:
                    raise ValueError("Supplied frozen normalization identity mismatch")
                providers[key] = own; inputs.extend(own.get("input_files", []))
            reference_path = path.with_name("REFERENCE_CONTRACT.json"); reference_blob = reference_path.read_bytes(); reference = json.loads(reference_blob)
            if any(manifest.get(k) != reference.get(k) or not str(manifest.get(k, "")) for k in audit.IDENTITY_KEYS):
                raise ValueError("Independent reference identity mismatch")
            recipe = patterns.training_identity(public, rid, manifest, registered["family"])
            inputs.extend(recipe.pop("input_files"))
            stats_path = _inside(public, manifest["public_stats_relative_path"])
            stats_sha = data.sha256_file(stats_path)
            inputs.append(dict(scope="public", file=stats_path.relative_to(public).as_posix(), sha256=stats_sha))
            inputs += [dict(scope="public", file=path.relative_to(public).as_posix(), sha256=hashlib.sha256(blob).hexdigest()),
                       dict(scope="public", file=reference_path.relative_to(public).as_posix(), sha256=hashlib.sha256(reference_blob).hexdigest()),
                       dict(scope="private", file=manifest["cache_relative_path"], sha256=manifest["archive_sha256"])]
            headers.append((path, manifest, reference, registered, asset, providers[key], recipe, stats_sha))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            excluded.append(dict(manifest=path.relative_to(public).as_posix(), reason=str(exc).replace(str(public), "<public>").replace(str(cache), "<cache>")))
    source_root = Path(__file__).resolve().parent
    sources = {name:data.sha256_file(source_root / name) for name in ("physical_metrics.py", "audit_predictions.py", "channel_patterns.py", "data_contract.py")}
    _new_json(output / "INPUT_SNAPSHOT.json", dict(sealed_unix_ns=time.time_ns(), sources=sources, split=split, input_files=inputs,
        normalizations=[dict(dataset=k[0], fold=k[1], scaler_sha=k[2], fit_end=v["fit_end"], mean=audit._fingerprint(np.asarray(v["mean"], np.float32)),
                             std=audit._fingerprint(np.asarray(v["std"], np.float32))) for k,v in providers.items()],
        new_target_windows=0, raw_Task_opened=False, normalized_primary_unchanged=True))
    scores, channel_rows, seen = [], [], set()
    for path, manifest, reference, registered, asset, own, recipe, stats_sha in headers:
        try:
            identity = (manifest["run_id"], manifest["cohort"])
            if identity in seen:
                raise ValueError("Duplicate run/cohort; no automatic score selection")
            arrays = audit._load_arrays(manifest, reference, cache)
            if list(map(str, arrays["channel_ids"])) != list(map(str, asset["columns"])):
                raise ValueError("Prediction channel order differs from frozen normalization/data")
            expected, normalized = audit._independent_statistics(arrays, manifest)
            stats_path = _inside(public, manifest["public_stats_relative_path"])
            if data.sha256_file(stats_path) != stats_sha:
                raise ValueError("Public error statistics changed after input seal")
            with gzip.open(stats_path, "rt", encoding="utf-8", newline="") as stream:
                audit._reaggregate(list(csv.DictReader(stream)), expected, manifest["score"])
            for field in ("mse", "mae"):
                audit._same_number(manifest["score"][field], normalized[field], field)
            audit._check_cost([manifest["cost_record"]], manifest["run_id"], reference)
            score, channels = physical_statistics(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"], own["std"])
            dataset = registered["dataset"]
            units = "flow counts (source feature0; 5-minute sample; no hourly conversion)" if dataset == "pems08" else "source-native channel units; heterogeneous sensor channels must be interpreted separately"
            common = {key:manifest[key] for key in audit.IDENTITY_KEYS}
            common.update(dataset=dataset, fold=registered.get("fold", "primary"), family=registered["family"], seed=int(registered.get("seed") or 0),
                          K=int(registered.get("latent_k") or 0), budget=patterns._budget(public, manifest["run_id"], manifest),
                          **recipe, canonical_scaler_fit_end=int(own["fit_end"]), physical_unit_note=units,
                          heterogeneous_channel_units=dataset in {"beijing", "jena", "robin"})
            scores.append(dict(**common, **score, normalized_primary_mse=normalized["mse"], normalized_primary_mae=normalized["mae"],
                               n_origins=normalized["n_origins"], horizon=normalized["horizon"], num_channels=normalized["num_channels"]))
            channel_rows.extend(dict(**common, channel=str(arrays["channel_ids"][row["channel_index"]]), **row) for row in channels)
            seen.add(identity)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            excluded.append(dict(manifest=path.relative_to(public).as_posix(), reason=str(exc).replace(str(public), "<public>").replace(str(cache), "<cache>")))
    _csv(output / "PHYSICAL_SCORES.csv", scores); _csv(output / "PHYSICAL_CHANNEL_ERRORS.csv.gz", channel_rows)
    unchanged = all(data.sha256_file(source_root / name) == digest for name,digest in sources.items())
    receipt = dict(status="AVAILABLE_AUXILIARY_METRICS" if scores and unchanged else "NO_VALID_AUXILIARY_METRICS", split=split,
        sources=sources, sources_unchanged=unchanged, input_snapshot_sha256=data.sha256_file(output / "INPUT_SNAPSHOT.json"),
        prediction_count=len(scores), channel_rows=len(channel_rows), excluded_prediction_inputs=excluded,
        output_files={p.name:data.sha256_file(p) for p in output.iterdir() if p.name.endswith((".csv", ".csv.gz"))},
        macro_rmse_definition="sqrt(mean_c(SSE_physical_c / N_c)); distinct mean_channel_rmse also reported",
        pooled_definition="sum observed physical error numerators / sum observed cells", dtype="independent NumPy float64",
        normalized_primary_unchanged=True, new_target_windows=0, raw_Task_opened=False, A60_statistics_refitted=False,
        inference_or_training_executed=False, GPU_execution=False, T_selection_lock_checked=split == "T_final",
        CPU_process_seconds=time.process_time()-cpu_started, wall_seconds=time.perf_counter()-started)
    _new_json(output / "PHYSICAL_METRIC_MANIFEST.json", receipt)
    links = [p.name for p in output.iterdir() if p.is_file()]
    (output / "README_KO.md").write_text("# 원단위 보조 오차\n\n검산된 기존 예측과 frozen A60 scaler만 사용했다. 주지표인 정규화 macro MSE를 바꾸지 않는다. macro RMSE는 macro MSE의 제곱근이며 채널 RMSE 평균과 별도다. 서로 단위가 다른 센서는 채널별 오차를 우선 해석한다. PeMS는 5분 flow counts이고 실제 달력·시간당 교통량을 추정하지 않는다.\n\n" + "\n".join(f"- [{name}]({name})" for name in sorted(links)) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True); parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", choices=("V_select", "T_final"), default="V_select")
    args = parser.parse_args(); print(generate(args.public_root, args.cache_root, args.split))
