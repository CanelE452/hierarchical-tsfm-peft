"""CPU functional future interventions; no target scoring or foundation loading."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import data_contract


DATASETS = ("beijing", "jena", "electricity", "pems08")
SOURCE_NAMES = ("information_audit.py", "data_contract.py", "fit_base.py", "training.py", "networks.py", "gpu_runtime.py", "evaluation.py")


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _fingerprint(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": data_contract.array_sha256(array)}


def perturb_copy(task, boundary):
    """Copy only the two data arrays, then overwrite future values without reading them."""
    boundary = int(boundary)
    if not 0 <= boundary <= len(task["physical"]):
        raise ValueError("Intervention boundary outside returned safe prefix")
    physical = task["physical"].copy(); observed = task["observed"].copy()
    n, c = physical[boundary:].shape
    physical[boundary:] = np.float32(100000) + np.arange(n, dtype=np.float32)[:, None] + np.arange(c, dtype=np.float32)[None] / np.float32(8)
    observed[boundary:] = ~observed[boundary:]
    return {"physical": physical, "observed": observed}


def audit_statistics(task, fit_end, k, statistics_fn=None, pca_fn=None):
    statistics_fn = statistics_fn or data_contract.fit_statistics
    pca_fn = pca_fn or data_contract.initialize_pca
    changed = perturb_copy(task, fit_end)
    original_minimal = {"physical": task["physical"], "observed": task["observed"]}
    mean, std, counts = statistics_fn(original_minimal, fit_end)
    mean_changed, std_changed, counts_changed = statistics_fn(changed, fit_end)
    pca = pca_fn(original_minimal, fit_end, k, mean, std)
    pca_changed = pca_fn(changed, fit_end, k, mean_changed, std_changed)
    basis, basis_changed = pca["E"], pca_changed["E"]
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    projector_changed = basis_changed.astype(np.float64) @ basis_changed.astype(np.float64).T
    checks = dict(mean_exact=bool(np.array_equal(mean, mean_changed)),
                  std_exact=bool(np.array_equal(std, std_changed)),
                  observed_counts_exact=bool(np.array_equal(counts, counts_changed)),
                  pca_center_exact=bool(np.array_equal(pca["pca_center"], pca_changed["pca_center"])),
                  pca_basis_exact=bool(np.array_equal(basis, basis_changed)),
                  pca_projector_exact=bool(np.array_equal(projector, projector_changed)),
                  pca_eigenvalues_exact=bool(np.array_equal(pca["eigenvalues"], pca_changed["eigenvalues"])))
    return {"status": "PASS" if all(checks.values()) else "FAIL", **checks,
            "fit_end": int(fit_end), "K": int(k), "perturbed_rows": int(len(task["physical"]) - fit_end),
            "physical_prefix_exact": bool(np.array_equal(task["physical"][:fit_end], changed["physical"][:fit_end], equal_nan=True)),
            "observed_prefix_exact": bool(np.array_equal(task["observed"][:fit_end], changed["observed"][:fit_end])),
            "mean": _fingerprint(mean), "std": _fingerprint(std), "pca_basis": _fingerprint(basis),
            "pca_projector": _fingerprint(projector), "pca_center": _fingerprint(pca["pca_center"]),
            "future_values_used_for_score": False}


def audit_histories(task, fixed_origins, mean, std, history_fn=None):
    history_fn = history_fn or data_contract.history_window
    rows = []
    length = int(task["L"])
    original_minimal = {"physical": task["physical"], "observed": task["observed"]}
    for origin in fixed_origins:
        origin = int(origin)
        changed = perturb_copy(task, origin)
        before, mask_before = history_fn(original_minimal, origin, mean, std, length=length)
        after, mask_after = history_fn(changed, origin, mean, std, length=length)
        row = {"origin": origin, "history_bounds_exclusive": [origin - length, origin],
               "intervention_bounds_exclusive": [origin, len(task["physical"])],
               "history_exact": bool(np.array_equal(before, after)), "mask_exact": bool(np.array_equal(mask_before, mask_after)),
               "history": _fingerprint(before), "mask": _fingerprint(mask_before),
               "input_shape": [1, length, int(task["physical"].shape[1])], "target_window_calls": 0}
        row["status"] = "PASS" if row["history_exact"] and row["mask_exact"] else "FAIL"
        rows.append(row)
    return {"status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL", "windows": rows}


def arguments_have_no_targets(names):
    return all(not any(token in name.lower() for token in ("target", "label", "future", "ground_truth")) and name.lower() not in {"y", "y_true"} for name in names)


def _find_function(tree, qualified):
    nodes = tree.body
    for part in qualified.split("."):
        found = next((node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part), None)
        if found is None:
            raise ValueError("Pinned source callable absent")
        nodes = found.body
    return found


def audit_source_calls():
    root = Path(__file__).resolve().parent
    targets = (("fit_base.py", "predict_model"), ("networks.py", "ForecastAdapter.forward"),
               ("gpu_runtime.py", "RowForecaster.forward"), ("gpu_runtime.py", "RowForecaster._invoke"),
               ("gpu_runtime.py", "FrozenBolt.__init__.predict"))
    rows = []; all_ok = True
    for source, qualified in targets:
        tree = ast.parse((root / source).read_text(encoding="utf-8"))
        function = _find_function(tree, qualified)
        args = [arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)]
        absent = arguments_have_no_targets(args)
        calls = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            expression = ast.unparse(node.func)
            if expression in {"foundation", "foundation_callable", "self.predict_rows", "model"}:
                call = {"call": expression, "arguments": [ast.unparse(arg) for arg in node.args],
                        "keywords": {item.arg: ast.unparse(item.value) for item in node.keywords}}
                calls.append(call)
                absent = absent and arguments_have_no_targets(list(call["keywords"]))
                if expression == "foundation_callable":
                    absent = absent and call["arguments"] == ["z"] and not call["keywords"]
                elif expression == "self.predict_rows":
                    absent = absent and call["arguments"] == ["rows"] and not call["keywords"]
                elif qualified == "FrozenBolt.__init__.predict":
                    absent = absent and not call["arguments"] and call["keywords"] == {"context": "rows"}
                elif expression == "foundation":
                    absent = absent and call["arguments"] == ["x"] and not call["keywords"]
        if qualified in {"ForecastAdapter.forward", "RowForecaster._invoke", "FrozenBolt.__init__.predict"} and len(calls) != 1:
            absent = False
        rows.append({"source": source, "function": qualified, "argument_names": args,
                     "target_argument_absent": bool(absent), "foundation_boundary_calls": calls})
        all_ok = all_ok and absent
    return {"status": "PASS" if all_ok else "FAIL", "functions": rows,
            "scope": "Selected Python source signatures and foundation-boundary expressions; not whole-program formal verification",
            "foundation_instantiated": False, "foundation_calls": 0}


def _selected_snapshot(public, cache, snapshot_path):
    if snapshot_path is None:
        return [], None
    snapshot_path = Path(snapshot_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = []
    for item in snapshot["rows"]:
        if item["status"] != "INCLUDED_COMPLETE_AT_SNAPSHOT":
            continue
        if item["family"] not in {"B0", "B2", "LAG_CAUSAL", "LAG_BIDIR"}:
            raise ValueError("Unexpected source encoder snapshot family")
        receipt_path = public / "runs" / item["run_id"] / "BASE_FIT_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        artifact = (cache / receipt["artifact_cache_relative_path"]).resolve()
        if not artifact.is_relative_to(cache) or data_contract.sha256_file(receipt_path) != item["receipt_sha256"]:
            raise ValueError("Encoder source receipt/cache boundary differs")
        if data_contract.sha256_file(artifact) != item["artifact_sha256"] or receipt["artifact_sha256"] != item["artifact_sha256"]:
            raise ValueError("Encoder artifact differs from predeclared source snapshot")
        rows.append({key: item[key] for key in ("run_id", "dataset", "family", "latent_k", "receipt_sha256", "artifact_sha256")})
        rows[-1]["artifact_cache_relative_path"] = receipt["artifact_cache_relative_path"]
    return rows, data_contract.sha256_file(snapshot_path)


def _audit_encoder(item, task, fixed_origins, cache):
    import torch
    import fit_base
    if torch.cuda.is_initialized():
        raise ValueError("Existing CUDA context forbidden")
    artifact = cache / item["artifact_cache_relative_path"]
    if data_contract.sha256_file(artifact) != item["artifact_sha256"]:
        raise ValueError("Encoder selected artifact changed after input seal")
    saved = torch.load(artifact, map_location="cpu", weights_only=False)
    norm, contract = saved["normalization"], saved["contract"]
    if contract["data_sha"] != task["data_sha256"]:
        raise ValueError("Encoder data SHA mismatch")
    if fit_base.scaler_fingerprint(norm["mean"], norm["std"]) != contract["scaler_sha"]:
        raise ValueError("Encoder scaler SHA mismatch")
    if int(norm["fit_end"]) != int(data_contract.split_ranges(task)["A_stop"][1]):
        raise ValueError("Encoder does not use declared A60 normalization")
    spec = dict(saved["spec"], device="cpu")
    model, rebuilt = fit_base.create_model(spec, task, int(norm["fit_end"]), object())
    if not np.array_equal(rebuilt["mean"], norm["mean"]) or not np.array_equal(rebuilt["std"], norm["std"]):
        raise ValueError("Rebuilt A60 scaler differs")
    model.load_state_dict(saved["model"], strict=True); model.eval()
    if any(not torch.equal(value, saved["model"][name]) for name, value in model.state_dict().items()):
        raise ValueError("Selected encoder state did not load exactly")
    state_before = data_contract.sha256_file(artifact)
    rows = []
    for origin in fixed_origins:
        changed = perturb_copy(task, int(origin))
        x, mask = data_contract.history_window(task, int(origin), norm["mean"], norm["std"])
        x_changed, mask_changed = data_contract.history_window(changed, int(origin), norm["mean"], norm["std"], length=task["L"])
        x, x_changed = torch.from_numpy(x[None]), torch.from_numpy(x_changed[None])
        with torch.no_grad():
            front, front_changed = model.front(x), model.front(x_changed)
            encoded, encoded_changed = model.encode(x), model.encode(x_changed)
        rows.append({"origin": int(origin), "front_exact": bool(torch.equal(front, front_changed)),
                     "encoded_exact": bool(torch.equal(encoded, encoded_changed)), "mask_exact": bool(np.array_equal(mask, mask_changed)),
                     "front_output": _fingerprint(front.numpy()), "encoder_output": _fingerprint(encoded.numpy()),
                     "encoder_input_shape": list(x.shape), "latent_output_shape": list(encoded.shape)})
    if data_contract.sha256_file(artifact) != state_before:
        raise ValueError("Selected artifact file changed during CPU encoder audit")
    return {"status": "PASS" if all(row["front_exact"] and row["encoded_exact"] and row["mask_exact"] for row in rows) else "FAIL",
            "run_id": item["run_id"], "dataset": item["dataset"], "family": spec["family"], "K": int(spec["latent_k"]),
            "checkpoint_sha256": item["artifact_sha256"], "data_sha256": task["data_sha256"],
            "scaler_sha256": contract["scaler_sha"], "checkpoint_training_source_sha": contract["source_sha"],
            "selected_search_epoch": saved.get("selected_search_epoch"), "windows": rows,
            "foundation_loaded": False, "foundation_calls": 0, "forecast_or_decoder_replay": False}


def run(public_root, cache_root, asset_root, encoder_snapshot=None):
    import torch
    import fit_base
    if torch.cuda.is_initialized():
        raise ValueError("Run in a fresh CPU process")
    torch.set_num_threads(2)
    public, cache = Path(public_root).resolve(), Path(cache_root).resolve()
    data_contract.ROOT, data_contract.CACHE = Path(asset_root).resolve(), cache
    output = public / "diagnostics" / ("information_" + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: data_contract.sha256_file(public / name) for name in SOURCE_NAMES}
    selected, input_encoder_sha = _selected_snapshot(public, cache, encoder_snapshot)
    tasks, task_rows = {}, []
    for name in DATASETS:
        task = data_contract.load_task(name, cache_dir=cache)
        task["physical"].setflags(write=False); task["observed"].setflags(write=False)
        tasks[name] = task
        bounds = data_contract.split_ranges(task)
        choices = data_contract.origins(bounds["V_select"], task["L"], task["H"], task["H"])
        fixed = [int(choices[index]) for index in (0, len(choices) // 2, len(choices) - 1)]
        task_rows.append({"dataset": name, "data_sha256": task["data_sha256"], "array_sha256": task["array_sha256"],
                          "shape": list(task["physical"].shape), "A60_end": int(bounds["A_stop"][1]),
                          "K": max(1, math.ceil(task["physical"].shape[1] / 4)), "fixed_V_origins": fixed,
                          "selection_rule": "First/middle/last metadata-only V stride-H origins; no errors examined",
                          "source_sha256": task["metadata"].get("source_sha256"),
                          "archive_deserialization_scope": task["metadata"].get("archive_deserialization_scope", "Prepared permitted-prefix NPZ members decompress as whole arrays"),
                          "pretraining_overlap_known": "unknown"})
    snapshot = {"sealed_unix_ns": time.time_ns(), "sources": sources, "tasks": task_rows,
                "encoders": selected, "encoder_source_snapshot_sha256": input_encoder_sha,
                "future_intervention": "Overwrite copies at/after boundary with large finite row/channel pattern; invert copied future masks",
                "target_read_for_score_or_selection": False, "new_T_evaluation": False, "real_F_calls": 0}
    _write_new(output / "INPUT_SNAPSHOT.json", snapshot)
    started = time.perf_counter(); cpu_started = time.process_time()
    datasets, encoders = [], []
    with patch.object(data_contract, "target_window", side_effect=AssertionError("Target access forbidden in information audit")):
        for row in task_rows:
            task = tasks[row["dataset"]]
            statistics = audit_statistics(task, row["A60_end"], row["K"])
            mean, std, _ = data_contract.fit_statistics(task, row["A60_end"])
            histories = audit_histories(task, row["fixed_V_origins"], mean, std)
            record = {"dataset": row["dataset"], "status": "PASS" if statistics["status"] == histories["status"] == "PASS" else "FAIL",
                      "data_sha256": task["data_sha256"], "scaler_sha256": fit_base.scaler_fingerprint(mean, std),
                      "statistics": statistics, "histories": histories, "original_arrays_unchanged":
                      data_contract.array_sha256(task["physical"]) == row["array_sha256"]["physical"] and
                      data_contract.array_sha256(task["observed"]) == row["array_sha256"]["observed"]}
            if not record["original_arrays_unchanged"]:
                record["status"] = "FAIL"
            datasets.append(record); _write_new(output / (row["dataset"] + ".json"), record)
        for item in selected:
            origins = next(row["fixed_V_origins"] for row in task_rows if row["dataset"] == item["dataset"])
            result = _audit_encoder(item, tasks[item["dataset"]], origins, cache)
            encoders.append(result); _write_new(output / (item["run_id"] + ".json"), result)
    source_calls = audit_source_calls()
    _write_new(output / "SOURCE_CALL_AUDIT.json", source_calls)
    sources_unchanged = all(data_contract.sha256_file(public / name) == digest for name, digest in sources.items())
    fields = ("dataset", "status", "data_sha256", "scaler_sha256", "A60_end", "K", "V_windows", "mean_exact", "std_exact", "pca_projector_exact", "original_arrays_unchanged")
    with (output / "INFORMATION_AUDIT.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for record in datasets:
            writer.writerow({"dataset": record["dataset"], "status": record["status"], "data_sha256": record["data_sha256"],
                             "scaler_sha256": record["scaler_sha256"], "A60_end": record["statistics"]["fit_end"], "K": record["statistics"]["K"],
                             "V_windows": len(record["histories"]["windows"]), "mean_exact": record["statistics"]["mean_exact"],
                             "std_exact": record["statistics"]["std_exact"], "pca_projector_exact": record["statistics"]["pca_projector_exact"],
                             "original_arrays_unchanged": record["original_arrays_unchanged"]})
    passed = all(row["status"] == "PASS" for row in datasets + encoders) and source_calls["status"] == "PASS" and sources_unchanged
    receipt = {"status": "PASS_TESTED_INFORMATION_PATHS" if passed else "FAIL_INFORMATION_PATH", "sources": sources,
               "input_snapshot_sha256": data_contract.sha256_file(output / "INPUT_SNAPSHOT.json"), "sources_unchanged": sources_unchanged,
               "datasets": len(datasets), "real_V_history_windows": sum(len(row["histories"]["windows"]) for row in datasets),
               "selected_encoder_checkpoints": len(encoders), "encoder_history_comparisons": sum(len(row["windows"]) for row in encoders),
               "data_fit_failures": sum(row["status"] != "PASS" for row in datasets), "encoder_failures": sum(row["status"] != "PASS" for row in encoders),
               "source_call_status": source_calls["status"], "target_window_calls": 0, "real_F_loaded": False, "real_F_calls": 0,
               "training_steps": 0, "new_T_evaluation": False, "registry_modified": False, "scores_modified": False,
               "cpu_process_seconds": time.process_time() - cpu_started, "wall_seconds": time.perf_counter() - started,
               "cuda_initialized": torch.cuda.is_initialized(), "CPU_threads": torch.get_num_threads(),
               "conclusion": "No future dependency detected in these tested fitting/history/P-E paths; not a universal no-leakage proof, forecast replay, performance gain, or independent generalization test",
               "remaining_uncertainty": ["Foundation pretraining overlap unknown", "Safe-prefix archives deserialize whole members before permitted slicing; not physical row isolation", "Legacy FP32 normalization/inversion ancestry remains", "Other paths and untested inputs are not exhaustively proven"]}
    _write_new(output / "INFORMATION_RECEIPT.json", receipt)
    text = ("# 실제 데이터 정보 사용 개입 검사\n\n"
            f"결과: `{receipt['status']}`. A60 통계/PCA {len(datasets)}데이터, V 과거 창 {receipt['real_V_history_windows']}개, "
            f"선택 checkpoint {len(encoders)}개의 P/E {receipt['encoder_history_comparisons']}개 비교를 수행했다. "
            "미래 구간의 복사본 값·mask를 바꾸어도 검사한 통계, PCA와 과거 입력/P/E가 정확히 같은지 확인했다.\n\n"
            "[INPUT_SNAPSHOT.json](INPUT_SNAPSHOT.json) · [집계 CSV](INFORMATION_AUDIT.csv) · "
            "[실행 영수증](INFORMATION_RECEIPT.json) · [F 경계 소스 검사](SOURCE_CALL_AUDIT.json)\n\n"
            "검사한 정보 경로에서 미래 의존성이 검출되지 않았다는 범위의 결과다. 실제 F·디코더 예측을 재실행하거나 Y로 성능을 평가하지 않았다. "
            "NPZ 멤버 전체 역직렬화와 FIT prefix 반환은 구분해야 하며 물리적인 행 격리를 보장하지 않는다. "
            "보호된 옛 TEST 파일을 새로 열지 않았고, task에 이미 허용된 prefix만 사용했다. 기반 모델 사전학습 중복과 legacy FP32 계보는 미확정으로 남는다.\n")
    with (output / "README_KO.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-root", type=Path, default=data_contract.CACHE)
    parser.add_argument("--asset-root", type=Path, default=data_contract.ROOT)
    parser.add_argument("--encoder-snapshot", type=Path)
    args = parser.parse_args()
    path = run(args.public_root, args.cache_root, args.asset_root, args.encoder_snapshot)
    print(path.relative_to(args.public_root.resolve()).as_posix())
