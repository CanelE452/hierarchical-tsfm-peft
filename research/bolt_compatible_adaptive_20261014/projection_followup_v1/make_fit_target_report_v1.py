"""Audited metadata-only comparison of FIT-future and past-PCA output spaces."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
SEEDS = ("92601", "92602")
SCALES = ("normalized", "physical_per_channel_diagnostic")
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_PINV_NATIVE", "FIT_TARGET_PINV_NATIVE")
COLORS = ("#666666", "#0072B2", "#D55E00", "#56B4E9", "#E69F00")
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}


def require(condition, label):
    if not condition: raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    path = Path(path).resolve()
    require(path.suffix.lower() == ".json", "JSON_METADATA_ONLY")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bind(reference, path, label):
    path = Path(path).resolve()
    require(path.suffix.lower() == ".json" and resolve(reference["path"]) == path and sha(path) == reference["sha256"], "EXACT_JSON_SHA_BINDING:" + label)


def load_verified(result_path, audit_path):
    result, audit = read(result_path), read(audit_path)
    require(result.get("status") == "COMPLETE" and audit.get("status") == "PASS", "COMPLETE_AND_INDEPENDENT_PASS_REQUIRED")
    bind(audit["producer"], result_path, "accepted result")
    expected = {"projection_states_checked": 4, "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2,
                "native_fits_checked": 8, "checkpoint_states_checked": 40, "cost_cells_checked": 20,
                "recorded_timer_samples_checked": 120, "saved_prediction_arrays_checked": 88,
                "output_subspace_diagnostic_cells_checked": 12, "fixed_D_real_method_inequalities_checked": 48,
                "bound_RAW_inequalities_checked": 0}
    require(all(audit["totals"].get(key) == value for key, value in expected.items()), "FULL_REGISTERED_AUDIT_SCOPE_REQUIRED")
    require(result["projection_fits"] == 2 and result["reused_projection_states"] == 2
            and result["projection_conditions"] == 4 and result["native_training_fits"] == 8, "MATCHED_NEW_TRAINING_COUNTS")
    for key in ("source_projection", "source_projection_audit"):
        require(audit[key] == result[key], "SAME_ORIGINAL_PROJECTION_CHAIN")
        bind(result[key], resolve(result[key]["path"]), key)
    require(result["encoder_fit_uses_FIT_true_future"] == {"PCA_PINV": False, "FIT_TARGET_PINV": True}, "SUPERVISED_TARGET_TRAINING_MUST_BE_EXPLICIT")
    require(result["original_LS_native_heads_reused_as_optimized_inverse_controls"] is False
            and result["canonical_native_batch_size"] == 32, "FRESH_MATCHED_NATIVE_HEADS_REQUIRED")
    require(result["F_before"] == result["F_after"] and result["new_backbone_training"] is False
            and result["prior_source_results_caches_and_selection_locks_unchanged"] is True, "FROZEN_BACKBONE_AND_PRIOR_PRESERVATION")
    require(result["protected_TEST_accessed"] is False and audit["protected_TEST_accessed"] is False
            and result["independent_holdout"] is False, "EXPOSED_DEV_SCOPE_REQUIRED")
    for dataset in DATASETS:
        data = result["datasets"][dataset]
        require(data["status"] == "COMPLETE" and set(data["projections"]) == set(PROJECTIONS), "EXACT_COMPLETE_PROJECTION_CONDITIONS")
        for route in PROJECTIONS:
            meta = data["projections"][route]["fit_metadata"]
            require(meta["TUNE_or_DEV_used_in_projection_or_decoder_fit"] is False
                    and meta["actual_future_target_used_in_projection_or_decoder_fit"] == (route == "FIT_TARGET_PINV"), "FIT_ONLY_SUPERVISED_SPACE_CONTRACT")
            for seed in SEEDS:
                native = data["projections"][route]["native"][seed]
                require(native["status"] == "COMPLETE" and native["metadata"]["updates"] == 600
                        and native["metadata"]["initial_weights_and_biases_zero"] is True
                        and native["metadata"]["seed_changes_initial_weights"] is False, "MATCHED_SIX_HUNDRED_UPDATE_ZERO_INIT_HEADS")
    return result, audit


def score_row(dataset, projection, split, predictor, seed, mask, scores, role, **extra):
    return [{"dataset": dataset, "projection": projection, "split": split, "predictor": predictor, "seed": seed,
             "mask": mask, "role": role, "metric_scale": scale,
             "this_MSE_is_a_fixed_D_minimum": predictor == "FIXED_D_MINIMUM" and scale == "normalized",
             **scores[scale], **extra} for scale in SCALES]


def collect_tables(result, audit):
    tables = {key: [] for key in ("COMPARISON", "MATCHED_COMPLETE_DIAGNOSTIC", "OUTPUT_SPACE_GEOMETRY", "CHECKPOINTS",
        "LEARNING_CURVE", "TRAINING_COST", "FIT_SPACE_METADATA", "FIT_TARGET_SPECTRUM", "RESTRICTED_PAST_VARIANCE",
        "WHOLE_COST", "WHOLE_COST_SAMPLES", "REGISTERED_ADOPTION")}
    for dataset, (channels, k, dev) in DATASETS.items():
        data = result["datasets"][dataset]
        for split in ("fit", "tune", dev):
            tables["COMPARISON"] += score_row(dataset, "NA", split, "RAW_BOLT", None, "all_original_observed", data["raw_scores"][split], "saved_raw_baseline")
        for route in PROJECTIONS:
            condition = data["projections"][route]
            meta = condition["fit_metadata"]
            tables["FIT_SPACE_METADATA"].append({"dataset": dataset, "projection": route, "C": channels, "K": k,
                "supervised_FIT_future_training": route == "FIT_TARGET_PINV", "past_only_training": route != "FIT_TARGET_PINV",
                "deployment_past_only": True, "TUNE_DEV_fit_used": False, "fit_metadata": meta,
                "state_artifact": condition["state_artifact"], "output_space_guarantees_predictability": False})
            if route == "FIT_TARGET_PINV":
                singular = meta["target_singularvalues"]
                total = meta["FIT_target_innovation_total_energy"]
                for index, value in enumerate(singular):
                    tables["FIT_TARGET_SPECTRUM"].append({"dataset": dataset, "component": index + 1, "K": k,
                        "singular_value": value, "squared_energy": value * value, "energy_fraction": value * value / total,
                        "retained_top_K": index < k, "FIT_complete_positions": meta["fit_complete_future_positions"],
                        "FIT_total_positions": meta["fit_total_future_positions"], "not_predictability_score": True})
                for index, value in enumerate(meta["retained_whitening_eigenvalues"]):
                    tables["RESTRICTED_PAST_VARIANCE"].append({"dataset": dataset, "component": index + 1,
                        "FIT_past_restricted_eigenvalue": value, "K": k, "relative_cut": meta["eigenvalue_relative_cut"],
                        "post_whitening_latent_covariance": meta["latent_fit_covariance"], "not_forecast_accuracy": True})
            for split in ("fit", "tune", dev):
                tables["COMPARISON"] += score_row(dataset, route, split, "BOLT", None, "all_original_observed", condition["bolt"]["scores"][split], "deployed_past_only")
            for seed in SEEDS:
                native = condition["native"][seed]
                metadata = native["metadata"]
                for split in ("fit", "tune", dev):
                    tables["COMPARISON"] += score_row(dataset, route, split, "NATIVE", seed, "all_original_observed", native["scores"][split], "fresh_matched_head", selected_update=metadata["selected_update"])
                for checkpoint in native["checkpoints"]:
                    tables["CHECKPOINTS"].append({"dataset": dataset, "projection": route, "seed": seed,
                        "update": checkpoint["update"], "selected": checkpoint["update"] == metadata["selected_update"],
                        **checkpoint["tune"], "state_sha256": checkpoint["state_sha256"],
                        "state_artifact": checkpoint["state_artifact"], "prediction_artifact": checkpoint["tune_prediction_artifact"]})
                for point in native["learning_curve"]:
                    tables["LEARNING_CURVE"].append({"dataset": dataset, "projection": route, "seed": seed, **point,
                        "loss_scope": "minibatch per-channel observed macro MSE; active channels", "not_full_FIT_loss": True})
                tables["TRAINING_COST"].append({"dataset": dataset, "projection": route, "seed": seed,
                    "train_wall_seconds": metadata["train_wall_seconds"], "updates": metadata["updates"],
                    "parameters": metadata["parameters"], "batch_size": metadata["batch_size"], "metadata": metadata,
                    "original_LS_head_reused": False, "seed_changes_initial_weights": False})
            for split in ("fit", "tune", dev):
                diagnostic = audit["output_subspace_diagnostics"][dataset][route][split]
                common = {"complete_target_positions": diagnostic["complete_target_positions"],
                    "total_target_positions": diagnostic["total_target_positions"], "fixed_D_minimum_objective_is_normalized_MSE": True,
                    "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "RAW_bound_applied": False}
                tables["MATCHED_COMPLETE_DIAGNOSTIC"] += score_row(dataset, route, split, "RAW_BOLT", None, "all_C_complete",
                    diagnostic["RAW_scores_same_complete_mask"], "saved_raw_same_mask", **common)
                tables["MATCHED_COMPLETE_DIAGNOSTIC"] += score_row(dataset, route, split, "FIXED_D_MINIMUM", None, "all_C_complete",
                    diagnostic["fixed_D_real_arithmetic_solution_scores_same_complete_mask"], "future_informed_fixed_D_real_minimum", **common)
                for predictor, value in diagnostic["methods_same_complete_mask"].items():
                    require(predictor in ("BOLT", "NATIVE_s92601", "NATIVE_s92602", "E_Y"), "EXACT_FOUR_DIAGNOSTIC_METHODS")
                    tables["MATCHED_COMPLETE_DIAGNOSTIC"] += score_row(dataset, route, split, predictor, None, "all_C_complete",
                        value["saved_prediction_scores_same_complete_mask"], "future_E_Y_diagnostic_only" if predictor == "E_Y" else "deployed_same_mask",
                        saved_minus_fixed_D_minimum_normalized_MSE=value["saved_minus_fixed_D_minimum_normalized_MSE"],
                        real_minus_fixed_D_minimum_normalized_MSE=value["real_minus_fixed_D_minimum_normalized_MSE"],
                        measured_FP32_rounding_MSE_allowance=value["measured_FP32_rounding_MSE_allowance"], **common)
                tables["OUTPUT_SPACE_GEOMETRY"].append({"dataset": dataset, "projection": route, "split": split,
                    **common, "fixed_D_minimum_normalized_MSE": diagnostic["fixed_D_minimum_normalized_MSE"],
                    "E_Y_minus_fixed_D_minimum_normalized_MSE": diagnostic["E_Y_minus_fixed_D_minimum_normalized_MSE"],
                    "geometry": diagnostic["geometry"]})
        rows = result["cost"][dataset]["rows"]
        require(len(rows) == 10 and {(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)}, "EXACT_ACTUAL_COST_CELLS")
        for row in rows:
            raw = next(value for value in rows if value["route"] == "RAW_BOLT" and value["batch"] == row["batch"])
            tables["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": row["repeats"],
                "median_seconds": row["median_seconds"], "median_seconds_per_window": row["median_seconds_per_window"],
                "ratio_of_RAW_medians": row["median_seconds"] / raw["median_seconds"], "median_paired_RAW_ratio": row["paired_RAW_median_ratio"],
                "seconds": row["seconds"], "origins": row["origins"], "memory_shared_process": row["memory"],
                "includes": row["whole_includes"], "excludes": row["whole_excludes"], "memory_scope": row["memory_scope"],
                "selected_metadata": row["selected_metadata"], "timed_outputs": row["timed_outputs"], "reference": row["reference"]})
            for index, sample in enumerate(row["samples"]):
                tables["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "position": sample["position"], "seconds": sample["seconds"],
                    "paired_RAW_ratio": row["paired_RAW_ratios"][index], "sample": sample})
        for route in ROUTES[1:]:
            projection = "PCA_PINV" if route.startswith("PCA_PINV") else "FIT_TARGET_PINV"
            metrics = (data["projections"][projection]["bolt"]["scores"]["tune"] if route.endswith("_BOLT")
                       else data["projections"][projection]["native"]["92601"]["scores"]["tune"])["normalized"]
            raw = data["raw_scores"]["tune"]["normalized"]
            cost = next(row for row in tables["WHOLE_COST"] if row["dataset"] == dataset and row["route"] == route and row["batch"] == 8)
            mse, mae, latency = metrics["mse"] / raw["mse"], metrics["mae"] / raw["mae"], cost["ratio_of_RAW_medians"]
            tables["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio": mse, "TUNE_MAE_ratio": mae,
                "B8_ratio_of_medians": latency, "all_registered_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .90,
                "actual_posthoc_method_selection_used": False})
    require(len(tables["COMPARISON"]) == 84 and len(tables["CHECKPOINTS"]) == 40 and len(tables["LEARNING_CURVE"]) == 4800
            and len(tables["MATCHED_COMPLETE_DIAGNOSTIC"]) == 144 and len(tables["WHOLE_COST_SAMPLES"]) == 120, "FULL_ACTUAL_METADATA_COHORT")
    return tables


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def rel(path, output):
    return Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix()


def choose(rows, **where):
    return next(row for row in rows if all(row[key] == value for key, value in where.items()))


def figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    folder = Path(output) / "figures"; folder.mkdir(exist_ok=False)
    paths = []
    def save(fig, stem):
        for suffix in ("png", "svg"):
            path = folder / (stem + "." + suffix); fig.savefig(path, dpi=165 if suffix == "png" else None, bbox_inches="tight"); paths.append(path)
        plt.close(fig)
    labels = ("Raw Bolt", "PCA / Bolt", "FIT target / Bolt", "PCA / native s1", "PCA / native s2", "FIT target / native s1", "FIT target / native s2")
    palette = ("#666666", "#0072B2", "#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]
            queries = [("NA", "RAW_BOLT", None), (PROJECTIONS[0], "BOLT", None), (PROJECTIONS[1], "BOLT", None)] + [(route, "NATIVE", seed) for route in PROJECTIONS for seed in SEEDS]
            values = [choose(tables["COMPARISON"], dataset=dataset, projection=p, split=split, predictor=h, seed=s, metric_scale="normalized")["mse"] for p, h, s in queries]
            ax.barh(np.arange(7), values, color=palette)
            for index, value in enumerate(values): ax.text(value + max(values) * .018, index, f"{value:.4f}", va="center")
            ax.set_xlim(0, max(values) * 1.22); ax.set_yticks(np.arange(7), labels); ax.invert_yaxis()
            ax.set_xlabel("Normalized macro MSE: all original observed targets"); ax.set_title(f"{DISPLAY[dataset]} / {split.upper()}"); ax.grid(axis="x", alpha=.2)
    fig.suptitle("FIT-future output directions vs past PCA: fresh matched native heads, same pinv(E) decoder", fontsize=13)
    save(fig, "01_actual_forecasts")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("fit", "tune", dev)):
            ax = axes[di, si]
            for mi, method in enumerate(("FIXED_D_MINIMUM", "BOLT", "NATIVE_s92601", "NATIVE_s92602")):
                values = [choose(tables["MATCHED_COMPLETE_DIAGNOSTIC"], dataset=dataset, projection=route, split=split, predictor=method, metric_scale="normalized")["mse"] for route in PROJECTIONS]
                x = np.arange(2) + (mi - 1.5) * .18
                bars = ax.bar(x, values, width=.17, color=("#CC79A7", "#0072B2", "#009E73", "#E69F00")[mi], label=method)
                ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=7)
            raw = choose(tables["MATCHED_COMPLETE_DIAGNOSTIC"], dataset=dataset, projection=PROJECTIONS[0], split=split, predictor="RAW_BOLT", metric_scale="normalized")
            ax.axhline(raw["mse"], color="#555555", linestyle="--", label="Raw: same mask")
            maximum = max(row["mse"] for row in tables["MATCHED_COMPLETE_DIAGNOSTIC"] if row["dataset"] == dataset and row["split"] == split and row["metric_scale"] == "normalized")
            ax.set_ylim(0, maximum * 1.25); ax.set_xticks(np.arange(2), ("Past PCA", "FIT future space")); ax.set_ylabel("Complete-C normalized MSE")
            ax.set_title(f"{DISPLAY[dataset]} / {split.upper()} / {raw['complete_target_positions']}/{raw['total_target_positions']} complete positions")
            ax.grid(axis="y", alpha=.2)
    handles, names = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, names, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle("A useful output span and a predictable latent future are different questions", fontsize=13)
    save(fig, "02_output_space_and_forecaster_gap")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, dataset in enumerate(DATASETS):
        for pi, route in enumerate(PROJECTIONS):
            ax = axes[di, pi]; loss_ax = ax.twinx()
            for seed, color in zip(SEEDS, ("#0072B2", "#D55E00")):
                points = sorted([row for row in tables["CHECKPOINTS"] if row["dataset"] == dataset and row["projection"] == route and row["seed"] == seed], key=lambda row: row["update"])
                ax.plot([row["update"] for row in points], [row["mse"] for row in points], "o-", color=color, label=f"TUNE MSE s{seed[-1]}")
                selected = next(row for row in points if row["selected"]); ax.scatter(selected["update"], selected["mse"], marker="*", s=150, color=color, edgecolor="black", zorder=5)
                curve = [row for row in tables["LEARNING_CURVE"] if row["dataset"] == dataset and row["projection"] == route and row["seed"] == seed]
                rolling = np.convolve([row["loss"] for row in curve], np.ones(25) / 25, mode="valid")
                loss_ax.plot(np.arange(25, 601), rolling, color=color, alpha=.24, linewidth=.9)
            ax.set_title(f"{DISPLAY[dataset]} / {route}"); ax.set_xlabel("Update (same 600 for both seeds)"); ax.set_ylabel("Full TUNE MSE / checkpoint selection")
            loss_ax.set_ylabel("Minibatch FIT loss: 25-update average", color="#777777"); ax.grid(alpha=.2)
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Stars are earliest minimum-TUNE checkpoints; seeds change minibatch order, not zero initialization", fontsize=12)
    save(fig, "03_learning_and_checkpoint_selection")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    for di, dataset in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di, bi]; rows = [choose(tables["WHOLE_COST"], dataset=dataset, batch=batch, route=route) for route in ROUTES]
            values = [row["median_seconds_per_window"] * 1000 for row in rows]
            ax.bar(np.arange(5), values, color=COLORS, alpha=.8)
            maximum = max(max(row["seconds"]) * 1000 / batch for row in rows)
            for index, row in enumerate(rows):
                samples = np.asarray(row["seconds"]) * 1000 / batch; ax.scatter(index + np.linspace(-.18, .18, 6), samples, color="black", s=13)
                ax.text(index, max(values[index], samples.max()) + maximum * .025, f"{values[index]:.2f} ms\nratio {row['ratio_of_RAW_medians']:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, maximum * 1.32); ax.set_xticks(np.arange(5), ("Raw Bolt", "PCA/Bolt", "Target/Bolt", "PCA/native", "Target/native"), rotation=15, ha="right")
            ax.set_title(f"{DISPLAY[dataset]} / B={batch} / six actual repeats"); ax.set_ylabel("Whole latency per window (ms)"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("One NEW trial: causal fill -> normalize -> E -> predictor -> D -> physical CPU return; native seed 92601", fontsize=12)
    save(fig, "04_whole_deployment_cost")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), layout="constrained")
    for di, (dataset, (_, k, _)) in enumerate(DATASETS.items()):
        ax = axes[di, 0]; rows = [row for row in tables["FIT_TARGET_SPECTRUM"] if row["dataset"] == dataset]
        ax.bar([row["component"] for row in rows], [row["energy_fraction"] for row in rows], color=["#0072B2" if row["retained_top_K"] else "#BBBBBB" for row in rows])
        retained = sum(row["energy_fraction"] for row in rows if row["retained_top_K"])
        ax.set_title(f"{DISPLAY[dataset]}: FIT future innovation, top K={k} energy {retained:.3f}"); ax.set_xlabel("Singular component"); ax.set_ylabel("Fraction of total future innovation energy"); ax.grid(axis="y", alpha=.2)
        ax = axes[di, 1]; rows = [row for row in tables["RESTRICTED_PAST_VARIANCE"] if row["dataset"] == dataset]
        ax.bar([row["component"] for row in rows], [row["FIT_past_restricted_eigenvalue"] for row in rows], color="#009E73")
        ax.set_title(f"{DISPLAY[dataset]}: past variance within target-selected span"); ax.set_xlabel("Restricted covariance component"); ax.set_ylabel("FIT past eigenvalue before whitening"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("FIT label supervision and past whitening diagnostics do not establish predictability", fontsize=12)
    save(fig, "05_FIT_space_diagnostics")
    return paths


def text_report(result, audit, tables, output, linked):
    summary = []
    for dataset, (_, _, dev) in DATASETS.items():
        old = choose(tables["COMPARISON"], dataset=dataset, projection="PCA_PINV", split=dev, predictor="BOLT", seed=None, metric_scale="normalized")
        new = choose(tables["COMPARISON"], dataset=dataset, projection="FIT_TARGET_PINV", split=dev, predictor="BOLT", seed=None, metric_scale="normalized")
        old_floor = choose(tables["OUTPUT_SPACE_GEOMETRY"], dataset=dataset, projection="PCA_PINV", split=dev)["fixed_D_minimum_normalized_MSE"]
        new_floor = choose(tables["OUTPUT_SPACE_GEOMETRY"], dataset=dataset, projection="FIT_TARGET_PINV", split=dev)["fixed_D_minimum_normalized_MSE"]
        summary.append(f"[확인] {DISPLAY[dataset]} {dev.upper()}에서 FIT-target 공간의 고정 D 정규화 MSE 최소값은 {new_floor:.9f}, PCA의 최소값은 {old_floor:.9f}이다. 실제 Bolt 연결 MSE는 각각 {new['mse']:.9f}, {old['mse']:.9f}이다. 출력 공간과 실제 예측의 방향은 이 두 비교를 따로 읽어 판단한다.")
    lines = ["# FIT 미래에 맞춘 출력 공간은 과거 PCA보다 예측에 도움이 되는가", "", *summary, "",
        "## 이번에 직접 시험한 것", "",
        "작은 K에서 고정 Bolt를 쓰면서 C 예측 정확도와 비용을 유지하려는 목적을 유지한다. 이번 대조는 압축을 못 하는 이유를 reconstruction 학습 하나로 설명하는 실험이 아니다. 기존 POINTWISE/MAIN과 native 경로에도 actual-Y 목적 학습이 있었다. 새롭게 시험한 좁은 질문은 ‘FIT 미래 변화가 많이 놓이는 출력 방향을 먼저 정하면, 과거 PCA 방향보다 같은 예측기가 잘 동작하는가’다. 상관된 채널이 많고 선형 readout 이득이 작다는 사실만으로 저차원 압축을 배제하지도 않는다.", "",
        "PCA_PINV는 기존 FIT 과거 PCA encoder/center를 그대로 재사용하고 D만 같은 pinv(E) 규칙으로 만든 control이다. FIT_TARGET_PINV는 완전히 관측된 C의 FIT 미래 위치에서 `Y - current-C anchor`의 비중심 rank-K SVD 방향을 선택한다. 선택한 공간 안에서 같은 FIT 과거의 covariance로 whitening한 encoder와 그 FP32 배포 encoder의 pseudoinverse decoder를 고정한다. 미래 평균이나 intercept는 맞추지 않는다. cutoff는 고정된 수치 계약이며 이번 실행의 결과를 보고 바꾸지 않았다. 이는 **supervised FIT-future 표현 학습**이고 비지도 학습이나 ‘학습에도 과거만 사용’한 표현이라고 부를 수 없다. D의 pinv 연산에 label을 직접 넣지는 않아도 supervised E에서 D를 만들므로 D 역시 FIT 미래의 간접 감독을 받는다. 학습 후 배포 encoder와 predictor의 입력은 과거만 사용한다. per-split 미래로 방향까지 고르는 이전 rank oracle와도 다르다.", "",
        "두 공간 모두 D = pinv(E32를 FP64로 해석, rcond=1e-15)을 FP32로 cast하는 규칙이다. past normalization·L512/H48·K·raw-current-C anchor·FIT/TUNE/개발 origins·원 mask·고정 Bolt를 맞췄다. C anchor 때문에 엄격한 K-only 압축은 아니다. 예측식은 FP32 `raw_last + (latent_future - encoded_history_last) @ D`이며 decoder bias는 innovation에서 제외한다. 이전 original-LS D로 학습한 head를 새 inverse-D의 최적화된 control로 재사용하지 않았다. PCA_PINV와 FIT_TARGET_PINV 모두 같은 구조의 native head 두 minibatch-order seed를 새로 600 updates 학습했다. zero initialization이므로 두 seed가 다른 초기 가중치의 독립 ensemble인 것도 아니다.", "",
        "## 같은 비교와 감독 신호의 경계", "",
        "Robin C17/K4와 Beijing C11/K2, FIT256·TUNE64·이미 노출된 개발 구간을 유지했다. target 방향은 complete-C FIT 미래 위치만으로 학습한다. 반면 head 손실은 원 관측된 미래 target 전체에 대해 채널별 관측 수로 나눈 정규화 macro MSE다. 따라서 complete-mask 공간 에너지 목적과 head의 full observed 채널 가중치가 정확히 같은 목적은 아니다. 데이터가 결측이면 이 차이가 결과 해석에 남는다. 겹치는 FIT 창의 반복 위치도 새로운 독립 표본처럼 세지 않는다. 과거 whitening에는 causal forward-filled FIT 과거가 들어가고 결측의 원 관측 여부는 따로 보존한다.", "",
        "future-label perturbation 검사는 실제 추출된 두 창에서 고정된 E/D/F/head forward가 target과 미래 mask 변경에 반응하지 않는지 확인한 범위다. 전체 원자료 미래 변경 후 학습부터 재실행한 누수 검사나 성능 개선 증명으로 바꾸지 않는다. TUNE은 정해진 checkpoint의 정규화 MSE 선택에만 쓰며 exact tie는 더 이른 update를 선택한다. 이번 실행의 개발 구간 및 실제 미래를 쓰는 oracle는 그 선택 이후에 계산했다. 그러나 이 후속 방법의 설계 동기는 이미 노출된 DEV·공간 최소값·rank oracle를 본 뒤 정했다. 따라서 탐색적·적응적 대조이며, FIT-only 행렬 학습이나 이번 checkpoint gate에 oracle를 사용하지 않은 사실을 ‘전체 방법 선정 동기에도 DEV/oracle 영향이 없었다’는 주장으로 바꾸지 않는다. 보호된 TEST는 읽지 않았고, 노출된 개발 결과는 독립 holdout 일반화 검증이 아니다. 출력 공간의 방향을 원래 사용자의 문장 그대로 승인받았다고 주장하는 대신 기록된 continuation 범위와 계획을 provenance로 연결한다.", "",
        "## 실제 full observed 예측 원수치", "", "```text", "Dataset / Split / Space / Predictor / Seed     normalized MSE      normalized MAE"]
    for row in tables["COMPARISON"]:
        if row["split"] != "fit" and row["metric_scale"] == "normalized":
            lines.append(f"{row['dataset']}/{row['split']}/{row['projection']}/{row['predictor']}/{row['seed']}  {row['mse']:.9f}  {row['mae']:.9f}")
    lines += ["```", "", "![Actual matched forecasts](figures/01_actual_forecasts.png)", "",
        "[확인] 위 표의 native는 새 D·E 조건에서 새로 학습한 비교다. 두 seed를 모두 표시했으며 DEV를 보고 좋은 seed만 채택하지 않았다. 전체 FIT/TUNE/DEV와 물리 단위 채널별 오차·관측 수는 COMPARISON CSV에 보존했다. 채널마다 단위가 다르므로 물리 단위 macro 평균은 보조 진단이고 공통 물리 효용의 최솟값이라고 할 수 없다. 고정된 단순 per-latent affine 512→48 head의 실패라면 모든 비선형·다변량·다른 시간 모델의 실패로 일반화할 수 없다.", "",
        "## 공간 일반화와 잠재 미래 예측 문제 분리", "",
        "고정 D 진단은 FIT/TUNE/DEV 각각의 알려진 `Y-currentC`를 배포 D32의 행공간에 실수 연산으로 최소제곱 투영한다. 같은 complete-C target 위치에서 나온 정규화 MSE만 해당 공간의 수학적 최소값이다. MAE나 물리 단위 MSE는 그 최소해의 점수일 뿐 각각의 최솟값이 아니다. FP32 roundoff 차이는 별도로 기록하고, Raw는 고정 D 공간에 갇힌 예측기가 아니므로 Raw에 하한 부등식을 적용하지 않는다. E(Y)는 알려진 실제 미래를 E로 인코딩한 비배포 진단이며 일반적인 최적 잠재 미래와 같은 것으로 가정하지 않는다.", "",
        "FIT-target 공간은 FIT의 미래 energy를 많이 보존하도록 골랐으므로 FIT에서의 좋은 공간 최소값은 학습 목적과 직접 연결된다. 그것만으로 TUNE/DEV 공간 일반화나 과거에서 잠재 미래를 알아낼 수 있다는 결론은 나오지 않는다. DEV의 고정 D 최소값 자체가 나쁘면 출력 공간 전이가 남은 문제이고, 최소값은 좋아도 실제 F/native가 크게 나쁘면 과거 정보·잠재 미래 예측·해당 predictor 학습 또는 표현 연결 문제를 더 분리해야 한다. 동일 mask에서 계산한 두 단계를 비교한다. 낮은 공간 최소값이 성능 개선의 달성값도 아니며, 모든 경우의 절대적인 오차 lower bound도 아니다.", "", "![Fixed output space and forecast gap](figures/02_output_space_and_forecaster_gap.png)", "",
        "[반례·추정] 큰 미래 변화 방향은 반드시 예측 가능한 방향이 아니다. 예를 들어 큰 독립 잡음이 한 채널에 놓이면 target SVD는 그 방향을 보존하지만 과거에는 그 잡음의 미래값을 알아낼 단서가 없다. 그러므로 이 supervised energy control이 실패하더라도 예측 목적 학습의 모든 형태를 부정할 수 없고, 성공하더라도 미래 에너지 자체가 인과적인 예측 정보임을 증명한 것은 아니다. 현재 출력 방향 선택은 과거로부터 예측 가능한 변화량을 직접 최적화한 학습이 아니며 dynamic graph나 contrastive 정렬도 없다.", "",
        "## 같은 학습량과 checkpoint 선택", "",
        "각 새 head는 batch4·AdamW 0.001·weight decay0·clip1·600 updates, CP0/100/200/400/600의 TUNE MSE 선택을 맞췄다. CP0는 zero latent innovation이므로 current-C persistence와 동일하다. 그림의 별은 TUNE 전체 평가의 선택 checkpoint이고 흐린 선은 25-update 평균의 minibatch FIT loss다. minibatch active-channel macro 손실과 전체 FIT loss를 같은 양으로 표시하지 않는다. E/D와 Bolt는 head 학습 중 고정되어 있다. 이 일정에서 안정적으로 학습됐는지와 목적에 맞는 표현인지는 별개다. seed별 학습 시간·parameter 수·gradient 원기록은 별도 CSV에 보존했다.", "", "![Learning and sealed checkpoint selection](figures/03_learning_and_checkpoint_selection.png)", "",
        "## 실제 전체 배포 비용과 채택 조건", "",
        "비용은 두 데이터 × B1/B8 × Raw/PCA Bolt/target Bolt/PCA native/target native의 20조건, 각 6회로 총120개 actual timer sample이다. native 비용은 seed92601 하나이며 seed92602의 비용으로 대신 말하지 않는다. causal fill·FIT normalize·E·실제 predictor forward·D·물리 단위 CPU return과 synchronization을 포함한다. 데이터·모델·state 로딩, 공간 SVD/whitening fitting, warmup·저장·hash·타이머 밖 검사는 제외한다. 학습 비용은 따로 표시한다. 단계별 profile이나 per-method 독점 memory를 측정하지 않았으며 F와 head가 공유된 프로세스의 RSS/VRAM 관측이다.", "",
        "첫 다섯 순서는 회전으로 방법별 위치를 균등하게 하고 여섯 번째는 seeded 추가 순서다. 여섯 순서 전체가 완벽하게 balanced인 것은 아니다. 같은 첫 B개의 개발 origins를 반복하므로 120번 측정은 독립 데이터 일반화 표본120개가 아니다. latency는 이번 새 시행 내부에서만 Raw와 비교한다. 이전 D-only 시행의 절대 시간과 연결해 개선을 주장하지 않는다. 사전 채택 조건은 같은 TUNE MSE·MAE 각각 Raw의1.01배 이하와 B8 중앙 latency 비율0.90 이하의 동시 충족이다. paired ratio의 중앙값도 CSV에 남기지만 ratio of medians 대신 gate에 쓰지 않는다.", "", "```text", "Dataset / Route / B     median ms/window     ratio of medians    paired median ratio"]
    for row in tables["WHOLE_COST"]:
        lines.append(f"{row['dataset']}/{row['route']}/B{row['batch']} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
    passes = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    lines += ["```", "", "![Actual whole deployment timing](figures/04_whole_deployment_cost.png)", "",
        f"[확인] 이 기록에서 사전 세 조건을 모두 충족하는 후보는 {passes}/{len(tables['REGISTERED_ADOPTION'])}개다. 이 숫자는 정해진 기준 재검산이고 DEV를 본 뒤 새 method 선택을 했다는 뜻이 아니다. 속도 또는 일부 metric 하나의 개선만으로 전체 목적이 해결됐다고 기록하지 않는다. 여섯 반복에서의 기술통계이고 배포 환경 전체의 유의한 속도 일반화를 입증한 것도 아니다.", "",
        "## FIT 공간의 수치적 진단", "",
        "FIT innovation spectrum의 top-K retained energy는 supervised 출력 방향이 어떤 에너지를 남기는지 보여준다. 해당 fraction은 예측 가능성 점수가 아니다. target-selected 공간의 FIT-past restricted covariance eigenvalue는 whitening이 가능한 past variance를 확인하는 다른 양이다. 두 spectrum은 미래 innovation과 과거 variance로 분모와 목적이 다르다. whitening 후 unit covariance가 성립하는 것도 구성된 수치 계약의 확인이고 시간 패턴 학습이나 미래 일반화 성공이 아니다. 방향 경계의 작은 singular gap과 overlap·결측의 선택 범위는 FIT metadata에 그대로 남긴다.", "", "![FIT label energy and restricted past variance](figures/05_FIT_space_diagnostics.png)", "",
        "## 지금 확인된 것과 미확정인 것", "",
        "[확인] 지정한 supervised FIT 공간을 만들고 과거 전용 E를 통해 고정 Bolt 및 같은 새 native head에 연결하는 비교를 구현·실제 추론·비용·검산까지 실행했다. 학습 label 사용은 명시적으로 구분했다. 부정적인 결과가 있어도 이 비교 조건을 비용 단계에서 생략하지 않았으며 기존 결과·모델·선택 lock을 덮어쓰지 않았다. 이번 완료 범위는 출력 방향 선택 대조와 정해진 predictor의 진단이다.", "",
        "[추정·다음 판단] FIT 방향이 DEV에도 충분한 공간을 주는지와 그 공간의 미래를 과거로 예측할 수 있는지를 먼저 나누어 봐야 한다. 조건이 유리한 공간이더라도 native와 Bolt 모두 gap이 남으면 단순 energy supervision에서 predictive supervision으로 바꾸는 대조, matched 출력 목적과 predictor의 표현 결합, 더 단순한 conditional baseline을 판단할 근거가 된다. 단, 이 다음 방법을 이미 실행했다거나 대조학습·동적 그래프·ExoTimer가 필요하다고 입증한 것은 아니다. 고정 Bolt를 유지할 근거는 같은 공간에서의 정확도·비용 비교로 평가하며 추가 방법 전체의 연구 해결을 선언하지 않는다. 논문 기여점은 가능한 주장과 실험 범위를 좁혀야 하며, 기존 실제 actual-Y 학습의 실패를 지워 새 방법만으로 해결됐다고 쓰지 않는다.", "",
        "## 원수치·검산·이전 결과", ""]
    for name, path in linked.items(): lines.append(f"- [{name}]({rel(path, output)}) — SHA256 `{sha(path)}`")
    for name, rows in tables.items(): lines.append(f"- [{name}: {len(rows)} rows](tables/{name}.csv)")
    lines += ["", "보고서 생성기는 JSON metadata만 읽는다. NPZ·checkpoint·모델을 로딩하거나 학습·F forward를 실행하지 않는다. 보고서 완료를 독립 일반화 검증 또는 전체 연구 목표 달성으로 바꾸지 않았고, TEST 및 자동 commit/push는 수행하지 않았다.", ""]
    return "\n".join(lines)


def generate(result_path, audit_path, main_report, rank_report, decoder_report, output):
    result, audit = load_verified(result_path, audit_path)
    prior = (main_report, rank_report, decoder_report)
    require(all(Path(path).is_file() and Path(path).suffix == ".md" for path in prior), "THREE_PRIOR_MD_REPORTS_REQUIRED")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    tables = collect_tables(result, audit); (output / "tables").mkdir()
    for name, rows in tables.items(): write_csv(output / "tables" / (name + ".csv"), rows)
    images = figures(tables, output)
    linked = {"actual producer": result_path, "independent audit": audit_path, "matched projection report": main_report,
              "rank oracle v2 report": rank_report, "decoder intervention v2 report": decoder_report, "report generator": Path(__file__)}
    (output / "REPORT_KO.md").write_text(text_report(result, audit, tables, output, linked), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": record(Path(__file__)), "producer": record(result_path), "audit": record(audit_path),
        "prior_reports": [record(path) for path in prior], "source_bindings": {name: record(path) for name, path in linked.items()},
        "metadata_only": True, "NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False,
        "supervised_FIT_future_training": True, "deployment_past_only": True, "past_only_or_unsupervised_training_claimed": False,
        "independent_generalization_claimed": False, "new_model_or_method_selected_posthoc": False,
        "fixed_D_MAE_or_physical_MSE_minimum_claimed": False, "RAW_bound_applied": False,
        "CSV_rows": {name: len(rows) for name, rows in tables.items()}, "figures": [rel(path, output) for path in images],
        "files": [{"path": rel(path, output), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()]}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("result", "audit", "main-report", "rank-report", "decoder-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    receipt = generate(args.result, args.audit, args.main_report, args.rank_report, args.decoder_report, args.output)
    print(json.dumps({"status": receipt["status"], "CSV_rows": receipt["CSV_rows"]}))
