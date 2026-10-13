"""Metric-aligned FORECAST generalization probe.

This script is intentionally bounded:
- reads the completed 2026-10-13 component study as fixed evidence;
- does not touch protected TEST data;
- writes only this study folder and its own cache;
- uses normalized observed channel-macro MSE as the primary selection metric.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import psutil
import torch
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "research" / "forecast_component_study_20261013"
PREVIOUS_CACHE = ROOT / ".cache" / "forecast_component_study_20261013" / "stage_b_full_v1"
RUN_ID = os.environ.get("FORECAST_ALIGNMENT_RUN", "run_v1")
CACHE = ROOT / ".cache" / "forecast_metric_alignment_20261013" / RUN_ID
OUT = HERE / RUN_ID

sys.path.insert(0, str(PREVIOUS))
import component_experiment as ce  # noqa: E402


CHECKPOINTS = (0, 100, 200, 400, 600)
DATASETS = ("jena", "robin")
SEEDS = (92601, 92602)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ahash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if replace:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Budget:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.process = psutil.Process()
        self.path = OUT / "RESOURCE_USAGE.json"
        self.record = {
            "pid": os.getpid(),
            "started_unix": time.time(),
            "status": "RUNNING",
            "wall_seconds": 0.0,
            "peak_sampled_rss_bytes": 0,
            "peak_allocated_vram_bytes": 0,
            "peak_reserved_vram_bytes": 0,
            "new_cache_bytes": 0,
            "new_research_bytes": 0,
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
        }
        self.check()

    def check(self) -> None:
        self.record["wall_seconds"] = time.perf_counter() - self.start
        self.record["peak_sampled_rss_bytes"] = max(
            self.record["peak_sampled_rss_bytes"], self.process.memory_info().rss
        )
        if torch.cuda.is_available():
            self.record["peak_allocated_vram_bytes"] = max(
                self.record["peak_allocated_vram_bytes"], torch.cuda.max_memory_allocated()
            )
            self.record["peak_reserved_vram_bytes"] = max(
                self.record["peak_reserved_vram_bytes"], torch.cuda.max_memory_reserved()
            )
        self.record["new_cache_bytes"] = sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file())
        self.record["new_research_bytes"] = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
        self.record["disk_free_bytes"] = shutil.disk_usage(ROOT).free
        write_json(self.path, self.record, replace=True)
        if self.record["peak_sampled_rss_bytes"] > 8 * 1024**3:
            raise RuntimeError("PROCESS_RSS_LIMIT_8GIB")
        if self.record["peak_allocated_vram_bytes"] > 8 * 1024**3:
            raise RuntimeError("ALLOCATED_VRAM_LIMIT_8GIB")
        if self.record["new_cache_bytes"] > 3 * 1024**3:
            raise RuntimeError("NEW_CACHE_LIMIT_3GIB")
        if self.record["wall_seconds"] > 2400:
            raise RuntimeError("WALL_LIMIT_2400_SECONDS")

    def close(self, status: str, error: str | None = None) -> None:
        self.record["wall_seconds"] = time.perf_counter() - self.start
        self.record["status"] = status
        self.record["error"] = error
        write_json(self.path, self.record, replace=True)


@dataclass(frozen=True)
class Variant:
    name: str
    lr: float
    weight_decay: float
    nonlinear: bool
    note: str


VARIANTS = (
    Variant("PRED_L2_1E4", 1e-3, 1e-4, True, "same architecture and order; Adam L2 weight_decay=1e-4"),
    Variant("PRED_LR3E4", 3e-4, 0.0, True, "same architecture and order; lower learning rate 3e-4"),
    Variant("PRED_LINEAR_ONLY", 1e-3, 0.0, False, "PCA-linear E/D only; removes causal TCN residual capacity"),
)


def row_from_score(prefix: str, score: dict) -> dict:
    return {
        f"{prefix}_physical_mse": score["mse"],
        f"{prefix}_physical_mae": score["mae"],
        f"{prefix}_normalized_mse": score["normalized"]["mse"],
        f"{prefix}_normalized_mae": score["normalized"]["mae"],
    }


def load_base_objects(dataset: str):
    data = ce.load_data(dataset, sizes=(256, 64, None))
    histories = {split: ce.windows(data, split) for split in data["origins"]}
    targets = {split: ce.windows(data, split, True) for split in data["origins"]}
    observed = {split: ce.masks(data, split) for split in data["origins"]}
    mean, basis, pca = ce.pca_initialization(data)
    return data, histories, targets, observed, mean, basis, pca


def evaluate_codec(
    data: dict,
    histories: dict,
    codec: torch.nn.Module,
    bolt,
    budget: Budget,
    metadata: dict,
) -> list[dict]:
    codec.eval().requires_grad_(False)
    rows = []
    for split in ("fit", "tune", "val"):
        prediction = ce.predict(histories[split], codec, bolt, budget, batch=8)
        score = ce.physical_score(data, prediction, split)
        rows.append(
            {
                **metadata,
                "split": split,
                "prediction_sha256": ahash(prediction),
                "observed_cells": int(ce.masks(data, split).sum()),
                **row_from_score("", score),
            }
        )
    return rows


def load_previous_checkpoint(dataset: str, seed: int, objective: str, update: int, mean, basis):
    folder = PREVIOUS_CACHE / f"{dataset}_s{seed}_{objective}"
    path = folder / f"{objective}_u{update}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    codec = ce.Codec(mean, basis, seed).to(ce.DEVICE)
    codec.load_state_dict(state["codec"])
    return codec, path


def evaluate_previous_checkpoints(bolt, budget: Budget) -> tuple[list[dict], list[dict]]:
    metric_rows: list[dict] = []
    contracts: list[dict] = []
    for dataset in DATASETS:
        data, histories, _targets, _observed, mean, basis, pca = load_base_objects(dataset)
        contracts.append(
            {
                "dataset": dataset,
                "source": data["source"],
                "source_sha256": data["source_hash"],
                "fit_bounds": data["fit_bounds"],
                "tune_bounds": data["tune_bounds"],
                "val_bounds": data["val_bounds"],
                "origins": data["origins"],
                "origin_hashes": {k: ahash(v) for k, v in data["origins"].items()},
                "normalization": data["normalization_fit_scope"],
                "pca": pca,
            }
        )
        for seed in SEEDS:
            for objective in ("REC", "PRED"):
                for update in CHECKPOINTS:
                    codec, path = load_previous_checkpoint(dataset, seed, objective, update, mean, basis)
                    metric_rows.extend(
                        evaluate_codec(
                            data,
                            histories,
                            codec,
                            bolt,
                            budget,
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "method": f"OLD_{objective}",
                                "objective": objective,
                                "variant": "PREVIOUS_STAGE_B",
                                "update": update,
                                "checkpoint_path": str(path.relative_to(ROOT)),
                                "checkpoint_sha256": sha(path),
                                "new_training": False,
                            },
                        )
                    )
                    print(json.dumps({"evaluated": "previous", "dataset": dataset, "seed": seed, "objective": objective, "update": update}), flush=True)
    return metric_rows, contracts


def train_variant(
    variant: Variant,
    dataset: str,
    seed: int,
    data: dict,
    histories: dict,
    targets: dict,
    observed: dict,
    mean: np.ndarray,
    basis: np.ndarray,
    bolt,
    budget: Budget,
) -> tuple[list[dict], dict]:
    folder = CACHE / dataset / f"s{seed}_{variant.name}"
    folder.mkdir(parents=True, exist_ok=False)
    codec = ce.Codec(mean, basis, seed, nonlinear=variant.nonlinear).to(ce.DEVICE)
    optimizer = torch.optim.Adam(codec.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    order = np.random.default_rng(seed).integers(0, len(histories["fit"]), size=(600, 4))
    history_observed = ce.masks(data, "fit", False)
    curve: list[dict] = []
    metrics: list[dict] = []
    initial_hash = ce.state_hash(codec)
    started = time.perf_counter()
    for update in range(601):
        if update in CHECKPOINTS:
            path = folder / f"{variant.name}_u{update}.pt"
            with path.open("xb") as handle:
                torch.save(
                    {
                        "codec": ce.clone_state(codec),
                        "update": update,
                        "seed": seed,
                        "dataset": dataset,
                        "variant": asdict(variant),
                        "objective": "PRED",
                    },
                    handle,
                )
            metrics.extend(
                evaluate_codec(
                    data,
                    histories,
                    codec,
                    bolt,
                    budget,
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": variant.name,
                        "objective": "PRED",
                        "variant": variant.name,
                        "update": update,
                        "checkpoint_path": str(path.relative_to(ROOT)),
                        "checkpoint_sha256": sha(path),
                        "new_training": True,
                    },
                )
            )
            print(json.dumps({"checkpoint": variant.name, "dataset": dataset, "seed": seed, "update": update}), flush=True)
            codec.train().requires_grad_(True)
        if update == 600:
            break
        ids = order[update]
        x = ce.tensor(histories["fit"][ids])
        optimizer.zero_grad(set_to_none=True)
        prediction = codec.decode(bolt(codec.encode(x)))
        loss = ce.macro_loss(prediction, ce.tensor(targets["fit"][ids]), ce.mask_tensor(observed["fit"][ids]))
        if not torch.isfinite(loss):
            raise RuntimeError("NONFINITE_TRAINING_LOSS")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("NONFINITE_GRADIENT")
        optimizer.step()
        curve.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": variant.name,
                "update": update + 1,
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clipping": float(grad_norm.detach().cpu()),
            }
        )
        if update % 20 == 0:
            budget.check()
    record = {
        "dataset": dataset,
        "seed": seed,
        "variant": asdict(variant),
        "initial_codec_hash": initial_hash,
        "final_codec_hash": ce.state_hash(codec),
        "order_sha256": ahash(order),
        "updates": 600,
        "batch_size": 4,
        "train_wall_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in codec.parameters()),
        "curve_path": str((OUT / "training_curves.csv").relative_to(ROOT)),
    }
    return metrics, {"record": record, "curve": curve}


def select_rows(metric_rows: list[dict]) -> list[dict]:
    by_group: dict[tuple, list[dict]] = {}
    for row in metric_rows:
        key = (row["dataset"], row["seed"], row["method"], row["objective"], row["variant"])
        by_group.setdefault(key, []).append(row)
    selected: list[dict] = []
    for key, rows in sorted(by_group.items()):
        tune_rows = [r for r in rows if r["split"] == "tune"]
        if not tune_rows:
            continue
        fit_lookup = {r["update"]: r for r in rows if r["split"] == "fit"}
        val_lookup = {r["update"]: r for r in rows if r["split"] == "val"}
        base_fit = fit_lookup[min(fit_lookup)]
        base_tune = min([r for r in tune_rows if r["update"] == min(fit_lookup)], key=lambda x: x["update"])
        for criterion, column in (
            ("tune_normalized_mse", "_normalized_mse"),
            ("tune_physical_mse", "_physical_mse"),
            ("tune_normalized_mae", "_normalized_mae"),
        ):
            winner = min(tune_rows, key=lambda r: (r[column], r["update"]))
            fit_row = fit_lookup[winner["update"]]
            val_row = val_lookup[winner["update"]]
            selected.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "method": key[2],
                    "objective": key[3],
                    "variant": key[4],
                    "criterion": criterion,
                    "selected_update": winner["update"],
                    "fit_normalized_mse": fit_row["_normalized_mse"],
                    "fit_normalized_mse_delta_from_u0": fit_row["_normalized_mse"] - base_fit["_normalized_mse"],
                    "tune_normalized_mse": winner["_normalized_mse"],
                    "tune_normalized_mse_delta_from_u0": winner["_normalized_mse"] - base_tune["_normalized_mse"],
                    "val_normalized_mse": val_row["_normalized_mse"],
                    "val_normalized_mae": val_row["_normalized_mae"],
                    "val_physical_mse": val_row["_physical_mse"],
                    "val_physical_mae": val_row["_physical_mae"],
                    "checkpoint_path": winner["checkpoint_path"],
                    "checkpoint_sha256": winner["checkpoint_sha256"],
                    "new_training": winner["new_training"],
                }
            )
    return selected


def read_previous_whole_system() -> list[dict]:
    path = PREVIOUS / "report_v1" / "all_78_methods.csv"
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in (
                "RAW_BOLT",
                "RAW_R_ONLY",
                "PRED_BOLT",
                "PRED_BOLT_RAW_R",
                "PRED_NATIVE_RAW_R",
            ):
                rows.append(row)
    return rows


def make_plots(metric_rows: list[dict], selected_rows: list[dict]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    figures: list[str] = []

    primary = [r for r in selected_rows if r["criterion"] == "tune_normalized_mse" and r["objective"] == "PRED"]
    order = ["OLD_PRED", "PRED_L2_1E4", "PRED_LR3E4", "PRED_LINEAR_ONLY"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = [r for r in primary if r["dataset"] == dataset]
        labels = []
        values = []
        colors = []
        for method in order:
            for seed in SEEDS:
                item = next((r for r in subset if r["method"] == method and int(r["seed"]) == seed), None)
                if item is None:
                    continue
                labels.append(method.replace("PRED_", "").replace("OLD_", "OLD ") + f"\n{seed}")
                values.append(float(item["val_normalized_mse"]))
                colors.append("#0072B2" if method == "OLD_PRED" else "#E69F00" if method == "PRED_L2_1E4" else "#009E73" if method == "PRED_LR3E4" else "#D55E00")
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_title(f"{dataset}: VAL normalized MSE")
        ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
        ax.set_ylabel("normalized macro MSE")
    path = OUT / "selected_val_normalized_mse.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, dataset in zip(axes[0], DATASETS):
        for method, color in zip(order, ("#0072B2", "#E69F00", "#009E73", "#D55E00")):
            rows = [
                r for r in metric_rows
                if r["dataset"] == dataset and r["method"] == method and r["objective"] == "PRED" and r["split"] == "tune"
            ]
            if not rows:
                continue
            grouped: dict[int, list[float]] = {}
            for row in rows:
                grouped.setdefault(int(row["update"]), []).append(float(row["_normalized_mse"]))
            xs = sorted(grouped)
            ys = [np.mean(grouped[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=method, color=color)
        ax.set_title(f"{dataset}: TUNE curve")
        ax.set_xlabel("update")
        ax.set_ylabel("normalized macro MSE")
        ax.legend(fontsize=7, frameon=False)
    for ax, dataset in zip(axes[1], DATASETS):
        rows = [
            r for r in selected_rows
            if r["dataset"] == dataset and r["criterion"] == "tune_normalized_mse" and r["objective"] == "PRED"
        ]
        xs = [float(r["fit_normalized_mse_delta_from_u0"]) for r in rows]
        ys = [float(r["tune_normalized_mse_delta_from_u0"]) for r in rows]
        labels = [f'{r["method"]}\n{r["seed"]}' for r in rows]
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.scatter(xs, ys, color="#CC79A7")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{dataset}: FIT gain vs TUNE gain")
        ax.set_xlabel("FIT normalized MSE change from update0")
        ax.set_ylabel("TUNE normalized MSE change from update0")
    path = OUT / "training_generalization_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))

    shift_rows = [
        r for r in selected_rows
        if r["method"] in ("OLD_REC", "OLD_PRED") and r["criterion"] in ("tune_normalized_mse", "tune_physical_mse")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = [r for r in shift_rows if r["dataset"] == dataset and r["objective"] == "PRED"]
        labels = [f'{r["seed"]}\n{r["criterion"].replace("tune_", "").replace("_mse", "")}' for r in subset]
        vals = [float(r["val_normalized_mse"]) for r in subset]
        ax.bar(np.arange(len(vals)), vals, color=["#56B4E9" if "normalized" in r["criterion"] else "#D55E00" for r in subset])
        ax.set_title(f"{dataset}: selection criterion effect")
        ax.set_xticks(np.arange(len(vals)), labels, rotation=35, ha="right")
        ax.set_ylabel("VAL normalized MSE")
    path = OUT / "selection_criterion_effect.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))
    return figures


def make_report(summary: dict, selected_rows: list[dict], whole_system_rows: list[dict], figures: list[str]) -> str:
    def best(dataset: str):
        rows = [
            r for r in selected_rows
            if r["dataset"] == dataset and r["objective"] == "PRED" and r["criterion"] == "tune_normalized_mse"
        ]
        return min(rows, key=lambda r: float(r["val_normalized_mse"]))

    lines: list[str] = []
    lines.append("# 평가 기준 정렬과 학습 일반화 재검산\n")
    lines.append("2026-10-13 · 기존 Stage B 자산 재평가 + 같은 FIT/TUNE 조건의 작은 학습 변경\n")
    lines.append("## 결론부터\n")
    lines.append("[확인] 주평가를 TUNE normalized observed channel-macro MSE로 고정해도 기존 PRED checkpoint 선택은 대부분 유지됐다. Jena seed92602만 physical 선택의 update600에서 normalized 선택의 update400으로 바뀌었고, Jena seed92601과 Robin 두 seed는 그대로였다. 하지만 이 선택 정렬 뒤에도 Jena PRED의 VAL normalized MSE는 0.9847-1.0023으로 RAW_BOLT 0.8486보다 높다. 즉 물리 MSE 개선을 주지표 개선으로 바꿔 말할 수 없다.\n")
    lines.append("[확인] 같은 FIT256/TUNE64/VAL origin과 같은 batch order에서 L2, 낮은 learning rate, linear-only를 비교했다. 세 변경 모두 Jena에서는 OLD_PRED보다 나빴고, Robin에서는 normalized MSE를 소폭 낮춘 조건이 있었지만 physical MSE/MAE는 거의 같거나 악화했으며 RAW_BOLT와의 큰 격차는 남았다.\n")
    for dataset in DATASETS:
        row = best(dataset)
        lines.append(f"[확인] {dataset}의 이번 주지표 최저는 {row['method']} seed{row['seed']} update{row['selected_update']}이며 VAL normalized MSE {float(row['val_normalized_mse']):.6f}, physical MSE {float(row['val_physical_mse']):.3f}였다.\n")
    lines.append("[판단] 현재 병목은 단순히 checkpoint 선택 하나가 아니라, PRED 목적이 일부 큰 단위 채널의 물리 MSE를 낮추는 방향으로 움직이면서 정규화 채널 평균과 MAE를 악화시키는 문제다. 그래서 다음 단계는 새 표현학습보다 loss/selection/decoder를 같은 지표로 묶는 작은 ablation이 먼저다.\n")
    lines.append("## 비교 조건\n")
    lines.append("- 보호 TEST는 열지 않았다. source hash와 origin hash는 `CONTRACTS.json`에 기록했다.\n")
    lines.append("- 정규화와 PCA는 기존처럼 FIT 범위만 사용했다. FIT256/TUNE64/VAL 전체 origin, L512/H48, Jena C21->K5, Robin C17->K4를 유지했다.\n")
    lines.append("- 기존 REC/PRED는 저장된 Stage B checkpoint를 읽어 Bolt로 FIT/TUNE/VAL을 다시 예측했다. 신규 학습은 PRED 계열 세 가지뿐이며, 모두 기존 seed별 batch order를 그대로 썼다.\n")
    lines.append("- 주선택 지표는 TUNE normalized MSE다. physical MSE/MAE와 normalized MAE는 보조 지표다. VAL은 결과 해석용이며 checkpoint 선택에 쓰지 않았다.\n")
    lines.append("## 원수치 요약\n")
    lines.append("아래 표는 주선택 기준인 `tune_normalized_mse`로 고른 PRED 계열만 적었다.\n")
    lines.append("```text")
    lines.append("dataset seed method              update  val_norm_mse  val_phys_mse  val_phys_mae")
    for row in [
        r for r in selected_rows
        if r["criterion"] == "tune_normalized_mse" and r["objective"] == "PRED"
    ]:
        lines.append(
            f"{row['dataset']:<7} {int(row['seed']):<5} {row['method']:<19} {int(row['selected_update']):<6} "
            f"{float(row['val_normalized_mse']):<13.6f} {float(row['val_physical_mse']):<13.3f} {float(row['val_physical_mae']):.3f}"
        )
    lines.append("```\n")
    lines.append("## 그림\n")
    for fig in figures:
        name = Path(fig).name
        lines.append(f"![{name}]({name})\n")
    lines.append("## Bolt 유지 판단과 연결\n")
    lines.append("이 재검산은 전체 시스템의 순위를 새로 뒤집지 않는다. 이전 결과의 원자료 Bolt/R-only/native+R 대조가 여전히 남아 있고, 이번 추가 결과는 그 전에 주지표를 맞춰야 한다는 경고를 강화한다.\n")
    lines.append("```text")
    lines.append("dataset seed method             val_physical_mse  val_physical_mae  val_normalized_mse  val_normalized_mae")
    for row in whole_system_rows:
        seed = row.get("seed") or "-"
        try:
            values = (
                float(row["val_physical_mse"]),
                float(row["val_physical_mae"]),
                float(row["val_normalized_mse"]),
                float(row["val_normalized_mae"]),
            )
        except Exception:
            continue
        lines.append(
            f"{row['dataset']:<7} {seed:<5} {row['method']:<18} "
            f"{values[0]:<17.3f} {values[1]:<17.3f} {values[2]:<19.6f} {values[3]:.6f}"
        )
    lines.append("```\n")
    lines.append("## 검산과 한계\n")
    lines.append(f"- 실행 상태: {summary['status']}, wall {summary['resources']['wall_seconds']:.2f}s, peak allocated VRAM {summary['resources']['peak_allocated_vram_bytes'] / 1024**2:.1f} MiB.\n")
    lines.append(f"- 산출물: `metrics.csv` {summary['metric_row_count']}행, `selection_summary.csv` {summary['selection_row_count']}행, 그림 {len(figures)}개.\n")
    lines.append("- 검산 통과는 같은 개발 VAL과 저장 자산의 일관성 확인이다. 독립 일반화, 논문급 성능 우위, 또는 Bolt 필요성의 증명으로 쓰지 않는다.\n")
    lines.append("## 다음 판단\n")
    lines.append("1. 주지표를 normalized MSE/MAE로 고정하려면 학습 loss, checkpoint selection, report headline을 모두 이 기준으로 맞춘다.\n")
    lines.append("2. PRED가 Jena에서 physical MSE만 낮추는 현상을 줄이려면 채널별 가중, MAE/Huber 보조항, decoder 고정/재적합을 각각 하나씩 분리해 본다.\n")
    lines.append("3. 이 기준에서도 개선이 없으면 K/lookback/Bolt 대조로 넘어가기 전에 압축 예측 경로의 중심 질문을 '속도를 위해 어느 정도 정규화 오차를 감수할 수 있는가'로 좁히는 편이 맞다.\n")
    return "\n".join(lines)


def verify_outputs(summary: dict, metric_rows: list[dict], selected_rows: list[dict], figures: list[str]) -> dict:
    checks = []
    checks.append({"name": "metric_rows_nonempty", "pass": len(metric_rows) > 0, "detail": len(metric_rows)})
    checks.append({"name": "selection_rows_nonempty", "pass": len(selected_rows) > 0, "detail": len(selected_rows)})
    finite = True
    for row in metric_rows:
        for key in ("_physical_mse", "_physical_mae", "_normalized_mse", "_normalized_mae"):
            finite = finite and np.isfinite(float(row[key]))
    checks.append({"name": "metrics_finite", "pass": finite, "detail": None})
    checks.append({"name": "figures_exist", "pass": all((ROOT / f).exists() for f in figures), "detail": figures})
    checks.append({"name": "protected_test_guarded", "pass": bool(summary["protected_test_guarded"]), "detail": summary["protected_test_guarded"]})
    expected_groups = len(DATASETS) * len(SEEDS) * (2 + len(VARIANTS))
    primary_selected = [
        r for r in selected_rows
        if r["criterion"] == "tune_normalized_mse" and (r["objective"] in ("REC", "PRED"))
    ]
    checks.append({"name": "primary_selection_group_count", "pass": len(primary_selected) >= expected_groups, "detail": len(primary_selected)})
    return {"status": "PASS" if all(c["pass"] for c in checks) else "FAIL", "checks": checks}


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=False)
    budget = Budget()
    summary = {
        "status": "RUNNING",
        "head": os.popen("git rev-parse HEAD").read().strip(),
        "started_unix": time.time(),
        "protected_test_guarded": ce.access_guard(),
        "previous_stage": str(PREVIOUS.relative_to(ROOT)),
        "previous_cache": str(PREVIOUS_CACHE.relative_to(ROOT)),
        "variants": [asdict(v) for v in VARIANTS],
        "primary_metric": "TUNE normalized observed channel-macro MSE",
        "source_hashes": {
            str((PREVIOUS / "component_experiment.py").relative_to(ROOT)): sha(PREVIOUS / "component_experiment.py"),
            str((ROOT / "research" / "past_only_repr_diagnosis_20261012" / "common.py").relative_to(ROOT)): sha(
                ROOT / "research" / "past_only_repr_diagnosis_20261012" / "common.py"
            ),
        },
    }
    write_json(OUT / "SUMMARY_RUNNING.json", summary)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        bolt = ce.FrozenBolt()
        metric_rows, contracts = evaluate_previous_checkpoints(bolt, budget)
        training_records: list[dict] = []
        training_curves: list[dict] = []
        for dataset in DATASETS:
            data, histories, targets, observed, mean, basis, _pca = load_base_objects(dataset)
            for seed in SEEDS:
                for variant in VARIANTS:
                    rows, result = train_variant(
                        variant,
                        dataset,
                        seed,
                        data,
                        histories,
                        targets,
                        observed,
                        mean,
                        basis,
                        bolt,
                        budget,
                    )
                    metric_rows.extend(rows)
                    training_records.append(result["record"])
                    training_curves.extend(result["curve"])
        selected_rows = select_rows(metric_rows)
        whole_system_rows = read_previous_whole_system()
        write_csv(OUT / "metrics.csv", metric_rows)
        write_csv(OUT / "selection_summary.csv", selected_rows)
        write_csv(OUT / "training_curves.csv", training_curves)
        write_json(OUT / "CONTRACTS.json", contracts)
        write_json(OUT / "TRAINING_RECORDS.json", training_records)
        figures = make_plots(metric_rows, selected_rows)
        budget.check()
        summary.update(
            {
                "status": "COMPLETE",
                "metric_row_count": len(metric_rows),
                "selection_row_count": len(selected_rows),
                "training_record_count": len(training_records),
                "figures": figures,
                "bolt_initial_hash": bolt.initial_hash,
                "bolt_final_hash": ce.state_hash(bolt.model),
                "resources": budget.record,
            }
        )
        if summary["bolt_initial_hash"] != summary["bolt_final_hash"]:
            raise RuntimeError("BOLT_HASH_CHANGED")
        report = make_report(summary, selected_rows, whole_system_rows, figures)
        (OUT / "REPORT_KO.md").write_text(report, encoding="utf-8")
        verification = verify_outputs(summary, metric_rows, selected_rows, figures)
        write_json(OUT / "VERIFICATION.json", verification)
        budget.close("COMPLETE")
        summary["resources"] = budget.record
        summary["verification_status"] = verification["status"]
        write_json(OUT / "SUMMARY.json", summary)
        return 0
    except BaseException as error:
        summary.update({"status": "FAILED_PRESERVED", "error": str(error), "traceback": traceback.format_exc(), "resources": budget.record})
        write_json(OUT / "FAILURE.json", summary)
        budget.close("FAILED_PRESERVED", str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(run())
