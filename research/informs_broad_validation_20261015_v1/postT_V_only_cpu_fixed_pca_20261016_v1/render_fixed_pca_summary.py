"""Render six audited fixed-PCA aggregates; default mode only validates metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys


FIELDS = (
    "run_id", "dataset", "label", "C", "K", "L", "H", "V_origins", "seed",
    "device", "threads", "metric_scale", "fold", "split", "cohort",
    "data_sha256", "scaler_sha256", "origins_sha256", "target_sha256",
    "observed_sha256", "channel_ids_sha256", "MSE", "MAE", "raw_run_id",
    "Raw_MSE", "Raw_MAE", "raw_full_V_device", "optimizer_updates",
    "training_exposures", "FP64_rescore_status", "replay_status", "replay_origins",
    "cost_B1_status", "cost_B8_status", "cost_B1_median_seconds",
    "cost_B1_p90_seconds", "cost_B8_median_seconds", "cost_B8_p90_seconds",
    "cost_warmup_trials", "cost_timed_trials", "cost_stage_trials",
    "F_row_microbatch", "checkpoint_inputs",
)
CONDITIONS = (
    ("beijing", 11, 9, 48, 43, "Beijing"),
    ("jena", 21, 16, 48, 43, "Jena"),
    ("electricity", 321, 41, 48, 50, "Electricity"),
    ("electricity", 321, 241, 48, 50, "Electricity"),
    ("pems08", 170, 22, 12, 148, "PeMS08"),
    ("pems08", 170, 128, 12, 148, "PeMS08"),
)
PASS_COST = "PASS_COST_NUMBERS_AND_METADATA_ONLY"


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def nonsymlink_path(value):
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Symlink paths are not accepted")
    return path.resolve()


def finite_number(value, field):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid finite nonnegative field: {field}")
    return number


def validate_rows(rows):
    if len(rows) != len(CONDITIONS):
        raise ValueError("Exactly six fixed-PCA conditions are required")
    by_id = {}
    for row in rows:
        if set(row) != set(FIELDS) or None in row or None in row.values():
            raise ValueError("The aggregate schema must match FIELDS exactly")
        if row["run_id"] in by_id:
            raise ValueError("Duplicate run ID")
        by_id[row["run_id"]] = row
    ordered = []
    matching = {}
    for dataset, channels, k, horizon, origins, name in CONDITIONS:
        run_id = f"postT_cpu_{dataset}_E1_PCA_FIXED_BOLT_k{k}_s92601_v1"
        if run_id not in by_id:
            raise ValueError("A registered fixed-PCA condition is missing")
        row = by_id[run_id]
        expected = {
            "dataset": dataset, "label": f"{name} K{k}", "C": str(channels),
            "K": str(k), "L": "512", "H": str(horizon), "V_origins": str(origins),
            "seed": "92601", "device": "cpu", "threads": "2", "fold": "primary",
            "split": "V_select", "metric_scale": "A_normalized_original_channels",
            "optimizer_updates": "0", "training_exposures": "0",
            "FP64_rescore_status": "PASS", "replay_status": "PASS", "replay_origins": "6",
            "cost_B1_status": PASS_COST, "cost_B8_status": PASS_COST,
            "cost_warmup_trials": "10", "cost_timed_trials": "20", "cost_stage_trials": "20",
            "F_row_microbatch": "128", "checkpoint_inputs": "False",
            "raw_run_id": f"primary_{dataset}_E1_RAW_BOLT_SMALL_k0_s92601",
            "raw_full_V_device": "historical_saved_V_not_claimed_CPU",
            "cohort": f"{dataset}:primary:V_select:stride{horizon}",
        }
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Condition identity, audit status, or protocol mismatch")
        for key in ("data_sha256", "scaler_sha256", "origins_sha256", "target_sha256",
                    "observed_sha256", "channel_ids_sha256"):
            if len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError("Malformed metadata SHA256")
        for key in ("MSE", "MAE", "Raw_MSE", "Raw_MAE", "cost_B1_median_seconds",
                    "cost_B1_p90_seconds", "cost_B8_median_seconds", "cost_B8_p90_seconds"):
            finite_number(row[key], key)
        for batch in (1, 8):
            if float(row[f"cost_B{batch}_p90_seconds"]) < float(row[f"cost_B{batch}_median_seconds"]):
                raise ValueError("p90 must not be below the median")
        signature = tuple(row[key] for key in (
            "C", "H", "V_origins", "cohort", "data_sha256", "scaler_sha256",
            "origins_sha256", "target_sha256", "observed_sha256", "channel_ids_sha256",
            "raw_run_id", "Raw_MSE", "Raw_MAE",
        ))
        if dataset in matching and matching[dataset] != signature:
            raise ValueError("Within-dataset K comparisons must share one evaluation problem")
        matching[dataset] = signature
        ordered.append(row)
    return ordered


def load_input(path, expected_sha):
    source = nonsymlink_path(path)
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError("Input must be a nonsymlink aggregate CSV")
    blob = source.read_bytes()
    if len(blob) > 256 * 1024 or sha256(blob) != expected_sha:
        raise ValueError("Aggregate size or SHA256 mismatch")
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != list(FIELDS):
        raise ValueError("CSV field order must match FIELDS")
    return source, blob, validate_rows(list(reader))


def render(source, blob, rows, output):
    destination = nonsymlink_path(output)
    if destination.exists() or not destination.parent.is_dir():
        raise FileExistsError("Output must be new and have an existing parent")
    source_code = Path(__file__).read_bytes()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 10, "svg.fonttype": "none", "svg.hashsalt": "fixed-pca-summary-v1",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    destination.mkdir(exist_ok=False)
    (destination / "INPUT_AGGREGATE.csv").write_bytes(blob)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ticks = list(range(len(rows)))
    labels = [f"{row['label']} / C{row['C']}\nH{row['H']}, V origins={row['V_origins']}" for row in rows]
    for ax, metric, letter in ((axes[0, 0], "MSE", "A"), (axes[0, 1], "MAE", "B")):
        pca = [float(row[metric]) for row in rows]
        raw = [float(row[f"Raw_{metric}"]) for row in rows]
        ax.barh([n - .18 for n in ticks], raw, height=.33, color="#B5B5B5", hatch="///", label="Historical Raw Bolt-small")
        ax.barh([n + .18 for n in ticks], pca, height=.33, color="#0072B2", label="Fixed PCA + frozen Bolt")
        right = max(pca + raw) * 1.2
        for n, pv, rv in zip(ticks, pca, raw):
            ax.text(rv + right * .012, n - .18, f"{rv:.4g}", va="center", fontsize=8)
            ax.text(pv + right * .012, n + .18, f"{pv:.4g}", va="center", fontsize=8)
        ax.set_xlim(0, right)
        ax.set_xlabel(f"{metric} (A-normalized original channels; lower is better)")
        ax.set_title(f"{letter}  Matched V {metric}", loc="left", fontweight="bold")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    for ax, batch, letter in ((axes[1, 0], 1, "C"), (axes[1, 1], 8, "D")):
        median = [float(row[f"cost_B{batch}_median_seconds"]) for row in rows]
        p90 = [float(row[f"cost_B{batch}_p90_seconds"]) for row in rows]
        ax.barh(ticks, median, height=.55, color="#009E73", label="Timed20 median")
        ax.plot(p90, ticks, "|", color="black", markersize=11, markeredgewidth=1.5, linestyle="none", label="Timed20 p90")
        right = max(p90) * 1.25
        for n, mv, qv in zip(ticks, median, p90):
            ax.text(qv + right * .015, n, f"{mv:.4g} / {qv:.4g}", va="center", fontsize=8)
        ax.set_xlim(0, right)
        ax.set_xlabel(f"Whole physical pipeline (seconds per batch of {batch}; median / p90)")
        ax.set_title(f"{letter}  CPU latency, batch {batch}", loc="left", fontweight="bold")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_yticks(ticks, labels)
        ax.set_ylim(len(rows) - .4, -.7)
        ax.xaxis.grid(True, color="#E6E6E6", linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Fixed PCA + frozen Bolt: V-only accuracy and CPU latency", fontsize=15, y=.975)
    fig.text(.5, .947, "Post-T development | seed 92601 | six PCA conditions, four reused Raw references | no CI", ha="center", fontsize=10)
    fig.text(.04, .052, "Matched data/scaler/cohort within each dataset; no cross-dataset average. Fixed transform: 0 optimizer updates.", fontsize=9)
    fig.text(.04, .034, "CPU: 2 threads, row128, warmup10/timed20/stage20. Only non-instrumented timed20 is plotted; p90 is not a confidence interval.", fontsize=9)
    fig.text(.04, .016, "No matched Raw CPU cost here: no speedup/Pareto claim. Full V FP64 audits and fresh CPU 6-origin replays passed; no independent generalization claim.", fontsize=9)
    fig.subplots_adjust(left=.16, right=.965, bottom=.135, top=.9, hspace=.35, wspace=.5)
    for suffix in ("svg", "png"):
        fig.savefig(destination / f"PCA_ACCURACY_CPU_COST.{suffix}", dpi=300,
                    metadata={"Date": None} if suffix == "svg" else {"Software": "render_fixed_pca_summary.py"})
    plt.close(fig)
    script_relative = Path(os.path.relpath(Path(__file__).resolve(), destination)).as_posix()
    readme = (
        "# 고정 PCA 6조건 그림\n\n"
        "![동일 자료의 Raw 대비 PCA 오차와 PCA CPU 지연시간](PCA_ACCURACY_CPU_COST.svg)\n\n"
        "입력은 저장 집계 CSV 6행뿐이다. 실제 새 추론·학습·비용 계측은 하지 않았다. "
        "MSE/MAE는 자료별 동일 V/data/scaler/cohort에 한정하고, 자료 간 평균은 만들지 않았다. "
        "Raw 4개 기준선을 K별로 재표시한 것이므로 독립 기준선 6개가 아니다. "
        "PCA의 optimizer update는 0이며 CPU 비용은 2 threads의 timed20 전체경로 median/p90다. "
        "p90는 신뢰구간이 아니며 stage_probe 시간은 섞지 않았다. "
        "같은 CPU Raw 비용이 없어 속도 배율·Pareto 우위는 주장하지 않는다. "
        "T 관측 후 개발·단일 seed·독립 일반화 미검증 범위를 유지한다.\n\n"
        "입력 원바이트는 [INPUT_AGGREGATE.csv](INPUT_AGGREGATE.csv), "
        "생성·출력 SHA는 [FIGURE_BINDING.json](FIGURE_BINDING.json)에 있다. "
        "PNG는 300 dpi로 함께 생성한 로컬 출력이며 게시 형식 선택은 별도다.\n\n"
        "이 결과 폴더에서 새 출력 폴더로 재생성할 수 있다:\n\n```text\n"
        f"python {script_relative} --input INPUT_AGGREGATE.csv --input-sha {sha256(blob)} --output ../RENDER_REPLAY_NEW --execute\n```\n"
    )
    (destination / "README_KO.md").write_text(readme, encoding="utf-8", newline="\n")
    if source.read_bytes() != blob or Path(__file__).read_bytes() != source_code:
        raise RuntimeError("Input or renderer changed while rendering")
    artifacts = []
    for name in ("INPUT_AGGREGATE.csv", "PCA_ACCURACY_CPU_COST.svg", "PCA_ACCURACY_CPU_COST.png", "README_KO.md"):
        payload = (destination / name).read_bytes()
        artifacts.append({"file": name, "bytes": len(payload), "sha256": sha256(payload)})
    binding = {
        "schema": "fixed_pca_six_aggregate_figure_v1", "status": "RENDERED_AGGREGATES_ONLY",
        "input_csv_sha256": sha256(blob), "input_csv_bytes": len(blob), "rows": len(rows),
        "run_ids": [row["run_id"] for row in rows], "renderer_sha256": sha256(source_code),
        "matplotlib_version": matplotlib.__version__, "png_dpi": 300, "artifacts": artifacts,
        "input_and_renderer_preserved": True, "new_inference_or_cost_measurement": False,
        "Raw_cost_comparison_available": False, "independent_generalization_claimed": False,
    }
    with (destination / "FIGURE_BINDING.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(binding, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.input) != bool(args.input_sha):
        parser.error("--input and --input-sha must be given together")
    if args.execute and (not args.input or not args.output):
        parser.error("--execute requires pinned input and a new output")
    rows = []
    if args.input:
        source, blob, rows = load_input(args.input, args.input_sha)
    if args.execute:
        result = render(source, blob, rows, args.output)
    else:
        result = {"status": "METADATA_ONLY", "input_rows_validated": len(rows),
                  "expected_rows": 6, "fields": list(FIELDS), "writes": 0,
                  "matplotlib_imported": "matplotlib" in sys.modules,
                  "numpy_imported": "numpy" in sys.modules, "torch_imported": "torch" in sys.modules}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
