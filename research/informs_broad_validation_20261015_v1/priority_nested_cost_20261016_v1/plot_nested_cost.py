"""Render only the three already audited N3 aggregate rows; default does no I/O."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


DEADLINE = 1792222550
MEMORY_LIMIT = 1024**3
INPUTS = {
    "RESULTS_WITH_COST_v1.csv": "e3d35a0ae6dbac212a4e5b7cfa589f71168ee1b990cbd0916b57e10d8a22a4c9",
    "INDEPENDENT_COST_REVIEW_ROOT_v1.json": "2d00f17d0520d6cd9482d304f4b403b5cf7da613d5faec9a9cd92483ac090c5b",
    "REPORT_INPUT_BINDINGS_v1.json": "2590c93f9ae0d4520a03c2e5dbf11818df5dca377ce1f3787e434e7d4ed8d40b",
}
COLORS = ("#1B9E77", "#377EB8", "#7851A9")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate(csv_text, audit, report):
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if (audit.get("schema") != "INDEPENDENT_NESTED_COST_REVIEW_v1" or audit.get("status") != "PASS"
            or report.get("status") != "REPORT_FROM_COMPLETED_COSTS_AND_REUSED_V"
            or report.get("whole_MASTER_complete") is not False or len(rows) != 3):
        raise ValueError("Exactly three already audited aggregate rows required")
    recorded_csv = next(r for r in report["outputs"] if r["relative_path"].endswith("/RESULTS_WITH_COST_v1.csv"))
    if recorded_csv["sha256"] != INPUTS["RESULTS_WITH_COST_v1.csv"]:
        raise ValueError("Original report must bind the exact unchanged CSV")
    if (audit["data_contract"]["A60_fit_end"] != 12622
            or audit["data_contract"]["physical_and_mask_input_hashes_same_across_K_for_each_batch"] is not True
            or audit["original_integrity"]["all48_current_bytes_match_old_plan"] is not True):
        raise ValueError("Existing same-cohort and input-integrity audit required")
    for row, k in zip(rows, (3, 6, 11)):
        identifier = "postT_cpu_beijing_E4_NESTED_PCA_k%d_s92601_v1" % k
        expected = dict(dataset="beijing", run_id=identifier, K=str(k), L="512", H="48", C="11",
                        V_origins="43", observed_labels="20539", accuracy_device="cpu", CPU_threads="2",
                        row_microbatch="128", new_optimizer_updates="0", PCA_fit_seconds="NOT_RECORDED")
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Original K, cohort, devices or fixed-transform scope differs")
        accuracy = next(a for a in audit["reused_accuracy"] if a["latent_k"] == k)
        for metric in ("MSE", "MAE"):
            value = float(row[metric])
            if not math.isfinite(value) or value != accuracy[metric.lower()]:
                raise ValueError("CSV accuracy differs from independently reviewed reused result")
        for batch in (1, 8):
            condition = next(a for a in audit["cost_conditions"] if a["latent_k"] == k and a["batch"] == batch)
            summary = next(s for s in condition["independent_summaries"] if s["phase"] == "timed")
            if condition["status"] != "PASS" or summary["n"] != 20:
                raise ValueError("Actual independent ordinary whole20 cost summary required")
            for suffix in ("median_seconds", "p90_seconds"):
                if float(row["B%d_%s" % (batch, suffix)]) != summary[suffix]:
                    raise ValueError("CSV latency differs from independently recomputed cost")
            if (float(row["B%d_resident_weights_cpu_MiB" % batch]) != 182.03032302856445
                    or condition["memory"]["resident_weights_cpu_MiB"] != 182.03032302856445
                    or int(row["B%d_F_streams_per_trial" % batch]) != batch*k
                    or int(row["B%d_F_calls_per_trial" % batch]) != 1):
                raise ValueError("Resident weights or actual foundation counter scope differs")
    return rows


class ResourceCheck:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = 0
        self.os_peak = 0
        self.stop = threading.Event()
        self.failed = False

    def check(self):
        memory = self.process.memory_info()
        self.peak = max(self.peak, memory.rss)
        self.os_peak = max(self.os_peak, getattr(memory, "peak_wset", memory.rss))
        if time.time() >= DEADLINE or max(self.peak, self.os_peak) >= MEMORY_LIMIT or self.failed:
            raise RuntimeError("Original deadline or own1GiB visualization memory limit reached")

    def __enter__(self):
        self.check()
        def sample():
            while not self.stop.wait(.05):
                try:
                    self.check()
                except RuntimeError:
                    self.failed = True
                    break
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *error):
        self.stop.set()
        self.thread.join(timeout=1)


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path = Path(os.environ["WINDIR"])/"Fonts/malgun.ttf"
    font = font_manager.FontProperties(fname=str(font_path)).get_name() if font_path.is_file() else "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False,
        "svg.fonttype": "none", "svg.hashsalt": "N3_NESTED_COST_v1", "axes.spines.top": False,
        "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    fig.subplots_adjust(left=.085, right=.97, bottom=.20, top=.82, wspace=.26, hspace=.49)
    fig.suptitle("고정 PCA: K가 커질 때 오차와 실제 추론 비용", x=.085, y=.965,
                 ha="left", fontsize=17, fontweight="bold", color="#172B3A")
    fig.text(.085, .912, "Beijing · C11 / L512 / H48 · 개발 V43 · 관측 labels 20,539 · 같은 A60 PCA basis", color="#455A64")
    fig.text(.085, .87, "정확도는 기존 전체 V 재사용 / 비용은 실제 CPU2 전체 경로 · 모든 축은 0부터", fontsize=9, color="#546E7A")
    panels = ((axes[0, 0], "MSE", "A   정규화 MSE", "MSE", 1),
              (axes[0, 1], "MAE", "B   정규화 MAE", "MAE", 1),
              (axes[1, 0], "B1_median_seconds", "C   B1 전체 latency", "밀리초 / batch", 1000),
              (axes[1, 1], "B8_median_seconds", "D   B8 전체 latency", "밀리초 / 8개 batch", 1000))
    artist_values = []
    for ax, key, title, ylabel, scale in panels:
        values = [float(row[key])*scale for row in rows]
        bars = ax.bar(range(3), values, width=.52, color=COLORS, zorder=3)
        artist_values.append(dict(column=key, multiplier=scale, plotted_values=[b.get_height() for b in bars]))
        top = max(values)
        if scale == 1000:
            p90_key = key.replace("median", "p90")
            p90 = [float(row[p90_key])*scale for row in rows]
            ax.errorbar(range(3), values, yerr=[[0.0]*3, [p-v for p, v in zip(p90, values)]],
                        fmt="none", ecolor="#263238", capsize=5, linewidth=1.2, zorder=4)
            artist_values.append(dict(column=p90_key, multiplier=scale, plotted_values=p90))
            top = max(p90)
        for index, value in enumerate(values):
            ax.text(index, (max(value, p90[index]) if scale == 1000 else value)+top*.045,
                    ("%.1f" if scale == 1000 else "%.4f") % value,
                    ha="center", va="bottom", fontweight="bold", fontsize=11, color="#263238")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(range(3), ["K = "+row["K"] for row in rows])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, top*1.24)
        ax.grid(axis="y", color="#E4E9ED", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
    fig.text(.085, .126, "비용 막대 = ordinary whole 20회 median / 위쪽 whisker = p90. B8은 8개 batch 전체 시간이며 request당 시간이 아니다.", fontsize=8.7)
    fig.text(.085, .092, "상주 F weights = 182.030323 MiB로 동일. K11도 회전 좌표 + inverse D이며 Raw 또는 학습 B2와 동일한 경로가 아니다.", fontsize=8.7)
    fig.text(.085, .058, "한 seed · 반복 개발 V · 독립 일반화 미검증. PCA fit 시간 미기록. 오차 감소나 검사 PASS를 보편적 우위로 해석하지 않는다.", fontsize=8.7, color="#536471")
    svg = io.BytesIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": "N3 aggregate visualization"})
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=145, metadata={"Software": "N3 aggregate visualization"})
    plt.close(fig)
    clean_svg = "\n".join(line.rstrip() for line in svg.getvalue().decode("utf-8").splitlines())+"\n"
    return clean_svg.encode(), png.getvalue(), artist_values, font


def execute(scope, preview):
    scope, preview = Path(scope).resolve(), Path(preview).resolve()
    if preview.is_relative_to(scope.parent):
        raise ValueError("PNG preview must remain outside the public campaign")
    paths = [scope/"figures/NESTED_COST_v1.svg", scope/"FIGURE_BINDINGS_v1.json", scope/"VISUALIZATION_KO_v1.md", preview]
    if any(p.exists() for p in paths):
        raise FileExistsError("Fresh visualization outputs only; preserve existing files")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("CPU/BLAS1 environment required")
    started = time.perf_counter()
    captures = {name: (scope/name).read_bytes() for name in INPUTS}
    if any(sha_bytes(blob) != INPUTS[name] for name, blob in captures.items()):
        raise ValueError("Original audited input SHA changed")
    source_sha = sha_bytes(Path(__file__).read_bytes())
    rows = validate(captures["RESULTS_WITH_COST_v1.csv"].decode(),
                    json.loads(captures["INDEPENDENT_COST_REVIEW_ROOT_v1.json"]),
                    json.loads(captures["REPORT_INPUT_BINDINGS_v1.json"]))
    with ResourceCheck() as resources:
        svg, png, artist_values, font = render(rows)
        for group in artist_values:
            expected = [float(row[group["column"]])*group["multiplier"] for row in rows]
            if group["plotted_values"] != expected:
                raise ValueError("Matplotlib artist values differ from unchanged exact CSV")
        if any(sha_bytes((scope/name).read_bytes()) != digest for name, digest in INPUTS.items()):
            raise ValueError("Visualization input changed during rendering")
        if sha_bytes(Path(__file__).read_bytes()) != source_sha:
            raise ValueError("Plot source changed during execution")
        resources.check()
    md = """# N3 고정 PCA K 비교 시각자료

![K3/6/11 정확도와 실제 CPU 비용](figures/NESTED_COST_v1.svg)

[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [원 보고](RESULTS_WITH_COST_KO_v1.md) · [독립 비용 검산](INDEPENDENT_COST_REVIEW_ROOT_v1.json) · [그림 결속](FIGURE_BINDINGS_v1.json) · [생성 코드](plot_nested_cost.py)

같은 A60 basis의 PCA K3/6/11에서 기존 전체 V43의 MSE·MAE와 이미 측정된 CPU2 B1/B8 전체 경로 latency를 표시했다. 막대는 ordinary whole20 median, 위쪽 whisker는 p90이다. B8은8개 batch 전체 시간이며 request당 시간으로 나누지 않았다. 표시 숫자만 반올림했고 CSV는 바꾸지 않았다.

이 개발 V에서는 K가 클수록 오차가 작고 latency는 길다. F 상주 weights는182.030323 MiB로 같아 입력 stream 감소를 backbone weights 경량화로 해석하지 않는다. K11도 회전된 좌표에서 Bolt를 거쳐 inverse D로 돌아오는 경로로 Raw/B2와 동일하지 않다. 한 seed·반복 개발 V이며 독립 일반화와 최종 방법 우위를 검증하지 않았다. 학습 optimizer0회와 PCA 제조 시간0초는 다르며 PCA fit 시간은 미기록이다.

기존 공개 집계3행·독립 감사6조건·보고 입력 결속만 읽었다. 새 학습·추론·모델/Task/NPZ·원자료/target·비용 재측정·네트워크·Git 작업은0이다. SVG만 게시용이고 PNG는 로컬 미리보기다. 기본 호출은 입력읽기·쓰기0, 명시 --execute는 새 출력만 허용한다.
"""
    svg_path, binding_path, md_path, _ = paths
    svg_path.parent.mkdir(exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    for path, blob in ((svg_path, svg), (preview, png), (md_path, md.encode())):
        with path.open("xb") as stream:
            stream.write(blob)
    elapsed = time.perf_counter()-started
    bindings = dict(schema="N3_NESTED_COST_FIGURE_BINDINGS_v1", status="PASS_EXISTING_AGGREGATE_VISUALIZATION",
        input_SHA_before_after_equal=True, input_SHA256=INPUTS, plot_source_sha256=source_sha,
        outputs=[dict(relative_path=path.relative_to(scope).as_posix(), bytes=len(blob), sha256=sha_bytes(blob))
                 for path, blob in ((svg_path, svg), (md_path, md.encode()))],
        local_preview=dict(published=False, bytes=len(png), sha256=sha_bytes(png)),
        plot_values_from_unchanged_CSV=artist_values, CSV_precision_preserved=True,
        plotted_accuracy_rows_reused=3, existing_audited_cost_conditions=6, new_accuracy_or_cost_measurements=0,
        cohort=dict(dataset="Beijing", C=11, L=512, H=48, V_origins=43, observed_labels=20539, seed=92601,
                    A60_fit_end=12622, fixed_nested_basis=True),
        timing_definition="B1/B8 whole batch milliseconds; median and p90 of20 ordinary timed trials; not per-request B8",
        font=font, CPU_BLAS_threads=1, deadline_unix=DEADLINE, elapsed_seconds=elapsed,
        sampled_own_peak_RSS_bytes=resources.peak, own_OS_peak_working_set_bytes=resources.os_peak,
        own_memory_limit_bytes=MEMORY_LIMIT, memory_scope="Own visualization process; not concurrent experiment",
        Torch_imported="torch" in sys.modules, model_Task_NPZ_raw_T_training_inference_cost_network_Git_operations=0,
        whole_MASTER_complete=False, independent_generalization_claimed=False,
        self_hash_policy="Bindings excludes itself; source and other outputs have exact byte SHA")
    with binding_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(bindings, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(status=bindings["status"], public_files_written=3, local_previews_written=1,
                elapsed_seconds=elapsed, sampled_own_peak_RSS_bytes=resources.peak,
                own_OS_peak_working_set_bytes=resources.os_peak, font=font, Torch_imported="torch" in sys.modules)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scope")
    parser.add_argument("--preview")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="NOT_EXECUTED", input_reads=0, files_written=0,
                              matplotlib_imported="matplotlib" in sys.modules)))
        return 0
    if not args.scope or not args.preview:
        parser.error("Explicit --execute --scope --preview required")
    print(json.dumps(execute(args.scope, args.preview), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
