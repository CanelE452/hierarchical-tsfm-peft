"""ROOT-only four-panel visualization of actual PeMS C32 Raw/G report rows.

Default zero I/O and lazy renderer import. Missing verified values remain NA.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import posixpath
import sys
import time
import types


NAME = "priority_pems_subset_20261016_v1"
COMMON = "priority_valid_minima_20261016_v1"
AUDITOR_SHA = "c2d99f40ffd7e5ed5ddff7c327aa3e003a5d06e660687f7fab9e8333c8d2e190"
BUILDER_SHA = "701dca855a385a49190a7583724f388753fa4cad31c7b8f4a815951283253dac"
INPUT_HELPER_SHA = "8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
STYLE_SHA = "50e25501857a9c3aad0c9f9902397c00075025d1b3556778539b0e400adaf47b"
COLUMNS = ("MSE", "MAE", "B1_median_seconds", "B8_median_seconds")
METHODS = (
    dict(family="RAW", run_id="postT_cpu_pems08_E1_SUBSET_RAW_k0_s92601_C32_priority_v1", label="Raw C32\nfrozen Bolt", color="#0072B2", hatch=""),
    dict(family="DIRECT_G", run_id="postT_cpu_pems08_E1_SUBSET_DIRECT_G_k0_s92601_C32_priority_v1", label="Direct G C32\nno foundation", color="#D55E00", hatch="//"))
CSV = "RAW_G_RESULTS_v1.csv"
REPORT_MD = "RAW_G_KO_v1.md"
REPORT_BIND = "RAW_G_BINDINGS_v1.json"
SVG = "figures/RAW_G_FOUR_PANEL_v1.svg"
VIS_MD = "RAW_G_VISUALIZATION_KO_v1.md"
VIS_BIND = "RAW_G_FIGURE_BINDINGS_v1.json"
CAPTION = ("C32 / L512 / H12 / V148 / seed92601 · 반복 개발 V · 제한 search와 선택 refit: 수렴·독립 일반화 미확정.\n"
    "Raw와 G는 구조·학습량이 다름. B2·압축 가치·Bolt 필요성의 충분한 검증이 아님. NA는 미완료이며 0이 아님.")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def module(path, expected, label):
    if sha(path) != expected:
        raise ValueError("Exact frozen helper, builder and style source SHA required")
    value = types.ModuleType(label); value.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), value.__dict__)
    return value


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Finite nonnegative actual report aggregate required")
    return result


def prepare_data(rows, audit, report):
    if (audit.get("schema") != "INDEPENDENT_PEMS_RAW_G_AUDIT_v1" or audit.get("status") not in ("PASS", "PARTIAL_VALID_EVIDENCE_PRESERVED")
            or audit.get("error") is not None or audit.get("audit_source_sha256") != AUDITOR_SHA
            or audit.get("input_SHA_before_after_equal") is not True or audit.get("source48_preserved") is not True
            or audit.get("whole_MASTER_complete") is not False or audit.get("independent_generalization_claimed") is not False
            or audit.get("full_C170_or_B2_or_Bolt_necessity_proven") is not False or audit.get("actual_current_owner_absence_bound") is not True
            or audit.get("costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals") is not True
            or report.get("schema") != "PEMS_RAW_G_REPORT_BINDINGS_v1" or report.get("status") != "FROM_ACTUAL_PEMS_RAW_G_AUDIT_" + audit["status"]
            or report.get("builder_source_sha256") != BUILDER_SHA or report.get("auditor_source_sha256") != AUDITOR_SHA
            or report.get("input_SHA_before_after_equal") is not True or report.get("source48_preserved") is not True
            or report.get("whole_MASTER_complete") is not False or report.get("independent_generalization_claimed") is not False
            or report.get("full_subset_experiments_complete") is not False):
        raise ValueError("Actual same-audit complete or honest partial C32 report required")
    actual = audit.get("rows", []); identifiers = {method["run_id"] for method in METHODS}
    if (not rows or len(rows) > 2 or len({row.get("run_id") for row in rows}) != len(rows)
            or not {row.get("run_id") for row in rows} <= identifiers
            or len(actual) != len(rows) or {row.get("run_id") for row in actual} != {row["run_id"] for row in rows}
            or report.get("completed_accuracy_rows") != len(rows) or report.get("completed_cost_rows") != len(audit.get("cost_conditions", []))):
        raise ValueError("Only actual verified report rows; no missing peer invented as an accuracy result")
    if audit["status"] == "PASS" and (len(rows) != 2 or len(audit.get("cost_conditions", [])) != 4 or audit.get("failures") != [] or audit.get("skipped") != []):
        raise ValueError("Complete status requires two accuracy rows and four actual costs")
    if audit["status"] != "PASS" and not (audit.get("failures") or audit.get("skipped")):
        raise ValueError("Partial display must preserve actual failure or skipped reasons")
    data = []; observed = set()
    for method in METHODS:
        selected = [row for row in rows if row["run_id"] == method["run_id"]]
        if not selected:
            data.append(dict(method=method, row=None, values={column: None for column in COLUMNS}, statuses={column: "NOT_COMPLETE_NO_VERIFIED_ACCURACY_ROW" for column in COLUMNS}))
            continue
        row = selected[0]; record = next(item for item in actual if item["run_id"] == row["run_id"])
        expected = dict(family=method["family"], C=32, L=512, H=12, seed=92601, V_origins=148, accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6")
        if any(str(row.get(key)) != str(value) for key, value in expected.items()) or record.get("family") != method["family"] or record.get("accuracy_status") != expected["accuracy_status"]:
            raise ValueError("Exact C32 V148 per-row identity and verified accuracy status required")
        count = int(row["observed_labels"])
        if not 0 < count <= 56832 or count != record.get("n_observed"):
            raise ValueError("Actual observed count must equal the audit, without a full-mask assumption")
        observed.add(count); values = {}; statuses = {}
        for column in ("MSE", "MAE"):
            value = number(row[column])
            if value != number(record[column.lower()]):
                raise ValueError("CSV accuracy must equal the original audited float exactly")
            values[column] = value; statuses[column] = row["accuracy_status"]
        for batch in (1, 8):
            column = "B" + str(batch) + "_median_seconds"; status = row["B" + str(batch) + "_status"]
            if status != record["cost_status"].get(str(batch)):
                raise ValueError("CSV cost status must preserve the audited same-batch status")
            cost = record["costs"].get(str(batch))
            if status == "PASS_ACTUAL_PHYSICAL_TRIALS":
                timed = [entry for entry in cost["summaries"] if entry.get("phase") == "timed" and entry.get("batch_size") == batch and entry.get("trials") == 20] if cost else []
                if cost is None or cost.get("trial_rows") != 50 or len(timed) != 1 or number(row[column]) != number(timed[0]["median_seconds"]):
                    raise ValueError("CSV whole median must equal the audited ordinary20 actual cost")
                values[column] = number(row[column])
            elif status == "NOT_COMPLETE" and cost is None and row.get(column) in (None, ""):
                values[column] = None
            else:
                raise ValueError("Missing or failed cost is unavailable; a zero is not a missing-value substitute")
            statuses[column] = status
        data.append(dict(method=method, row=row, values=values, statuses=statuses))
    if len(observed) != 1:
        raise ValueError("Present rows must share the actual observed C32 cohort")
    return data


def validate_artists(artists, data):
    if len(artists) != 4 or {artist.get("column") for artist in artists} != set(COLUMNS):
        raise ValueError("Exactly four panel bindings required")
    for artist in artists:
        column = artist["column"]; present = [(index, item) for index, item in enumerate(data) if item["values"][column] is not None]
        missing = [item["method"]["run_id"] for item in data if item["values"][column] is None]
        if (artist.get("positions") != [index for index, _ in present]
                or artist.get("run_ids") != [item["method"]["run_id"] for _, item in present]
                or artist.get("values") != [item["values"][column] for _, item in present]
                or artist.get("missing_run_ids") != missing
                or artist.get("row_statuses") != [item["statuses"][column] for item in data]):
            raise ValueError("Actual bar values, missing positions and statuses must equal unchanged CSV aggregates")


def draw_panels(axes, data):
    if len(axes) != 4:
        raise ValueError("Four axes required")
    artists = []; titles = ("A   정규화 MSE", "B   정규화 MAE", "C   CPU2 B1 warm whole median", "D   CPU2 B8 warm whole median")
    for ax, column, title in zip(axes, COLUMNS, titles):
        present = [(index, item) for index, item in enumerate(data) if item["values"][column] is not None]
        positions = [index for index, _ in present]; values = [item["values"][column] for _, item in present]
        bars = ax.bar(positions, values, width=.52, color=[item["method"]["color"] for _, item in present], zorder=3)
        for bar, (_, item) in zip(bars, present):
            bar.set_hatch(item["method"]["hatch"])
        artists.append(dict(column=column, positions=positions, run_ids=[item["method"]["run_id"] for _, item in present], values=[float(bar.get_height()) for bar in bars],
            missing_run_ids=[item["method"]["run_id"] for item in data if item["values"][column] is None], row_statuses=[item["statuses"][column] for item in data]))
        top = max(max(values), 1e-12) if values else 1.
        for index, item in enumerate(data):
            value = item["values"][column]
            ax.text(index, top * .07 if value is None else value + top * .035, "NA\n미완료" if value is None else f"{value:.4g}", ha="center", va="bottom", fontsize=10, color="#536471" if value is None else "#263238")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel(column if column in ("MSE", "MAE") else "초 / 전체 batch")
        ax.set_xticks(range(2), [method["label"] for method in METHODS]); ax.set_xlim(-.6, 1.6); ax.set_ylim(0, top * 1.3)
        ax.grid(axis="y", color="#E4E9ED", linewidth=.7, zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="both", length=0)
    validate_artists(artists, data)
    return artists


def render(data, csv_sha, audit_sha, audit_status):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path = Path(os.environ["WINDIR"]) / "Fonts/malgun.ttf"
    if not font_path.is_file():
        raise FileNotFoundError("Existing Malgun font required; no installation")
    font = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font, "font.size": 9, "axes.unicode_minus": False, "svg.fonttype": "none", "svg.hashsalt": "RAW_G_FOUR_PANEL_v1", "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)); fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.26, hspace=.7, wspace=.3)
    try:
        fig.suptitle("PeMS08 C32: Raw와 직접 G의 정확도와 실제 warm 비용", x=.08, y=.96, ha="left", fontsize=16, fontweight="bold", color="#172B3A")
        fig.text(.08, .916, "같은 C32 / L512 / H12 / V148 / seed92601 · CPU2 · " + audit_status, fontsize=10, color="#455A64")
        fig.text(.08, .878, "B1/B8은 전체 입력 batch의 초 · 미검증 값은 NA와 빈 자리 · 표시 라벨만 반올림 · 모든 축 0 시작", fontsize=9, color="#455A64")
        fig.text(.08, .841, "CSV SHA " + csv_sha[:12] + " · 같은 독립 감사 SHA " + audit_sha[:12], fontsize=8.5, color="#536471")
        artists = draw_panels(list(axes.flat), data)
        fig.text(.08, .18, CAPTION, fontsize=8.8, linespacing=1.65, color="#536471")
        fig.text(.08, .1, "warm timer는 cold model load를 제외함 · B8을 8로 나누지 않음 · CI/seed 분산을 추가하지 않음", fontsize=8.8, color="#536471")
        fig.text(.08, .06, "upstream 미래 보간·전처리 계보 UNKNOWN · 원 보고서 우선 · 그림 실패는 다른 실험의 중단 gate가 아님 · MASTER 미완료", fontsize=8.5, color="#536471")
        svg, png = io.BytesIO(), io.BytesIO()
        fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": "Audited PeMS C32 Raw/G report aggregates"})
        fig.savefig(png, format="png", dpi=145, metadata={"Software": "Audited PeMS C32 Raw/G report aggregates"})
        clean = ("\n".join(line.rstrip() for line in svg.getvalue().decode("utf-8").splitlines()) + "\n").encode()
        return clean, png.getvalue(), artists, font, sha(font_path)
    finally:
        plt.close(fig)


def csv_bytes(builder, rows):
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=builder.FIELDS, lineterminator="\n"); writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def execute(public, cache, plot_sha, report_sha, audit_relative, audit_sha, csv_sha):
    public, cache = Path(public).resolve(), Path(cache).resolve(); scope = public / NAME
    svg_path, md_path, binding_path = scope / SVG, scope / VIS_MD, scope / VIS_BIND
    preview = cache / NAME / "visualization/RAW_G_FOUR_PANEL_v1.png"
    if any(path.exists() for path in (svg_path, md_path, binding_path, preview)):
        raise FileExistsError("Existing visualization preserved; CreateNew outputs only")
    if Path(__file__).resolve() != scope / "plot_raw_g_v2.py" or sha(__file__) != plot_sha:
        raise ValueError("ROOT exact current plot source SHA and canonical path required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Explicit CPU/BLAS1 visualization environment required")
    if not audit_relative.startswith(NAME + "/") or not audit_relative.endswith(".json"):
        raise ValueError("Explicit actual same-namespace N6 audit JSON required")
    helper = module(public / COMMON / "build_beijing_report.py", INPUT_HELPER_SHA, "raw_g_plot_inputs")
    builder = module(scope / "build_raw_g_report_v2.py", BUILDER_SHA, "raw_g_plot_builder")
    style = module(public / COMMON / "plot_beijing.py", STYLE_SHA, "raw_g_plot_resources")
    if time.time() >= style.DEADLINE:
        raise TimeoutError("Original worker deadline reached")
    inputs = helper.Inputs(public, cache)
    direct = ((NAME + "/plot_raw_g_v2.py", plot_sha), (NAME + "/audit_raw_g_v2.py", AUDITOR_SHA), (NAME + "/build_raw_g_report_v2.py", BUILDER_SHA),
        (COMMON + "/build_beijing_report.py", INPUT_HELPER_SHA), (COMMON + "/plot_beijing.py", STYLE_SHA), (NAME + "/" + REPORT_BIND, report_sha), (audit_relative, audit_sha))
    for name, digest in direct:
        inputs.capture("public", name, digest)
    report = inputs.json("public", NAME + "/" + REPORT_BIND, report_sha); audit = inputs.json("public", audit_relative, audit_sha)
    builder.validate_audit(audit)
    if (report.get("audit_sha256") != audit_sha or report.get("audit_relative_path") != audit_relative
            or report.get("transitive_input_manifest") != dict(root="public", relative_path=audit_relative, sha256=audit_sha, field="inputs")):
        raise ValueError("Canonical report must bind the same actual audit and its full input manifest")
    outputs = report.get("outputs", [])
    if len(outputs) != 2 or {row["relative_path"] for row in outputs} != {NAME + "/" + CSV, NAME + "/" + REPORT_MD}:
        raise ValueError("Actual canonical CSV and Korean MD must precede the figure")
    for row in (*report.get("inputs", []), *audit.get("inputs", [])):
        inputs.capture(row["root"], row["relative_path"], row["sha256"])
    for row in outputs:
        inputs.capture("public", row["relative_path"], row["sha256"])
        if inputs.path("public", row["relative_path"]).stat().st_size != row["bytes"]:
            raise ValueError("Canonical output bytes must equal report bindings")
    if any(inputs.path("public", name).exists() for name in audit.get("missing_public_native_paths", [])):
        raise ValueError("Audit-observed absent native output changed before visualization")
    csv_relative = NAME + "/" + CSV; blob = inputs.path("public", csv_relative).read_bytes()
    if inputs.files[("public", csv_relative)] != csv_sha:
        raise ValueError("ROOT actual CSV SHA must equal the canonical report output")
    actual_rows = builder.rows_from_audit(audit)
    if blob != csv_bytes(builder, actual_rows) or inputs.path("public", NAME + "/" + REPORT_MD).read_bytes() != builder.markdown(audit, actual_rows, audit_relative, audit_sha).encode("utf-8"):
        raise ValueError("Canonical CSV and MD bytes must preserve exact frozen-builder outputs")
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8")))
    if reader.fieldnames != list(builder.FIELDS):
        raise ValueError("Exact canonical CSV header required")
    rows = list(reader); data = prepare_data(rows, audit, report)
    if not inputs.preserved():
        raise ValueError("Inputs changed before renderer")
    started = time.perf_counter()
    with style.ResourceCheck() as resources:
        svg, png, artists, font, font_sha = render(data, csv_sha, audit_sha, audit["status"])
        if not inputs.preserved():
            raise ValueError("Inputs changed during renderer")
        resources.check()
    audit_link = posixpath.relpath(audit_relative, NAME)
    md = ("# PeMS08 C32 Raw·직접 G 시각자료\n\n![정규화 정확도와 CPU2 warm whole 비용](" + SVG + ")\n\n"
        "[실험 보고](" + REPORT_MD + ") · [원정밀도 CSV](" + CSV + ") · [그림 결속](" + VIS_BIND + ") · [코드](plot_raw_g_v2.py) · [같은 독립 감사](" + audit_link + ")\n\n"
        "감사 상태: `" + audit["status"] + "`. Raw/G 두 자리를 고정하되 실제 검증된 보고 행만 bar로 그렸다. 누락된 peer나 실패한 Raw batch 비용은 NA로 표시하며 가짜 0이나 완료 행을 만들지 않는다.\n\n"
        "MSE가 주지표이며 MAE와 CPU2 B1/B8 ordinary whole20 median을 함께 표시했다. B8은 8개 입력 batch 전체의 초이고 8로 나누지 않았다. 표시 라벨만 반올림했으며 모든 실제 bar 높이와 상태는 canonical CSV 원 수치와 exact 대사했다. CI·seed 분산·새 p90을 추가하지 않았다.\n\n"
        + CAPTION.replace("\n", " ") + "\n\n"
        "Raw는 동결 Chronos-Bolt C32와 optimizer0, G는 F 없이 채널 공유 512→32→12 GELU innovation·현재값 anchor다. 구조·학습량이 다르므로 이 그림을 B2·압축의 가치·Bolt 필요성의 충분한 증거로 확대하지 않는다. 실제 선택 refit 길이와 학습 상태는 원 표를 따른다. upstream 미래 보간·전처리 계보는 UNKNOWN이다.\n\n"
        "warm whole timer는 cold model load를 제외한다. 그림에서 CPU parameter+buffer resident·RSS·F 단독 resident를 대입하거나 새 측정하지 않는다. canonical audit와 보고의 입력 SHA를 재검사했고 원수치/trial 재집계·Task/원자료/NPZ/CP/모델 해석·학습·추론·비용 재측정·프로세스 제어·Git·network는 하지 않았다. 공개 SVG와 private cache PNG를 별도로 만들었다. 원 보고서가 먼저이며 그림 실패는 다른 실험의 중단 gate가 아니다. 전체 MASTER와 원 subset 전체 실험은 미완료다.\n")
    if not svg_path.parent.resolve().is_relative_to(scope.resolve()) or not preview.parent.resolve().is_relative_to(cache):
        raise ValueError("Literal figure output directories must stay within the declared roots")
    svg_path.parent.mkdir(exist_ok=True); preview.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((svg_path, svg), (preview, png), (md_path, md.encode("utf-8"))):
        with path.open("xb") as stream:
            stream.write(value)
    if not inputs.preserved():
        raise ValueError("Inputs changed; partial figure files preserved without a completed binding")
    proof = dict(schema="PEMS_RAW_G_FIGURE_BINDINGS_v1", status="FROM_ACTUAL_PEMS_RAW_G_AUDIT_" + audit["status"], actual_audit_status=audit["status"],
        plot_source_sha256=plot_sha, builder_source_sha256=BUILDER_SHA, auditor_source_sha256=AUDITOR_SHA, audit_sha256=audit_sha,
        report_bindings_sha256=report_sha, CSV_sha256=csv_sha, input_SHA_before_after_equal=True, source48_preserved=True,
        inputs=[dict(root="public", relative_path=name, sha256=inputs.files[("public", name)]) for name in sorted({name for name, _ in direct} | {row["relative_path"] for row in outputs})],
        transitive_input_manifests=[dict(root="public", relative_path=name, sha256=digest, field="inputs") for name, digest in ((audit_relative, audit_sha), (NAME + "/" + REPORT_BIND, report_sha))],
        outputs=[dict(relative_path=path.relative_to(scope).as_posix(), sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for path, value in ((svg_path, svg), (md_path, md.encode("utf-8")))],
        private_preview=dict(published=False, cache_relative_path=preview.relative_to(cache).as_posix(), sha256=hashlib.sha256(png).hexdigest(), bytes=len(png)),
        artist_values_from_unchanged_CSV=artists, artist_groups=4, artist_scalar_count=sum(len(artist["values"]) for artist in artists),
        verified_accuracy_rows=len(rows), missing_display_slots_are_not_accuracy_rows=True, CSV_precision_preserved=True, missing_values_drawn_as_zero=False,
        actual_renderer_executed=True, new_accuracy_or_cost_measurements=0, p90_or_CI_added=False, font=font, font_file_sha256=font_sha,
        CPU_BLAS_threads=1, outer_CPU_maximum=2, outer_RSS_maximum_bytes=3 * 1024**3, own_RSS_limit_bytes=style.MEMORY_LIMIT,
        sampled_own_peak_RSS_bytes=resources.peak, own_OS_peak_working_set_bytes=resources.os_peak, deadline_unix=style.DEADLINE,
        elapsed_seconds=time.perf_counter() - started, whole_MASTER_complete=False, full_subset_experiments_complete=False,
        independent_generalization_claimed=False, full_C170_or_B2_or_Bolt_necessity_proven=False, report_first=True, plot_failure_blocks_other_paths=False,
        self_hash_policy="Bindings excludes itself; final publication inventory binds its exact bytes")
    with binding_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return dict(status=proof["status"], public_files_written=3, private_previews_written=1, artist_groups=4, artist_scalar_count=proof["artist_scalar_count"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute", action="store_true"); parser.add_argument("--root-triggered", action="store_true")
    for key in ("public-root", "cache-root", "plot-sha", "report-binding-sha", "audit-relative", "audit-sha", "csv-sha"):
        parser.add_argument("--" + key)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED", input_reads=0, files_written=0, actual_render=0,
            numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules, matplotlib_imported="matplotlib" in sys.modules,
            future_actual_audit_sha256=None, future_report_binding_sha256=None, future_CSV_sha256=None))); return 0
    if not args.root_triggered or any(getattr(args, key) is None for key in ("public_root", "cache_root", "plot_sha", "report_binding_sha", "audit_relative", "audit_sha", "csv_sha")):
        parser.error("ROOT exact current plot, actual audit/report-binding/CSV SHA, explicit roots and fresh outputs required")
    print(json.dumps(execute(args.public_root, args.cache_root, args.plot_sha, args.report_binding_sha, args.audit_relative, args.audit_sha, args.csv_sha))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
