"""Capture the completed E3 A_stop milestone, or redraw its public CSV only."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
NAME = HERE.name
PLAN_SHA = "fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35"
COLORS = {"B0": "#666666", "B2": "#0072B2", "LAG_CAUSAL": "#D55E00", "LAG_BIDIR": "#009E73"}
LABELS = {"B0": "B0: pointwise", "B2": "B2: full-window mixer", "LAG_CAUSAL": "Past-directed lags",
          "LAG_BIDIR": "Within-window bidirectional lags"}
EPOCHS = (0., .25, 1., 3.)
UPDATES = (0, 156, 622, 1863)
EXPOSURES = (0, 2481, 9923, 29769)
CURVE_FIELDS = ("epoch", "exposures", "updates", "astop_mse", "segment_seconds", "clipped_steps")
FIELDS = ("run_id", "family", *CURVE_FIELDS, "selected_by_Astop")
SOURCE_NAMES = ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")
CSV_NAME = "SEARCH_CURVES_THROUGH_E3.csv"
SVG_NAME = "SEARCH_MILESTONE_E3.svg"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run_id(family):
    return "postT_cpu_jena_E3_" + family + "_k6_s92601_v1"


def selected_curve(curves):
    best, value = None, math.inf
    for row in curves:
        if row["astop_mse"] < value:
            best, value = row, row["astop_mse"]
    return best


def validate_curves(curves):
    if len(curves) != 4 or [row.get("epoch") for row in curves] != list(EPOCHS):
        raise ValueError("Exactly the completed 0/0.25/1/3 endpoints are required")
    for index, row in enumerate(curves):
        if set(row) != set(CURVE_FIELDS):
            raise ValueError("Original curve fields differ")
        if any(isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) for key in CURVE_FIELDS):
            raise ValueError("Curve values must be finite numeric scalars")
        if row["updates"] != UPDATES[index] or row["exposures"] != EXPOSURES[index]:
            raise ValueError("The fixed endpoint update/exposure schedule differs")
        if row["astop_mse"] < 0 or row["segment_seconds"] < 0 or int(row["clipped_steps"]) != row["clipped_steps"] or row["clipped_steps"] < 0:
            raise ValueError("Negative metric/time or invalid clipping count")
        segment_updates = UPDATES[index] - (UPDATES[index-1] if index else 0)
        if row["clipped_steps"] > segment_updates or (index == 0 and row["segment_seconds"] != 0):
            raise ValueError("Initial or segment clipping/time contract differs")
    return selected_curve(curves)


def expected_contract(plan, condition):
    task = plan["task_contract"]
    axes = dict(L=task["L"], H=task["H"], C=task["C"], columns=task["columns"],
                fold=condition["spec"]["fold"], source_array_fingerprints=task["array_sha256"])
    return dict(source_sha=fingerprint({name: plan["source39"][name] for name in SOURCE_NAMES}),
                config_sha=fingerprint(dict(spec=condition["spec"], axes=axes)), data_sha=task["data_sha256"],
                model_sha=plan["foundation_tensor_sha256"], scaler_sha=task["search_scaler_sha256"])


def validate_events(events, plan, condition):
    if len(events) != 4:
        raise ValueError("All four immutable endpoint events are required")
    contract = events[-1]["contract"]
    expected = expected_contract(plan, condition)
    if (set(contract) != set(expected) | {"initialization_sha"} or any(contract.get(key) != value for key, value in expected.items())
            or len(contract["initialization_sha"]) != 64):
        raise ValueError("Completed checkpoint source/config/data/model/scaler contract differs")
    for index, event in enumerate(events):
        if (event["status"], event["epoch"], event["updates"], event["exposures"], event["contract"]) != (
                "SEGMENT_COMPLETE", EPOCHS[index], UPDATES[index], EXPOSURES[index], contract):
            raise ValueError("Checkpoint and completed endpoint event differ")
    return contract


def validate_saved(saved, events, plan, condition):
    if saved["contract"] != validate_events(events, plan, condition):
        raise ValueError("Saved checkpoint and completed events have different contracts")
    if saved["updates"] != 1863 or saved["sampler"]["exposures"] != 29769 or saved["extra"]["completed_epochs"] != list(EPOCHS):
        raise ValueError("Incomplete checkpoint endpoint or sampler")
    curves = saved["extra"]["curves"]
    selected = validate_curves(curves)
    if saved["extra"]["best_epoch"] != selected["epoch"] or saved["extra"]["best_mse"] != selected["astop_mse"]:
        raise ValueError("Saved best checkpoint differs from original strict-less-than A_stop selection")
    return [dict(run_id=condition["run_id"], family=condition["family"], **row,
                 selected_by_Astop=int(row is selected)) for row in curves]


def read_records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Public aggregate CSV schema differs")
        records = list(reader)
    if len(records) != 16 or {row["family"] for row in records} != set(COLORS):
        raise ValueError("Exactly four families and sixteen endpoint rows are required")
    result = []
    for family in COLORS:
        rows = [row for row in records if row["family"] == family]
        if any(row["run_id"] != run_id(family) for row in rows):
            raise ValueError("Public CSV run identity differs")
        curves = [{key: float(row[key]) for key in CURVE_FIELDS} for row in rows]
        selected = validate_curves(curves)
        for row, curve in zip(rows, curves):
            if row["selected_by_Astop"] not in {"0", "1"} or int(row["selected_by_Astop"]) != int(curve is selected):
                raise ValueError("Public CSV selection marker differs")
            result.append(dict(run_id=row["run_id"], family=family, **curve, selected_by_Astop=int(row["selected_by_Astop"])))
    return result


def require_new_output(output, public=None, preview=None, cache=None):
    if output.exists():
        raise FileExistsError("Preserve previous snapshot directory")
    if public is not None and (output == public / NAME / "diagnostics" or not output.is_relative_to(public / NAME / "diagnostics")):
        raise ValueError("Capture output must be a new descendant of the separate diagnostic root")
    if preview is not None:
        if preview.suffix.lower() != ".png" or preview.exists() or not preview.parent.is_dir():
            raise ValueError("Preview must be a new PNG in an existing private directory")
        if cache is not None and not preview.is_relative_to(cache):
            raise ValueError("Capture preview must stay in the explicit private cache root")


def draw(csv_path, output, preview=None):
    rows = read_records(csv_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "search-milestone-e3-v1"})
    fig, (axis, info) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.25, 1.25]})
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.24, wspace=.22)
    for family, color in COLORS.items():
        points = [row for row in rows if row["family"] == family]
        axis.plot([row["updates"] for row in points], [row["astop_mse"] for row in points], marker="o", markersize=5,
                  linewidth=2, color=color, label=LABELS[family])
        selected = next(row for row in points if row["selected_by_Astop"])
        axis.scatter(selected["updates"], selected["astop_mse"], s=125, marker="D", facecolors="none", edgecolors=color, linewidths=1.7, zorder=4)
    values = [row["astop_mse"] for row in rows]
    padding = max((max(values)-min(values))*.10, .01)
    axis.set(xlabel="Completed optimizer updates (same exposure schedule)", ylabel="A_stop channel-macro MSE",
             xticks=list(UPDATES), xticklabels=["0\ninit", "156\n0.25 ep", "622\n1 ep", "1863\n3 ep"],
             ylim=(max(0, min(values)-padding), max(values)+padding))
    axis.grid(axis="y", alpha=.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    info.axis("off")
    info.text(0, 1, "MATCHED CPU DEVELOPMENT\n\nJena: L=512, H=48, C=21, K=6\nSeed=92601; frozen Bolt-small\nSame decoder, loss and exposure schedule\nSearch complete through 3 epochs\n29,769 origin exposures per condition\n\nDiamond: A_stop-selected endpoint", va="top", fontsize=9.2, linespacing=1.4)
    table_rows = []
    for family in COLORS:
        points = [row for row in rows if row["family"] == family]
        selected = next(row for row in points if row["selected_by_Astop"])
        table_rows.append([{"LAG_CAUSAL": "Past lags", "LAG_BIDIR": "Bidir lags"}.get(family, family),
                           "%.6f" % points[-1]["astop_mse"], "%g / %.6f" % (selected["epoch"], selected["astop_mse"])])
    table = info.table(cellText=table_rows, colLabels=["Encoder", "Last MSE", "Selected epoch / MSE"],
                       colWidths=[.25, .25, .50], cellLoc="left", colLoc="left", bbox=[0, .02, 1, .36])
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#dddddd")
    fig.suptitle("Completed 3-epoch search — development A_stop milestone", x=.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(.07, .045, "Post-T exploratory development; one seed, no uncertainty interval. Different encoder capacities and boundaries.\n"
             "This A_stop snapshot establishes no final V result, convergence, deployment cost, independent generalization or original GPU-gap completion.", fontsize=9)
    svg = output / SVG_NAME
    with svg.open("xb") as stream:
        fig.savefig(stream, format="svg", metadata={"Date": None}, bbox_inches="tight")
    if preview is not None:
        with preview.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(svg_sha256=sha(svg), public_csv_sha256=sha(csv_path), private_preview_sha256=sha(preview) if preview else None)


def capture(public, cache, asset, plan_sha):
    if plan_sha != PLAN_SHA:
        raise ValueError("An explicit original frozen E3 plan SHA is required")
    spec = importlib.util.spec_from_file_location("milestone_original_cpu_runner", HERE / "cpu_lag_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    plan = runner.metadata(public, cache, asset, plan_sha)
    snapshots = []
    for condition in plan["conditions"]:
        identity = condition["run_id"]
        paths = [public / NAME / "runs" / identity / "search" / ("event_%06d.json" % index) for index in range(4)]
        event_hashes = [sha(path) for path in paths]
        events = [read(path) for path in paths]
        validate_events(events, plan, condition)
        checkpoint = cache / NAME / "runs" / identity / "search/checkpoint_000003.pt"
        snapshots.append((condition, paths, event_hashes, events, checkpoint, sha(checkpoint)))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    import torch
    torch.set_num_threads(2)
    records, inputs = [], []
    for condition, paths, event_hashes, events, checkpoint, checkpoint_hash in snapshots:
        identity = condition["run_id"]
        if sha(checkpoint) != checkpoint_hash or [sha(path) for path in paths] != event_hashes:
            raise ValueError("Completed snapshot/event bytes changed before deserialization")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        records.extend(validate_saved(saved, events, plan, condition))
        if sha(checkpoint) != checkpoint_hash or [sha(path) for path in paths] != event_hashes:
            raise ValueError("Completed snapshot/event bytes changed while capturing")
        inputs.append(dict(run_id=identity, checkpoint_cache_relative_path=checkpoint.relative_to(cache).as_posix(),
            checkpoint_sha256=checkpoint_hash, contract=saved["contract"], events=[dict(public_relative_path=path.relative_to(public).as_posix(),
            sha256=value) for path, value in zip(paths, event_hashes)]))
        del saved
    for _, paths, event_hashes, _, checkpoint, checkpoint_hash in snapshots:
        if sha(checkpoint) != checkpoint_hash or [sha(path) for path in paths] != event_hashes:
            raise ValueError("An input changed before the full capture completed")
    runner.metadata(public, cache, asset, plan_sha)
    return records, dict(status="CAPTURED_COMPLETED_E3_SEARCH_METADATA_ONLY", inputs=inputs, plan_sha256=plan_sha,
        runner_sha256=sha(HERE / "cpu_lag_runner.py"), capture_source_sha256=sha(__file__), captured_unix=time.time(),
        actual_completed_checkpoint_deserializations=4, actual_foundation_or_model_construction_or_forward_or_fit=0,
        actual_Task_or_target_array_load=0, checkpoint_weights_published=False, post_T_exploratory_development=True,
        original_source39_immutable9_and_postT_registry_preserved=True, final_V_result=False, convergence_established=False,
        deployment_cost_measured=False, independent_holdout_validation=False, original_GPU_gap_completed=False,
        selection_rule="Original strict-less-than A_stop minimum over initial0,0.25,1,3; earliest tied endpoint retained")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public", "cache", "asset"):
        parser.add_argument("--" + root + "-root", type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args(argv)
    if not args.capture and args.input_csv is None:
        print(json.dumps(dict(status="METADATA_ONLY_USE_CAPTURE_OR_PUBLIC_CSV", endpoints=list(EPOCHS), expected_rows=16,
            torch_imported="torch" in sys.modules, actual_science_or_array_access=0, writes=0)))
        return
    if args.output is None or (args.capture and args.input_csv is not None):
        parser.error("Choose capture or public CSV and supply a new output directory")
    output, preview = args.output.resolve(), args.preview.resolve() if args.preview else None
    if args.capture:
        if not all((args.public_root, args.cache_root, args.asset_root, args.plan_sha)):
            parser.error("Capture requires original explicit roots and frozen plan SHA")
        public, cache, asset = [path.resolve() for path in (args.public_root, args.cache_root, args.asset_root)]
        require_new_output(output, public, preview, cache)
        records, evidence = capture(public, cache, asset, args.plan_sha)
        output.mkdir(parents=True, exist_ok=False)
        csv_path = output / CSV_NAME
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        new(output / "EVIDENCE.json", dict(evidence, public_csv_sha256=sha(csv_path)))
    else:
        require_new_output(output, preview=preview)
        csv_path = args.input_csv.resolve()
        read_records(csv_path)
        output.mkdir(parents=True, exist_ok=False)
    result = draw(csv_path, output, preview)
    new(output / "FIGURE_BINDING.json", dict(status="RENDERED_PUBLIC_E3_AGGREGATE", **result,
        source_sha256=sha(__file__), csv_basename=csv_path.name, svg_basename=SVG_NAME, row_count=16,
        one_seed=True, uncertainty_interval=False, final_V_result=False, post_T_exploratory_development=True,
        checkpoint_deserializations_in_this_invocation=4 if args.capture else 0))
    print(json.dumps(dict(status="E3_SEARCH_MILESTONE_CAPTURE_OR_RENDER_COMPLETE", **result,
                         torch_imported="torch" in sys.modules)))


if __name__ == "__main__":
    main()
