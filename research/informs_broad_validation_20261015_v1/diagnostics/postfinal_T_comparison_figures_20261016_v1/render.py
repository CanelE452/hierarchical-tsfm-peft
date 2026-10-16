"""Plot preselected T roles from the audited flat score table; no model execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

DATASETS = ("beijing", "jena", "electricity", "pems08")
ROLES = ("G", "Hpast", "H_B0", "H_B2", "RawH", "AlphaFit", "RawSmall", "LoRA", "Ridge")
PINS = {
    "SCORES.csv": "715531ac419070c91fbdbf9d0c4defb4af78ec0c539b4974ef830fc64b703766",
    "SELECTION_LOCK.json": "c3493746a14b3ac7b6291841847a25a74cb3060d059ee9962929bd13390a51f1",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role(spec):
    family = spec["family"]
    if family == "DIRECT_G":
        return "G"
    if family == "ALPHA":
        return "AlphaFit"
    if family == "COMBINATION":
        if spec["variant"] == "past_control":
            return "Hpast"
        if spec["variant"] == "conditional":
            return {"B0": "H_B0", "B2": "H_B2", "RAW": "RawH"}.get(spec["b_route"]["family"])
    if family == "RAW" and spec["model_size"] == "small":
        return "RawSmall"
    return {"LORA": "LoRA", "RAW_RIDGE": "Ridge"}.get(family)


def prepare(public):
    for name, digest in PINS.items():
        if sha(public / name) != digest:
            raise ValueError("Frozen chart input changed: " + name)
    lock = json.loads((public / "SELECTION_LOCK.json").read_bytes())
    if not lock["locked"]:
        raise ValueError("T selection is not locked")
    with (public / "SCORES.csv").open(encoding="utf-8-sig", newline="") as stream:
        scores = {row["run_id"]: row for row in csv.DictReader(stream) if row["split"] == "T_final"}
    if set(scores) != set(lock["allowed_run_ids"]) or len(scores) != 46:
        raise ValueError("Score IDs differ from the exact frozen T46")
    found = {}
    for rid, binding in lock["route_bindings"].items():
        spec = binding["route_spec"]
        name = role(spec)
        if name is None:
            continue
        key = (spec["dataset"], name)
        if key in found:
            raise ValueError("Ambiguous predeclared role: " + str(key))
        row = scores[rid]
        for metric in ("mse", "mae"):
            if not math.isfinite(float(row[metric])) or float(row[metric]) < 0:
                raise ValueError("Invalid audited metric")
        found[key] = dict(dataset=key[0], role=name, **{field: row[field] for field in (
            "run_id", "split", "cohort", "mse", "mae", "n_observed", "n_origins", "num_channels", "horizon", "model_sha", "scaler_sha")})
    if set(found) != {(dataset, name) for dataset in DATASETS for name in ROLES}:
        raise ValueError("Missing or unexpected predeclared chart role")
    for dataset in DATASETS:
        g = lock["route_bindings"][found[dataset, "G"]["run_id"]]["route_spec"]
        for name in ROLES[1:6]:
            spec = lock["route_bindings"][found[dataset, name]["run_id"]]["route_spec"]
            if spec["g_route"] != g:
                raise ValueError("Chart comparison changes the fixed G")
        if len({(found[dataset, name]["cohort"], found[dataset, name]["scaler_sha"], found[dataset, name]["n_observed"]) for name in ROLES}) != 1:
            raise ValueError("Chart conditions have different target cohorts")
    return [found[dataset, name] for dataset in DATASETS for name in ROLES]


def render(public, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(__file__).resolve().parent / ("snapshot_" + str(time.time_ns()))
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path"})
    colors = ("#777777", "#56B4E9", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222", "#E69F00", "#9467BD")
    for metric in ("mse", "mae"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, dataset in zip(axes.flat, DATASETS):
            group = [row for row in rows if row["dataset"] == dataset]
            values = [float(row[metric]) for row in group]
            bars = ax.bar(range(len(ROLES)), values, color=colors)
            for bar in list(bars)[6:]:
                bar.set_hatch("//")
            ax.set_xticks(range(len(ROLES)), ROLES, rotation=27, ha="right", fontsize=8)
            ax.set_ylim(0, max(values) * 1.2)
            ax.set_ylabel("A60-normalized channel-macro " + metric.upper())
            first = group[0]
            ax.set_title(dataset + " / origins=" + first["n_origins"] + " / C=" + first["num_channels"] + " / H=" + first["horizon"])
            ax.grid(axis="y", alpha=.2)
            for index, value in enumerate(values):
                ax.annotate(f"{value:.4f}", (index, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        fig.suptitle("Locked T46: fixed G controls and strong separate alternatives / " + metric.upper(), fontsize=14)
        fig.text(.02, .025, "First six roles share exactly the same frozen G. Hatched Raw/LoRA/Ridge are separate practical alternatives.\n"
                 "Fixed role order; every predeclared role retained. One seed / one cutoff / no CI; no T-dependent model or case selection.", fontsize=9)
        fig.subplots_adjust(left=.08, right=.98, top=.9, bottom=.14, hspace=.43, wspace=.26)
        for suffix in ("png", "svg"):
            fig.savefig(output / ("locked_T_" + metric.upper() + "." + suffix), dpi=300, facecolor="white")
        plt.close(fig)
    with (output / "PLOT_VALUES.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, digest in PINS.items():
        if sha(public / name) != digest:
            raise ValueError("Chart input changed during rendering")
    manifest = dict(status="COMPLETE_PREDECLARED_T_ROLE_PLOTS", rows=36, chart_roles=ROLES,
        input_sha256=PINS, source_sha256=sha(Path(__file__)), split="T_final", source_scores_independently_audited=True,
        source_artifacts_modified=False, GPU_or_NPZ_models_accessed=False, inference_or_training=False,
        T_method_K_seed_or_case_selection=False, confidence_intervals_or_generalization_claimed=False,
        outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(output.iterdir())})
    with (output / "FIGURE_MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    public = args.public_root.resolve()
    rows = prepare(public)
    if args.render:
        print(render(public, rows).relative_to(public).as_posix())
    else:
        print(json.dumps(dict(status="METADATA_ONLY_NO_RENDER", rows=len(rows), roles=ROLES, source_sha256=sha(Path(__file__)))))
