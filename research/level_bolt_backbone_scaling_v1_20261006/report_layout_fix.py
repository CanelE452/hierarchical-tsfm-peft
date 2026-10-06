"""Presentation-only headroom correction; sealed TEST/scoring/cost code unchanged."""
import shutil
from runtime import CACHE, HERE, Job, artifact, read_json, save_json
from report_controls import (ARMS, ALTERNATIVES, DATASETS, DATASET_LABELS, COLORS,
    MARKERS, MIB, LABELS, score, cost, load_sources, decide, point_frontier)

def make_figures(data, decisions, frontiers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    sources = {name: artifact(HERE / name) for name in
               ("accuracy_comparison.csv", "paired_comparisons.csv", "cost_comparison.csv")}
    manifest = {"schema": "level_bolt_allocation_figures_v1", "status": "generated_from_complete_tables",
                "sources": sources, "code": artifact(__file__), "no_inference": True,
                "no_resampling": True, "zero_axes": True,
                "whiskers": "Observed all-selected-instance/block range; not a confidence interval",
                "figure_rows": [], "decisions": decisions, "point_frontiers": frontiers}
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
        fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.20, wspace=.25)
        fig.suptitle(DATASET_LABELS[dataset] + ": fixed Chronos-Bolt allocation", y=.97, fontsize=15)
        used = []
        for arm in ARMS:
            a, c = score(data, dataset, arm), cost(data, dataset, arm)
            used.append({"dataset": dataset, "arm": arm,
                         "accuracy_csv_record": a["_csv_record"], "accuracy_csv_line": a["_csv_line"],
                         "cost_csv_record": c["_csv_record"], "cost_csv_line": c["_csv_line"],
                         "period": "combined", "batch": 4, "mse": a["mse"],
                         "origins_s": c["mean_seed_median_origins_s"],
                         "peak_allocated_bytes": c["mean_seed_median_peak_allocated_bytes"]})
            for ax, key, range_key, divisor in (
                    (axes[0], "mean_seed_median_origins_s", "block_median_origins_s_range", 1),
                    (axes[1], "mean_seed_median_peak_allocated_bytes", "allocated_range_bytes", MIB)):
                x = c[key]
                if x is None:
                    continue
                bounds = c.get(range_key)
                x /= divisor
                ax.scatter(x, a["mse"], color=COLORS[arm], marker=MARKERS[arm], s=85,
                           edgecolors="black", linewidths=.6, zorder=4, label=LABELS[arm])
                if bounds is not None:
                    ax.hlines(a["mse"], bounds[0] / divisor, bounds[1] / divisor,
                              color=COLORS[arm], linewidth=1.6, zorder=3)
                    ax.plot([bounds[0] / divisor, bounds[1] / divisor], [a["mse"], a["mse"]],
                            linestyle="none", marker="|", color=COLORS[arm], markersize=8)
        for ax in axes:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0, top=max(score(data, dataset, a)["mse"] for a in ARMS) * 1.12)
            ax.set_ylabel("Combined channel-macro MSE (lower is better)")
            ax.grid(True, color="#cccccc", linewidth=.6, alpha=.7)
        axes[0].set_title("Accuracy and batch-4 throughput")
        axes[0].set_xlabel("Batch-4 origins/s (higher is better)")
        axes[1].set_title("Accuracy and batch-4 allocated memory")
        axes[1].set_xlabel("Batch-4 peak allocated MiB (lower is better)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .905), ncol=4,
                   frameon=False, columnspacing=1.8)
        foot = ("FP32, TF32 off, RTX 4070; full CPU-input to CPU-output scope.\n"
                "Whiskers: observed block range, not CI. Previously exposed evaluation; fixed models/K.\n"
                "Sources: accuracy_comparison.csv " + sources["accuracy_comparison.csv"]["sha256"][:12]
                + "; cost_comparison.csv " + sources["cost_comparison.csv"]["sha256"][:12])
        fig.text(.5, .035, foot, ha="center", va="bottom", fontsize=8, color="#333333")
        outputs = []
        for suffix in ("png", "svg"):
            path = directory / f"allocation_{dataset}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white")
            outputs.append(artifact(path))
        plt.close(fig)
        manifest["figure_rows"].append({"dataset": dataset, "outputs": outputs, "plotted_rows": used})
    save_json(directory / "manifest.json", manifest)
    return manifest

if __name__ == "__main__":
    with Job("cpu", "publication_figure_headroom", metadata={"fit_count": 0,
             "model_inference": False, "bootstrap_draws": 0,
             "only_presentation_axes": True}) as job:
        destination = HERE / "figures" / "layout_correction.json"
        if destination.exists():
            raise FileExistsError("Headroom correction already completed")
        old = read_json(HERE / "figures" / "manifest.json")
        before = CACHE / "figures_before_layout_fix"
        before.mkdir(parents=True, exist_ok=False)
        for f in (HERE / "figures").iterdir():
            if f.is_file():
                shutil.copyfile(f, before / f.name)
        data = load_sources()
        decisions = [decide(data, d, a) for d in DATASETS for a in ALTERNATIVES]
        frontiers = {d: point_frontier(data, d) for d in DATASETS}
        new = make_figures(data, decisions, frontiers)
        assert new["sources"] == old["sources"]
        assert [r["plotted_rows"] for r in new["figure_rows"]] == [r["plotted_rows"] for r in old["figure_rows"]]
        new["original_report_code"] = old["code"]
        new["presentation_y_limits"] = {d: [0, max(score(data,d,a)["mse"] for a in ARMS)*1.12] for d in DATASETS}
        save_json(HERE / "figures" / "manifest.json", new)
        save_json(destination, {"reason": "Visual inspection found marker clipping at automatic y upper limit",
            "only_presentation_axes": True, "change": "Add12percent y headroom; zero axes retained",
            "scientific_seal_changed": False, "test_settings_changed": False,
            "fit_count": 0, "model_inference": False, "new_resampling": False,
            "original_report_code": old["code"], "renderer": artifact(__file__),
            "before_manifest": artifact(before / "manifest.json"),
            "after_manifest": artifact(HERE / "figures" / "manifest.json"),
            "identical_CSV_sources": old["sources"], "plotted_values_identical": True,
            "preserved_old_figures": [artifact(before / (d["dataset"].join(["allocation_", ".png"]))) for d in old["figure_rows"]],
            "expected_y_limits": new["presentation_y_limits"]})
        print("Presentation headroom corrected; same CSV sources and plotted values")
