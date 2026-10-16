"""Render aggregate report CSV only; no datasets, checkpoints or models."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("csv", "svg", "preview"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY", input_reads=0, plots_created=0)))
        return 0
    if not all((args.csv, args.svg, args.preview)):
        parser.error("--execute requires aggregate --csv and new --svg/--preview paths")
    if any(path.exists() for path in (args.svg, args.preview)):
        raise FileExistsError("Existing figures are preserved")
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    def selected(dataset, role):
        found = [row for row in rows if row["dataset"] == dataset and row["role"] == role]
        if len(found) != 1:
            raise ValueError("Exactly one aggregate comparison row required: " + role)
        return found[0]

    jena = [
        (92601, selected("jena", "R3_B0"), selected("jena", "R4_B2")),
        (92602, selected("jena", "B0_SEED92602"), selected("jena", "B2_SEED92602")),
    ]
    parent = selected("pems08", "B2_PARENT_CPU_FULL")
    head = selected("pems08", "FIXED_B2_NO_BOLT_HEAD")
    if any(row["accuracy_device"] != "cpu" for row in (parent, head)):
        raise ValueError("The PeMS paired plot requires full V accuracy on CPU for both routes")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.18, hspace=.83, wspace=.34)
    fig.suptitle("INFORMS priority follow-up: development V", fontsize=16, y=.985)
    fig.text(.09, .93, "Jena: same V43, H=48, C=21, K=6; two training seeds only", fontsize=11)
    colors, markers = ("#0072B2", "#D55E00"), ("o", "s")
    for axis, metric, title in zip(axes[0], ("MSE", "MAE"), ("A. Normalized channel-macro MSE", "B. Normalized channel-macro MAE")):
        values = [[float(row[metric]) for row in pair[1:]] for pair in jena]
        for index, (seed, baseline, candidate) in enumerate(jena):
            axis.plot([0, 1], values[index], marker=markers[index], color=colors[index], lw=1.8, ms=6, label=str(seed))
            for method in (0, 1):
                value = values[index][method]
                other = values[1-index][method]
                offset = 8 if value >= other else -16
                axis.annotate(f"{value:.6f}", (method, value), xytext=(0, offset), textcoords="offset points",
                              ha="center", fontsize=9, color=colors[index])
        axis.set_xticks([0, 1], ["B0", "B2"])
        axis.set_xlim(-.3, 1.3)
        axis.set_ylim(min(map(min, values))*.955, max(map(max, values))*1.045)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend(title="Training seed", frameon=False, loc="lower left")
    epochs = "; ".join(f"{seed}: {baseline['selected_refit_epoch']}/{candidate['selected_refit_epoch']}"
                       for seed, baseline, candidate in jena)
    fig.text(.09, .555, "A_stop-selected B0/B2 refit epochs — " + epochs + ". No convergence or independent-test claim.", fontsize=9)

    fig.text(.09, .485, "PeMS08: same frozen B2 P/E/D and full V148 on CPU; H=12, C=170, K=43", fontsize=11)
    mse = [float(parent["MSE"]), float(head["MSE"])]
    bars = axes[1, 0].bar([0, 1], mse, color=[colors[0], "#7F7F7F"], width=.6)
    bars[1].set_hatch("//")
    axes[1, 0].set_xticks([0, 1], ["Frozen Bolt", "No-Bolt linear head"])
    axes[1, 0].set_ylim(0, max(mse)*1.28)
    axes[1, 0].set_ylabel("Normalized channel-macro MSE")
    axes[1, 0].set_title("C. Accuracy with fixed final P/E/D", loc="left")
    axes[1, 0].grid(axis="y", alpha=.2)
    for bar, value in zip(bars, mse):
        axes[1, 0].annotate(f"{value:.6f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=9)

    x = np.arange(2)
    for index, (label, row, color) in enumerate((("Frozen Bolt", parent, colors[0]), ("Linear head", head, "#7F7F7F"))):
        latencies = [1000*float(row[f"B{batch}_median_seconds"]) for batch in (1, 8)]
        bars = axes[1, 1].bar(x+(index-.5)*.34, latencies, width=.34, color=color, label=label,
                              hatch="//" if index else None)
        for bar, value in zip(bars, latencies):
            axes[1, 1].annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 4),
                              textcoords="offset points", ha="center", fontsize=8)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(.25, 10000)
    axes[1, 1].set_xticks(x, ["Batch 1", "Batch 8"])
    axes[1, 1].set_ylabel("CPU2 latency (ms / batch; log)")
    axes[1, 1].set_title("D. Actual full-path inference cost", loc="left")
    axes[1, 1].legend(frameon=False, loc="upper left", fontsize=8)
    axes[1, 1].grid(axis="y", alpha=.2)
    fig.text(.09, .085, "PeMS parent: 1-epoch limited screen; fixed head adds 3 epochs. Training and model loading are separate.\n"
             "Cost: 10 warmups / 20 timed / 20 separate stage probes, CPU2, batch latency.\n"
             "Both datasets reuse development V. Results do not establish independent generalization or overall superiority.", fontsize=9, va="bottom")
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    for path in (args.svg, args.preview):
        path.parent.mkdir(parents=True, exist_ok=True)
    svg = io.StringIO()
    fig.savefig(svg, format="svg", facecolor="white", metadata={"Description": "Aggregate CSV SHA256="+digest})
    with args.svg.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(line.rstrip() for line in svg.getvalue().splitlines())+"\n")
    fig.savefig(args.preview, format="png", dpi=150, facecolor="white", metadata={"Description": "Aggregate CSV SHA256="+digest})
    plt.close(fig)
    print(json.dumps(dict(status="CREATED", aggregate_csv_sha256=digest, plots=1, exports=2, model_or_raw_data_reads=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
